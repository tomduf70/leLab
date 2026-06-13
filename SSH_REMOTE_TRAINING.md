# SSH remote training runner

This fork adds a third training **compute target** to LeLab, alongside `local`
(this machine) and `hf_cloud` (paid HF Jobs): **`ssh_remote`**, which dispatches
a training to a remote GPU host over SSH.

In this setup the LeLab UI runs on a **Jetson Orin** (`si-robotics-desktop`,
slow for training) and the training is deported to a **dedicated PC**
(`robotic-ai`, RTX 5060 Ti 16 GB). The dataset and checkpoints travel through
the **Hugging Face Hub** — SSH is used *only* to launch and monitor the remote
`lerobot-train` process.

## How it works

```
┌─────────────── Orin (LeLab UI :8000) ───────────────┐        ┌──── robotic-ai (RTX 5060 Ti) ────┐
│ SshJobRunner (lelab/runners/ssh_remote.py)          │  SSH   │ ~/projects/lerobot-test          │
│  • ships launch.sh + wrapper.py (base64 over stdin) │ ─────► │   .venv/bin/lerobot-train        │
│  • setsid nohup → detached training                 │        │   (detached, survives SSH drop)  │
│  • tail -F log  → live logs + metrics               │ ◄───── │   writes pid / rc / log files    │
│  • poll pidfile/rc → liveness + exit code           │        │   pushes checkpoints → HF Hub    │
└──────────────────────┬──────────────────────────────┘        └──────────────┬───────────────────┘
                       │  lists checkpoints from the Hub        pulls dataset  │
                       └───────────────►  Hugging Face Hub  ◄──────────────────┘
```

- **Detached execution.** The trainer runs under `setsid nohup`, so it survives
  an SSH drop or even an Orin restart. Two worker threads tail its log and poll
  its liveness; on restart, `JobRegistry` re-attaches (everything derives from
  the job id, nothing extra is persisted).
- **Hub-based sync** (same as the cloud runner). The remote pulls the dataset by
  `--dataset.repo_id`; checkpoints are pushed back to the Hub under
  `checkpoints/<step>/pretrained_model/` by the shared uploader wrapper, so the
  LeLab UI browses them live — identical to HF Cloud jobs.
- **No shell-quoting hell.** `launch.sh` and `wrapper.py` are shipped
  base64-encoded over SSH stdin; remote paths are absolute (no `$HOME`/tilde
  expansion needed).
- **One job at a time** on the remote GPU (enforced in `JobRegistry.start`).

Key files: [`lelab/runners/ssh_remote.py`](lelab/runners/ssh_remote.py) (the
runner), [`lelab/jobs.py`](lelab/jobs.py) (`JobTarget`/`JobRecord`/`JobRegistry`
wiring), [`frontend/src/components/training/config/TargetCard.tsx`](frontend/src/components/training/config/TargetCard.tsx)
(the UI option).

## One-time setup

### 1. Orin → robotic-ai SSH access

The runner shells out to `ssh robotic-ai …`, so that host alias must resolve and
authenticate non-interactively from the Orin.

```bash
# cloudflared (tunnel to robotic-ai.iscol.fr) — arm64 via Cloudflare's apt repo
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared jammy main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install -y cloudflared

# SSH key (no passphrase, so the runner can launch non-interactively)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "orin-lelab-runner@si-robotics-desktop"
```

`~/.ssh/config` on the Orin:

```
Host robotic-ai
    HostName robotic-ai.iscol.fr
    User tomduf
    ProxyCommand cloudflared access ssh --hostname %h
    IdentityFile ~/.ssh/id_ed25519
```

Then authorize the key on robotic-ai (run from a machine that already has access):

```bash
ssh robotic-ai 'umask 077; mkdir -p ~/.ssh && echo "<contents of orin ~/.ssh/id_ed25519.pub>" >> ~/.ssh/authorized_keys'
```

Verify: `ssh robotic-ai hostname` should print `robotic-ai` with no prompt.

### 2. robotic-ai prerequisites

- `lerobot` installed in `~/projects/lerobot-test/.venv` (console script
  `.venv/bin/lerobot-train`). Currently **lerobot 0.5.0**, Python 3.12.
- **FFmpeg 6** for video decoding (`torchcodec` needs it). On Ubuntu 24.04 a
  bare `libavutil`/`libavcodec` is not enough — install the full set, or
  `torchcodec` fails to load any core and training aborts at the first batch:
  ```bash
  sudo apt-get install -y ffmpeg   # pulls libavformat60, libavfilter9, libswscale7, libavdevice60…
  ```
- A **Hugging Face token with write access** (`hf auth login`) so checkpoints
  push to the Hub.
- For W&B logging: `wandb login` on robotic-ai (key in `~/.netrc`).

The Orin also needs an HF token (`hf auth login`) so the UI can list the pushed
checkpoints. Both machines should be the same HF account.

### 3. Hard-coded remote layout (MVP)

These constants live at the top of
[`lelab/runners/ssh_remote.py`](lelab/runners/ssh_remote.py) — change them to
target a different host:

| Constant | Value |
|---|---|
| `SSH_HOST` | `robotic-ai` |
| `REMOTE_HOME` | `/home/tomduf` |
| `REMOTE_PROJECT_DIR` | `/home/tomduf/projects/lerobot-test` |
| `REMOTE_TRAIN_BIN` | `…/.venv/bin/lerobot-train` |
| `REMOTE_PYTHON` | `…/.venv/bin/python` |
| `REMOTE_HF_USER` | `tomduf70` |

## Using it

From the LeLab UI: **Training → Compute target → "Remote — robotic-ai · RTX
5060 Ti"**, pick a dataset, **Start**. You're taken to the live monitoring page
(logs + loss/step curves). Checkpoints appear under the job once pushed.

## Accessing the UI remotely (from the Mac)

LeLab binds `127.0.0.1:8000`. **Do not** open it by IP over plain HTTP — the
frontend calls `crypto.randomUUID()`, which browsers only expose in a *secure
context* (HTTPS or `localhost`), so an IP origin renders a blank page. Use an SSH
tunnel and open it via `localhost`:

```bash
# on the Mac
ssh -N -L 8000:127.0.0.1:8000 admin-si@192.168.1.134
# then open http://localhost:8000   (localhost, NOT the IP)
```

## Troubleshooting

- **Training aborts at the first batch with a `torchcodec` / `libav*.so` error** →
  FFmpeg is incomplete on robotic-ai; `sudo apt-get install -y ffmpeg`.
- **`EvalConfig: eval batch size > eval episodes`** → handled: the runner clamps
  `eval_batch_size` to `eval_n_episodes` (lerobot 0.5 validates this even when
  eval is disabled).
- **Checkpoints show empty right after a run** → 30 s list cache; they appear on
  the next poll.
- **Blank UI page from the Mac** → you opened it by IP; use the SSH tunnel +
  `localhost` (see above).

## Known limitations / follow-ups

- Remote host/paths/HF-user are hard-coded constants (no UI/env config yet).
- `torchcodec` is the fast default once FFmpeg is fixed; pyav is a fallback.
- `dist/` is rebuilt by CI on `main`; no need to commit it from a feature branch.
