# LeLab sur Jetson Orin — nœud d'inférence GPU (conteneur)

Comment faire tourner LeLab **avec GPU** sur le Jetson Orin NX (Seeed reComputer,
JetPack 6.2.1 / L4T r36.4.3, CUDA 12.6).

## Pourquoi un conteneur (et pas du natif)

Validé le 2026-07-13 : l'inférence GPU en **natif est impossible** sur cette carte.

- lerobot 0.6 **et** lelab exigent **Python ≥ 3.12**.
- Les wheels PyTorch-CUDA pour Orin (`pypi.jetson-ai-lab.io/jp6/cu126`) sont
  **cp310 uniquement** — aucune cp312.
- Aucun Python natif ne satisfait donc les deux à la fois. (En natif py3.10 on a
  bien un torch-CUDA fonctionnel, mais lelab refuse de s'installer : py<3.12.)

L'écosystème NVIDIA/dusty-nv résout ça par le **conteneur** : l'image
`dustynv/lerobot:r36.4.0-cu128-24.04` est basée sur Ubuntu 24.04 → **Python 3.12**
avec un **torch-CUDA cp312** compilé dans l'image. Sur Linux natif, Docker n'est
pas une VM (namespaces/cgroups) : **aucune perte de perf** — GPU, USB et réseau
sont en accès direct.

## L'image `lelab-orin`

Construite en committant le conteneur validé. Contenu vérifié :

| Composant | Version | GPU |
|---|---|---|
| Python | 3.12.3 | — |
| torch | 2.7.0 (cu128) | **CUDA True, device "Orin"** ✓ |
| lerobot | 0.6.0 | — |
| transformers | 5.5.4 | (corrigé, voir plus bas) |
| torchvision / numpy | 0.22.0 / 2.2.5 | interop numpy↔cuda OK ✓ |
| lelab | éditable depuis `/opt/lelab` | serveur HTTP 200 ✓ |

Reconstruction si perdue : voir [`Dockerfile`](./Dockerfile).

### Le correctif transformers (important)

lerobot 0.6 impose `huggingface-hub>=1.0`, mais le `transformers` 4.51 de l'image
de base exige `hub<1.0` → `ImportError` au démarrage du serveur. Corrigé en
montant transformers dans la plage attendue par lerobot 0.6 :
`pip install -U 'transformers>=5.4.0,<5.6.0'` (→ 5.5.4). Déjà appliqué dans
`lelab-orin`.

## Lancer

```bash
# 1. libérer le port 8000 : couper le service natif (sans GPU)
systemctl --user disable --now lelab.service

# 2. lancer le conteneur GPU
./deploy/orin/run.sh
```

Accès : `http://localhost:8000` (tunnel Cloudflare `lelab.iscol.fr` ou Tailscale
`100.91.247.98` — jamais par IP locale). Voir [`run.sh`](./run.sh) pour les
variables (`LELAB_PORT`, etc.).

## Données & auth HF — tout vit sur l'hôte (persistant)

Le conteneur est **jetable** : `docker rm` ne perd aucune donnée. Tout ce qui
compte est monté depuis l'hôte et survit à la destruction du conteneur :

| Hôte | Conteneur | Contenu |
|---|---|---|
| `~/lelab-dev` | `/opt/lelab` | code (éditable) |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | token HF, calibrations, datasets, cache modèles |

**Piège HF_HOME (corrigé dans `run.sh`)** : l'image de base fixe
`HF_HOME=/data/models/huggingface` (chemin interne jetable). Sans override,
huggingface_hub et lerobot cherchent token/calibrations/datasets à ce chemin →
token « non configuré », calibrations introuvables (alors que les fichiers sont
bien montés !). `run.sh` repointe `HF_HOME` et `HF_LEROBOT_HOME` vers le cache
monté. Vérifié : `whoami → tomduf70`, calibrations des 2 bras détectées, datasets
enregistrés écrits dans `~/.cache/huggingface/lerobot` (sur l'hôte).

## Passage du hardware — À VALIDER

`run.sh` monte `/dev` + autorise les majeurs via cgroup (166=ttyACM bras,
81=video caméras, 188=ttyUSB), ce qui **survit au hotplug** (rebrancher le bras
sans relancer le conteneur).

> ⚠️ **Non testé en conditions réelles** : au 2026-07-13 le bras SO-101 et les
> caméras étaient rangés (voyage). Le GPU, l'install et le serveur sont validés ;
> le pont USB/caméras reste à confirmer au retour. Marche à suivre : rebrancher
> bras + caméra, lancer `run.sh`, puis dans l'UI vérifier la détection des ports
> (`/dev/ttyACM0/1`) et le flux caméra. Les ports/calib sont persistés via
> `~/.cache/huggingface` (monté).

## Réserves connues à surveiller au runtime

- **huggingface-hub 1.x** : corrigé via transformers 5.5.4, mais toute policy VLA
  (pi0, smolvla, groot…) réexerce ce chemin — re-tester au premier chargement.
- **torchcodec 0.11.1** (décodage vidéo dataset) installé sur aarch64 — à valider
  au premier accès dataset.
