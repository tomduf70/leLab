#!/usr/bin/env bash
#
# Lance LeLab (UI + inférence GPU) sur Jetson Orin dans le conteneur lelab-orin.
#
# Contexte : sur cette carte (Seeed reComputer Orin NX 16G, JetPack 6.2.1 /
# L4T r36.4.3), l'inférence GPU en natif est IMPOSSIBLE — lerobot 0.6 exige
# Python >=3.12 alors que les wheels PyTorch-CUDA de jetson-ai-lab sont cp310
# seulement. Le conteneur dustynv/lerobot (Ubuntu 24.04 -> Python 3.12 +
# torch-CUDA cp312) lève ce verrou. Voir README.md.
#
# Docker sur Linux natif = namespaces/cgroups, PAS une VM : le GPU (--runtime
# nvidia), l'USB (--device / cgroup) et le réseau (--network host) sont en accès
# direct, sans surcoût de performance.
#
# Prérequis :
#   - image lelab-orin:latest présente (docker images lelab-orin)
#     -> sinon la reconstruire : voir Dockerfile dans ce dossier
#   - le service lelab natif NE doit PAS tenir le port 8000 :
#       systemctl --user disable --now lelab.service
#
set -euo pipefail

IMAGE="${LELAB_IMAGE:-lelab-orin:latest}"
NAME="${LELAB_NAME:-lelab}"
PORT="${LELAB_PORT:-8000}"
# Racine du repo lelab (deux niveaux au-dessus de ce script)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_CACHE="${HOME}/.cache/huggingface"

mkdir -p "${HF_CACHE}"

# --- passage du hardware (bras SO-101 + caméras USB) --------------------------
# On monte /dev en entier + on autorise les majeurs de périphériques via cgroup,
# ce qui survit au HOTPLUG (débrancher/rebrancher le bras sans relancer le
# conteneur). Majeurs : 166 = /dev/ttyACM* (bras Feetech via USB CDC-ACM),
# 81 = /dev/video* (caméras V4L2), 188 = /dev/ttyUSB* (au cas où).
# NB : le GPU passe par --runtime nvidia (nœuds /dev/nvidia* injectés).
HW_ARGS=(
  -v /dev:/dev
  --device-cgroup-rule 'c 166:* rmw'
  --device-cgroup-rule 'c 81:* rmw'
  --device-cgroup-rule 'c 188:* rmw'
)

# --- HF_HOME / HF_LEROBOT_HOME (IMPORTANT) ------------------------------------
# L'image de base fixe HF_HOME=/data/models/huggingface (chemin INTERNE). Sans
# override, huggingface_hub et lerobot chercheraient token + calibrations +
# datasets à ce chemin interne (jetable) au lieu du cache monté -> token "non
# configuré", calibrations introuvables. On repointe donc HF vers le montage
# hôte : tout (token, calib des 2 bras, datasets enregistrés, cache modèles)
# vit alors sur l'hôte et survit à la destruction du conteneur.
HF_ENV=(
  -e HF_HOME=/root/.cache/huggingface
  -e HF_LEROBOT_HOME=/root/.cache/huggingface/lerobot
)

# --- SSH vers robotic-ai (training déporté) -----------------------------------
# Le runner ssh_remote fait `ssh robotic-ai` via l'alias de ~/.ssh/config (avec
# ProxyCommand cloudflared). On monte ~/.ssh, ~/.cloudflared (lecture seule, à
# /host-*) + le binaire cloudflared. Au démarrage (LAUNCH ci-dessous) on RECOPIE
# ces dossiers dans /root/... avec chown root : le conteneur tourne en root
# (uid 0) et ssh refuse une config/clé possédée par un autre uid (« bad
# ownership »). La copie vit dans la couche jetable du conteneur.
SSH_ARGS=(
  -v "${HOME}/.ssh":/host-ssh:ro
  -v /usr/local/bin/cloudflared:/usr/local/bin/cloudflared:ro
)
[ -d "${HOME}/.cloudflared" ] && SSH_ARGS+=( -v "${HOME}/.cloudflared":/host-cloudflared:ro )
# Config locale LeLab hors git (ex. robotic_ai_wake_url : webhook Home Assistant
# qui allume la prise du PC d'entraînement). root lit sans souci un fichier 600.
[ -d "${HOME}/.config/lelab" ] && SSH_ARGS+=( -v "${HOME}/.config/lelab":/root/.config/lelab:ro )

# Script de démarrage : prépare le SSH root-owned puis lance le serveur.
# $PORT vient de l'env du conteneur (-e PORT plus bas) ; ici c'est du littéral.
LAUNCH='
install -d -m 700 /root/.ssh
cp -a /host-ssh/. /root/.ssh/ 2>/dev/null || true
chown -R root:root /root/.ssh && chmod 700 /root/.ssh && chmod 600 /root/.ssh/* 2>/dev/null || true
if [ -d /host-cloudflared ]; then
  install -d -m 700 /root/.cloudflared
  cp -a /host-cloudflared/. /root/.cloudflared/ 2>/dev/null || true
  chown -R root:root /root/.cloudflared
fi
cd /opt/lelab && exec uvicorn lelab.server:app --host 0.0.0.0 --port "$PORT"
'

# -it seulement en interactif (une unit systemd n'a pas de TTY)
TTY_ARGS=()
[ -t 1 ] && TTY_ARGS=(-it)

exec docker run --rm "${TTY_ARGS[@]}" \
  --name "${NAME}" \
  --runtime nvidia \
  --network host \
  "${HW_ARGS[@]}" \
  "${HF_ENV[@]}" \
  "${SSH_ARGS[@]}" \
  -e PORT="${PORT}" \
  -v "${REPO}":/opt/lelab \
  -v "${HF_CACHE}":/root/.cache/huggingface \
  "${IMAGE}" \
  bash -lc "${LAUNCH}"
