# Reconstruction reproductible de l'image lelab-orin.
#
# L'image de travail a été figée par `docker commit` du conteneur validé le
# 2026-07-13. Ce Dockerfile documente / permet de la reconstruire à partir de
# la base dustynv officielle si elle est perdue.
#
# Base : Ubuntu 24.04 => Python 3.12 ; torch 2.7.0 CUDA cp312 (compilé cu128,
# tourne en forward-compat sur le driver CUDA 12.6 de JetPack 6.2). Testé :
# torch.cuda.is_available() == True, device "Orin", interop numpy2<->cuda OK.
FROM dustynv/lerobot:r36.4.0-cu128-24.04

# La base configure pip avec l'index Jetson (jetson-ai-lab) comme index
# PRINCIPAL, qui n'a pas setuptools/wheel/etc. On rajoute PyPI en secours,
# sinon le build des deps de lerobot 0.6 échoue (ResolutionImpossible).
ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple

# Le repo lelab est monté à /opt/lelab au runtime (install éditable). git refuse
# un repo appartenant à un autre uid ("dubious ownership") -> on l'autorise.
RUN git config --global --add safe.directory /opt/lelab

# Installe lelab en éditable (tire lerobot 0.6.0 depuis git, garde le torch-CUDA
# de la base car 0.6.0 exige torch>=2.7,<2.11, satisfait par 2.7.0), PUIS monte
# transformers dans la plage attendue par lerobot 0.6 (>=5.4,<5.6). Sans ça, le
# transformers 4.51 de la base exige huggingface-hub<1.0 alors que lerobot 0.6
# impose hub>=1.0 -> ImportError au démarrage du serveur.
#
# Note : ce RUN suppose le code lelab disponible au build. En pratique on
# préfère committer le conteneur (code monté), ce Dockerfile sert de mémo :
#   docker run -d --name lelab-build -v <repo>:/opt/lelab <base> sleep infinity
#   docker exec -e PIP_EXTRA_INDEX_URL=https://pypi.org/simple lelab-build \
#     bash -lc 'cd /opt/lelab && pip install -e . && \
#               pip install -U "transformers>=5.4.0,<5.6.0"'
#   docker commit lelab-build lelab-orin:latest
#
# Rappel des versions validées : lerobot 0.6.0, transformers 5.5.4,
# tokenizers 0.22.2, huggingface-hub 1.23.0, torchvision 0.22.0, numpy 2.2.5.
