# Image de l'application : API (uvicorn) et interface (Streamlit).
# Une seule image pour les deux services de docker-compose.yml ; seule la
# commande de lancement change. Aucun secret dans l'image : la clé Anthropic
# et le mot de passe de la base sont fournis au lancement (env_file).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface

# Utilisateur sans droits d'administration, et dossier du cache Hugging Face
# (monté en volume : le modèle e5 n'est téléchargé qu'une fois).
RUN useradd --create-home app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache

WORKDIR /app

# PyTorch en version CPU : la version par défaut sous Linux embarque CUDA
# (cartes graphiques), plusieurs Go inutiles ici. Même version qu'en local.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# Dépendances du projet, installées avant le code : tant que pyproject.toml ne
# change pas, cette couche est reprise du cache et la reconstruction est rapide.
# Installation « éditable » : le code reste dans /app/src, ce qui garde juste
# le chemin de data/sources.csv (calculé depuis l'emplacement du code).
COPY pyproject.toml README.md ./
COPY src/fire_safety_rag/__init__.py src/fire_safety_rag/__init__.py
RUN pip install -e ".[embeddings,ui]"

COPY src ./src
COPY app ./app
COPY data/sources.csv ./data/sources.csv

USER app

EXPOSE 8000 8501

# Commande par défaut : l'API. Le service "ui" la remplace par Streamlit.
CMD ["uvicorn", "fire_safety_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
