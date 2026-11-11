"""Configuration de l'application, chargée depuis l'environnement et le fichier .env."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres de l'application.

    Les valeurs sont lues par ordre de priorité :
    1. les variables d'environnement du système ;
    2. le fichier ``.env`` à la racine du projet ;
    3. les valeurs par défaut définies ci-dessous.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Modèle d'embedding Hugging Face. Changer de modèle impose de revectoriser
    # tout le corpus : les vecteurs de deux modèles ne sont pas comparables.
    embedding_model: str = "intfloat/multilingual-e5-base"
    # Seuil de refus : si le meilleur passage a une similarité cosinus
    # inférieure, on refuse de répondre sans appeler le LLM. Valeur mesurée le
    # 10/11/2026 avec e5-base (evaluation/threshold.py) : 0 question trouvée
    # sur 19 refusée, 6 questions hors corpus sur 10 refusées. Dépend du
    # modèle d'embedding : à remesurer si embedding_model change.
    refusal_min_score: float = 0.84
    # Reranker (cross-encodeur local) : relit la question avec chaque candidat
    # de la recherche vectorielle et garde les meilleurs. Vide = désactivé.
    # Ses scores sont sur une autre échelle : le seuil de refus est à
    # remesurer avant de l'activer dans la commande de réponse.
    reranker_model: str | None = None
    # Nombre de candidats retenus par la recherche vectorielle avant le reranker.
    rerank_candidates: int = 30

    # LLM (API Anthropic). La clé n'a pas de valeur par défaut : elle n'est
    # exigée qu'au premier appel, pour que le reste du projet fonctionne sans.
    # Elle est lue ici et passée explicitement au client : pydantic-settings
    # ne recopie pas le fichier .env dans les variables d'environnement.
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-haiku-5-5"
    # Profondeur de réflexion du modèle, facturée comme de la sortie : "low"
    # suffit pour répondre à partir de quelques passages.
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    # Plafond de tokens générés (réflexion + réponse).
    llm_max_tokens: int = 2000
    llm_timeout_seconds: float = 60.0
    # Nouvelles tentatives du SDK sur erreur passagère (429, 5xx, réseau).
    llm_max_retries: int = 2

    # Base PostgreSQL (mêmes variables que docker-compose.yml). Le mot de passe
    # n'a pas de valeur par défaut : il n'est exigé qu'à la connexion, pour que
    # le reste du projet fonctionne sans base. SecretStr le masque dans les
    # affichages et les logs ("**********").
    postgres_user: str = "fire_user"
    postgres_password: SecretStr | None = None
    postgres_db: str = "fire_safety"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432

    # Adresse de l'API, utilisée par l'interface Streamlit. Dans Docker
    # (phase 5), ce sera le nom du service : http://api:8000.
    api_url: str = "http://127.0.0.1:8000"


@lru_cache
def get_settings() -> Settings:
    """Retourne la configuration, lue une seule fois puis mise en cache."""
    return Settings()
