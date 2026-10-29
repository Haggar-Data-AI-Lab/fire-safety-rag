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

    # Base PostgreSQL (mêmes variables que docker-compose.yml). Le mot de passe
    # n'a pas de valeur par défaut : il n'est exigé qu'à la connexion, pour que
    # le reste du projet fonctionne sans base. SecretStr le masque dans les
    # affichages et les logs ("**********").
    postgres_user: str = "fire_user"
    postgres_password: SecretStr | None = None
    postgres_db: str = "fire_safety"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432


@lru_cache
def get_settings() -> Settings:
    """Retourne la configuration, lue une seule fois puis mise en cache."""
    return Settings()
