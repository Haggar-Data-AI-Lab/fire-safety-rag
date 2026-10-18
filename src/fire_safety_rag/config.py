"""Configuration de l'application, chargée depuis l'environnement et le fichier .env."""

from functools import lru_cache
from typing import Literal

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


@lru_cache
def get_settings() -> Settings:
    """Retourne la configuration, lue une seule fois puis mise en cache."""
    return Settings()
