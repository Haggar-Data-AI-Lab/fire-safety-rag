"""Tests du module de configuration."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from fire_safety_rag.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isole chaque test : dossier vide, variables d'environnement nettoyées, cache vidé."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_defaults_without_env_file() -> None:
    """Sans .env ni variable système, les valeurs par défaut s'appliquent."""
    settings = Settings()

    assert settings.app_env == "dev"
    assert settings.log_level == "INFO"
    assert settings.embedding_model == "intfloat/multilingual-e5-base"


def test_reads_env_file(isolated_env: Path) -> None:
    """Les valeurs du fichier .env sont prises en compte."""
    (isolated_env / ".env").write_text("LOG_LEVEL=DEBUG\n", encoding="utf-8")

    settings = Settings()

    assert settings.log_level == "DEBUG"


def test_invalid_value_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Une valeur non autorisée bloque le chargement de la configuration."""
    monkeypatch.setenv("LOG_LEVEL", "BAVARD")

    with pytest.raises(ValidationError):
        Settings()


def test_env_var_overrides_env_file(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Une variable d'environnement système est prioritaire sur le fichier .env."""
    (isolated_env / ".env").write_text("APP_ENV=dev\n", encoding="utf-8")
    monkeypatch.setenv("APP_ENV", "prod")

    settings = Settings()

    assert settings.app_env == "prod"


def test_get_settings_is_cached() -> None:
    """get_settings() renvoie toujours la même instance."""
    assert get_settings() is get_settings()
