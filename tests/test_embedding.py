"""Tests unitaires et d'intégration pour le module embedder."""

import math
from datetime import date

import pytest

from fire_safety_rag.embedding.embedder import (
    embed_passages,
    embed_query,
    format_passage,
    format_query,
)
from fire_safety_rag.ingestion.chunker import Chunk


def _chunk(article_id: str, text: str) -> Chunk:
    """Construit un Chunk minimal pour les tests."""
    return Chunk(
        document_id="doc",
        article_id=article_id,
        version="VERSION EN VIGUEUR DEPUIS LE 01/05/2008",
        valid_from=date(2008, 5, 1),
        valid_until=None,
        history=(),
        chunk_index=0,
        text=text,
        page_start=1,
        page_end=1,
    )


class TestFormat:
    """Tests des fonctions de mise en forme (sans modèle)."""

    def test_format_passage_article(self) -> None:
        """Un article reçoit le préfixe e5 et l'en-tête "Article X"."""
        chunk = _chunk("GN 1", "Classement des établissements")
        assert format_passage(chunk) == "passage: Article GN 1\nClassement des établissements"

    def test_format_passage_annexe(self) -> None:
        """Une annexe garde son titre tel quel, sans "Article" devant."""
        chunk = _chunk("Annexe I", "Contenu de l'annexe.")
        assert format_passage(chunk) == "passage: Annexe I\nContenu de l'annexe."

    def test_format_query(self) -> None:
        """Une question reçoit le préfixe e5 "query: "."""
        assert format_query("Largeur des dégagements ?") == "query: Largeur des dégagements ?"


class TestEmbeddingModel:
    """Tests d'intégration avec le vrai modèle (sautés sans l'extra "embeddings").

    Le premier lancement télécharge le modèle (~1 Go) dans ~/.cache/huggingface.
    """

    @pytest.fixture(autouse=True)
    def _requires_sentence_transformers(self) -> None:
        """Saute ces tests si sentence-transformers n'est pas installé (cas de la CI)."""
        pytest.importorskip("sentence_transformers")

    def test_dimension_et_normalisation(self) -> None:
        """Les vecteurs font 768 dimensions et sont normalisés (norme 1)."""
        (vector,) = embed_passages([_chunk("R4227-4", "Les dégagements sont toujours libres.")])
        assert len(vector) == 768
        assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, abs_tol=1e-3)

    def test_question_proche_du_bon_passage(self) -> None:
        """Une question sur les dégagements est plus proche du passage qui en parle."""
        degagements, rideaux = embed_passages(
            [
                _chunk(
                    "R4227-5",
                    "La largeur des dégagements ne doit pas être inférieure à 0,80 mètre.",
                ),
                _chunk(
                    "AM 13",
                    "Les rideaux de scènes doivent être en matériaux de catégorie M1.",
                ),
            ]
        )
        question = embed_query("Quelle est la largeur minimale d'un dégagement ?")

        # Vecteurs normalisés : le produit scalaire est la similarité cosinus
        def similarity(vector: list[float]) -> float:
            return sum(a * b for a, b in zip(question, vector, strict=True))

        assert similarity(degagements) > similarity(rideaux)
