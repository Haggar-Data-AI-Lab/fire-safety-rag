"""Tests unitaires du contrôle des citations (réponses simulées, sans LLM)."""

from datetime import date

from fire_safety_rag.generation.citations import Source, check_citations, cited_numbers
from fire_safety_rag.generation.prompt import REFUSAL_SENTENCE
from fire_safety_rag.ingestion.chunker import Chunk
from fire_safety_rag.storage.store import SearchResult


def _result(article_id: str, document_id: str = "erp") -> SearchResult:
    """Construit un SearchResult minimal (page 12, en vigueur depuis le 01/03/2020)."""
    chunk = Chunk(
        document_id=document_id,
        article_id=article_id,
        version="",
        valid_from=date(2020, 3, 1),
        valid_until=None,
        history=(),
        chunk_index=0,
        text="texte",
        page_start=12,
        page_end=12,
    )
    return SearchResult(chunk=chunk, score=0.9)


TITLES = {"erp": "Règlement ERP", "ssiap": "Arrêté SSIAP"}
RESULTS = [_result("MS 21"), _result("GN 1"), _result("Annexe I", "ssiap")]


def _ref(heading: str, title: str = "Règlement ERP") -> str:
    """Référence attendue pour un passage de test."""
    return f"{title} — {heading} (page 12, en vigueur depuis le 01/03/2020)"


class TestCitedNumbers:
    """Tests de l'extraction des numéros."""

    def test_formes_reconnues(self) -> None:
        """[1], [2][3] et [1, 4] sont reconnus."""
        assert cited_numbers("A [1]. B [2][3]. C [1, 4].") == {1, 2, 3, 4}

    def test_plage_non_reconnue(self) -> None:
        """Une plage [1-3] et du texte entre crochets ne comptent pas."""
        assert cited_numbers("Voir [1-3] et [note].") == set()


class TestCheckCitations:
    """Tests de check_citations."""

    def test_citations_valides_triees_sans_doublon(self) -> None:
        """Les sources sont triées par numéro, sans doublon, avec leur référence."""
        checked = check_citations("B [2]. A [1]. Encore [2].", RESULTS, TITLES)
        assert checked.sources == (
            Source(1, _ref("Article MS 21")),
            Source(2, _ref("Article GN 1")),
        )
        assert checked.invalid_numbers == ()
        assert not checked.uncited
        assert not checked.is_refusal

    def test_reference_d_annexe(self) -> None:
        """Une annexe garde son titre, sans "Article" devant."""
        checked = check_citations("Voir [3].", RESULTS, TITLES)
        assert checked.sources == (Source(3, _ref("Annexe I", "Arrêté SSIAP")),)

    def test_numero_invalide(self) -> None:
        """Un numéro hors de 1..n est signalé ; les numéros valides restent."""
        checked = check_citations("A [1]. B [7]. C [0].", RESULTS, TITLES)
        assert [s.number for s in checked.sources] == [1]
        assert checked.invalid_numbers == (0, 7)
        assert not checked.uncited

    def test_seulement_invalide_est_sans_citation(self) -> None:
        """Sans aucun numéro valide, la réponse est signalée sans citation."""
        checked = check_citations("Selon [9], oui.", RESULTS, TITLES)
        assert checked.sources == ()
        assert checked.invalid_numbers == (9,)
        assert checked.uncited

    def test_reponse_sans_citation(self) -> None:
        """Une réponse qui ne cite rien est signalée."""
        checked = check_citations("Oui, deux sorties suffisent.", RESULTS, TITLES)
        assert checked.uncited
        assert checked.sources == ()

    def test_refus(self) -> None:
        """La phrase de refus (même entre guillemets) donne un refus sans source."""
        checked = check_citations(f"« {REFUSAL_SENTENCE} » [1]", RESULTS, TITLES)
        assert checked.is_refusal
        assert checked.sources == ()
        assert not checked.uncited
