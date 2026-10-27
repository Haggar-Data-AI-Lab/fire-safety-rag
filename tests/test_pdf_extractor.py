"""Tests unitaires et d'intégration pour le module pdf_extractor."""

from pathlib import Path

import pytest

from fire_safety_rag.ingestion.pdf_extractor import PageText, clean_page_text, extract_pages

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
CODE_TRAVAIL_PDF = DATA_RAW / "code-travail-r4227-incendie.pdf"


class TestCleanPageText:
    """Tests unitaires de la fonction clean_page_text."""

    def test_supprime_ligne_horodatage(self) -> None:
        """La ligne d'horodatage Légifrance est supprimée."""
        text = (
            "Article R4227-29\n"
            "19/10/2026 20:35 Chapitre VII : Risques d'incendies - Légifrance\n"
            "https://www.legifrance.gouv.fr/codes/id/LEGISCTA000018532149 9/20"
        )
        result = clean_page_text(text)
        assert "19/10/2026" not in result
        assert "Légifrance" not in result
        assert "Article R4227-29" in result

    def test_supprime_ligne_url_pagination(self) -> None:
        """La ligne URL + pagination Légifrance est supprimée."""
        text = (
            "L'employeur prend les mesures nécessaires.\n"
            "https://www.legifrance.gouv.fr/codes/id/LEGISCTA000018532149 9/20"
        )
        result = clean_page_text(text)
        assert "legifrance.gouv.fr" not in result
        assert "L'employeur prend les mesures nécessaires." in result

    def test_conserve_version_en_vigueur(self) -> None:
        """Les lignes VERSION EN VIGUEUR sont conservées."""
        text = (
            "Article R4227-28\n"
            "VERSION EN VIGUEUR DEPUIS LE 01/05/2008\n"
            "L'employeur prend les mesures nécessaires."
        )
        result = clean_page_text(text)
        assert "VERSION EN VIGUEUR DEPUIS LE 01/05/2008" in result

    def test_conserve_ligne_creation_modification(self) -> None:
        """Les lignes Création/Modifié par sont conservées."""
        text = (
            "Article R4227-28\n"
            "VERSION EN VIGUEUR DEPUIS LE 01/05/2008\n"
            "Création Décret n°2008-244 du 7 mars 2008 - art. (V)\n"
            "L'employeur prend les mesures nécessaires."
        )
        result = clean_page_text(text)
        assert "Création Décret n°2008-244 du 7 mars 2008 - art. (V)" in result

    def test_conserve_date_reglementaire(self) -> None:
        """Une ligne de texte contenant une date ne doit pas être supprimée."""
        text = "Décret n°2008-244 du 7 mars 2008 fixe les dispositions applicables."
        result = clean_page_text(text)
        assert result == "Décret n°2008-244 du 7 mars 2008 fixe les dispositions applicables."

    def test_conserve_ligne_horodatage_sans_fin_legifrance(self) -> None:
        """Une ligne qui commence par JJ/MM/AAAA HH:MM mais ne se termine pas
        par '- Légifrance' ou '…' doit être conservée (faux positif potentiel)."""
        text = (
            "19/10/2026 20:35 Cette ligne ressemble à un horodatage mais n'est pas du bruit.\n"
            "Article R4227-29"
        )
        result = clean_page_text(text)
        assert "19/10/2026 20:35 Cette ligne ressemble à un horodatage" in result
        assert "Article R4227-29" in result

    def test_texte_sans_bruit_inchange(self) -> None:
        """Un texte sans aucune ligne de bruit est retourné tel quel."""
        text = (
            "Article R4227-29\nLe premier secours contre l'incendie est assuré par des extincteurs."
        )
        result = clean_page_text(text)
        assert result == text.strip()

    def test_normalise_lignes_vides_excessives(self) -> None:
        """Plus de 2 lignes vides consécutives sont réduites à 1."""
        text = "Paragraphe A.\n\n\n\nParagraphe B."
        result = clean_page_text(text)
        assert "Paragraphe A." in result
        assert "Paragraphe B." in result
        assert "\n\n\n" not in result

    def test_nettoyage_complet_page_reelle(self) -> None:
        """Simule une page complète avec les deux lignes de bruit en fin de page."""
        text = (
            "Article R4227-28\n"
            "VERSION EN VIGUEUR DEPUIS LE 01/05/2008\n"
            "Création Décret n°2008-244 du 7 mars 2008 - art. (V)\n"
            "L'employeur prend les mesures nécessaires pour que tout commencement d'incendie\n"
            "puisse être rapidement et efficacement combattu.\n"
            "19/10/2026 20:35 Chapitre VII : Risques d'incendies et d'explosions"
            " et évacuation (Articles R4227-1 à R4227-57) - Légifrance\n"
            "https://www.legifrance.gouv.fr/codes/id/LEGISCTA000018532149 9/20"
        )
        result = clean_page_text(text)
        assert "19/10/2026" not in result
        assert "legifrance.gouv.fr" not in result
        assert "VERSION EN VIGUEUR DEPUIS LE 01/05/2008" in result
        assert "Création Décret n°2008-244 du 7 mars 2008 - art. (V)" in result
        assert "L'employeur prend les mesures" in result


class TestExtractPages:
    """Test d'intégration de la fonction extract_pages."""

    @pytest.mark.skipif(
        not CODE_TRAVAIL_PDF.is_file(),
        reason="Corpus absent : les PDF ne sont pas versionnés (voir README)",
    )
    def test_integration_code_travail(self) -> None:
        """Extraction complète de code-travail-r4227-incendie.pdf."""
        pages = list(extract_pages(CODE_TRAVAIL_PDF, document_id="code-travail-r4227"))

        # Nombre de pages correct
        assert len(pages) == 20

        # Numérotation de 1 à 20
        assert [p.page_number for p in pages] == list(range(1, 21))

        # Tous les éléments sont des PageText avec le bon document_id
        assert all(isinstance(p, PageText) for p in pages)
        assert all(p.document_id == "code-travail-r4227" for p in pages)

        # Aucune page ne contient l'URL Légifrance après nettoyage
        for p in pages:
            assert "legifrance.gouv.fr" not in p.text

        # La première page contient l'article R4227-1
        assert "R4227-1" in pages[0].text
