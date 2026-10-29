"""Tests unitaires pour le module chunker."""

from fire_safety_rag.ingestion.chunker import Article, split_into_articles
from fire_safety_rag.ingestion.pdf_extractor import PageText


def _single_page(text: str, document_id: str = "doc") -> list[PageText]:
    """Construit un document d'une seule page (page 1) à partir d'un texte."""
    return [PageText(document_id=document_id, page_number=1, text=text)]


class TestSplitIntoArticles:
    """Tests unitaires de la fonction split_into_articles."""

    def test_decoupe_deux_articles(self) -> None:
        """Deux articles consécutifs donnent deux Article complets."""
        text = (
            "Article R4227-4\n"
            "VERSION EN VIGUEUR DEPUIS LE 01/05/2008\n"
            "Les établissements comportent des dégagements.\n"
            "Ces dégagements sont toujours libres.\n"
            "Article R4227-5\n"
            "VERSION EN VIGUEUR DEPUIS LE 01/05/2008\n"
            "La largeur des dégagements est calculée."
        )
        result = split_into_articles(_single_page(text, "code-travail"))
        assert result == [
            Article(
                document_id="code-travail",
                article_id="R4227-4",
                version="VERSION EN VIGUEUR DEPUIS LE 01/05/2008",
                text=(
                    "Les établissements comportent des dégagements.\n"
                    "Ces dégagements sont toujours libres."
                ),
                page_start=1,
                page_end=1,
            ),
            Article(
                document_id="code-travail",
                article_id="R4227-5",
                version="VERSION EN VIGUEUR DEPUIS LE 01/05/2008",
                text="La largeur des dégagements est calculée.",
                page_start=1,
                page_end=1,
            ),
        ]

    def test_identifiant_en_plusieurs_mots(self) -> None:
        """Un identifiant de plusieurs mots (ex. "CTS 31 bis") est reconnu."""
        text = (
            "Article CTS 31 bis\n"
            "VERSION EN VIGUEUR DEPUIS LE 07/04/2002\n"
            "Les chapiteaux sont implantés sur un terrain plat."
        )
        result = split_into_articles(_single_page(text))
        assert [article.article_id for article in result] == ["CTS 31 bis"]

    def test_ignore_titre_sans_ligne_version(self) -> None:
        """Une ligne "Article 3 :" non suivie de VERSION n'est pas un titre."""
        text = (
            "Article 2\n"
            "VERSION EN VIGUEUR DEPUIS LE 01/01/2006\n"
            "Modèle de convention :\n"
            "Article 3 :\n"
            "Le montant de la prestation est de ..."
        )
        result = split_into_articles(_single_page(text))
        assert len(result) == 1
        assert result[0].article_id == "2"
        assert "Article 3 :" in result[0].text

    def test_ignore_preambule(self) -> None:
        """Le texte situé avant le premier article est ignoré."""
        text = (
            "Code du travail\n"
            "VERSION EN VIGUEUR AU 19/10/2026\n"
            "Article R4227-1\n"
            "VERSION EN VIGUEUR DEPUIS LE 01/05/2008\n"
            "Les dispositions du présent chapitre s'appliquent."
        )
        result = split_into_articles(_single_page(text))
        assert len(result) == 1
        assert "Code du travail" not in result[0].text

    def test_texte_sans_article(self) -> None:
        """Un texte sans titre d'article renvoie une liste vide."""
        assert split_into_articles(_single_page("Sommaire\nChapitre VII")) == []

    def test_aucune_page(self) -> None:
        """Une liste de pages vide renvoie une liste vide."""
        assert split_into_articles([]) == []

    def test_article_sur_deux_pages(self) -> None:
        """Un article qui continue sur la page suivante couvre les deux pages."""
        pages = [
            PageText(
                document_id="doc",
                page_number=1,
                text=(
                    "Article R4227-4\n"
                    "VERSION EN VIGUEUR DEPUIS LE 01/05/2008\n"
                    "Les établissements comportent des dégagements."
                ),
            ),
            PageText(document_id="doc", page_number=2, text="Ces dégagements sont libres."),
        ]
        result = split_into_articles(pages)
        assert len(result) == 1
        assert (result[0].page_start, result[0].page_end) == (1, 2)
        assert result[0].text == (
            "Les établissements comportent des dégagements.\nCes dégagements sont libres."
        )

    def test_article_suivant_commence_sur_nouvelle_page(self) -> None:
        """Un article qui finit en bas de page s'arrête à cette page."""
        pages = [
            PageText(
                document_id="doc",
                page_number=1,
                text=("Article R4227-4\nVERSION EN VIGUEUR DEPUIS LE 01/05/2008\nTexte A."),
            ),
            PageText(
                document_id="doc",
                page_number=2,
                text=("Article R4227-5\nVERSION EN VIGUEUR DEPUIS LE 01/05/2008\nTexte B."),
            ),
        ]
        result = split_into_articles(pages)
        assert [(a.page_start, a.page_end) for a in result] == [(1, 1), (2, 2)]
