"""Tests unitaires pour le module chunker."""

from fire_safety_rag.ingestion.chunker import Article, split_into_articles


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
        result = split_into_articles(text, "code-travail")
        assert result == [
            Article(
                document_id="code-travail",
                article_id="R4227-4",
                version="VERSION EN VIGUEUR DEPUIS LE 01/05/2008",
                text=(
                    "Les établissements comportent des dégagements.\n"
                    "Ces dégagements sont toujours libres."
                ),
            ),
            Article(
                document_id="code-travail",
                article_id="R4227-5",
                version="VERSION EN VIGUEUR DEPUIS LE 01/05/2008",
                text="La largeur des dégagements est calculée.",
            ),
        ]

    def test_identifiant_en_plusieurs_mots(self) -> None:
        """Un identifiant de plusieurs mots (ex. "CTS 31 bis") est reconnu."""
        text = (
            "Article CTS 31 bis\n"
            "VERSION EN VIGUEUR DEPUIS LE 07/04/2002\n"
            "Les chapiteaux sont implantés sur un terrain plat."
        )
        result = split_into_articles(text, "erp")
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
        result = split_into_articles(text, "ssiap")
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
        result = split_into_articles(text, "code-travail")
        assert len(result) == 1
        assert "Code du travail" not in result[0].text

    def test_texte_sans_article(self) -> None:
        """Un texte sans titre d'article renvoie une liste vide."""
        assert split_into_articles("Sommaire\nChapitre VII", "doc") == []
