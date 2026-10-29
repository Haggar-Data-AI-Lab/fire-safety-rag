"""Tests unitaires pour le module chunker."""

from datetime import date

import pytest

from fire_safety_rag.ingestion.chunker import (
    Article,
    parse_version,
    split_article,
    split_into_articles,
)
from fire_safety_rag.ingestion.pdf_extractor import PageText


def _single_page(text: str, document_id: str = "doc") -> list[PageText]:
    """Construit un document d'une seule page (page 1) à partir d'un texte."""
    return [PageText(document_id=document_id, page_number=1, text=text)]


def _make_article(lines: list[str], pages: list[int]) -> Article:
    """Construit un Article à partir de ses lignes et de la page de chacune."""
    return Article(
        document_id="doc",
        article_id="GN 1",
        version="VERSION EN VIGUEUR DEPUIS LE 01/01/2020",
        valid_from=date(2020, 1, 1),
        valid_until=None,
        history=(),
        text="\n".join(lines),
        page_start=pages[0],
        page_end=pages[-1],
        line_pages=tuple(pages),
    )


class TestParseVersion:
    """Tests unitaires de la fonction parse_version."""

    def test_format_depuis_le(self) -> None:
        """Le format "DEPUIS LE" donne une date de début et pas de date de fin."""
        assert parse_version("VERSION EN VIGUEUR DEPUIS LE 01/05/2008") == (
            date(2008, 5, 1),
            None,
        )

    def test_format_du_au(self) -> None:
        """Le format "DU ... AU ..." donne une date de début et une date de fin."""
        assert parse_version("VERSION EN VIGUEUR DU 15/08/1980 AU 01/06/2027") == (
            date(1980, 8, 15),
            date(2027, 6, 1),
        )

    def test_format_inconnu_leve_une_erreur(self) -> None:
        """Une ligne de version inconnue lève ValueError au lieu d'être ignorée."""
        with pytest.raises(ValueError, match="non reconnue"):
            parse_version("VERSION EN VIGUEUR AU 19/10/2026")


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
                valid_from=date(2008, 5, 1),
                valid_until=None,
                history=(),
                text=(
                    "Les établissements comportent des dégagements.\n"
                    "Ces dégagements sont toujours libres."
                ),
                page_start=1,
                page_end=1,
                line_pages=(1, 1),
            ),
            Article(
                document_id="code-travail",
                article_id="R4227-5",
                version="VERSION EN VIGUEUR DEPUIS LE 01/05/2008",
                valid_from=date(2008, 5, 1),
                valid_until=None,
                history=(),
                text="La largeur des dégagements est calculée.",
                page_start=1,
                page_end=1,
                line_pages=(1,),
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

    def test_annexe_separee_de_l_article_precedent(self) -> None:
        """Une annexe forme sa propre unité au lieu d'allonger l'article précédent."""
        text = (
            "Article 17\n"
            "VERSION EN VIGUEUR DEPUIS LE 10/07/2024\n"
            "Le présent arrêté sera publié au Journal officiel.\n"
            "Annexes (Articles Annexe I à Annexe XIII)\n"
            "Annexe I\n"
            "VERSION EN VIGUEUR DEPUIS LE 10/07/2024\n"
            "Contenu de l'annexe I."
        )
        result = split_into_articles(_single_page(text))
        assert [article.article_id for article in result] == ["17", "Annexe I"]
        assert "Contenu de l'annexe I." not in result[0].text
        assert result[1].text == "Contenu de l'annexe I."

    def test_annexe_rattachee_a_un_article(self) -> None:
        """Un titre "Annexe à l'article PO 11" est reconnu tel quel."""
        text = (
            "Annexe à l'article PO 11\n"
            "VERSION EN VIGUEUR DEPUIS LE 30/10/2011\n"
            "Tableau des distances."
        )
        result = split_into_articles(_single_page(text))
        assert [article.article_id for article in result] == ["Annexe à l'article PO 11"]

    def test_sommaire_des_annexes_ignore(self) -> None:
        """La ligne de sommaire "Annexes (...)" n'est jamais un titre."""
        text = (
            "Annexes (Articles Annexe I à Annexe V)\n"
            "VERSION EN VIGUEUR DEPUIS LE 22/06/2011\n"
            "Texte."
        )
        assert split_into_articles(_single_page(text)) == []

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
        assert result[0].line_pages == (1, 2)
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


class TestSplitArticle:
    """Tests unitaires de la fonction split_article."""

    @staticmethod
    def _lines(count: int) -> list[str]:
        """Renvoie `count` lignes de 10 caractères terminées par un point."""
        return [f"ligne {index:03d}." for index in range(count)]

    def test_article_court_donne_un_morceau(self) -> None:
        """Un article qui tient dans max_chars donne un seul morceau identique."""
        article = _make_article(["Première ligne.", "Seconde ligne."], [4, 5])
        result = split_article(article)
        assert len(result) == 1
        assert result[0].chunk_index == 0
        assert result[0].text == article.text
        assert (result[0].page_start, result[0].page_end) == (4, 5)

    def test_aucun_morceau_ne_depasse_max_chars(self) -> None:
        """Sans chevauchement, les morceaux respectent la taille et couvrent tout."""
        lines = self._lines(10)
        result = split_article(_make_article(lines, [1] * 10), max_chars=35, overlap_chars=0)
        assert all(len(chunk.text) <= 35 for chunk in result)
        assert [chunk.chunk_index for chunk in result] == list(range(len(result)))
        assert "\n".join(chunk.text for chunk in result) == "\n".join(lines)

    def test_coupe_apres_fin_de_phrase(self) -> None:
        """La coupe recule jusqu'à la dernière ligne qui termine une phrase."""
        lines = ["Phrase un est finie.", "suite", "de texte"]
        result = split_article(_make_article(lines, [1, 1, 1]), max_chars=28, overlap_chars=0)
        assert [chunk.text for chunk in result] == ["Phrase un est finie.", "suite\nde texte"]

    def test_ignore_fin_de_phrase_si_morceau_trop_court(self) -> None:
        """Une fin de phrase qui laisserait un morceau trop court est ignorée."""
        lines = ["Court.", "aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]
        result = split_article(_make_article(lines, [1] * 4), max_chars=30, overlap_chars=0)
        assert result[0].text == "Court.\naaaaaaaaaa\nbbbbbbbbbb"

    def test_chevauchement(self) -> None:
        """Le morceau suivant reprend la dernière ligne du précédent."""
        result = split_article(
            _make_article(self._lines(6), [1] * 6), max_chars=32, overlap_chars=12
        )
        assert len(result) >= 2
        assert result[1].text.splitlines()[0] == result[0].text.splitlines()[-1]

    def test_ligne_trop_longue_forme_un_morceau(self) -> None:
        """Une ligne plus longue que max_chars forme un morceau à elle seule."""
        lines = ["x" * 50, "fin."]
        result = split_article(_make_article(lines, [1, 1]), max_chars=20)
        assert [chunk.text for chunk in result] == ["x" * 50, "fin."]

    def test_pages_propres_a_chaque_morceau(self) -> None:
        """Chaque morceau porte les pages de ses propres lignes."""
        result = split_article(
            _make_article(self._lines(4), [3, 3, 4, 5]), max_chars=21, overlap_chars=0
        )
        assert [(chunk.page_start, chunk.page_end) for chunk in result] == [(3, 3), (4, 5)]


class TestHistorique:
    """Tests du retrait des lignes d'historique Légifrance (via split_into_articles)."""

    @staticmethod
    def _article(*body: str) -> Article:
        """Découpe un document d'un seul article dont le contenu est `body`."""
        text = "\n".join(["Article GN 1", "VERSION EN VIGUEUR DEPUIS LE 01/01/2020", *body])
        (article,) = split_into_articles(_single_page(text))
        return article

    def test_ligne_modifie_par_deplacee_dans_history(self) -> None:
        """Une ligne "Modifié par" quitte le texte et passe dans history."""
        article = self._article(
            "Modifié par Arrêté du 7 février 2022 - art. 3",
            "Classement des établissements",
        )
        assert article.history == ("Modifié par Arrêté du 7 février 2022 - art. 3",)
        assert article.text == "Classement des établissements"
        assert article.line_pages == (1,)

    def test_plusieurs_lignes_d_historique(self) -> None:
        """Toutes les lignes d'historique consécutives en tête sont captées."""
        article = self._article(
            "Création Arrêté du 4 juin 1982 (V)",
            "Modifié par Arrêté du 10 octobre 2005 - art. Annexe, v. init.",
            "Transféré par Arrêté du 11 décembre 2009 - art.",
            "Texte réglementaire.",
        )
        assert len(article.history) == 3
        assert article.text == "Texte réglementaire."

    def test_ligne_coupee_recollee(self) -> None:
        """Une parenthèse ouverte en fin de ligne est refermée par la ligne suivante."""
        article = self._article(
            "Modifié par Arrêté du 23 décembre 1996 - art. Annexe, v. init. (en dernier",
            "lieu)",
            "Texte réglementaire.",
        )
        assert article.history == (
            "Modifié par Arrêté du 23 décembre 1996 - art. Annexe, v. init. (en dernier lieu)",
        )
        assert article.text == "Texte réglementaire."

    def test_ligne_creation_au_milieu_non_touchee(self) -> None:
        """Une ligne commençant par "Création" au milieu du texte est conservée."""
        article = self._article(
            "Les exploitants doivent prévoir :",
            "Création d'un registre de sécurité.",
        )
        assert article.history == ()
        assert "Création d'un registre de sécurité." in article.text
