"""Interface web de l'assistant (Streamlit).

La page appelle l'API (fire_safety_rag.ui.client) : elle ne contient aucune
logique de recherche ni de génération, seulement l'affichage.

Lancement (API déjà démarrée avec uvicorn, extra "ui" installé) :
    streamlit run app/streamlit_app.py
"""

import streamlit as st

from fire_safety_rag.config import get_settings
from fire_safety_rag.ui.client import ApiAnswer, ApiError, ask_api

EXAMPLES = (
    "Combien d'extincteurs faut-il dans des locaux de travail ?",
    "Combien d'heures dure la formation SSIAP 1 ?",
    "À partir de quelle hauteur un immeuble d'habitation devient-il un IGH ?",
)


def show_answer(answer: ApiAnswer) -> None:
    """Affiche la réponse, ses sources et ses avertissements."""
    if answer.is_refusal:
        st.info(answer.reponse)
        st.caption(
            "L'assistant ne répond qu'à partir des textes de son corpus ; "
            "aucun passage ne permet de répondre à cette question."
        )
    else:
        st.markdown(answer.reponse)

    for warning in answer.avertissements:
        st.warning(warning)

    if answer.sources:
        st.subheader("Sources")
        # Numéros conservés tels quels : ils correspondent aux [n] de la réponse
        for source in answer.sources:
            st.markdown(f"**[{source.numero}]** {source.reference}")

    st.divider()
    st.caption(answer.avertissement_juridique)


def main() -> None:
    """Construit la page."""
    st.set_page_config(page_title="Assistant sécurité incendie", page_icon="🧯")
    st.title("Assistant réglementation sécurité incendie")
    st.write(
        "Posez une question sur la réglementation : Code du travail (incendie), "
        "règlement de sécurité des ERP, Code de la construction (sécurité incendie), "
        "arrêté SSIAP. Chaque réponse cite les articles utilisés."
    )
    with st.expander("Exemples de questions"):
        for example in EXAMPLES:
            st.markdown(f"- {example}")

    with st.form("question"):
        question = st.text_area("Votre question", max_chars=1000)
        submitted = st.form_submit_button("Demander")

    if not submitted:
        return
    if not question.strip():
        st.warning("Saisissez une question.")
        return

    with st.spinner("Recherche dans les textes…"):
        try:
            answer = ask_api(question, get_settings().api_url)
        except ApiError as exc:
            st.error(str(exc))
            return
    show_answer(answer)


main()
