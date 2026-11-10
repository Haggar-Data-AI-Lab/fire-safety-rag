"""Évaluation des réponses : chaîne complète sur le jeu de questions.

Chaque question (corpus et hors corpus) passe par answer_question. Le
résultat est enregistré, ligne par ligne, dans un fichier JSONL : on ne paie
les appels qu'une fois, puis les mesures se recalculent à partir du fichier.

Mesures automatiques :
- refus justifiés : questions hors corpus refusées ;
- refus à tort : questions "trouvées" (bon article dans le top k) refusées ;
- questions "manquées" : refusées ou non ;
- bon article cité : la réponse cite un article attendu ;
- citations défaillantes : numéro inconnu, aucune citation, réponse coupée ;
- coût réel.

La fidélité (la réponse dit-elle ce que disent les passages ?) se juge par
relecture : une fiche CSV est produite, avec une colonne "fidele" à remplir.

Utilisation (base indexée, extra "embeddings", clé API dans .env) :
    python -m fire_safety_rag.evaluation.answers [--limit N]
    python -m fire_safety_rag.evaluation.answers --from-file notes/reponses-AAAA-MM-JJ-HHMM.jsonl
"""

import argparse
import csv
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from fire_safety_rag.config import get_settings
from fire_safety_rag.embedding.embedder import embed_query
from fire_safety_rag.evaluation.evaluate import DEFAULT_QUESTIONS, EvalQuestion, load_questions
from fire_safety_rag.evaluation.metrics import first_relevant_rank
from fire_safety_rag.evaluation.threshold import (
    DEFAULT_OFF_CORPUS,
    FOUND,
    MISSED,
    OFF_CORPUS,
    load_off_corpus,
)
from fire_safety_rag.generation.answer import (
    INCOMPLETE,
    REFUSED_BY_LLM,
    REFUSED_BY_SCORE,
    Answer,
    answer_question,
    load_titles,
)
from fire_safety_rag.generation.llm import generate
from fire_safety_rag.ingestion.pipeline import PROJECT_ROOT
from fire_safety_rag.storage.database import connect
from fire_safety_rag.storage.store import SearchResult, search

NOTES_DIR = PROJECT_ROOT / "notes"
REFUSALS = (REFUSED_BY_SCORE, REFUSED_BY_LLM)
# Colonnes de la fiche de relecture à remplir à la main.
REVIEW_COLUMNS = ("fidele", "repond_question", "commentaire")


@dataclass(frozen=True)
class AnswerRecord:
    """Résultat enregistré pour une question.

    Attributes:
        id: identifiant de la question.
        group: FOUND, MISSED ou OFF_CORPUS.
        question: texte de la question.
        expected: articles attendus ("document_id:article_id"), vide hors corpus.
        retrieved: articles des passages retrouvés, dans l'ordre.
        top_score: score du meilleur passage.
        status: statut de la réponse (voir generation.answer).
        text: texte de la réponse du LLM ("" s'il n'a pas été appelé).
        cited: articles des passages cités valides.
        sources: références lisibles des passages cités ("[n] titre — article…").
        invalid_numbers: numéros cités inconnus.
        uncited: réponse sans aucune citation valide.
        input_tokens: tokens envoyés (0 sans appel).
        output_tokens: tokens générés (0 sans appel).
        cost_usd: coût estimé de l'appel (0 sans appel).
    """

    id: str
    group: str
    question: str
    expected: list[str]
    retrieved: list[str]
    top_score: float
    status: str
    text: str
    cited: list[str]
    sources: list[str]
    invalid_numbers: list[int]
    uncited: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _key(result: SearchResult) -> str:
    """Clé "document_id:article_id" d'un passage."""
    return f"{result.chunk.document_id}:{result.chunk.article_id}"


def record_answer(question: EvalQuestion, answer: Answer) -> AnswerRecord:
    """Transforme le résultat d'une question en ligne enregistrable.

    Args:
        question: question évaluée (avec ses articles attendus).
        answer: résultat de answer_question.

    Returns:
        La ligne à enregistrer.
    """
    retrieved = [_key(result) for result in answer.results]
    if not question.expected:
        group = OFF_CORPUS
    elif first_relevant_rank(retrieved, question.expected) is not None:
        group = FOUND
    else:
        group = MISSED

    checked, response = answer.checked, answer.response
    sources = checked.sources if checked is not None else ()
    return AnswerRecord(
        id=question.id,
        group=group,
        question=question.question,
        expected=sorted(question.expected),
        retrieved=retrieved,
        top_score=answer.top_score,
        status=answer.status,
        text=response.text if response is not None else "",
        cited=[_key(answer.results[s.number - 1]) for s in sources],
        sources=[f"[{s.number}] {s.reference}" for s in sources],
        invalid_numbers=list(checked.invalid_numbers) if checked is not None else [],
        uncited=checked.uncited if checked is not None else False,
        input_tokens=response.input_tokens if response is not None else 0,
        output_tokens=response.output_tokens if response is not None else 0,
        cost_usd=(response.cost_usd or 0.0) if response is not None else 0.0,
    )


def _ratio(part: int, total: int) -> str:
    """Affiche "part / total"."""
    return f"{part} / {total}"


def summarize(records: Sequence[AnswerRecord]) -> dict[str, str]:
    """Calcule les mesures automatiques.

    Args:
        records: lignes enregistrées.

    Returns:
        Mesure -> valeur affichable ("x / n" ou montant).
    """
    by_group = {g: [r for r in records if r.group == g] for g in (FOUND, MISSED, OFF_CORPUS)}
    refused = {g: sum(1 for r in rs if r.status in REFUSALS) for g, rs in by_group.items()}
    answered = [r for r in records if r.group != OFF_CORPUS and r.status not in REFUSALS]
    good_citation = sum(1 for r in answered if set(r.cited) & set(r.expected))
    calls = [r for r in records if r.input_tokens > 0]
    total_cost = sum(r.cost_usd for r in records)
    return {
        "Refus justifiés (hors corpus refusées)": _ratio(
            refused[OFF_CORPUS], len(by_group[OFF_CORPUS])
        ),
        "  dont refus par le score": _ratio(
            sum(1 for r in by_group[OFF_CORPUS] if r.status == REFUSED_BY_SCORE),
            len(by_group[OFF_CORPUS]),
        ),
        "Refus à tort (trouvées refusées)": _ratio(refused[FOUND], len(by_group[FOUND])),
        "Manquées refusées": _ratio(refused[MISSED], len(by_group[MISSED])),
        "Bon article cité (réponses du corpus)": _ratio(good_citation, len(answered)),
        "Réponses avec citation inconnue": _ratio(
            sum(1 for r in records if r.invalid_numbers), len(records)
        ),
        "Réponses sans citation": _ratio(sum(1 for r in records if r.uncited), len(records)),
        "Réponses incomplètes": _ratio(
            sum(1 for r in records if r.status == INCOMPLETE), len(records)
        ),
        "Appels au LLM": str(len(calls)),
        "Coût total": f"{total_cost:.5f} $",
        "Coût moyen par appel": f"{total_cost / len(calls):.5f} $" if calls else "—",
    }


def append_record(path: Path, record: AnswerRecord) -> None:
    """Ajoute une ligne au fichier JSONL (créé si besoin)."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[AnswerRecord]:
    """Relit un fichier JSONL produit par append_record."""
    with path.open(encoding="utf-8") as f:
        return [AnswerRecord(**json.loads(line)) for line in f if line.strip()]


def write_review_sheet(path: Path, records: Iterable[AnswerRecord]) -> None:
    """Écrit la fiche de relecture (CSV ";" lisible par Excel en français).

    Les réponses refusées par le score n'y figurent pas : il n'y a rien à relire.
    Colonnes à remplir (oui / partiel / non) :
    - "fidele" : chaque affirmation est appuyée par les passages cités ;
    - "repond_question" : la réponse répond à la question, avec le bon texte.
    Une réponse peut être fidèle à un passage sans répondre à la question
    (passage d'une autre réglementation, par exemple).
    """
    columns = ["id", "groupe", "statut", "question", "attendu", "reponse", "sources"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([*columns, *REVIEW_COLUMNS])
        for r in records:
            if r.status == REFUSED_BY_SCORE:
                continue
            writer.writerow(
                [
                    r.id,
                    r.group,
                    r.status,
                    r.question,
                    " ; ".join(r.expected),
                    r.text.strip(),
                    "\n".join(r.sources),
                    *([""] * len(REVIEW_COLUMNS)),
                ]
            )


def main() -> None:
    """Lance l'évaluation (ou relit un fichier), affiche les mesures, écrit la fiche."""
    parser = argparse.ArgumentParser(description="Évalue les réponses sur le jeu de questions.")
    parser.add_argument("--limit", type=int, help="n'évaluer que les N premières questions")
    parser.add_argument(
        "--from-file", type=Path, help="relire un fichier JSONL existant, sans aucun appel"
    )
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    for noisy in ("httpx", "httpx2", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Heure dans le nom : un nouveau lancement n'écrase jamais des réponses déjà payées
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    if args.from_file:
        records = read_records(args.from_file)
        jsonl_path = args.from_file
    else:
        questions = load_questions(DEFAULT_QUESTIONS) + load_off_corpus(DEFAULT_OFF_CORPUS)
        if args.limit is not None:
            questions = questions[: args.limit]
        NOTES_DIR.mkdir(exist_ok=True)
        suffix = f"-essai{args.limit}" if args.limit is not None else ""
        jsonl_path = NOTES_DIR / f"reponses-{stamp}{suffix}.jsonl"
        titles = load_titles()
        records = []
        with connect(settings) as connection:

            def retrieve(question: str, k: int) -> list[SearchResult]:
                return search(connection, embed_query(question), settings.embedding_model, k=k)

            for question in questions:
                answer = answer_question(
                    question.question,
                    retrieve,
                    titles,
                    lambda prompt: generate(prompt, settings=settings),
                    settings.refusal_min_score,
                )
                record = record_answer(question, answer)
                append_record(jsonl_path, record)  # enregistré tout de suite : rien n'est perdu
                records.append(record)
                print(f"  {record.id:4} {record.group:12} {record.status}")

    review_path = jsonl_path.with_name(jsonl_path.stem.replace("reponses", "relecture") + ".csv")
    # Une fiche existante peut contenir une relecture : ne jamais l'écraser
    if review_path.exists():
        review_note = "conservée (supprimez-la pour la regénérer)"
    else:
        write_review_sheet(review_path, records)
        review_note = "créée"

    print(
        f"\n{len(records)} questions ({settings.llm_model}, seuil {settings.refusal_min_score:g})"
    )
    for name, value in summarize(records).items():
        print(f"  {name:42} {value}")
    print(f"\nRéponses : {jsonl_path}")
    print(f"Fiche de relecture {review_note} : {review_path}")


if __name__ == "__main__":
    main()
