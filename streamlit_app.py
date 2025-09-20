"""Streamlit UI für das bio-inspirierte Hippocampus-System."""
from __future__ import annotations

from typing import List

import streamlit as st

import bio_hippocampal_snn_ctx_theta_cmp_feedback_ctxlearn_hardgate as hippo
from book_ingestion_vsa_adapter_plus import (
    BookAdapter,
    LearningEvent,
    SentenceEvent,
)


st.set_page_config(page_title="Bio-Strukturelles SNN", layout="wide")


def _init_state() -> None:
    if "system" not in st.session_state:
        hyper = hippo.Hyper()
        system = hippo.HippocampalSystem(hyper, seed=101)
        adapter = BookAdapter(system=system)
        st.session_state.system = system
        st.session_state.adapter = adapter
        st.session_state.learning_log: List[LearningEvent] = []
        st.session_state.books = []


def _reset_state() -> None:
    for key in ("system", "adapter", "learning_log", "books"):
        if key in st.session_state:
            del st.session_state[key]
    _init_state()


_init_state()


def _format_sentence_stats(sentences: List[SentenceEvent]) -> str:
    return " | ".join(
        f"{idx+1}: DG={sent.dg_activations} → {sent.text}" for idx, sent in enumerate(sentences)
    )


def _system_metrics() -> None:
    system: hippo.HippocampalSystem = st.session_state.system
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Engramme", len(system.engrams))
        st.metric("Replay-Temp", f"{system.replay_temp:.2f}")
    with col2:
        avg_mismatch = sum(system.ca1_last_mismatch) / max(1, len(system.ca1_last_mismatch))
        st.metric("CA1-Mismatch", f"{avg_mismatch:.4f}")
        st.metric("Hard-Gate aktiv", "Ja" if system.hard_gate_countdown > 0 else "Nein")
    with col3:
        st.metric("CA3-Plastizität", f"{system.CA3.feedback_plasticity_gate:.3f}")
        st.metric("CA1-Plastizität", f"{system.CA1.feedback_plasticity_gate:.3f}")


with st.sidebar:
    st.header("Steuerung")
    if st.button("System zurücksetzen", type="primary"):
        _reset_state()
        st.success("System wurde zurückgesetzt.")

    st.subheader("Gelernte Bücher")
    if st.session_state.books:
        for info in st.session_state.books:
            st.caption(
                f"📘 {info['name']} – Sätze: {info['sentences']} | Absätze: {info['paragraphs']} | Kapitel: {info['chapters']}"
            )
    else:
        st.caption("Noch keine Bücher eingelernt.")


st.title("Bio-Strukturelle Hippocampus Streamlit App")

st.markdown(
    """
    Diese Anwendung erlaubt das Hochladen von Textbüchern, das Lernen über den
    biologisch inspirierten Hippocampus-Stack sowie das Stellen von Abfragen.
    Alle relevanten Lern- und Abfrageinformationen werden transparent angezeigt.
    """
)

st.header("1. Buch hochladen & lernen")

uploaded = st.file_uploader("Textdatei hochladen", type=["txt"], accept_multiple_files=False)
chapter_size = st.number_input("Maximale Satzanzahl pro Kapitel", min_value=4, max_value=200, value=40, step=4)
para_size = st.number_input("Sätze pro Absatz", min_value=2, max_value=30, value=6, step=1)

if uploaded is not None:
    text_bytes = uploaded.getvalue()
    try:
        text = text_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = text_bytes.decode("latin-1")
else:
    text = ""

if st.button("Lernphase starten", disabled=not text):
    adapter: BookAdapter = st.session_state.adapter
    sentences, paras, chapters = adapter.split_book(text, int(chapter_size), int(para_size))
    if not sentences:
        st.warning("Die Datei enthält keinen Text.")
    else:
        total_paragraphs = sum(len(chapter) for chapter in chapters)
        progress = st.progress(0)
        progress_state = {"count": 0}
        new_events: List[LearningEvent] = []

        def _progress_cb(event: LearningEvent) -> None:
            progress_state["count"] += 1
            new_events.append(event)
            progress.progress(min(1.0, progress_state["count"] / max(1, total_paragraphs)))

        with st.spinner("Lernen läuft…"):
            adapter.ingest_book(
                text,
                chapter_size_sents=int(chapter_size),
                para_size_sents=int(para_size),
                progress_cb=_progress_cb,
            )
        st.session_state.learning_log.extend(new_events)
        st.session_state.books.append(
            {
                "name": uploaded.name,
                "sentences": len(sentences),
                "paragraphs": len(paras),
                "chapters": len(chapters),
            }
        )
        st.success(
            f"Lernen abgeschlossen: {len(sentences)} Sätze → {total_paragraphs} Absätze in {len(chapters)} Kapiteln."
        )

if st.session_state.learning_log:
    st.subheader("Lernprotokoll")
    events_table = []
    for ev in st.session_state.learning_log:
        events_table.append(
            {
                "Kapitel": ev.chapter_index + 1,
                "Absatz": ev.paragraph_index + 1,
                "Sätze": len(ev.sentences),
                "Kontext-Neuronen": len(ev.ctx_ids),
                "Belohnung": f"{ev.reward:.3f}",
                "Kohärenz": f"{ev.coherence:.3f}",
                "Engramm": ev.engram_tag,
                "DG-Größe": ev.engram_size,
                "Replay-Temp": f"{ev.replay_temp:.2f}",
                "Hard-Gate": "Ja" if ev.hard_gate_active else "Nein",
                "CA3 aktiv": ev.ca3_active,
                "CA1 aktiv": ev.ca1_active,
                "CA1-Mismatch": f"{ev.avg_ca1_mismatch:.4f}",
            }
        )
    st.dataframe(events_table, use_container_width=True)

    with st.expander("Details je Absatz"):
        for ev in st.session_state.learning_log:
            st.markdown(
                f"**Kapitel {ev.chapter_index + 1}, Absatz {ev.paragraph_index + 1}** – Kontextneuronen: {len(ev.ctx_ids)}"
            )
            st.caption(
                f"Plastizitätsgates CA3/CA1: {ev.ca3_feedback_gate:.3f} / {ev.ca1_feedback_gate:.3f} | "
                f"Wachstum CA3/CA1: {ev.ca3_growth_events} / {ev.ca1_growth_events}"
            )
            st.write(_format_sentence_stats(ev.sentences))
            st.divider()

st.header("2. Systemstatus")
_system_metrics()

st.header("3. Abfragen & Analysen")

adapter = st.session_state.adapter

recall_tab, relation_tab, stim_tab = st.tabs(["Absatzsuche", "Relationen", "Hippocampus-Stimulation"])

with recall_tab:
    recall_query = st.text_area("Textabfrage", height=120)
    topk = st.slider("Top-k Ergebnisse", min_value=1, max_value=10, value=5)
    if st.button("Absätze abrufen", key="recall", disabled=not recall_query.strip()):
        results = adapter.recall_paragraph_like(recall_query, topk=topk)
        if results:
            st.table({"Tag": [tag for tag, _ in results], "Ähnlichkeit": [f"{sim:.3f}" for _, sim in results]})
        else:
            st.info("Keine Ergebnisse im Cortex gefunden.")

with relation_tab:
    col_a, col_b, col_c = st.columns(3)
    subj = col_a.text_input("Subjekt")
    verb = col_b.text_input("Verb")
    obj = col_c.text_input("Objekt")
    if st.button("Relation prüfen", key="relation", disabled=not (subj and verb and obj)):
        pos, neg, conf = adapter.ask_relation(subj, verb, obj)
        st.write(f"Positive Evidenz: {pos} | Negative Evidenz: {neg} | Konfidenz: {conf:.3f}")

with stim_tab:
    stim_text = st.text_area("Satz für Hippocampus-Stimulation", height=120)
    paragraph_options = [
        f"Kapitel {ev.chapter_index + 1}, Absatz {ev.paragraph_index + 1}" for ev in st.session_state.learning_log
    ]
    context_idx = st.selectbox(
        "Kontext auswählen", options=list(range(len(paragraph_options))), format_func=lambda i: paragraph_options[i]
    ) if paragraph_options else (None)
    ticks_after = st.slider("Ticks nach Stimulation", min_value=1, max_value=15, value=8)
    if st.button(
        "Stimuliere", key="stim", disabled=not (stim_text.strip() and paragraph_options)
    ):
        enc = adapter.encoder.encode_sentence(stim_text)
        ctx_ids = st.session_state.learning_log[context_idx].ctx_ids if context_idx is not None else []
        dg_ids = adapter._stimulate_sentence(enc, ctx_ids, ticks_after=ticks_after)
        ca3_active = len(st.session_state.system.CA3.active_exc_ids())
        ca1_active = len(st.session_state.system.CA1.active_exc_ids())
        avg_mismatch = sum(st.session_state.system.ca1_last_mismatch) / max(
            1, len(st.session_state.system.ca1_last_mismatch)
        )
        st.write(
            {
                "DG-Neuronen aktiviert": len(dg_ids),
                "CA3 aktiv": ca3_active,
                "CA1 aktiv": ca1_active,
                "Replay-Temp": round(st.session_state.system.replay_temp, 3),
                "Hard-Gate aktiv": st.session_state.system.hard_gate_countdown > 0,
                "CA1-Mismatch": round(avg_mismatch, 5),
            }
        )
        st.info(
            "Der Satz wurde zusammen mit dem gewählten Kontext präsentiert. "
            "Die obenstehenden Werte zeigen den aktuellen Netzwerkzustand."
        )

st.markdown("---")
st.caption("Hinweis: Diese Anwendung nutzt ausschließlich Standardbibliothek und Streamlit.")
