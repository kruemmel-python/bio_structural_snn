"""Streamlit-UI für den bio-inspirierten Buch-Adapter."""
from __future__ import annotations
import queue
import threading
import time
from typing import Dict, List

import pandas as pd
import streamlit as st


def _rerun() -> None:
    """Kompatibler Re-Run-Aufruf für verschiedene Streamlit-Versionen."""

    rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun is not None:
        rerun()

import bio_hippocampal_snn_ctx_theta_cmp_feedback_ctxlearn_hardgate as hippo
from book_ingestion_vsa_adapter_plus_cli import (
    DEMO_TEXT,
    InstrumentedBookAdapter,
    ascii_plot,
)


TRAINING_SEED = 404
REFRESH_DELAY_SEC = 0.6


def _ensure_state() -> None:
    state = st.session_state
    if "metrics_queue" not in state:
        state.metrics_queue = queue.Queue()
    if "status_queue" not in state:
        state.status_queue = queue.Queue()
    if "metrics_history" not in state:
        state.metrics_history: List[Dict[str, float]] = []
    if "interaction_logs" not in state:
        state.interaction_logs: List[str] = []
    if "training_thread" not in state:
        state.training_thread = None
    if "training_running" not in state:
        state.training_running = False
    if "training_status" not in state:
        state.training_status = "Bereit"
    if "training_error" not in state:
        state.training_error = ""
    if "hippo_system" not in state:
        state.hippo_system = None
    if "adapter" not in state:
        state.adapter = None
    if "book_text" not in state:
        state.book_text = ""
    if "book_text_area" not in state:
        state.book_text_area = state.book_text


def _create_adapter(metric_queue: queue.Queue) -> InstrumentedBookAdapter:
    h = hippo.Hyper()
    system = hippo.HippocampalSystem(h, seed=TRAINING_SEED)
    adapter = InstrumentedBookAdapter(system=system)
    adapter.clear_metric_callbacks()

    def _emit(snapshot: Dict[str, float]) -> None:
        metric_queue.put(snapshot)

    adapter.add_metric_callback(_emit)
    st.session_state.hippo_system = system
    st.session_state.adapter = adapter
    return adapter


def _start_training(text: str, chapter_size: int, para_size: int) -> None:
    state = st.session_state
    state.metrics_history = []
    metrics_queue: queue.Queue = queue.Queue()
    status_queue: queue.Queue = queue.Queue()
    state.metrics_queue = metrics_queue
    state.status_queue = status_queue
    adapter = _create_adapter(metrics_queue)

    def _worker() -> None:
        status_queue.put({"type": "status", "value": "Training läuft..."})
        try:
            adapter.ingest_book(text, chapter_size_sents=chapter_size, para_size_sents=para_size)
            status_queue.put({"type": "status", "value": "Training abgeschlossen"})
        except Exception as exc:  # pragma: no cover - nur zur Anzeige in der UI
            status_queue.put({"type": "error", "value": str(exc)})
        finally:
            status_queue.put({"type": "done", "value": None})

    worker = threading.Thread(target=_worker, name="book_training", daemon=True)
    state.training_thread = worker
    state.training_running = True
    state.training_status = "Training läuft..."
    state.training_error = ""
    worker.start()


def _drain_queues() -> None:
    state = st.session_state
    metrics = state.metrics_queue
    while not metrics.empty():
        state.metrics_history.append(metrics.get())
    status_queue = state.status_queue
    while not status_queue.empty():
        msg = status_queue.get()
        mtype = msg.get("type")
        if mtype == "status":
            state.training_status = msg.get("value", "")
        elif mtype == "error":
            state.training_error = msg.get("value", "")
        elif mtype == "done":
            state.training_running = False


def _render_header() -> None:
    st.title("Bio-inspirierter Buch-Lerner – Streamlit UI")
    st.caption(
        "Lade ein Buch hoch oder nutze den Demo-Text, starte das Training und verfolge live die Metriken."
    )


def _render_book_input() -> None:
    state = st.session_state
    with st.expander("📘 Buchquelle", expanded=True):
        col_upload, col_demo = st.columns([2, 1])
        with col_upload:
            uploaded = st.file_uploader("Buch (.txt) hochladen", type=["txt"])
            if uploaded is not None:
                text = uploaded.read().decode("utf-8", errors="ignore")
                state.book_text = text
                state.book_text_area = text
                st.success("Datei übernommen.")
        with col_demo:
            if st.button("Demo-Text laden"):
                state.book_text = DEMO_TEXT
                state.book_text_area = DEMO_TEXT
                _rerun()
        text_value = st.text_area(
            "Buchinhalt (bearbeitbar)",
            value=state.book_text_area,
            height=220,
            key="book_text_area_widget",
        )
        state.book_text = text_value
        state.book_text_area = text_value


def _render_training_controls() -> None:
    state = st.session_state
    with st.expander("⚙️ Training steuern", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            chapter_size = st.number_input("Sätze pro Kapitel", min_value=5, max_value=200, value=40, step=5)
        with col2:
            para_size = st.number_input("Sätze pro Absatz", min_value=2, max_value=40, value=6, step=1)
        with col3:
            st.write("Status:")
            st.info(state.training_status)
            if state.training_error:
                st.error(f"Fehler: {state.training_error}")
        start_disabled = state.training_running
        if st.button("🚀 Training starten", disabled=start_disabled):
            if not state.book_text.strip():
                st.warning("Bitte einen Buchtext angeben.")
            else:
                _start_training(state.book_text, int(chapter_size), int(para_size))
                _rerun()
        if st.button("🔄 System neu initialisieren", disabled=state.training_running):
            state.book_text_area = state.book_text
            state.metrics_history = []
            state.training_status = "Bereit"
            state.training_error = ""
            state.adapter = None
            state.hippo_system = None
            st.success("Systemzustand zurückgesetzt.")


def _render_metrics() -> None:
    state = st.session_state
    adapter: InstrumentedBookAdapter | None = state.adapter
    metrics_history = state.metrics_history
    with st.expander("📈 Live-Metriken", expanded=True):
        if metrics_history:
            df = pd.DataFrame(metrics_history).set_index("step")
            st.line_chart(df[["coherence", "mismatch"]])
            st.line_chart(df[["replay_temp", "hard_gate"]])
            st.bar_chart(df[["engrams"]])
            st.dataframe(df, use_container_width=True)
        else:
            st.write("Noch keine Metriken – starte das Training.")
        if adapter and adapter.metrics.steps:
            cols = st.columns(3)
            last_idx = -1
            cols[0].metric("Schritt", adapter.metrics.steps[last_idx])
            cols[1].metric("Kohärenz", f"{adapter.metrics.coherence[last_idx]:.4f}")
            cols[2].metric("Mismatch", f"{adapter.metrics.mismatch[last_idx]:.4f}")
            st.write(f"Engramme: {adapter.metrics.engrams[last_idx]} | Replay-T: {adapter.metrics.replay_T[last_idx]:.3f} | Hard-Gate: {adapter.metrics.hardgate[last_idx]}")
            ascii_cols = st.columns(2)
            ascii_cols[0].code(ascii_plot(adapter.metrics.coherence, label="Kohärenz", width=48, height=8))
            ascii_cols[0].code(ascii_plot(adapter.metrics.mismatch, label="CA1-Mismatch", width=48, height=8))
            ascii_cols[1].code(ascii_plot([float(x) for x in adapter.metrics.engrams], label="#Engramme", width=48, height=8))
            ascii_cols[1].code(ascii_plot(adapter.metrics.replay_T, label="Replay-Temp", width=48, height=8))
            st.code(ascii_plot([float(x) for x in adapter.metrics.hardgate], label="Hard-Gate", width=72, height=8))
            csv_text = adapter.metrics.to_csv_text()
            st.download_button(
                "📥 Metriken als CSV herunterladen",
                data=csv_text,
                file_name="training_metrics.csv",
                mime="text/csv",
            )
        if adapter and adapter.system:
            st.write(f"Gespeicherte Engramme im System: {len(adapter.system.engrams)}")


def _render_interactions() -> None:
    state = st.session_state
    adapter: InstrumentedBookAdapter | None = state.adapter
    disable_actions = state.training_running or not adapter
    with st.expander("🧠 Nach dem Training interagieren", expanded=True):
        if not adapter:
            st.info("Nach Trainingsende stehen hier Interaktionen zur Verfügung.")
            return
        st.caption("Nutze die Funktionen des CLI-Tools jetzt direkt in der Web-Oberfläche.")
        with st.form("nearest_form", clear_on_submit=False):
            st.subheader("Semantisch ähnliche Absätze")
            query = st.text_input("Satz eingeben", key="nearest_query")
            submitted = st.form_submit_button("Top 5 finden", disabled=disable_actions)
            if submitted:
                try:
                    res = adapter.recall_paragraph_like(query, topk=5)
                except Exception as exc:
                    st.error(f"Fehler: {exc}")
                else:
                    if not res:
                        st.warning("Keine Ergebnisse.")
                    else:
                        for tag, sim in res:
                            meta = adapter.get_paragraph_metadata(tag) if adapter else None
                            if meta:
                                st.write(f"{meta['chapter_label']} – {tag}: Ähnlichkeit {sim:.3f}")
                                st.caption(meta["text"])
                            else:
                                st.write(f"{tag}: Ähnlichkeit {sim:.3f}")
                        state.interaction_logs.append(f"nearest('{query}') -> {len(res)} Treffer")
        with st.form("relation_form", clear_on_submit=False):
            st.subheader("Relation abfragen")
            col1, col2, col3 = st.columns(3)
            subj = col1.text_input("Subjekt", key="rel_subj")
            verb = col2.text_input("Verb", key="rel_verb")
            obj = col3.text_input("Objekt", key="rel_obj")
            submitted = st.form_submit_button("Relation prüfen", disabled=disable_actions)
            if submitted:
                try:
                    pos, neg, conf = adapter.ask_relation(subj, verb, obj)
                except Exception as exc:
                    st.error(f"Fehler: {exc}")
                else:
                    st.write(f"Positiv: {pos}, Negativ: {neg}, Konfidenz: {conf:.2f}")
                    state.interaction_logs.append(f"rel({subj},{verb},{obj}) -> conf {conf:.2f}")
        with st.form("replay_form", clear_on_submit=False):
            st.subheader("Gezieltes Replay anstoßen")
            replay_col1, replay_col2 = st.columns([1, 2])
            chapter_idx = replay_col1.number_input("Kapitelindex", min_value=0, value=0, step=1, disabled=disable_actions)
            paragraph_ids = replay_col2.text_input("Absatz-IDs (Komma-getrennt)", key="replay_ids")
            submitted = st.form_submit_button("Replay starten", disabled=disable_actions)
            if submitted:
                try:
                    ids = [int(x.strip()) for x in paragraph_ids.split(",") if x.strip()]
                    adapter.targeted_replay(int(chapter_idx), ids)
                except Exception as exc:
                    st.error(f"Fehler: {exc}")
                else:
                    st.success("Replay ausgelöst.")
                    state.interaction_logs.append(
                        f"replay(chapter={chapter_idx}, ids={ids})"
                    )
        if state.interaction_logs:
            st.subheader("Aktionsprotokoll")
            st.write("\n".join(state.interaction_logs[-10:]))


def main() -> None:
    st.set_page_config(page_title="Bio-SNN Buchtrainer", layout="wide")
    _ensure_state()
    _drain_queues()
    _render_header()
    _render_book_input()
    _render_training_controls()
    _render_metrics()
    _render_interactions()
    if st.session_state.training_running:
        time.sleep(REFRESH_DELAY_SEC)
        _rerun()


if __name__ == "__main__":
    main()
