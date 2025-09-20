from __future__ import annotations
# ------------------------------------------------------------
# Buch-Lernen (VSA + Hippocampus) mit:
# - erweiterten Parse-Heuristiken
# - Rollenlernen (Hebb)
# - Cortex<->Hippo Doppelschleife
# - Kompressions-Policies
# + NEU: Metrik-Logging, CSV-Export, ASCII-Plots, interaktive CLI
# Nur Standardbibliothek, Python 3.12
# ------------------------------------------------------------

import argparse
import csv
import io
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional

# >>> ggf. anpassen: Import deines Hippocampus-Stacks und Adapters (Plus)
import bio_hippocampal_snn_ctx_theta_cmp_feedback_ctxlearn_hardgate as hippo
import book_ingestion_vsa_adapter_plus as adapter_plus  # die vorherige Datei aus meiner letzten Antwort

# ==============================
# 1) ASCII-Plot & Metrik-Logger
# ==============================

def _min_max(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (0.0, 1.0)
    mn = min(xs)
    mx = max(xs)
    if mx - mn < 1e-9:
        mx = mn + 1.0
    return mn, mx

def ascii_plot(series: List[float], width: int = 64, height: int = 10, label: str = "") -> str:
    """
    Einfache ASCII-„Heatmap“-Zeitreihe: y=0..height-1, x=0..width-1
    skaliert auf min..max. Keine externen Libs.
    """
    if not series:
        return f"{label}: (keine Daten)\n"
    # verdichte/strecke auf width
    N = len(series)
    if N <= width:
        xs = series + [series[-1]] * (width - N)
    else:
        # Mittelung in Blöcken
        block = N / width
        xs = []
        acc = 0.0
        cnt = 0
        idx_target = block
        j = 0
        while len(xs) < width:
            acc += series[j]
            cnt += 1
            if cnt >= idx_target or j == N-1:
                xs.append(acc/cnt)
                acc = 0.0
                cnt = 0
            j = min(N-1, j+1)

    mn, mx = _min_max(xs)
    # Raster
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for x, v in enumerate(xs):
        y = 0 if mx-mn < 1e-9 else int((v - mn) / (mx - mn) * (height - 1))
        y = max(0, min(height - 1, y))
        # zeichne Säule bis y
        for yy in range(y + 1):
            ch = "▁▂▃▄▅▆▇█"[min(7, max(0, int(yy / max(1, height-1) * 7)))]
            grid[height - 1 - yy][x] = ch
    # Achseninfo
    lines = [f"{label}  min={mn:.4f}  max={mx:.4f}"]
    for row in grid:
        lines.append("".join(row))
    return "\n".join(lines) + "\n"

@dataclass
class MetricsLogger:
    coherence: List[float] = field(default_factory=list)
    mismatch:  List[float] = field(default_factory=list)
    engrams:   List[int]   = field(default_factory=list)
    replay_T:  List[float] = field(default_factory=list)
    hardgate:  List[int]   = field(default_factory=list)
    familiarity: List[float] = field(default_factory=list)
    steps:     List[int]   = field(default_factory=list)

    def log(self, step: int, coh: float, mis: float, eng: int, temp: float, hg: int, fam: float) -> None:
        self.steps.append(step)
        self.coherence.append(coh)
        self.mismatch.append(mis)
        self.engrams.append(eng)
        self.replay_T.append(temp)
        self.hardgate.append(hg)
        self.familiarity.append(fam)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step","coherence","mismatch","engrams","replay_temp","hard_gate","familiarity"])
            for i in range(len(self.steps)):
                w.writerow([
                    self.steps[i],
                    f"{self.coherence[i]:.6f}",
                    f"{self.mismatch[i]:.6f}",
                    self.engrams[i],
                    f"{self.replay_T[i]:.6f}",
                    self.hardgate[i],
                    f"{self.familiarity[i]:.6f}",
                ])

    def to_csv_text(self) -> str:
        """Liefert identische CSV-Daten als String (für Downloads)."""
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["step","coherence","mismatch","engrams","replay_temp","hard_gate","familiarity"])
        for i in range(len(self.steps)):
            w.writerow([
                self.steps[i],
                f"{self.coherence[i]:.6f}",
                f"{self.mismatch[i]:.6f}",
                self.engrams[i],
                f"{self.replay_T[i]:.6f}",
                self.hardgate[i],
                f"{self.familiarity[i]:.6f}",
            ])
        return buf.getvalue()

    def as_dicts(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for i in range(len(self.steps)):
                rows.append(
                    {
                        "step": self.steps[i],
                        "coherence": self.coherence[i],
                        "mismatch": self.mismatch[i],
                        "engrams": self.engrams[i],
                        "replay_temp": self.replay_T[i],
                        "hard_gate": self.hardgate[i],
                        "familiarity": self.familiarity[i],
                    }
                )
        return rows

# ==============================
# 2) Instrumentierter Adapter (wrappt den „Plus“-Adapter)
# ==============================

class InstrumentedBookAdapter(adapter_plus.BookAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = MetricsLogger()
        self._step_counter = 0
        self._metric_callbacks: List[Callable[[Dict[str, float]], None]] = []

    def add_metric_callback(self, cb: Callable[[Dict[str, float]], None]) -> None:
        """Registriere Callback, der bei jedem Log ein Snapshot-Dikt erhält."""
        if cb is None:
            return
        self._metric_callbacks.append(cb)

    def clear_metric_callbacks(self) -> None:
        """Entferne alle registrierten Callbacks (z. B. beim Reset)."""
        self._metric_callbacks.clear()

    def _log_now(self):
        # aktuelle Werte holen
        coh = self.cortex.coherence_score()
        mis = sum(self.system.ca1_last_mismatch) / max(1, len(self.system.ca1_last_mismatch))
        eng = len(self.system.engrams)
        temp = self.system.replay_temp
        hg = self.system.hard_gate_countdown
        fam = getattr(self.system, "average_familiarity", 0.0)
        self.metrics.log(self._step_counter, coh, mis, eng, temp, hg, fam)
        if self._metric_callbacks:
            payload = {
                "step": self.metrics.steps[-1],
                "coherence": self.metrics.coherence[-1],
                "mismatch": self.metrics.mismatch[-1],
                "engrams": self.metrics.engrams[-1],
                "replay_temp": self.metrics.replay_T[-1],
                "hard_gate": self.metrics.hardgate[-1],
                "familiarity": self.metrics.familiarity[-1],
            }
            for cb in list(self._metric_callbacks):
                try:
                    cb(payload)
                except Exception:
                    # externe Konsumenten sollen das Logging nicht blockieren
                    pass

    # Override: kleine Hooks an sinnvollen Stellen
    def _stimulate_sentence(self, enc: adapter_plus.EncodedSentence, ctx_ids: List[int], ticks_after: int = 6) -> None:
        super()._stimulate_sentence(enc, ctx_ids, ticks_after)
        self._step_counter += ticks_after + 2  # grobe Schritte (CTX + STIM + ticks)
        self._log_now()

    def compress_chapter(self, chapter_idx: int) -> None:
        super().compress_chapter(chapter_idx)
        self._step_counter += 1
        self._log_now()

    def monitor_and_intervene(self, chapter_idx: int) -> None:
        before = self.system.hard_gate_countdown
        super().monitor_and_intervene(chapter_idx)
        after = self.system.hard_gate_countdown
        # wenn eingegriffen wurde, Fortschritt loggen
        if after != before or after > 0:
            self._step_counter += 1
            self._log_now()

    def ingest_book(
        self,
        text: str,
        chapter_size_sents: int = 40,
        para_size_sents: int = 6,
        *,
        reset_state: bool = True,
    ) -> None:
        # Anfangslog
        self._log_now()
        super().ingest_book(text, chapter_size_sents, para_size_sents, reset_state=reset_state)
        # Abschlusslog
        self._step_counter += 1
        self._log_now()

    def export_brain_state(self) -> bytes:
        payload = self.export_state_dict()
        payload["metrics"] = self.metrics
        payload["step_counter"] = self._step_counter
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def import_brain_state(self, data: bytes) -> None:
        payload = pickle.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Ungültiges Gehirnformat: Erwartet Wörterbuch.")
        metrics = payload.pop("metrics", None)
        step_counter = payload.pop("step_counter", 0)
        super().import_state_dict(payload)
        if isinstance(metrics, MetricsLogger):
            self.metrics = metrics
            current = getattr(self.metrics, "familiarity", [])
            if len(current) < len(self.metrics.steps):
                current = list(current) + [0.0 for _ in range(len(self.metrics.steps) - len(current))]
            self.metrics.familiarity = current
        else:
            self.metrics = MetricsLogger()
        self._step_counter = int(step_counter) if step_counter is not None else 0
        # Falls importierte Metriken leer sind, aktuellen Zustand loggen, damit UI Werte hat.
        if not self.metrics.steps:
            self._log_now()

# ==============================
# 3) CLI
# ==============================

HELP_TEXT = """
Befehle:
  nearest <Satz>                 – Top-Absatz-Tags (semantisch ähnlich) zeigen
  rel <Subj> <Verb> <Obj>        – Relationszähler & Konfidenz anzeigen
  stats                          – letzte Metriken kurz anzeigen
  plot                           – ASCII-Plots (Kohärenz, Mismatch, Engramme, Replay-T, Hard-Gate)
  savecsv <pfad.csv>             – Metriken als CSV exportieren
  replay <kapitel> <p1,p2,...>   – gezieltes Replay für Absätze starten
  help                           – diese Hilfe
  exit                           – beenden
"""

def print_stats(inst: InstrumentedBookAdapter) -> None:
    m = inst.metrics
    if not m.steps:
        print("Noch keine Metriken.")
        return
    i = -1
    print(f"Schritt: {m.steps[i]}")
    print(f"Coherence: {m.coherence[i]:.4f}")
    print(f"CA1-Mismatch: {m.mismatch[i]:.4f}")
    print(f"Engramme: {m.engrams[i]}")
    print(f"Replay-Temp: {m.replay_T[i]:.2f}")
    print(f"Hard-Gate: {m.hardgate[i]}")
    print(f"Familiarität: {m.familiarity[i]:.2f}")

def print_plots(inst: InstrumentedBookAdapter) -> None:
    m = inst.metrics
    print(ascii_plot(m.coherence, label="Kohärenz", width=64, height=8))
    print(ascii_plot(m.mismatch,  label="CA1-Mismatch", width=64, height=8))
    # Engramme als float wandeln für Plot
    print(ascii_plot([float(x) for x in m.engrams], label="#Engramme", width=64, height=8))
    print(ascii_plot(m.replay_T,  label="Replay-Temp", width=64, height=8))
    print(ascii_plot([float(x) for x in m.hardgate], label="Hard-Gate", width=64, height=8))
    print(ascii_plot(m.familiarity, label="Familiarität", width=64, height=8))

def run_cli(inst: InstrumentedBookAdapter) -> None:
    print("\nInteraktive CLI – tippe 'help' für Hilfe, 'exit' zum Beenden.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTschüss.")
            break
        if not line:
            continue
        if line.lower() in {"exit","quit"}:
            print("Tschüss.")
            break
        if line.lower() in {"help","?"}:
            print(HELP_TEXT)
            continue
        if line.startswith("nearest "):
            q = line[len("nearest "):].strip()
            res = inst.recall_paragraph_like(q, topk=5)
            for tag, sim in res:
                print(f" {tag:>12s}   sim={sim:.3f}")
            continue
        if line.startswith("rel "):
            parts = line.split()
            if len(parts) < 4:
                print("Nutzen: rel <Subj> <Verb> <Obj>")
                continue
            subj, verb, obj = parts[1], parts[2], " ".join(parts[3:])
            pos, neg, conf = inst.ask_relation(subj, verb, obj)
            print(f"Pos:{pos}  Neg:{neg}  Konf:{conf:.2f}")
            continue
        if line == "stats":
            print_stats(inst)
            continue
        if line == "plot":
            print_plots(inst)
            continue
        if line.startswith("savecsv "):
            path = line[len("savecsv "):].strip()
            try:
                inst.metrics.to_csv(path)
                print(f"Gespeichert: {path}")
            except Exception as e:
                print(f"Fehler beim Speichern: {e}")
            continue
        if line.startswith("replay "):
            parts = line.split()
            if len(parts) < 3:
                print("Nutzen: replay <kapitelIndex> <p1,p2,...>")
                continue
            try:
                ch = int(parts[1])
                plist = [int(x) for x in parts[2].split(",") if x]
                inst.targeted_replay(ch, plist)
                print("Gezieltes Replay ausgelöst.")
            except Exception as e:
                print(f"Fehler: {e}")
            continue
        print("Unbekannter Befehl. 'help' zeigt Hilfe.")

# ==============================
# 4) Main
# ==============================

DEMO_TEXT = """
Kapitel Eins. Der Hund bellt im Garten. Die Katze schläft auf dem Sofa. Der Hund jagt die Katze.
Später bellt der Hund nicht. Die Katze läuft in den Garten. Der Junge steht auf und ruft.
Kapitel Zwei. Der Mond scheint über dem See. Der Hund schläft. Die Katze jagt nicht den Hund.
"""

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Buch-Lernen (VSA + Hippocampus) – CLI & Metriken")
    parser.add_argument("--book", type=str, default="", help="Pfad zu buch.txt (UTF-8)")
    parser.add_argument("--demo", action="store_true", help="Demo-Text verwenden (Default, wenn --book fehlt)")
    parser.add_argument("--chapter-size", type=int, default=40, help="Sätze pro Kapitel (Chunking)")
    parser.add_argument("--para-size", type=int, default=6, help="Sätze pro Absatz (Chunking)")
    args = parser.parse_args(argv)

    h = hippo.Hyper()
    sys = hippo.HippocampalSystem(h, seed=404)
    inst = InstrumentedBookAdapter(system=sys)

    if args.book and os.path.exists(args.book):
        with open(args.book, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"-> Lese Buch: {args.book}")
    else:
        print("-> Demo-Text wird verwendet. (Nutze --book pfad/zu/buch.txt für eigenes Buch)")
        text = DEMO_TEXT

    t0 = time.time()
    inst.ingest_book(text, chapter_size_sents=args.chapter_size, para_size_sents=args.para_size)
    dt = time.time() - t0
    print(f"\nIngestion fertig. Dauer: {dt:.2f}s   Engramme: {len(sys.engrams)}")
    print_stats(inst)
    print("\nKurze Plots (ASCII):")
    print_plots(inst)

    run_cli(inst)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
