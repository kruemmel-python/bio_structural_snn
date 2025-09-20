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
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

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
    steps:     List[int]   = field(default_factory=list)

    def log(self, step: int, coh: float, mis: float, eng: int, temp: float, hg: int) -> None:
        self.steps.append(step)
        self.coherence.append(coh)
        self.mismatch.append(mis)
        self.engrams.append(eng)
        self.replay_T.append(temp)
        self.hardgate.append(hg)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step","coherence","mismatch","engrams","replay_temp","hard_gate"])
            for i in range(len(self.steps)):
                w.writerow([self.steps[i], f"{self.coherence[i]:.6f}", f"{self.mismatch[i]:.6f}",
                            self.engrams[i], f"{self.replay_T[i]:.6f}", self.hardgate[i]])

# ==============================
# 2) Instrumentierter Adapter (wrappt den „Plus“-Adapter)
# ==============================

class InstrumentedBookAdapter(adapter_plus.BookAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = MetricsLogger()
        self._step_counter = 0

    def _log_now(self):
        # aktuelle Werte holen
        coh = self.cortex.coherence_score()
        mis = sum(self.system.ca1_last_mismatch) / max(1, len(self.system.ca1_last_mismatch))
        eng = len(self.system.engrams)
        temp = self.system.replay_temp
        hg = self.system.hard_gate_countdown
        self.metrics.log(self._step_counter, coh, mis, eng, temp, hg)

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

    def ingest_book(self, text: str, chapter_size_sents: int = 40, para_size_sents: int = 6) -> None:
        # Anfangslog
        self._log_now()
        super().ingest_book(text, chapter_size_sents, para_size_sents)
        # Abschlusslog
        self._step_counter += 1
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

def print_plots(inst: InstrumentedBookAdapter) -> None:
    m = inst.metrics
    print(ascii_plot(m.coherence, label="Kohärenz", width=64, height=8))
    print(ascii_plot(m.mismatch,  label="CA1-Mismatch", width=64, height=8))
    # Engramme als float wandeln für Plot
    print(ascii_plot([float(x) for x in m.engrams], label="#Engramme", width=64, height=8))
    print(ascii_plot(m.replay_T,  label="Replay-Temp", width=64, height=8))
    print(ascii_plot([float(x) for x in m.hardgate], label="Hard-Gate", width=64, height=8))

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
