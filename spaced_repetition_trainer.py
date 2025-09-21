from __future__ import annotations
# ============================================================
# Spaced Repetition Trainer (SM-2-inspiriert) für dein Projekt
# - Wiederholungsplanung über "Steps" (keine Echtzeit nötig)
# - Nutzt Adapter-Metriken: coherence(), ca1_last_mismatch
# - Kontextvariation (leichter Jitter über Pseudo-Kontext)
# - Mismatch/Coherence -> Intervallanpassung, gezieltes Replay
# - Schlaf-/Replay-Blöcke nach Batches (Konsolidierung)
# Nur Standardbibliothek, Python 3.12
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random

# Dein vorhandenes (vereinfachtes) Adapter-Modul:
import book_ingestion_vsa_adapter_plus as ap  # BookAdapter nutzt SimpleCortex & system-Stub


# --------------------------- Repräsentation -----------------------------

@dataclass
class RehearsalItem:
    tag: str                  # "chapXX_paraYY" (Absatz) oder "chapXX" (Kapitel)
    chapter_idx: int
    para_idx: Optional[int]   # None für Kapitel-Wiederholung
    due_at_step: int = 0
    reps: int = 0
    interval: int = 20
    ease: float = 2.3         # SM-2-ähnliche „Leichtigkeit“


@dataclass
class ScheduleConfig:
    min_interval: int = 20
    max_interval: int = 10_000
    seed: int = 123
    # Qualitätsgewichtung: hoch für Kohärenz, niedrig für Mismatch
    w_coh: float = 4.0
    w_mis: float = 5.0


# ----------------------------- Scheduler -------------------------------

class SpacedScheduler:
    """Einfacher SM-2-inspirierter Planer auf 'Steps' statt Uhrzeit."""

    def __init__(self, cfg: ScheduleConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def _quality(self, coherence: float, mismatch: float) -> int:
        """
        0..5 (wie SM-2). Mismatch wird invers belohnt, Kohärenz direkt.
        Hart: starker Mismatch -> sehr schlechte Note.
        """
        if mismatch > 0.20:
            return 1
        if coherence < 0.55:
            return 2
        score = self.cfg.w_coh * max(0.0, min(1.0, coherence)) + self.cfg.w_mis * max(0.0, 1.0 - min(1.0, mismatch * 5))
        if score > 7.5:
            return 5
        if score > 6.5:
            return 4
        if score > 5.5:
            return 3
        return 2

    def seed_interval(self, policy: str) -> int:
        base = self.cfg.min_interval
        match policy:
            case "ebbinghaus":
                # grobe Stufen: kurz / mittel / länger
                return self.rng.choice([base, base * 10, base * 60])
            case "fixed":
                return max(base, 200)
            case "adaptive" | _:
                return base

    def update(self, item: RehearsalItem, coherence: float, mismatch: float, now_step: int) -> None:
        q = self._quality(coherence, mismatch)
        if q < 3:
            item.reps = 0
            item.interval = max(self.cfg.min_interval, item.interval // 2)
            item.ease = max(1.3, item.ease - 0.2)
        else:
            item.reps += 1
            if item.reps == 1:
                item.interval = max(self.cfg.min_interval, item.interval)
            elif item.reps == 2:
                item.interval = max(self.cfg.min_interval, int(item.interval * 2.5))
            else:
                item.ease = min(2.8, item.ease + 0.05)
                item.interval = min(self.cfg.max_interval, int(item.interval * item.ease))
        jitter = int(0.1 * item.interval)
        item.due_at_step = now_step + item.interval + self.rng.randint(-jitter, +jitter)


# ------------------------------- Trainer -------------------------------

@dataclass
class Trainer:
    adapter: ap.BookAdapter
    scheduler: SpacedScheduler = field(default_factory=lambda: SpacedScheduler(ScheduleConfig()))
    replay_after_n_items: int = 6         # Batchgröße für Schlaf-/Replay-Block
    ctx_jitter_strength: int = 2          # kleine Kontextvariation
    max_steps: int = 50_000               # Schutz, falls jemand unendliche Epochen setzt
    log: List[Dict[str, float]] = field(default_factory=list)

    # ---- Hilfen (Metriken/Stimuli) ----
    def _metrics(self) -> Tuple[float, float, int, float, int, float]:
        coh = self.adapter.cortex.coherence_score()
        mis = sum(self.adapter.system.ca1_last_mismatch) / max(1, len(self.adapter.system.ca1_last_mismatch))
        eng = len(self.adapter.system.engrams)
        temp = getattr(self.adapter.system, "replay_temp", 1.0)
        hg = getattr(self.adapter.system, "hard_gate_countdown", 0)
        fam = getattr(self.adapter.system, "average_familiarity", 0.0)
        return coh, mis, eng, temp, hg, fam

    def _log_now(self, step: int) -> None:
        coh, mis, eng, temp, hg, fam = self._metrics()
        self.log.append({
            "step": float(step),
            "coherence": float(coh),
            "mismatch": float(mis),
            "engrams": float(eng),
            "replay_temp": float(temp),
            "hard_gate": float(hg),
            "familiarity": float(fam),
        })

    def _paragraphs(self) -> List[Tuple[str, int, int]]:
        """
        Liefert Liste (tag, chapter_idx, para_idx).
        Greift auf die öffentliche Struktur des vereinfachten Adapters zu.
        """
        out: List[Tuple[str, int, int]] = []
        for p in getattr(self.adapter, "_paragraphs", []):
            out.append((p.tag, p.chapter_idx, p.paragraph_idx))
        return out

    def _stimulate_paragraph(self, tag: str) -> None:
        """
        Re-stimuliert einen Absatz, indem wir seine Sätze erneut durch den internen
        Stimulus-Pfad schicken. Das ist *absichtlich* „nah am Hirn“ (Wieder-Aktivierung).
        """
        para = self.adapter._paragraph_lookup.get(tag)  # bewusst intern; didaktisch in Ordnung
        if not para:
            return

        def jitter(ids: List[int]) -> List[int]:
            ids = ids[:]
            for _ in range(max(0, self.ctx_jitter_strength)):
                if ids:
                    ids[self.scheduler.rng.randrange(len(ids))] = self.scheduler.rng.randrange(997)
            return ids

        for enc in para.sentences:
            ctx_ids = jitter(enc.ctx_ids)
            self.adapter._stimulate_sentence(enc, ctx_ids, ticks_after=4)

    def _sleep_replay(self, rounds: int = 2) -> None:
        """
        Der vereinfachte Adapter hat keinen echten Hippocampus-Schlaf.
        Wir emulieren Konsolidierung, indem wir die "Replay-Temperatur" leicht senken
        und optional gezieltes Replay triggern.
        """
        base = getattr(self.adapter.system, "replay_temp", 1.0)
        self.adapter.system.replay_temp = max(0.05, base * 0.93)
        engrams = list(getattr(self.adapter.system, "engrams", []))
        if not engrams:
            return
        picks = [e for e in sorted(engrams, key=lambda x: x.get("ts", 0.0))[:2]]
        for e in picks:
            tag = e.get("tag", "")
            meta = self.adapter.get_paragraph_metadata(tag)
            if meta:
                self.adapter.targeted_replay(meta["chapter_index"], [meta["paragraph_index"]])

    # ---- Öffentliches API ----
    def build_queue(self, include_chapter_nodes: bool = True) -> List[RehearsalItem]:
        q: List[RehearsalItem] = []
        seen_ch: set[int] = set()
        for tag, ch, pi in self._paragraphs():
            q.append(RehearsalItem(tag=tag, chapter_idx=ch, para_idx=pi))
            seen_ch.add(ch)
        if include_chapter_nodes:
            for ch in sorted(seen_ch):
                q.append(RehearsalItem(tag=f"chap{ch:02d}", chapter_idx=ch, para_idx=None))
        return q

    def repeat(self, epochs: int = 3, policy: str = "ebbinghaus") -> None:
        """
        Führt echtes Wiederholen durch. Jede Epoche:
         - Pick das jeweils fällige Item
         - Stimuliere Absatz (mit leichtem Kontext-Jitter)
         - Messe Kohärenz/Mismatch -> Update Intervall
         - Jede N Items: „Schlaf“
        """
        queue = self.build_queue(include_chapter_nodes=True)
        for it in queue:
            it.interval = self.scheduler.seed_interval(policy)
            it.due_at_step = it.interval

        items_since_sleep = 0
        step = 0
        self._log_now(step)

        for _ in range(max(1, epochs)):
            while step < self.max_steps:
                due = [it for it in queue if it.due_at_step <= step]
                if not due:
                    step += 10
                    if step % 100 == 0:
                        self._log_now(step)
                    continue

                due.sort(key=lambda it: it.due_at_step)
                item = due[0]

                self._stimulate_paragraph(item.tag)

                coh, mis, *_ = self._metrics()
                if mis > 0.08 or coh < 0.6:
                    meta = self.adapter.get_paragraph_metadata(item.tag)
                    if meta:
                        self.adapter.targeted_replay(meta["chapter_index"], [meta["paragraph_index"]])

                self.scheduler.update(item, coherence=coh, mismatch=mis, now_step=step)

                items_since_sleep += 1
                if items_since_sleep >= self.replay_after_n_items:
                    self._sleep_replay(rounds=2)
                    items_since_sleep = 0

                step += max(5, item.interval // 4)
                self._log_now(step)

                if all(it.interval > 2000 for it in queue):
                    break

            self.adapter.system.replay_temp = max(0.05, self.adapter.system.replay_temp * 0.9)
            step = 0
            self._log_now(step)
