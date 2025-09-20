from __future__ import annotations
"""Simplified book ingestion adapter with deterministic text heuristics.

This module provides the ``BookAdapter`` class that was referenced by the
Streamlit UI and the CLI instrumentation but was missing from the repository.
The goal is not to emulate the full hippocampal pipeline from the original
research code, but to supply a lightweight, fully self-contained adapter that
exposes the same public surface area so that the UI can train a book, obtain
metrics and answer interactive queries.

The implementation uses simple bag-of-words statistics to approximate
coherence, mismatch and engram behaviour:

* Sentences are tokenised using regular expressions and grouped into
  paragraphs/chapters according to the requested sizes.
* Each sentence update refreshes a miniature ``SimpleCortex`` structure which
  tracks lexical overlap to derive a coherence score.
* ``HippocampalSystem`` instances receive dynamically attached attributes such
  as ``ca1_last_mismatch`` or ``engrams`` if they are not already present so
  the rest of the project can continue to consume them transparently.
* Query helpers ``recall_paragraph_like`` and ``ask_relation`` operate on the
  stored paragraph metadata using cosine similarity and simple pattern
  matching.

The goal of this shim is to keep all higher level functionality working while
remaining dependency free and easy to reason about.
"""

from dataclasses import dataclass, field
import math
import random
import re
import time
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def _normalise_word(word: str) -> str:
    return word.lower()


def _sentence_tokens(sentence: str) -> List[str]:
    return [_normalise_word(tok) for tok in TOKEN_RE.findall(sentence)]


@dataclass
class EncodedSentence:
    """Container mirroring the structure used by the instrumentation layer."""

    text: str
    tokens: List[str]
    ctx_ids: List[int] = field(default_factory=list)
    paragraph_tag: str = ""


class SimpleCortex:
    """Very small helper that mimics a cortex ``coherence_score`` call."""

    def __init__(self, memory: int = 32) -> None:
        self._history: List[Counter[str]] = []
        self._memory = memory

    def observe(self, tokens: Iterable[str]) -> None:
        bag = Counter(tokens)
        self._history.append(bag)
        if len(self._history) > self._memory:
            self._history.pop(0)

    def coherence_score(self) -> float:
        if len(self._history) < 2:
            return 1.0 if self._history else 0.0
        # Average pairwise cosine similarity across the history window.
        sims = []
        for i in range(1, len(self._history)):
            sims.append(_cosine(self._history[i - 1], self._history[i]))
        if not sims:
            return 0.0
        return sum(sims) / len(sims)


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Paragraph:
    tag: str
    chapter_idx: int
    paragraph_idx: int
    text: str
    sentences: List[EncodedSentence]
    token_counts: Counter[str]


def _chunk(seq: Sequence[str], size: int) -> List[List[str]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def _ensure_attr(obj: object, name: str, default):
    if not hasattr(obj, name):
        setattr(obj, name, default)
    return getattr(obj, name)


class BookAdapter:
    """Minimal yet functional book ingestion pipeline.

    The original project provided a much richer adapter written in pure Python
    which is not part of this repository. To keep the rest of the tooling
    operational this replacement focuses on the observable behaviour that the
    Streamlit UI and the CLI need.
    """

    def __init__(self, system: Optional[object] = None) -> None:
        self.system = system or object()
        self.cortex = SimpleCortex()
        self._paragraphs: List[Paragraph] = []
        self._last_tokens: Counter[str] = Counter()
        self._rng = random.Random(1234)

        # Ensure the downstream instrumentation always finds the expected fields
        _ensure_attr(self.system, "ca1_last_mismatch", [0.0])
        _ensure_attr(self.system, "engrams", [])
        _ensure_attr(self.system, "replay_temp", 1.0)
        _ensure_attr(self.system, "hard_gate_countdown", 0)

    # ------------------------------------------------------------------
    # High level ingestion API
    # ------------------------------------------------------------------
    def ingest_book(self, text: str, chapter_size_sents: int = 40, para_size_sents: int = 6) -> None:
        sentences = [s.strip() for s in re.split(r"[\n\.\?!]+", text) if s.strip()]
        if not sentences:
            return

        chapters = _chunk(sentences, chapter_size_sents)
        timestamp = time.time()
        self._paragraphs = []
        self.system.engrams.clear()

        for ch_idx, chapter in enumerate(chapters):
            paragraphs = _chunk(chapter, para_size_sents)
            for para_idx, para_sentences in enumerate(paragraphs):
                tag = f"chap{ch_idx:02d}_para{para_idx:02d}"
                encoded_sentences: List[EncodedSentence] = []
                for sentence in para_sentences:
                    tokens = _sentence_tokens(sentence)
                    ctx_ids = self._context_ids_for(tokens)
                    enc = EncodedSentence(
                        text=sentence,
                        tokens=tokens,
                        ctx_ids=ctx_ids,
                        paragraph_tag=tag,
                    )
                    self._stimulate_sentence(enc, ctx_ids)
                    encoded_sentences.append(enc)
                paragraph_counter = Counter(token for s in encoded_sentences for token in s.tokens)
                paragraph = Paragraph(
                    tag=tag,
                    chapter_idx=ch_idx,
                    paragraph_idx=para_idx,
                    text=" ".join(p.text for p in encoded_sentences),
                    sentences=encoded_sentences,
                    token_counts=paragraph_counter,
                )
                self._paragraphs.append(paragraph)
                self.system.engrams.append(
                    {
                        "ids": list(range(len(encoded_sentences))),
                        "tag": tag,
                        "saliency": max(0.05, min(1.0, self.cortex.coherence_score())),
                        "ts": timestamp,
                    }
                )
                timestamp += 1.0
            self.compress_chapter(ch_idx)
            self.monitor_and_intervene(ch_idx)

    # ------------------------------------------------------------------
    # Interactive helpers mirroring the original adapter
    # ------------------------------------------------------------------
    def recall_paragraph_like(self, query: str, topk: int = 5) -> List[Tuple[str, float]]:
        tokens = Counter(_sentence_tokens(query))
        if not tokens:
            return []
        scored = []
        for para in self._paragraphs:
            score = _cosine(tokens, para.token_counts)
            if score > 0.0:
                scored.append((para.tag, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

    def ask_relation(self, subj: str, verb: str, obj: str) -> Tuple[int, int, float]:
        subj_l = _normalise_word(subj)
        verb_l = _normalise_word(verb)
        obj_l = _normalise_word(obj)
        positive = 0
        negative = 0
        for para in self._paragraphs:
            tokens = [tok for s in para.sentences for tok in s.tokens]
            if subj_l in tokens and verb_l in tokens and obj_l in tokens:
                # Prefer an ordered match, otherwise treat as uncertain.
                if _ordered_contains(tokens, [subj_l, verb_l, obj_l]):
                    positive += 1
                else:
                    negative += 1
            elif subj_l in tokens and verb_l in tokens:
                negative += 1
        confidence = positive / (positive + negative) if (positive + negative) else 0.0
        return positive, negative, confidence

    def targeted_replay(self, chapter_idx: int, paragraph_indices: Sequence[int]) -> None:
        """Mimic a replay by nudging the replay temperature and gate."""
        self.system.replay_temp = max(0.1, self.system.replay_temp * 0.9)
        if paragraph_indices:
            self.system.hard_gate_countdown = max(self.system.hard_gate_countdown - 1, 0)

    # ------------------------------------------------------------------
    # Hooks overridden by ``InstrumentedBookAdapter``
    # ------------------------------------------------------------------
    def _stimulate_sentence(self, enc: EncodedSentence, ctx_ids: List[int], ticks_after: int = 6) -> None:
        tokens = Counter(enc.tokens)
        self.cortex.observe(enc.tokens)
        mismatch = 1.0 - _cosine(tokens, self._last_tokens)
        self.system.ca1_last_mismatch = [mismatch]
        self._last_tokens = tokens

        # Temperature gradually adapts based on lexical novelty.
        base = getattr(self.system, "replay_temp", 1.0)
        self.system.replay_temp = max(0.05, min(5.0, base * (1.0 + 0.1 * (mismatch - 0.5))))
        # Hard gate activates if novelty is too high.
        if mismatch > 0.8:
            self.system.hard_gate_countdown = 3
        elif self.system.hard_gate_countdown > 0:
            self.system.hard_gate_countdown -= 1

    def compress_chapter(self, chapter_idx: int) -> None:
        # Slightly decay replay temperature after a chapter to mimic consolidation.
        self.system.replay_temp = max(0.1, self.system.replay_temp * 0.95)

    def monitor_and_intervene(self, chapter_idx: int) -> None:
        # Randomly clear the hard gate to emulate operator interventions.
        if self.system.hard_gate_countdown > 0 and self._rng.random() < 0.3:
            self.system.hard_gate_countdown -= 1

    # Internal helpers -------------------------------------------------
    def _context_ids_for(self, tokens: Sequence[str]) -> List[int]:
        # Stable hash to derive pseudo context neuron ids.
        ctx_ids = {abs(hash(tok)) % 997 for tok in tokens}
        if not ctx_ids:
            return [0]
        return sorted(ctx_ids)


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------

def _ordered_contains(tokens: Sequence[str], pattern: Sequence[str]) -> bool:
    if len(pattern) > len(tokens):
        return False
    it = iter(tokens)
    try:
        for target in pattern:
            while True:
                current = next(it)
                if current == target:
                    break
        return True
    except StopIteration:
        return False

