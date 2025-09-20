from __future__ import annotations

# ------------------------------------------------------------
# Buch-Lernen: VSA/HRR-Encoder, semantischer Mini-Cortex, Sequenzen
# und Adapter -> HippocampalSystem (DG/Ctx/Stimulation + Replay).
# Nur Standardbibliothek, Python 3.12.
# ------------------------------------------------------------

import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# >>> PASSE DIES AN: Name deines Hippocampus-Moduls
import bio_hippocampal_snn_ctx_theta_cmp_feedback_ctxlearn_hardgate as hippo

# ==============================
# 1) VSA / HRR (ohne NumPy, rein Python)
# ==============================

@dataclass
class VSA:
    dim: int = 1024
    rng_seed: int = 123

    def __post_init__(self):
        random.seed(self.rng_seed)
        self.lexicon: Dict[str, List[int]] = {}
        # Basis-Rollen / Positionsvektoren
        self.roles: Dict[str, List[int]] = {
            "SUBJ": self.rand_vec(),
            "VERB": self.rand_vec(),
            "OBJ":  self.rand_vec(),
            "POS":  self.rand_vec(),   # Basis für Positionskodierung
            "NEXT": self.rand_vec(),   # Sequenz-Link
            "SENT": self.rand_vec(),
            "PARA": self.rand_vec(),
            "CHAP": self.rand_vec(),
        }

    # ±1-Randomvektor
    def rand_vec(self) -> List[int]:
        return [1 if random.random() < 0.5 else -1 for _ in range(self.dim)]

    # Superposition (Mehrheitszeichen, bei Gleichheit zufällig)
    def superpose(self, *vecs: List[int]) -> List[int]:
        s = [0]*self.dim
        for v in vecs:
            for i,x in enumerate(v):
                s[i] += x
        out = []
        for i, val in enumerate(s):
            if val > 0: out.append(1)
            elif val < 0: out.append(-1)
            else: out.append(1 if random.random() < 0.5 else -1)
        return out

    # Zirkulare Faltung (HRR-Binding) – O(n^2) Variante, reicht didaktisch
    def bind(self, a: List[int], b: List[int]) -> List[int]:
        n = self.dim
        out = [0]*n
        for i in range(n):
            s = 0
            for j in range(n):
                k = (i - j) % n
                s += a[j] * b[k]
            out[i] = 1 if s >= 0 else -1
        return out

    # Zirkulare Korrelation (ungefähr inverse Bindung)
    def unbind(self, a: List[int], b: List[int]) -> List[int]:
        n = self.dim
        out = [0]*n
        for i in range(n):
            s = 0
            for j in range(n):
                k = (i + j) % n
                s += a[j] * b[k]
            out[i] = 1 if s >= 0 else -1
        return out

    # Positionsvektor (Power der POS-Rolle)
    def pos_vec(self, idx: int) -> List[int]:
        v = self.roles["POS"]
        out = v
        for _ in range(max(0, idx-1)):
            out = self.bind(out, v)
        return out

    def word_vec(self, w: str) -> List[int]:
        key = w.lower()
        if key not in self.lexicon:
            self.lexicon[key] = self.rand_vec()
        return self.lexicon[key]

    # Kosinus-Ähnlichkeit für ±1-Vektoren = normalisierte Punktzahl
    def cosine(self, a: List[int], b: List[int]) -> float:
        dot = sum(x*y for x,y in zip(a,b))
        return dot / float(self.dim)

# ==============================
# 2) Sehr simpler Parser & Encoder
# ==============================

SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')
WORD_SPLIT = re.compile(r'\s+')
PUNCT = re.compile(r'[^\wäöüÄÖÜß\-]')

@dataclass
class EncodedSentence:
    sent_vec: List[int]
    words: List[str]
    triples: List[Tuple[str,str,str]]  # (subj, verb, obj)

class SimpleEncoder:
    """
    Minimale, erklärbare Pipeline:
    - Sätze: split per . ! ? (heuristisch)
    - Wörter: whitespace; Reinigung rudimentär, keine ML
    - Rollenbindung: SUBJ/VERB/OBJ (heuristisch: erstes Substantiv ~ SUBJ, erstes Verb ~ VERB, folgendes Substantiv ~ OBJ)
    - Positionskodierung bindet Wort mit POS^i
    - Satzvektor = superpose( SENT ⊗ (Summe(Positionen ⊗ Wort)) , Rollenbindungen )
    """
    def __init__(self, vsa: VSA):
        self.vsa = vsa

    def clean_word(self, w: str) -> str:
        return PUNCT.sub('', w).strip('-_').lower()

    def guess_pos(self, w: str) -> str:
        # super-simple Heuristik: Endungen '-en' => Verbkandidat (Infinitiv), großgeschriebenes erstes Wort im Satz => Nomen (Deutsch)
        # Didaktischer Zweck, KEIN echtes POS-Tagging
        if len(w) > 2 and w.endswith('en'):
            return 'V'
        if len(w) > 1 and w[0].isupper():
            return 'N'
        # Fallbacks
        if len(w) > 3 and w.endswith('ung'):
            return 'N'
        return 'X'

    def to_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        return SENT_SPLIT.split(text)

    def encode_sentence(self, sent: str) -> EncodedSentence:
        toks = [self.clean_word(w) for w in WORD_SPLIT.split(sent) if w.strip()]
        toks = [t for t in toks if t]  # remove empty
        words = toks[:]

        # Positionskodierung
        pos_vectors = []
        for i, w in enumerate(toks):
            wv = self.vsa.word_vec(w)
            pv = self.vsa.pos_vec(i+1)
            pos_vectors.append(self.vsa.bind(wv, pv))

        base = self.vsa.roles["SENT"]
        bag = self.vsa.superpose(*pos_vectors) if pos_vectors else self.vsa.rand_vec()
        sent_core = self.vsa.bind(base, bag)

        # Rollenbindungen (heuristisch)
        subj = None; verb = None; obj = None
        for w in toks:
            if subj is None and self.guess_pos(w) == 'N':
                subj = w
            if verb is None and self.guess_pos(w) == 'V':
                verb = w
            if obj is None and subj and verb and self.guess_pos(w) == 'N' and w != subj:
                obj = w
        triples = []
        role_bindings = []
        if subj and verb and obj:
            triples.append((subj, verb, obj))
            role_bindings.append(self.vsa.bind(self.vsa.roles["SUBJ"], self.vsa.word_vec(subj)))
            role_bindings.append(self.vsa.bind(self.vsa.roles["VERB"], self.vsa.word_vec(verb)))
            role_bindings.append(self.vsa.bind(self.vsa.roles["OBJ"],  self.vsa.word_vec(obj)))

        sent_vec = self.vsa.superpose(sent_core, *role_bindings) if role_bindings else sent_core
        return EncodedSentence(sent_vec=sent_vec, words=words, triples=triples)

# ==============================
# 3) Semantischer Mini-Cortex (Clean-Up Memory + Relationsnetz)
# ==============================

@dataclass
class CortexMemory:
    vsa: VSA
    # speichert Sätze/Absätze/Kapitel als Vektoren
    store: List[Tuple[str, List[int]]] = field(default_factory=list)  # (tag, vec)
    # einfaches Relationsnetz: Kanten-Counts und Negationshinweise
    relations: Dict[Tuple[str,str,str], int] = field(default_factory=dict)
    negations: Dict[Tuple[str,str,str], int] = field(default_factory=dict)

    def add_vector(self, tag: str, vec: List[int]) -> None:
        self.store.append((tag, vec))

    def nearest(self, q: List[int], k: int = 5) -> List[Tuple[str, float]]:
        sims = [(tag, self.vsa.cosine(q, v)) for tag, v in self.store]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def add_triples(self, triples: List[Tuple[str,str,str]], sent_text: str) -> None:
        # sehr simpel: Negation, wenn „nicht/kein“ im Satztext
        neg = (' nicht ' in (' ' + sent_text.lower() + ' ')) or (' kein ' in (' ' + sent_text.lower() + ' '))
        for t in triples:
            self.relations[t] = self.relations.get(t, 0) + 1
            if neg:
                self.negations[t] = self.negations.get(t, 0) + 1

    def coherence_score(self) -> float:
        # Widerspruch, wenn Negationszähler nahe/über Positivzähler
        if not self.relations: return 1.0
        inconsist = 0
        total = 0
        for t, c in self.relations.items():
            n = self.negations.get(t, 0)
            total += c
            if n >= max(1, c//2):  # grobe Heuristik
                inconsist += 1
        return max(0.0, 1.0 - inconsist / max(1, len(self.relations)))

# ==============================
# 4) Projektion VSA → DG (sparse) & Kontext → Ctx
# ==============================

@dataclass
class SparseProjector:
    dg_size: int
    ctx_size: int
    k_active: int = 12
    rng_seed: int = 777

    def __post_init__(self):
        random.seed(self.rng_seed)
        # fester Random-„Projektor“ pro Index in VSA
        self.perm_dg = list(range(self.dg_size))
        random.shuffle(self.perm_dg)
        self.perm_ctx = list(range(self.ctx_size))
        random.shuffle(self.perm_ctx)

    def to_dg_ids(self, vec: List[int]) -> List[int]:
        """
        Mappe ±1-VSA-Vektor deterministisch auf k DG-IDs:
        - wähle die k höchsten Absolutbeiträge über stabile Hash-Fenster
          (hier: simple Segmentierung + Mehrheitszeichen)
        """
        n = len(vec)
        segs = self.k_active
        seg_len = n // segs
        ids = []
        for s in range(segs):
            start = s * seg_len
            end = n if s == segs-1 else (s+1) * seg_len
            # Summenzeichen in Segment
            score = sum(vec[start:end])
            # stabiler Index aus Segmentnummer
            idx = self.perm_dg[s]
            ids.append(idx)
        return ids

    def context_ids(self, chapter_idx: int, para_idx: int) -> List[int]:
        # sehr simple Kontexte: ein paar Ctx-Zellen pro (Kapitel, Absatz)
        base = (chapter_idx * 13 + para_idx * 7) % self.ctx_size
        ids = [(base + i*3) % self.ctx_size for i in range(6)]
        # permutieren für Verteilung
        ids = [self.perm_ctx[i % self.ctx_size] for i in ids]
        return ids

# ==============================
# 5) Sequenzen & Hierarchie (Satz→Absatz→Kapitel)
# ==============================

@dataclass
class SequenceController:
    vsa: VSA
    cortex: CortexMemory

    def chain_sentences(self, sent_vecs: List[List[int]]) -> List[int]:
        """
        Erzeuge Absatz-Zeiger als Superposition der Sätze + Sequenz-Links:
        para = PARA ⊗ superpose( SENT_i ⊗ NEXT^i )
        """
        linked = []
        for i, sv in enumerate(sent_vecs, start=1):
            link = self.vsa.bind(sv, self.vsa.pos_vec(i))            # Positionsbindung als NEXT^i-Ersatz
            linked.append(link)
        bag = self.vsa.superpose(*linked) if linked else self.vsa.rand_vec()
        return self.vsa.bind(self.vsa.roles["PARA"], bag)

    def chain_paragraphs(self, para_vecs: List[List[int]]) -> List[int]:
        linked = []
        for i, pv in enumerate(para_vecs, start=1):
            link = self.vsa.bind(pv, self.vsa.pos_vec(i))
            linked.append(link)
        bag = self.vsa.superpose(*linked) if linked else self.vsa.rand_vec()
        return self.vsa.bind(self.vsa.roles["CHAP"], bag)

# ==============================
# 6) Buch-Adapter: Ingestion → Hippocampus
# ==============================

@dataclass
class BookAdapter:
    system: hippo.HippocampalSystem
    vsa: VSA = field(default_factory=lambda: VSA(dim=1024, rng_seed=42))
    encoder: SimpleEncoder = None
    cortex: CortexMemory = None
    projector: SparseProjector = None
    seqctl: SequenceController = None

    def __post_init__(self):
        if self.encoder is None: self.encoder = SimpleEncoder(self.vsa)
        if self.cortex  is None: self.cortex  = CortexMemory(self.vsa)
        if self.projector is None:
            self.projector = SparseProjector(
                dg_size=len(self.system.DG.exc),
                ctx_size=self.system.h.ctx_size,
                k_active=max(8, int(0.03*len(self.system.DG.exc)))
            )
        if self.seqctl is None: self.seqctl = SequenceController(self.vsa, self.cortex)

    # --- Kern: einen Satz einspeisen ---
    def _stimulate_sentence(self, enc: EncodedSentence, ctx_ids: List[int], ticks_after: int = 6) -> None:
        dg_ids = self.projector.to_dg_ids(enc.sent_vec)
        # Kontext aktivieren (Ctx → Plastizität)
        self.system.deliver(hippo.Event(hippo.EventKind.CTX, data={"ctx_ids": ctx_ids}))
        # DG stimulieren (Satz)
        self.system.deliver(hippo.Event(hippo.EventKind.STIMULUS, data={"dg_ids": dg_ids}))
        # ein paar freie Schritte zur Rekurrenz/Plastik
        for _ in range(ticks_after):
            self.system.deliver(hippo.Event(hippo.EventKind.TICK))

    # --- Mismatch/Kohärenz → Comparator-Feedback (ergänzend) ---
    def _coherence_to_reward(self, coherence: float) -> float:
        # skalieren: hohe Kohärenz -> leicht positiv; sehr niedrige -> leicht negativ (hier: Dopamin nur positiv → kleiner Hack: Plastizitätsgates übernehmen die Bremse)
        return max(0.0, (coherence - 0.5)) * 0.6

    # --- Hauptpipeline ---
    def ingest_book(self, text: str, chapter_size_sents: int = 40, para_size_sents: int = 6) -> None:
        sentences = self.encoder.to_sentences(text)
        if not sentences:
            return
        # chunking in Absätze/Kapitel
        paras: List[List[str]] = []
        for i in range(0, len(sentences), para_size_sents):
            paras.append(sentences[i:i+para_size_sents])

        chapters: List[List[List[str]]] = []
        for i in range(0, len(paras), max(1, chapter_size_sents // para_size_sents)):
            chapters.append(paras[i:i + max(1, chapter_size_sents // para_size_sents)])

        # pro Kapitel
        for ci, chapter in enumerate(chapters):
            para_vecs = []
            for pi, para in enumerate(chapter):
                ctx_ids = self.projector.context_ids(ci, pi)
                sent_vecs = []
                for sent in para:
                    enc = self.encoder.encode_sentence(sent)
                    sent_vecs.append(enc.sent_vec)
                    # Relationsnetz updaten
                    self.cortex.add_triples(enc.triples, sent)
                    # Satz einspeisen
                    self._stimulate_sentence(enc, ctx_ids, ticks_after=6)
                # Absatz-Zeiger erzeugen und im Cortex ablegen
                para_vec = self.seqctl.chain_sentences(sent_vecs)
                self.cortex.add_vector(f"CH{ci}-P{pi}", para_vec)
                para_vecs.append(para_vec)

                # episodisches Engramm im Hippo: speichere die DG-IDs des Absatz-Zeigers
                dg_para = self.projector.to_dg_ids(para_vec)
                self.system.store_assembly(dg_para, tag=f"CH{ci}-P{pi}", saliency=1.0)

                # nach jedem Absatz: kleine Belohnung je nach Kohärenz
                coh = self.cortex.coherence_score()
                rew = self._coherence_to_reward(coh)
                if rew > 0:
                    self.system.deliver(hippo.Event(hippo.EventKind.REWARD, data={"magnitude": rew}))

            # Kapitel-Zeiger
            chap_vec = self.seqctl.chain_paragraphs(para_vecs)
            self.cortex.add_vector(f"CH{ci}", chap_vec)

            # nach jedem Kapitel: Schlaf-Replay mit dynamischer Temperatur/Hard-Gate
            self.system.deliver(hippo.Event(hippo.EventKind.SLEEP, data={"rounds": 2}))

    # --- Abfrage/„Verstehen“: nearest Nachbarn, Completionen, einfache Relation-Fragen ---
    def recall_paragraph_like(self, query_text: str, topk: int = 5) -> List[Tuple[str, float]]:
        enc = self.encoder.encode_sentence(query_text)
        return self.cortex.nearest(enc.sent_vec, k=topk)

    def ask_relation(self, subj: str, verb: str, obj: str) -> Tuple[int, int, float]:
        pos = self.cortex.relations.get((subj.lower(), verb.lower(), obj.lower()), 0)
        neg = self.cortex.negations.get((subj.lower(), verb.lower(), obj.lower()), 0)
        conf = (pos - neg) / max(1, pos + neg)
        return pos, neg, conf

# ==============================
# 7) Minimal-Demo (nutzt dein Hippocampus-System)
# ==============================

DEMO_TEXT = """
Kapitel Eins. Der Hund bellt. Die Katze schläft. Der Hund jagt die Katze.
Im Garten ist es ruhig. Später bellt der Hund nicht. Die Katze läuft.
Kapitel Zwei. Der Mond scheint. Der Hund schläft. Die Katze jagt nicht den Hund.
"""

def demo():
    h = hippo.Hyper()
    sys = hippo.HippocampalSystem(h, seed=101)
    adapter = BookAdapter(system=sys)

    # Buch einlesen (hier DEMO_TEXT statt buch.txt)
    adapter.ingest_book(DEMO_TEXT, chapter_size_sents=8, para_size_sents=4)

    # Einfache Abfragen
    print("\n-- Recall-ähnliche Absätze (Nearest) --")
    for tag, sim in adapter.recall_paragraph_like("Der Hund schläft im Garten.", topk=3):
        print(tag, f"sim={sim:.3f}")

    print("\n-- Relations-Frage (Hund jagt Katze?) --")
    pos, neg, conf = adapter.ask_relation("Hund","jagen","Katze")
    print(f"Pos:{pos} Neg:{neg} Konf:{conf:.2f}")

    # Completion-Test: Query als Satz → DG-Cue + passender Kontext
    q_enc = adapter.encoder.encode_sentence("Der Hund bellt.")
    ctx_ids = adapter.projector.context_ids(0,0)
    adapter._stimulate_sentence(q_enc, ctx_ids, ticks_after=8)
    for _ in range(8): sys.deliver(hippo.Event(hippo.EventKind.TICK))
    print("\nCA3 aktive EXC:", len(sys.CA3.active_exc_ids()), "| CA1 aktive EXC:", len(sys.CA1.active_exc_ids()))
    print("Replay-Temperatur dyn.:", f"{adapter.system.replay_temp:.2f}")

if __name__ == "__main__":
    demo()
