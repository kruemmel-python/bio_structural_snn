from __future__ import annotations
# ------------------------------------------------------------
# Buch-Lernen (erweitert):
# - VSA/HRR-Symbolik
# - Verbessertes Parsing (deutsche Heuristiken)
# - Hebb-Rollenlernen (Kontext/Position -> SUBJ/OBJ)
# - Cortex<->Hippo-Doppelschleife (Mismatch-getriggertes, kontext-spez. Replay + Hard-Gate)
# - Kompressions-Policies (Cortex-Verdichtung, Hippo-"Pruning" via Salienzabsenkung)
# Nur Standardbibliothek, Python 3.12
# ------------------------------------------------------------

import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# >>> ggf. anpassen: Import deines Hippocampus-Stacks
import bio_hippocampal_snn_ctx_theta_cmp_feedback_ctxlearn_hardgate as hippo

# ==============================
# 1) VSA / HRR (wie zuvor)
# ==============================

@dataclass
class VSA:
    dim: int = 1024
    rng_seed: int = 123
    def __post_init__(self):
        random.seed(self.rng_seed)
        self.lexicon: Dict[str, List[int]] = {}
        self.roles: Dict[str, List[int]] = {
            "SUBJ": self.rand_vec(),
            "VERB": self.rand_vec(),
            "OBJ":  self.rand_vec(),
            "POS":  self.rand_vec(),
            "NEXT": self.rand_vec(),
            "SENT": self.rand_vec(),
            "PARA": self.rand_vec(),
            "CHAP": self.rand_vec(),
        }
    def rand_vec(self) -> List[int]:
        return [1 if random.random() < 0.5 else -1 for _ in range(self.dim)]
    def superpose(self, *vecs: List[int]) -> List[int]:
        s = [0]*self.dim
        for v in vecs:
            for i,x in enumerate(v):
                s[i] += x
        out = []
        for val in s:
            if val > 0: out.append(1)
            elif val < 0: out.append(-1)
            else: out.append(1 if random.random()<0.5 else -1)
        return out
    def bind(self, a: List[int], b: List[int]) -> List[int]:
        n = self.dim
        out = [0]*n
        for i in range(n):
            s = 0
            for j in range(n):
                k = (i - j) % n
                s += a[j]*b[k]
            out[i] = 1 if s >= 0 else -1
        return out
    def unbind(self, a: List[int], b: List[int]) -> List[int]:
        n = self.dim
        out = [0]*n
        for i in range(n):
            s = 0
            for j in range(n):
                k = (i + j) % n
                s += a[j]*b[k]
            out[i] = 1 if s >= 0 else -1
        return out
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
    def cosine(self, a: List[int], b: List[int]) -> float:
        dot = sum(x*y for x,y in zip(a,b))
        return dot / float(self.dim)

# ==============================
# 2) Parser mit erweiterten DE-Heuristiken
# ==============================

SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')
WORD_SPLIT = re.compile(r'\s+')
PUNCT = re.compile(r'[^\wäöüÄÖÜß\-]')

GER_PREPS = {
    # häufige Präpositionen (Kasus-Hinweise nicht voll genutzt, aber Marker)
    "in","im","am","an","auf","über","unter","vor","hinter",
    "neben","zwischen","mit","ohne","für","gegen","um","durch","zu","vom","vom","von","aus","nach","bei","seit","bis"
}
GER_NEG = {"nicht","kein","keine","keinen","keinem","keiner","keines"}
GER_ART_NOM = {"der","die","das"}        # Nominativ (Subjekt-Kandidat)
GER_ART_ACC = {"den","die","das"}        # Akkusativ (Objekt-Kandidat, mask. den)
SEPARABLE_PREFIXES = {"auf","ein","aus","zu","an","mit","nach","vor","bei","nach","um"}

def is_capitalized(w: str) -> bool:
    return len(w)>0 and w[0].isupper()

def looks_verb_inf(w: str) -> bool:
    return len(w)>3 and (w.endswith("en") or w.endswith("ern") or w.endswith("eln"))

def looks_noun(w: str) -> bool:
    # sehr grob: Großschreibung oder typische Endungen
    return is_capitalized(w) or w.endswith(("ung","keit","heit","nis","tum","schaft"))

@dataclass
class EncodedSentence:
    sent_vec: List[int]
    words: List[str]
    triples: List[Tuple[str,str,str]]

@dataclass
class RoleLearner:
    """
    Hebb-Matrizen:
      - ctx_role[(chapter_idx, para_idx)][role] -> Gewicht
      - pos_role[pos_bin][role] -> Gewicht
    Lernen langsam, Einfluss als additive Bias bei Rollenwahl.
    """
    lr: float = 0.02          # Lernrate
    decay: float = 0.001      # Vergessen
    pos_bins: int = 6
    ctx_role: Dict[Tuple[int,int], Dict[str, float]] = field(default_factory=dict)
    pos_role: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def _pos_bin(self, pos_index: int, sent_len: int) -> int:
        if sent_len <= 0: return 0
        frac = pos_index / max(1, sent_len-1)
        return min(self.pos_bins-1, int(frac * self.pos_bins))

    def bias_for(self, chapter_idx: int, para_idx: int, pos_index: int, sent_len: int, role: str) -> float:
        ctx = self.ctx_role.get((chapter_idx, para_idx), {})
        pb = self.pos_role.get(self._pos_bin(pos_index, sent_len), {})
        return ctx.get(role, 0.0) + pb.get(role, 0.0)

    def learn_from(self, chapter_idx: int, para_idx: int, subj_pos: int, obj_pos: int, sent_len: int, reward: float) -> None:
        # LTP/LTD-artig: positives reward -> verstärke SUBJ/OBJ-Bias an den jeweiligen Positionen & Kontext
        # negatives reward -> schwäche sie ab (hier reward ∈ [-1..+1], wir nutzen in Pipeline nur ≥0, aber vorbereitet)
        sign = 1.0 if reward >= 0 else -1.0
        upd = self.lr * abs(reward)
        # Kontext
        dctx = self.ctx_role.setdefault((chapter_idx, para_idx), {"SUBJ":0.0,"OBJ":0.0})
        dctx["SUBJ"] = dctx["SUBJ"]*(1-self.decay) + sign*upd
        dctx["OBJ"]  = dctx["OBJ"]*(1-self.decay) + sign*upd
        # Position
        sb = self._pos_bin(subj_pos, sent_len)
        ob = self._pos_bin(obj_pos,  sent_len)
        dpos_s = self.pos_role.setdefault(sb, {"SUBJ":0.0,"OBJ":0.0})
        dpos_o = self.pos_role.setdefault(ob, {"SUBJ":0.0,"OBJ":0.0})
        dpos_s["SUBJ"] = dpos_s["SUBJ"]*(1-self.decay) + sign*upd
        dpos_o["OBJ"]  = dpos_o["OBJ"] *(1-self.decay) + sign*upd

class ImprovedEncoder:
    """
    Parser/Encoder mit DE-Heuristiken und Rollenlernen-Bias.
    """
    def __init__(self, vsa: VSA, role_learner: RoleLearner):
        self.vsa = vsa
        self.rl = role_learner

    def clean_word(self, w: str) -> str:
        return PUNCT.sub('', w).strip('-_')

    def to_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text: return []
        return SENT_SPLIT.split(text)

    def _split_words(self, sent: str) -> List[str]:
        toks = [self.clean_word(w) for w in WORD_SPLIT.split(sent) if w.strip()]
        toks = [t for t in toks if t]
        return toks

    def _detect_negation(self, toks: List[str]) -> bool:
        low = [t.lower() for t in toks]
        return any(t in GER_NEG for t in low)

    def _merge_separable_verbs(self, toks: List[str]) -> List[str]:
        """
        Trenne/merze sehr grob trennbare Verben wie 'aufstehen' vs. 'steht ... auf'.
        Heuristik: wenn letzter Token ein Präfix aus SEPARABLE_PREFIXES ist und in der Mitte ein Verbkandidat steht.
        """
        if len(toks) < 3: return toks
        last = toks[-1].lower()
        if last in SEPARABLE_PREFIXES:
            # suche Verbkandidat
            for i in range(1, len(toks)-1):
                if looks_verb_inf(toks[i].lower()) or toks[i].lower().endswith(('t','st')):
                    merged = toks[:]
                    merged[i] = last + merged[i]  # 'auf' + 'stehen' -> 'aufstehen' (oder 'läuft' -> 'aufläuft' – nur Heuristik)
                    return merged[:-1]
        return toks

    def _roles_from_articles(self, toks: List[str]) -> Tuple[Optional[int], Optional[int]]:
        """
        Nutze (Pro-)Artikel als Kasusmarker:
        - erster 'der/die/das' vor Nomen => Subjekt-Kandidat
        - 'den' vor Nomen => Objekt-Kandidat (mask. Akkusativ)
        """
        subj_i = None
        obj_i  = None
        low = [t.lower() for t in toks]
        for i, t in enumerate(low):
            if t in GER_ART_NOM and i+1 < len(toks) and looks_noun(toks[i+1]):
                if subj_i is None:
                    subj_i = i+1
            if t in GER_ART_ACC and i+1 < len(toks) and looks_noun(toks[i+1]):
                if obj_i is None and toks[i+1] != toks[subj_i] if subj_i is not None else True:
                    obj_i = i+1
        return subj_i, obj_i

    def _pick_subj_obj(self, toks: List[str], chapter_idx: int, para_idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Liefere Indizes (subj, verb, obj). Kombiniert:
        - Artikel-Heuristik
        - Positions-/Kontext-Bias aus RoleLearner
        - Fallbacks: erstes großgeschriebenes Nomen als Subjekt, erstes Verb als Verb, nächstes Nomen als Objekt
        """
        sent_len = len(toks)
        low = [t.lower() for t in toks]
        # 1) Verb
        verb_i = None
        for i, w in enumerate(low):
            if looks_verb_inf(w) or w.endswith(("t","st")):
                verb_i = i; break
        # 2) Artikel-Heuristik
        subj_i, obj_i = self._roles_from_articles(toks)

        # 3) Bias aus Rollenlernen (Kontext/Position)
        # Wir wählen aus Kandidaten die mit maximalem Bias, falls unbestimmt.
        if subj_i is None:
            candidates = [i for i,w in enumerate(toks) if looks_noun(w)]
            if candidates:
                best = max(candidates, key=lambda i: self.rl.bias_for(chapter_idx, para_idx, i, sent_len, "SUBJ"))
                subj_i = best
        if obj_i is None and subj_i is not None:
            candidates = [i for i,w in enumerate(toks) if looks_noun(w) and i != subj_i]
            if candidates:
                best = max(candidates, key=lambda i: self.rl.bias_for(chapter_idx, para_idx, i, sent_len, "OBJ"))
                obj_i = best

        # 4) Fallbacks
        if subj_i is None:
            for i,w in enumerate(toks):
                if looks_noun(w): subj_i = i; break
        if verb_i is None:
            for i,w in enumerate(low):
                if looks_verb_inf(w) or w.endswith(("t","st")): verb_i = i; break
        if obj_i is None and subj_i is not None:
            for i,w in enumerate(toks):
                if looks_noun(w) and i != subj_i: obj_i = i; break
        return subj_i, verb_i, obj_i

    def encode_sentence(self, sent: str, chapter_idx: int, para_idx: int) -> EncodedSentence:
        toks = self._split_words(sent)
        toks = self._merge_separable_verbs(toks)
        neg = self._detect_negation(toks)

        # Positionsbindung + „Bag of bound positions“
        pos_bound = []
        for i, w in enumerate(toks, start=1):
            wv = self.vsa.word_vec(w.lower())
            pv = self.vsa.pos_vec(i)
            pos_bound.append(self.vsa.bind(wv, pv))
        bag = self.vsa.superpose(*pos_bound) if pos_bound else self.vsa.rand_vec()
        sent_core = self.vsa.bind(self.vsa.roles["SENT"], bag)

        # Rollen (mit Lernen-Bias)
        subj_i, verb_i, obj_i = self._pick_subj_obj(toks, chapter_idx, para_idx)
        role_bindings = []
        triples: List[Tuple[str,str,str]] = []
        if subj_i is not None and verb_i is not None and obj_i is not None:
            subj = toks[subj_i].lower()
            verb = toks[verb_i].lower()
            obj  = toks[obj_i].lower()
            triples.append((subj, verb, obj))
            role_bindings.append(self.vsa.bind(self.vsa.roles["SUBJ"], self.vsa.word_vec(subj)))
            role_bindings.append(self.vsa.bind(self.vsa.roles["VERB"], self.vsa.word_vec(verb)))
            role_bindings.append(self.vsa.bind(self.vsa.roles["OBJ"],  self.vsa.word_vec(obj)))
        # Negation als leichtes Störsignal (biologisch plausibel: interne Modulation)
        if neg:
            role_bindings.append(self.vsa.bind(self.vsa.word_vec("negation"), self.vsa.pos_vec(1)))

        sent_vec = self.vsa.superpose(sent_core, *role_bindings) if role_bindings else sent_core
        return EncodedSentence(sent_vec=sent_vec, words=[t.lower() for t in toks], triples=triples)

# ==============================
# 3) Cortex (Clean-Up + Relationen) mit Nutzungshistorie
# ==============================

@dataclass
class CortexMemory:
    vsa: VSA
    store: List[Tuple[str, List[int]]] = field(default_factory=list)  # (tag, vec)
    relations: Dict[Tuple[str,str,str], int] = field(default_factory=dict)
    negations: Dict[Tuple[str,str,str], int] = field(default_factory=dict)
    # Nutzung/Alter pro Tag:
    uses: Dict[str, int] = field(default_factory=dict)
    ts: Dict[str, float] = field(default_factory=dict)

    def add_vector(self, tag: str, vec: List[int]) -> None:
        self.store.append((tag, vec))
        self.uses[tag] = 0
        self.ts[tag] = time.time()

    def touch(self, tag: str) -> None:
        self.uses[tag] = self.uses.get(tag, 0) + 1
        self.ts[tag] = time.time()

    def nearest(self, q: List[int], k: int = 5) -> List[Tuple[str, float]]:
        sims = [(tag, self.vsa.cosine(q, v)) for tag, v in self.store]
        sims.sort(key=lambda x: x[1], reverse=True)
        for tag,_ in sims[:k]:
            self.touch(tag)
        return sims[:k]

    def add_triples(self, triples: List[Tuple[str,str,str]], sent_text: str) -> None:
        low = ' ' + sent_text.lower() + ' '
        neg = any(n in low for n in (" nicht ", " kein "))
        for t in triples:
            self.relations[t] = self.relations.get(t, 0) + 1
            if neg:
                self.negations[t] = self.negations.get(t, 0) + 1

    def coherence_score(self) -> float:
        if not self.relations: return 1.0
        inconsist = 0
        for t, c in self.relations.items():
            n = self.negations.get(t, 0)
            if n >= max(1, c//2):
                inconsist += 1
        return max(0.0, 1.0 - inconsist / max(1, len(self.relations)))

    def contradictory_tags(self, topk: int = 3) -> List[str]:
        """
        Suche Tags (Absätze), die vermutlich Widersprüche enthalten:
        heuristisch: Wörter 'nicht/kein' nahe starker Relationseinträge.
        Wir approximieren über 'letzte k eingefügte Absätze' als Kandidaten.
        """
        # simple Heuristik: nimm die ältesten Absätze (höhere Fehlerchance) und solche mit geringer Kohärenz insgesamt.
        # Da wir keinen Satz->Tag Index halten, fallback: sortiere nach Alter (älter zuerst).
        candidates = sorted(self.ts.items(), key=lambda x: x[1])  # alt -> neu
        return [tag for tag,_ in candidates[:topk]]

# ==============================
# 4) Sparse-Projektion & Kontext
# ==============================

@dataclass
class SparseProjector:
    dg_size: int
    ctx_size: int
    k_active: int = 12
    rng_seed: int = 777
    def __post_init__(self):
        random.seed(self.rng_seed)
        self.perm_dg = list(range(self.dg_size)); random.shuffle(self.perm_dg)
        self.perm_ctx = list(range(self.ctx_size)); random.shuffle(self.perm_ctx)
    def to_dg_ids(self, vec: List[int]) -> List[int]:
        n = len(vec)
        segs = self.k_active
        seg_len = n // segs
        ids = []
        for s in range(segs):
            # deterministische Segmentwahl -> DG-Index
            ids.append(self.perm_dg[s])
        return ids
    def context_ids(self, chapter_idx: int, para_idx: int) -> List[int]:
        base = (chapter_idx * 13 + para_idx * 7) % self.ctx_size
        ids = [(base + i*3) % self.ctx_size for i in range(6)]
        return [self.perm_ctx[i % self.ctx_size] for i in ids]

# ==============================
# 5) Sequenzen & Hierarchie
# ==============================

@dataclass
class SequenceController:
    vsa: VSA
    cortex: CortexMemory
    def chain_sentences(self, sent_vecs: List[List[int]]) -> List[int]:
        linked = []
        for i, sv in enumerate(sent_vecs, start=1):
            link = self.vsa.bind(sv, self.vsa.pos_vec(i))
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
# 6) Hauptadapter mit Doppelschleife & Kompression
# ==============================

@dataclass
class BookAdapter:
    system: hippo.HippocampalSystem
    vsa: VSA = field(default_factory=lambda: VSA(dim=1024, rng_seed=42))
    role_learner: RoleLearner = field(default_factory=lambda: RoleLearner(lr=0.02, decay=0.001, pos_bins=6))
    encoder: ImprovedEncoder = None
    cortex: CortexMemory = None
    projector: SparseProjector = None
    seqctl: SequenceController = None

    # Kompressions-Policy
    compress_every_paras: int = 8         # nach wie vielen Absätzen wird komprimiert
    compress_keep_top: int = 6            # pro Kapitel: wie viele häufig/genutzte behalten
    compress_archive_tag: str = "ARCHIVE" # Archiv-Präfix

    # Doppelschleife/Trigger
    mismatch_trigger: float = 0.05        # CA1-Mismatch-Schwelle
    coh_low_trigger: float = 0.65         # Kohärenzgrenze (darunter -> gezieltes Replay)
    target_replay_rounds: int = 2

    def __post_init__(self):
        if self.encoder is None: self.encoder = ImprovedEncoder(self.vsa, self.role_learner)
        if self.cortex  is None: self.cortex  = CortexMemory(self.vsa)
        if self.projector is None:
            self.projector = SparseProjector(
                dg_size=len(self.system.DG.exc),
                ctx_size=self.system.h.ctx_size,
                k_active=max(8, int(0.03*len(self.system.DG.exc)))
            )
        if self.seqctl is None: self.seqctl = SequenceController(self.vsa, self.cortex)

    # ---------- Stimulation ----------
    def _stimulate_sentence(self, enc: EncodedSentence, ctx_ids: List[int], ticks_after: int = 6) -> None:
        dg_ids = self.projector.to_dg_ids(enc.sent_vec)
        self.system.deliver(hippo.Event(hippo.EventKind.CTX, data={"ctx_ids": ctx_ids}))
        self.system.deliver(hippo.Event(hippo.EventKind.STIMULUS, data={"dg_ids": dg_ids}))
        for _ in range(ticks_after):
            self.system.deliver(hippo.Event(hippo.EventKind.TICK))

    def _reward_from_coherence(self, coherence: float) -> float:
        # linear: 0.65 -> 0.0; 1.0 -> ~0.21 (mit 0.6 gain)
        return max(0.0, (coherence - 0.65)) * 0.6

    # ---------- Doppelschleife: gezieltes Replay ----------
    def _context_specific_hard_gate(self, enable: bool) -> None:
        """
        Kontextspezifisches Hard-Gate approximieren:
        - globales Hard-Gate des Systems kurz aktivieren (Countdown hochsetzen)
        - Wachstum temporär aus
        - Plastizitätsgates runter
        """
        if enable:
            self.system.hard_gate_countdown = max(self.system.hard_gate_countdown, self.system.h.hard_gate_cooldown_steps)
            self.system.CA3.growth_enabled = False
            self.system.CA1.growth_enabled = False
            self.system.CA3.feedback_plasticity_gate = self.system.h.hard_gate_min_plasticity
            self.system.CA1.feedback_plasticity_gate = self.system.h.hard_gate_min_plasticity
        else:
            self.system.CA3.growth_enabled = True
            self.system.CA1.growth_enabled = True

    def targeted_replay(self, chapter_idx: int, para_indices: List[int]) -> None:
        """
        Wähle die angegebenen Absätze gezielt zum Replay aus:
        - aktiviere passendes Kontextmuster
        - triggere Replay-Schritte mit gesetztem Hard-Gate
        """
        self._context_specific_hard_gate(True)
        try:
            for pi in para_indices:
                ctx_ids = self.projector.context_ids(chapter_idx, pi)
                tag = f"CH{chapter_idx}-P{pi}"
                # Engramm dazu finden (simple Suche)
                engrams = [e for e in self.system.engrams if e.tag == tag]
                if not engrams:
                    continue
                # Temperatur etwas erhöhen (explorativ, um Widerspruch aufzulösen)
                self.system.replay_temp = min(self.system.h.replay_temp_max, self.system.replay_temp * 1.2)
                # gezieltes Replay: wir stimulieren DG direkt mit dem gespeicherten Assembly
                for _ in range(self.target_replay_rounds):
                    for _ in range(self.system.h.replay_steps):
                        self.system.deliver(hippo.Event(hippo.EventKind.CTX, data={"ctx_ids": ctx_ids}))
                        self.system.deliver(hippo.Event(hippo.EventKind.STIMULUS, data={"dg_ids": engrams[0].ids}))
                        self.system.deliver(hippo.Event(hippo.EventKind.TICK))
        finally:
            # Hard-Gate zurücknehmen; System normalisieren
            self._context_specific_hard_gate(False)

    def monitor_and_intervene(self, chapter_idx: int) -> None:
        """
        Prüfe CA1-Mismatch + Kohärenz. Bei Problemen: gezieltes Replay widersprüchlicher Absätze.
        """
        mismatch = sum(self.system.ca1_last_mismatch) / max(1, len(self.system.ca1_last_mismatch))
        coherence = self.cortex.coherence_score()
        if mismatch >= self.mismatch_trigger or coherence < self.coh_low_trigger:
            # Kandidaten: widersprüchliche/alte Absätze
            cand_tags = self.cortex.contradictory_tags(topk=3)
            para_indices = []
            for t in cand_tags:
                m = re.match(r"CH(\d+)-P(\d+)", t)
                if m and int(m.group(1)) == chapter_idx:
                    para_indices.append(int(m.group(2)))
            if para_indices:
                self.targeted_replay(chapter_idx, para_indices)

    # ---------- Kompression ----------
    def compress_chapter(self, chapter_idx: int) -> None:
        """
        Verdichte seltene/alte Absatz-Vektoren eines Kapitels:
        - Behalte die 'compress_keep_top' meistgenutzten Tags
        - Superponiere den Rest zu einem ARCHIVE-Vektor
        - Senke Hippo-Salienz der entfernten Engramme (biologisch: weniger Replay -> Zerfall/Pruning)
        """
        # sammle Kapitel-Tags
        chapter_tags = [(tag, v) for tag, v in self.cortex.store if tag.startswith(f"CH{chapter_idx}-P")]
        if len(chapter_tags) <= self.compress_keep_top:
            return
        # sortiere nach Nutzung (absteigend), behalte top
        used_sorted = sorted(chapter_tags, key=lambda tv: self.cortex.uses.get(tv[0], 0), reverse=True)
        keep = set(t for t,_ in used_sorted[:self.compress_keep_top])
        drop = [(t,v) for t,v in used_sorted[self.compress_keep_top:]]

        # ARCHIVE-Vektor bilden
        if drop:
            archive_vec = self.vsa.superpose(*[v for _,v in drop])
            arch_tag = f"CH{chapter_idx}-{self.compress_archive_tag}"
            # ersetze/füge Archiv
            # entferne 'drop' aus store
            self.cortex.store = [(t,v) for (t,v) in self.cortex.store if t not in (set(t for t,_ in drop))]
            self.cortex.add_vector(arch_tag, archive_vec)

            # Hippo: senke Salienz zugehöriger Engramme ~0 (kein explizites Löschen; Zerfall durch Nicht-Replay)
            for t,_ in drop:
                for e in self.system.engrams:
                    if e.tag == t:
                        e.saliency = 1e-6  # praktisch nie mehr gesampelt
            # optional: kleine Belohnung, um Konsolidierung des Archivs zu fördern
            self.system.deliver(hippo.Event(hippo.EventKind.REWARD, data={"magnitude": 0.2}))

    # ---------- Haupt-Ingestion ----------
    def ingest_book(self, text: str, chapter_size_sents: int = 40, para_size_sents: int = 6) -> None:
        sentences = self.encoder.to_sentences(text)
        if not sentences: return

        paras: List[List[str]] = []
        for i in range(0, len(sentences), para_size_sents):
            paras.append(sentences[i:i+para_size_sents])

        chapters: List[List[List[str]]] = []
        span = max(1, chapter_size_sents // para_size_sents)
        for i in range(0, len(paras), span):
            chapters.append(paras[i:i+span])

        seen_paras = 0
        for ci, chapter in enumerate(chapters):
            para_vecs = []
            for pi, para in enumerate(chapter):
                ctx_ids = self.projector.context_ids(ci, pi)
                sent_vecs = []
                for s_idx, sent in enumerate(para):
                    enc = self.encoder.encode_sentence(sent, ci, pi)
                    sent_vecs.append(enc.sent_vec)
                    # Relationen sammeln
                    self.cortex.add_triples(enc.triples, sent)
                    # Satz einspeisen
                    self._stimulate_sentence(enc, ctx_ids, ticks_after=6)

                # Absatz-Zeiger
                para_vec = self.seqctl.chain_sentences(sent_vecs)
                tag = f"CH{ci}-P{pi}"
                self.cortex.add_vector(tag, para_vec)
                self.cortex.touch(tag)
                para_vecs.append(para_vec)

                # Hippo: episodisches Engramm
                dg_para = self.projector.to_dg_ids(para_vec)
                self.system.store_assembly(dg_para, tag=tag, saliency=1.0)

                # Rollenlernen belohnen (coherence -> reward)
                coh = self.cortex.coherence_score()
                rew = self._reward_from_coherence(coh)
                if rew > 0:
                    # positives RPE → RoleLearner verstärken (wenn Tripel vorhanden, grob Positionsannahmen)
                    # Für Heuristik: SUBJ=1. Wort mit Nomen, OBJ=erstes Nomen != SUBJ
                    toks = self.encoder._split_words(sent if para else "")
                    subj_i, verb_i, obj_i = self.encoder._pick_subj_obj(toks, ci, pi)
                    if subj_i is not None and obj_i is not None:
                        self.role_learner.learn_from(ci, pi, subj_i, obj_i, len(toks), reward=+min(1.0, rew))
                    self.system.deliver(hippo.Event(hippo.EventKind.REWARD, data={"magnitude": rew}))

                # Monitoring & ggf. gezieltes Replay bei Mismatch/Kohärenz-Problem
                self.monitor_and_intervene(ci)

                seen_paras += 1
                if seen_paras % self.compress_every_paras == 0:
                    self.compress_chapter(ci)

            # Kapitel-Zeiger
            chap_vec = self.seqctl.chain_paragraphs(para_vecs)
            self.cortex.add_vector(f"CH{ci}", chap_vec)
            # Nach Kapitel: Schlaf
            self.system.deliver(hippo.Event(hippo.EventKind.SLEEP, data={"rounds": 2}))

    # ---------- Abfrage ----------
    def recall_paragraph_like(self, query_text: str, topk: int = 5) -> List[Tuple[str, float]]:
        enc = self.encoder.encode_sentence(query_text, 0, 0)
        return self.cortex.nearest(enc.sent_vec, k=topk)

    def ask_relation(self, subj: str, verb: str, obj: str) -> Tuple[int, int, float]:
        key = (subj.lower(), verb.lower(), obj.lower())
        pos = self.cortex.relations.get(key, 0)
        neg = self.cortex.negations.get(key, 0)
        conf = (pos - neg) / max(1, pos + neg)
        return pos, neg, conf

# ==============================
# 7) Mini-Demo
# ==============================

DEMO_TEXT = """
Kapitel Eins. Der Hund bellt im Garten. Die Katze schläft auf dem Sofa. Der Hund jagt die Katze.
Später bellt der Hund nicht. Die Katze läuft in den Garten. Der Junge steht auf und ruft.
Kapitel Zwei. Der Mond scheint über dem See. Der Hund schläft. Die Katze jagt nicht den Hund.
"""

def demo():
    h = hippo.Hyper()
    sys = hippo.HippocampalSystem(h, seed=303)
    adapter = BookAdapter(system=sys)

    adapter.ingest_book(DEMO_TEXT, chapter_size_sents=8, para_size_sents=4)

    print("\n-- Recall-Absätze (Nearest) --")
    for tag, sim in adapter.recall_paragraph_like("Der Hund schläft im Garten.", topk=3):
        print(tag, f"sim={sim:.3f}")

    print("\n-- Relation: Hund jagt Katze? --")
    pos, neg, conf = adapter.ask_relation("Hund","jagen","Katze")
    print(f"Pos:{pos} Neg:{neg} Konf:{conf:.2f}")

    print("\n-- Netzstatus --")
    print("CA3 act:", len(sys.CA3.active_exc_ids()), "| CA1 act:", len(sys.CA1.active_exc_ids()))
    print("Replay-Temp dyn.:", f"{sys.replay_temp:.2f}")
    print("Engramme:", len(sys.engrams))

if __name__ == "__main__":
    demo()
