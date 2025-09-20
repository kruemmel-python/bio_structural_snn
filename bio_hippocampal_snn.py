from __future__ import annotations

# ------------------------------------------------------------
# Biologisch inspirierter Hippocampus-Stack:
# DG (Pattern-Separation) -> CA3 (Attraktor / Completion) -> CA1 (Comparator)
# - Spiking (LIF)
# - I-Neurone (Inhibition, WTA)
# - STDP + Eligibility + Dopamin (Dreifaktor)
# - Strukturelles Wachstum + Pruning
# - Synaptische Skalierung (Homeostase)
# - Energie-Budget (Spikekosten, Regeneration, Schlaf)
# - Schlaf-Replay (Salienz x Rezenz)
# ------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Union
from enum import Enum, auto
import random
import time
import math

# ==============================
# 0) Ereignisse & Hilfen
# ==============================

class EventKind(Enum):
    STIMULUS = auto()  # externer Reiz (DG-Input)
    REWARD   = auto()  # Belohnungssignal (Dopamin / RPE)
    SLEEP    = auto()  # Schlaf/Replay
    TICK     = auto()  # Zeitschritt ohne speziellen Input

@dataclass
class Event:
    kind: EventKind
    data: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

# ==============================
# 1) Hyperparameter
# ==============================

@dataclass
class Hyper:
    # Neuron (LIF)
    threshold_exc: float = 1.0
    threshold_inh: float = 0.9
    leak: float = 0.95
    refractory: int = 2

    # Populationsgrößen
    dg_size: int = 300
    ca3_exc: int = 350
    ca3_inh: int = 60
    ca1_exc: int = 300
    ca1_inh: int = 50

    # Sparsity / Selektivität
    dg_activity_ratio: float = 0.04  # extrem sparsam (Separation)
    ca3_cue_drive: float = 1.1
    ca1_drive: float = 0.9

    # Konnektivität
    init_conn: float = 0.04
    w_init: Tuple[float, float] = (-0.03, 0.03)
    w_max_exc: float = 1.6
    w_max_inh: float = 1.2
    max_out_degree: int = 40
    max_in_degree: int = 70

    # STDP / Eligibility / Dopamin (Dreifaktor)
    base_lr: float = 0.01
    tau_trace: float = 0.90       # STDP-Spuren-Zerfall
    tau_elig: float = 0.95        # Eligibility-Trace Zerfall
    tau_dopa: float = 0.98        # Dopaminspur Zerfall (langsam)
    dop_gain: float = 0.8         # Stärke des Dopamin-Gates
    lr_floor: float = 0.003       # Minimaler Lernanteil

    # Energie/Metabolik
    spike_energy_cost: float = 1.0
    idle_energy_cost: float = 0.02
    energy_recover_awake: float = 0.05
    energy_recover_sleep: float = 0.25
    energy_max: float = 5.0
    energy_gate: float = 0.6      # unterhalb: excitability drosseln
    growth_energy_cost: float = 0.6

    # Vergessen & Pruning & Skalierung
    weight_decay: float = 0.002
    pruning_floor: float = 0.02
    scaling_interval: int = 50         # alle X steps skalieren
    scaling_target_sum: float = 5.0    # Ziel-Summe eingehender EXC-Gewichte

    # Replay
    replay_steps: int = 8
    replay_batch: int = 3

# ==============================
# 2) Grundbausteine
# ==============================

class CellType(Enum):
    EXC = auto()
    INH = auto()

@dataclass
class Neuron:
    kind: CellType
    threshold: float
    leak: float
    refractory: int
    ref_count: int = 0
    fired: bool = False
    v: float = 0.0

    # Energie
    energy: float = 5.0
    energy_max: float = 5.0

    # Feuerraten-Homeostase
    fire_count: int = 0
    win_count: int = 0
    target_rate: float = 0.08

    # Dopaminspur (postsynaptisch)
    dopa_trace: float = 0.0

    def reset_spike(self) -> None:
        self.fired = False

    def step(self, input_current: float, h: Hyper, sleeping: bool) -> bool:
        # Grundumsatz
        self.energy = max(0.0, self.energy - h.idle_energy_cost)

        if self.ref_count > 0:
            self.ref_count -= 1
            self.v *= self.leak
            self.fired = False
        else:
            gate = 1.0 if self.energy >= h.energy_gate else 0.6
            self.v = self.v * self.leak + gate * input_current
            thr = self.threshold
            if self.v >= thr and self.energy >= h.spike_energy_cost:
                self.fired = True
                self.v = 0.0
                self.ref_count = self.refractory
                self.energy -= h.spike_energy_cost
                self.fire_count += 1
            else:
                self.fired = False

        # Regeneration
        rec = h.energy_recover_sleep if sleeping else h.energy_recover_awake
        self.energy = min(self.energy_max, self.energy + rec)

        # Fensterzählung
        self.win_count += 1
        return self.fired

    def rate_dev(self) -> float:
        if self.win_count == 0:
            return 0.0
        return (self.fire_count / self.win_count) - self.target_rate

    def reset_window(self) -> None:
        self.fire_count = 0
        self.win_count = 0

@dataclass
class Synapse:
    pre: int
    post: int
    w: float
    inhibitory: bool

    # Spuren / Eligibility
    pre_trace: float = 0.0
    post_trace: float = 0.0
    elig: float = 0.0

    def decay(self, h: Hyper) -> None:
        self.pre_trace *= h.tau_trace
        self.post_trace *= h.tau_trace
        self.elig *= h.tau_elig
        # passives Vergessen
        if self.w > 0:
            self.w = max(0.0, self.w - h.weight_decay)
        else:
            self.w = min(0.0, self.w + h.weight_decay)

# ==============================
# 3) Teilnetz (Layer) mit Inhibition
# ==============================

class Layer:
    """
    Ein Layer enthält EXC- und INH-Neurone + Synapsen:
    - rekurrent EXC->EXC
    - EXC->INH (treibt Inhibition)
    - INH->EXC (globalere Hemmung, WTA)
    Optional: FEEDFORWARD-Input von vorheriger Schicht.
    """
    def __init__(self, n_exc: int, n_inh: int, h: Hyper, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.h = h
        self.exc: List[Neuron] = [
            Neuron(CellType.EXC, h.threshold_exc, h.leak, h.refractory, energy=h.energy_max, energy_max=h.energy_max)
            for _ in range(n_exc)
        ]
        self.inh: List[Neuron] = [
            Neuron(CellType.INH, h.threshold_inh, h.leak, h.refractory, energy=h.energy_max, energy_max=h.energy_max)
            for _ in range(n_inh)
        ]
        self.syn: List[Synapse] = []
        self.out_adj: Dict[int, List[int]] = {}
        self.in_adj: Dict[int, List[int]] = {}
        self.sleeping: bool = False

        # rekurrente Initialkanten (EXC->EXC, EXC->INH, INH->EXC)
        self._rand_connect_exc_exc(self.exc, self.h.init_conn)
        self._rand_connect_exc_inh(self.exc, self.inh, self.h.init_conn)
        self._rand_connect_inh_exc(self.inh, self.exc, self.h.init_conn)

        # Zähler für Synaptische Skalierung
        self._steps = 0

    # ---- Verbindungshelfer ----

    def _add_syn(self, pre: int, post: int, w: float, inhibitory: bool) -> Optional[int]:
        # Degree-Budgets prüfen (nur für EXC-Eingänge relevant für Skalierung)
        if not inhibitory and len(self.in_adj.get(post, [])) >= self.h.max_in_degree:
            return None
        if len(self.out_adj.get(pre, [])) >= self.h.max_out_degree:
            return None
        idx = len(self.syn)
        self.syn.append(Synapse(pre, post, w, inhibitory))
        self.out_adj.setdefault(pre, []).append(idx)
        self.in_adj.setdefault(post, []).append(idx)
        return idx

    def _rand_connect_exc_exc(self, exc: List[Neuron], p: float) -> None:
        n = len(exc)
        for i in range(n):
            for j in range(n):
                if i == j: 
                    continue
                if random.random() < p:
                    self._add_syn(i, j, random.uniform(*self.h.w_init), inhibitory=False)

    def _rand_connect_exc_inh(self, exc: List[Neuron], inh: List[Neuron], p: float) -> None:
        nE, nI = len(exc), len(inh)
        EOFF = 0  # EXC index-Offset in layer-local
        IOFF = len(exc)  # INH "start" (wir mappen in globalem Indexraum, aber layer-lokal bleiben wir 0..)
        # Wir modellieren hier im Layer-Lokalraum: EXC 0..nE-1, INH separat über Negativindex trick?
        # -> Einfach: wir halten Synapsen-IDs im EXC/INH-Raum getrennt und mappen im Step.
        # Zur Vereinfachung: wir nummerieren EXC zuerst, dann INH im selben Indexraum.
        # Daher müssen INH-IDs real existieren; wir simulieren das, indem wir EXC+INH als ein Feld denken.
        # In der Praxis erzeugen wir EXC->INH: pre in [0..nE-1], post in [nE..nE+nI-1]
        for i in range(nE):
            for j in range(nI):
                if random.random() < p:
                    post = nE + j
                    self._add_syn(i, post, random.uniform(0.02, 0.08), inhibitory=False)

    def _rand_connect_inh_exc(self, inh: List[Neuron], exc: List[Neuron], p: float) -> None:
        nE, nI = len(exc), len(inh)
        for i in range(nI):
            for j in range(nE):
                if random.random() < p:
                    pre = len(exc) + i
                    self._add_syn(pre, j, -abs(random.uniform(0.02, 0.08)), inhibitory=True)

    # ---- Laufzeit ----

    def _collect_currents(self) -> List[float]:
        # Gemeinsamer Indexraum: 0..E-1 (EXC), E..E+I-1 (INH)
        nE, nI = len(self.exc), len(self.inh)
        currents = [0.0] * (nE + nI)
        # EXC/INH feuern?
        fired_mask = [False] * (nE + nI)
        for i, n in enumerate(self.exc):
            fired_mask[i] = n.fired
        for i, n in enumerate(self.inh):
            fired_mask[nE + i] = n.fired

        for s in self.syn:
            if s.pre < 0:
                continue
            if fired_mask[s.pre]:
                currents[s.post] += s.w
        return currents

    def stimulate_exc(self, ids: List[int], drive: float) -> None:
        for i in ids:
            if 0 <= i < len(self.exc):
                self.exc[i].v += drive

    def step(self) -> Tuple[List[int], List[int]]:
        """Ein Schritt: Ströme sammeln, Neurone updaten, STDP/Eligibility/Pruning/Scaling."""
        nE, nI = len(self.exc), len(self.inh)
        # Reset fired Flags
        for n in self.exc: n.reset_spike()
        for n in self.inh: n.reset_spike()

        currents = self._collect_currents()

        fired_exc: List[int] = []
        fired_inh: List[int] = []

        # Update EXC
        for i, n in enumerate(self.exc):
            if n.step(currents[i], self.h, sleeping=self.sleeping):
                fired_exc.append(i)
        # Update INH
        for i, n in enumerate(self.inh):
            if n.step(currents[nE + i], self.h, sleeping=self.sleeping):
                fired_inh.append(nE + i)  # gemischter Indexraum

        # Dreifaktor-STDP + Zerfall + Pruning
        self._plasticity_update(fired_exc, fired_inh)

        # Synaptische Skalierung gelegentlich
        self._steps += 1
        if self._steps % self.h.scaling_interval == 0:
            self._synaptic_scaling()

        return fired_exc, fired_inh

    def _plasticity_update(self, fired_exc: List[int], fired_inh: List[int]) -> None:
        h = self.h
        fired = set([*fired_exc, *fired_inh])
        # Lokale Dopaminspuren (postsynaptisch) zerfallen
        for n in self.exc:
            n.dopa_trace *= h.tau_dopa
        for n in self.inh:
            n.dopa_trace *= h.tau_dopa

        for idx, s in enumerate(self.syn):
            if s.pre < 0:
                continue
            # Spuren- und Gewichts-Zerfall
            s.decay(h)

            pre_fired = s.pre in fired
            post_fired = s.post in fired

            if pre_fired:
                s.pre_trace += 1.0
                # Post vor Pre -> LTD-Anteil in Eligibility
                s.elig -= 0.5 * s.post_trace
            if post_fired:
                s.post_trace += 1.0
                # Pre vor Post -> LTP-Anteil in Eligibility
                s.elig += 1.0 * s.pre_trace

            # Post-synaptische Dopaminspur (RPE) moduliert effektives Update
            # lokaler LR mit Feuerraten-Homeostase:
            post_is_exc = s.post < len(self.exc) + len(self.inh) and (s.post < len(self.exc))
            post_neuron = self.exc[s.post] if post_is_exc else self.inh[s.post - len(self.exc)]
            lr_scale = max(h.lr_floor, 1.0 - 0.5 * max(0.0, post_neuron.rate_dev()))
            lr = lr_scale * h.base_lr

            # Dreifaktor-Update: Δw ∝ lr * sigmoid(dopa_trace) * elig
            dopa_gain = 1.0 + h.dop_gain * post_neuron.dopa_trace
            delta = lr * dopa_gain * s.elig

            # Inhibitorische Gewichte bleiben ≤ 0, Exzitatorische ≥ 0
            if s.inhibitory:
                s.w = min(0.0, s.w + delta)
                s.w = max(-h.w_max_inh, s.w)
            else:
                s.w = max(0.0, s.w + delta)
                s.w = min(h.w_max_exc, s.w)

            # Pruning
            if abs(s.w) < h.pruning_floor:
                self._kill_syn(idx)

    def _kill_syn(self, idx: int) -> None:
        s = self.syn[idx]
        # Entferne Referenzen
        if idx in self.out_adj.get(s.pre, []):
            self.out_adj[s.pre].remove(idx)
        if idx in self.in_adj.get(s.post, []):
            self.in_adj[s.post].remove(idx)
        # Tombstone
        self.syn[idx] = Synapse(-1, -1, 0.0, inhibitory=False)

    def _synaptic_scaling(self) -> None:
        """Skaliert eingehende EXC-Gewichte pro EXC-Neuron auf Zielsumme."""
        target = self.h.scaling_target_sum
        nE = len(self.exc)
        # Eingehende Kanten jedes EXC sammeln
        incoming: Dict[int, List[int]] = {i: [] for i in range(nE)}
        for idx, s in enumerate(self.syn):
            if s.pre < 0:
                continue
            # nur EXC->EXC skalieren
            if not s.inhibitory and s.post < nE:
                incoming[s.post].append(idx)

        for post, idxs in incoming.items():
            if not idxs: 
                continue
            ssum = sum(max(0.0, self.syn[i].w) for i in idxs)
            if ssum <= 1e-6:
                continue
            scale = target / ssum
            # moderate Skalierung begrenzen, damit nichts explodiert
            scale = max(0.5, min(2.0, scale))
            for i in idxs:
                self.syn[i].w = min(self.h.w_max_exc, max(0.0, self.syn[i].w * scale))

    # ---- Metriken/Abfragen ----

    def active_exc_ids(self) -> List[int]:
        return [i for i, n in enumerate(self.exc) if n.fired]

    def reset_windows(self) -> None:
        for n in self.exc + self.inh:
            n.reset_window()

# ==============================
# 4) Hippocampales System
# ==============================

@dataclass
class Assembly:
    ids: List[int]
    tag: str = ""
    saliency: float = 1.0
    ts: float = field(default_factory=time.time)

class HippocampalSystem:
    """
    DG -> CA3 -> CA1 Pipeline:
    - DG: Pattern-Separation; wir injizieren hier externe Stimuli (sehr sparsam).
    - CA3: rekurrente Completion via Layer (EXC/INH).
    - CA1: Comparator (Feedforward von DG und CA3); generiert Mismatch-Aktivität.
    """
    def __init__(self, h: Optional[Hyper] = None, seed: Optional[int] = 7):
        self.h = h or Hyper()
        if seed is not None:
            random.seed(seed)

        # Layers
        self.DG = Layer(self.h.dg_size, 0, self.h, seed=seed)         # DG ohne INH (einfacher Separator)
        self.CA3 = Layer(self.h.ca3_exc, self.h.ca3_inh, self.h, seed=seed+1)
        self.CA1 = Layer(self.h.ca1_exc, self.h.ca1_inh, self.h, seed=seed+2)

        # Feedforward-Verbindungen DG->CA3 (stark, aber spärlich)
        self.ff_dg_ca3: List[Synapse] = []
        self._connect_ff(self.DG, self.CA3, prob=0.06, w_range=(0.05, 0.12), inhibitory=False)
        # Feedforward DG/CA3 -> CA1
        self.ff_dg_ca1: List[Synapse] = []
        self.ff_ca3_ca1: List[Synapse] = []
        self._connect_ff(self.DG, self.CA1, prob=0.05, w_range=(0.03, 0.08), inhibitory=False, store="dg_ca1")
        self._connect_ff(self.CA3, self.CA1, prob=0.06, w_range=(0.04, 0.10), inhibitory=False, store="ca3_ca1")

        # Engramm-Store für Replay (auf DG/CA3 bezogen)
        self.engrams: List[Assembly] = []

        # Schlaf-Flag
        self.sleeping: bool = False

    # ---- Feedforward-Verbindungen (vereinfachte Synapsen ohne Strukturgrowth hier) ----

    def _connect_ff(
        self,
        src: Layer,
        dst: Layer,
        prob: float,
        w_range: Tuple[float, float],
        inhibitory: bool,
        store: Optional[str] = None
    ) -> None:
        syns: List[Synapse] = []
        n_src = len(src.exc)
        n_dst = len(dst.exc)
        for i in range(n_src):
            for j in range(n_dst):
                if random.random() < prob:
                    w = random.uniform(*w_range)
                    syns.append(Synapse(pre=i, post=j, w=w if not inhibitory else -abs(w), inhibitory=inhibitory))
        if store == "dg_ca1":
            self.ff_dg_ca1 = syns
        elif store == "ca3_ca1":
            self.ff_ca3_ca1 = syns
        else:
            self.ff_dg_ca3 = syns

    # ---- Laufzeit der Pipeline ----

    def _apply_ff(self, syns: List[Synapse], src: Layer, dst: Layer) -> None:
        # addiere Ströme aus src.EXC auf dst.EXC
        src_fired = src.active_exc_ids()
        fired_set = set(src_fired)
        for s in syns:
            if s.pre in fired_set:
                dst.exc[s.post].v += s.w

    def present_to_DG(self, stimulus_ids: List[int]) -> None:
        """Externer Input: selektive DG-Neurone depolarisieren (Separation)."""
        # Starke Sparsity: wir aktivieren nur den Schnitt aus stimulus_ids und DG
        # (Annahme: stimulus_ids sind DG-Indices)
        ids = [i for i in stimulus_ids if 0 <= i < len(self.DG.exc)]
        self.DG.stimulate_exc(ids, drive=1.2)

    def step(self) -> None:
        # DG Schritt (ohne Inhibition hier)
        self.DG.sleeping = self.sleeping
        self.DG.step()

        # Feedforward DG->CA3
        self._apply_ff(self.ff_dg_ca3, self.DG, self.CA3)
        # CA3 rekurrent
        self.CA3.sleeping = self.sleeping
        self.CA3.step()

        # Feedforward DG/CA3 -> CA1
        self._apply_ff(self.ff_dg_ca1, self.DG, self.CA1)
        self._apply_ff(self.ff_ca3_ca1, self.CA3, self.CA1)
        # CA1 Schritt (Comparator-Dynamik in EXC/INH)
        self.CA1.sleeping = self.sleeping
        self.CA1.step()

        # Fenster ggf. resetten (z. B. nach Schlafblock)
        # (wird im Schlaf-Replay am Ende gemacht)

    # ---- Belohnung / Dopamin (RPE) ----

    def reward(self, magnitude: float = 0.5) -> None:
        """Belohnung -> Dopaminspuren der postsynaptischen Zellen erhöhen (RPE-Proxy)."""
        # Einfacher Proxy: CA3 und CA1 postsynaptische dopa_trace anheben
        for n in self.CA3.exc + self.CA3.inh + self.CA1.exc + self.CA1.inh:
            n.dopa_trace = min(1.0, n.dopa_trace + magnitude)

    # ---- Replay (Schlaf) ----

    def store_assembly(self, ids: List[int], tag: str = "", saliency: float = 1.0) -> None:
        # ids beziehen sich auf DG-EXC-Indices (Input-Muster)
        ids = [i for i in ids if 0 <= i < len(self.DG.exc)]
        self.engrams.append(Assembly(ids=ids, tag=tag, saliency=saliency))

    def sleep_replay(self, rounds: int = 2) -> None:
        if not self.engrams:
            return
        self.sleeping = True
        for _ in range(rounds):
            # priorisiert nach Salienz x Rezenz
            now = time.time()
            weights = []
            for a in self.engrams:
                rec = 1.0 / (1.0 + (now - a.ts))
                weights.append(max(1e-6, a.saliency * rec))
            total = sum(weights)
            probs = [w/total for w in weights]

            picks = random.choices(self.engrams, weights=probs, k=min(self.h.replay_batch, len(self.engrams)))
            for a in picks:
                # sanftes Replay
                for _ in range(self.h.replay_steps):
                    self.present_to_DG(a.ids)
                    self.step()

            # Homeostat-Fenster nach jedem Block zurücksetzen
            self.DG.reset_windows()
            self.CA3.reset_windows()
            self.CA1.reset_windows()
        self.sleeping = False

    # ---- Ereignisse ----

    def deliver(self, ev: Event) -> None:
        match ev.kind:
            case EventKind.STIMULUS:
                ids = list(ev.data.get("dg_ids", []))
                self.present_to_DG(ids)
                self.step()
            case EventKind.REWARD:
                mag = float(ev.data.get("magnitude", 0.5))
                self.reward(magnitude=mag)
            case EventKind.SLEEP:
                rounds = int(ev.data.get("rounds", 3))
                self.sleep_replay(rounds=rounds)
            case EventKind.TICK:
                self.step()
            case _:
                raise ValueError(f"Unbekanntes Event: {ev.kind!r}")

# ==============================
# 5) Hilfen & Demo
# ==============================

def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / len(A | B)

def sparse_ids(n: int, ratio: float, seed: Optional[int] = None) -> List[int]:
    if seed is not None:
        random.seed(seed)
    k = max(1, int(n * ratio))
    return random.sample(range(n), k=k)

def demo() -> None:
    h = Hyper()
    sys = HippocampalSystem(h, seed=11)

    # --- Drei „Erfahrungen“ als DG-Assemblies (extrem sparsam) ---
    A1 = sparse_ids(h.dg_size, h.dg_activity_ratio, seed=1)
    A2 = sparse_ids(h.dg_size, h.dg_activity_ratio, seed=2)
    A3 = sparse_ids(h.dg_size, h.dg_activity_ratio, seed=3)

    # Lernen im Wachzustand (DG->CA3->CA1 fließt)
    for _ in range(6):
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids": A1}))
        for _ in range(5): sys.deliver(Event(EventKind.TICK))
    sys.store_assembly(A1, tag="A1", saliency=1.4)

    for _ in range(6):
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids": A2}))
        for _ in range(5): sys.deliver(Event(EventKind.TICK))
    sys.store_assembly(A2, tag="A2", saliency=1.0)

    for _ in range(6):
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids": A3}))
        for _ in range(5): sys.deliver(Event(EventKind.TICK))
    sys.store_assembly(A3, tag="A3", saliency=0.8)

    # Belohnung (RPE) -> Dopaminspuren ↑
    sys.deliver(Event(EventKind.REWARD, data={"magnitude": 0.6}))

    # Schlaf-Replay zur Konsolidierung
    sys.deliver(Event(EventKind.SLEEP, data={"rounds": 3}))

    # Pattern-Completion-Test:
    # wir cauen nur 40% der DG-IDs an und lassen Rekurrenz in CA3 wirken.
    def cue_and_complete(dg_ids: List[int], frac: float = 0.4) -> Tuple[List[int], List[int]]:
        k = max(1, int(len(dg_ids) * frac))
        cue = random.sample(dg_ids, k=k)
        # kurzer Cue + freie Dynamik
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids": cue}))
        for _ in range(3): sys.deliver(Event(EventKind.TICK))
        for _ in range(12): sys.deliver(Event(EventKind.TICK))
        # Lese CA3/CA1 aktive EXC aus
        ca3_act = sys.CA3.active_exc_ids()
        ca1_act = sys.CA1.active_exc_ids()
        return ca3_act, ca1_act

    ca3_1, ca1_1 = cue_and_complete(A1, 0.4)
    ca3_2, ca1_2 = cue_and_complete(A2, 0.4)
    ca3_3, ca1_3 = cue_and_complete(A3, 0.4)

    # Für Vergleich mappen wir DG-Targets grob auf CA3 mittels Feedforward-Abdeckung:
    # Hier nutzen wir Proxy: DG-IDs selbst (Ground truth) vs. CA3-Aktivität über FF-Graph ist
    # in dieser Mini-Demo nicht 1:1, daher reporten wir einfach Aktivitäts-Kohärenzen:
    print("Completion-Proxy (CA3 Aktivität Größenverhältnis):",
          f"A1 {len(ca3_1)}  A2 {len(ca3_2)}  A3 {len(ca3_3)}")
    print("CA1 Aktivität (Comparator-Proxy):",
          f"A1 {len(ca1_1)}  A2 {len(ca1_2)}  A3 {len(ca1_3)}")

if __name__ == "__main__":
    demo()
