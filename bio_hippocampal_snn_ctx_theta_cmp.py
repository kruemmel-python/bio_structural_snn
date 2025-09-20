from __future__ import annotations

# ------------------------------------------------------------
# Biologisch inspirierter Hippocampus-Stack (erweitert):
# DG (Pattern-Separation) -> CA3 (Attraktor/Completion) -> CA1 (Comparator)
# Zusätze:
# - Kontext-Gating (eigene Ctx-Population modifiziert lokale Plastizität)
# - Theta/Gamma-Zeitfenster für STDP/Eligibility/Dopamin (Oszillator)
# - CA1-Mismatch-Subpopulation (dedizierte Comparator-Neurone, ±(DG, CA3))
# - Strukturelles Wachstum (EXC->EXC) + Energie-Budget + Overlap-Block
# - Synaptische Skalierung, Energie, Vergessen/Pruning, Schlaf-Replay
# ------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
from enum import Enum, auto
import random
import time
import math

# ==============================
# 0) Ereignisse & Hilfen
# ==============================

class EventKind(Enum):
    STIMULUS = auto()   # externer Reiz (DG-Input)
    CTX      = auto()   # Kontext-Reiz (Ctx-Input)
    REWARD   = auto()   # Belohnungssignal (Dopamin/RPE)
    SLEEP    = auto()   # Schlaf/Replay
    TICK     = auto()   # Zeitschritt ohne speziellen Input

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
    ctx_size: int = 60               # Kontext-Population (eigener Layer)
    cmp_size: int = 80               # CA1-Comparator-Neurone (dedizierte Subpop)

    # Sparsity / Selektivität
    dg_activity_ratio: float = 0.04  # extrem sparsam (Separation)

    # Konnektivität
    init_conn: float = 0.04
    w_init: Tuple[float, float] = (-0.03, 0.03)
    w_max_exc: float = 1.6
    w_max_inh: float = 1.2
    max_out_degree: int = 40
    max_in_degree: int = 70

    # STDP / Eligibility / Dopamin (Dreifaktor)
    base_lr: float = 0.01
    tau_trace: float = 0.90        # STDP-Spuren-Zerfall
    tau_elig: float = 0.95         # Eligibility-Zerfall
    tau_dopa: float = 0.98         # Dopaminspur (postsynaptisch) Zerfall
    dop_gain: float = 0.8          # Stärke des Dopamin-Gates
    lr_floor: float = 0.003        # Minimaler Lernanteil

    # Kontext-Gating
    ctx_to_ca3_gain: float = 0.8   # skaliert lokale Plastizität in CA3
    ctx_to_ca1_gain: float = 0.8   # skaliert lokale Plastizität in CA1
    ctx_decay: float = 0.92        # Zerfall der ctx_gate pro Schritt
    ctx_drive: float = 1.2         # Depolarisation der Kontext-Neurone
    ctx_gate_min: float = 0.5      # Untergrenze für Plastizitäts-Skalierung
    ctx_gate_max: float = 1.8      # Obergrenze

    # Theta/Gamma
    theta_period: int = 12         # Schritte pro Theta-Zyklus
    gamma_period: int = 3          # Schritte pro Gamma-Zyklus
    theta_potent_window: Tuple[float, float] = (0.2, 0.6)  # Phase in [0..1]: LTP-freundlich
    gamma_gate_strength: float = 0.6                         # zusätzlicher Gate-Faktor

    # Energie/Metabolik
    spike_energy_cost: float = 1.0
    idle_energy_cost: float = 0.02
    energy_recover_awake: float = 0.05
    energy_recover_sleep: float = 0.25
    energy_max: float = 5.0
    energy_gate: float = 0.6
    growth_energy_cost: float = 0.6

    # Vergessen & Pruning & Skalierung
    weight_decay: float = 0.002
    pruning_floor: float = 0.02
    scaling_interval: int = 50
    scaling_target_sum: float = 5.0

    # Replay
    replay_steps: int = 8
    replay_batch: int = 3

    # Strukturelles Wachstum
    growth_overlap_block: float = 0.4
    growth_init_range: Tuple[float, float] = (0.01, 0.04)

# ==============================
# 2) Oszillator (Theta/Gamma-Gates)
# ==============================

@dataclass
class Oscillator:
    theta_period: int
    gamma_period: int
    t: int = 0  # Schrittzähler

    def step(self) -> None:
        self.t += 1

    def theta_phase(self) -> float:
        # Phase in [0..1]
        return (self.t % self.theta_period) / self.theta_period

    def gamma_phase(self) -> float:
        return (self.t % self.gamma_period) / self.gamma_period

    def gates(self, h: Hyper) -> Tuple[float, float]:
        """
        Liefert (stdp_gate, bias_gate).
        stdp_gate skaliert die Effektivität der Plastizität (0..1+).
        bias_gate kann LTP/LTD-Balance leicht verschieben (hier Gamma-Anteil).
        """
        th = self.theta_phase()
        gm = self.gamma_phase()

        # Theta-LTP-Fenster
        t0, t1 = h.theta_potent_window
        theta_gate = 1.0 if (t0 <= th <= t1) else 0.4  # außerhalb schwächeres Lernen

        # Gamma-Feinmodulation (Perioden-Fenster)
        gamma_gate = 1.0 if gm < 0.5 else h.gamma_gate_strength

        stdp_gate = theta_gate * gamma_gate
        # bias_gate könnte z.B. in LTP-Richtung leicht gewichten; hier vereinfachend 1.0
        return stdp_gate, 1.0

# ==============================
# 3) Grundbausteine: Neuron, Synapse, Comparator-Neuron
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

    # Kontext-Gate (plastizitäts-modulierend)
    ctx_gate: float = 1.0

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

        # Kontext-Gate zerfällt leicht Richtung 1.0
        self.ctx_gate = 1.0 + (self.ctx_gate - 1.0) * h.ctx_decay

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

# Comparator-Neuron (CA1-Mismatch-Subpopulation): einfacher LIF mit signiertem Input
@dataclass
class ComparatorNeuron:
    threshold: float = 1.0
    leak: float = 0.95
    refractory: int = 2
    v: float = 0.0
    ref_count: int = 0
    fired: bool = False

    def step(self, input_current: float) -> bool:
        if self.ref_count > 0:
            self.ref_count -= 1
            self.v *= self.leak
            self.fired = False
            return False
        self.v = self.v * self.leak + input_current
        if self.v >= self.threshold:
            self.fired = True
            self.v = 0.0
            self.ref_count = self.refractory
            return True
        self.fired = False
        return False

# ==============================
# 4) Layer mit Inhibition, Wachstum, Kontext-Modulation, Oszillator-Gates
# ==============================

class Layer:
    """
    EXC/INH-Population mit:
    - rekurrenten Synapsen (EXC->EXC, EXC->INH, INH->EXC)
    - Dreifaktor-Plastizität (STDP×Eligibility×Dopamin)
    - strukturellem Wachstum (EXC->EXC) im Wachzustand
    - synaptischer Skalierung
    - Kontext-Gating (ctx_gate moduliert lr/dopa_gain lokal)
    - Oszillator-Gates (theta/gamma) für Plastizität
    """
    def __init__(self, n_exc: int, n_inh: int, h: Hyper, osc: Oscillator, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.h = h
        self.osc = osc
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

        # initiale Rekurrenz
        self._rand_connect_exc_exc(self.exc, self.h.init_conn)
        self._rand_connect_exc_inh(self.exc, self.inh, self.h.init_conn)
        self._rand_connect_inh_exc(self.inh, self.exc, self.h.init_conn)

        self._steps = 0
        self.grown_count: int = 0

        # Kontext-Kopplungsstärke pro Postsynapse (wird vom System gesetzt)
        self.ctx_gain_local: float = 1.0  # wird in CA3/CA1 spezifisch gesetzt

    # ---- Verbindungshelfer ----

    def _add_syn(self, pre: int, post: int, w: float, inhibitory: bool) -> Optional[int]:
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
                if i == j: continue
                if random.random() < p:
                    self._add_syn(i, j, random.uniform(*self.h.w_init), inhibitory=False)

    def _rand_connect_exc_inh(self, exc: List[Neuron], inh: List[Neuron], p: float) -> None:
        nE, nI = len(exc), len(inh)
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
        nE, nI = len(self.exc), len(self.inh)
        currents = [0.0] * (nE + nI)
        fired_mask = [False] * (nE + nI)
        for i, n in enumerate(self.exc): fired_mask[i] = n.fired
        for i, n in enumerate(self.inh): fired_mask[nE + i] = n.fired
        for s in self.syn:
            if s.pre < 0: continue
            if fired_mask[s.pre]:
                currents[s.post] += s.w
        return currents

    def stimulate_exc(self, ids: List[int], drive: float) -> None:
        for i in ids:
            if 0 <= i < len(self.exc):
                self.exc[i].v += drive

    def step(self) -> Tuple[List[int], List[int]]:
        nE, nI = len(self.exc), len(self.inh)
        for n in self.exc: n.reset_spike()
        for n in self.inh: n.reset_spike()

        currents = self._collect_currents()
        fired_exc: List[int] = []
        fired_inh: List[int] = []

        for i, n in enumerate(self.exc):
            if n.step(currents[i], self.h, sleeping=self.sleeping):
                fired_exc.append(i)
        for i, n in enumerate(self.inh):
            if n.step(currents[nE + i], self.h, sleeping=self.sleeping):
                fired_inh.append(nE + i)

        # STDP/Eligibility/Pruning mit Oszillator- und Kontext-Gates
        self._plasticity_update(fired_exc, fired_inh)

        # Strukturelles Wachstum (nur wach)
        if not self.sleeping and len(fired_exc) >= 2:
            self._growth_rule_exc_exc(fired_exc)

        # Synaptische Skalierung
        self._steps += 1
        if self._steps % self.h.scaling_interval == 0:
            self._synaptic_scaling()

        return fired_exc, fired_inh

    def _plasticity_update(self, fired_exc: List[int], fired_inh: List[int]) -> None:
        h = self.h
        stdp_gate, bias_gate = self.osc.gates(h)  # zeitliche Gating-Faktoren
        fired = set([*fired_exc, *fired_inh])

        # Dopaminspuren und Kontext-Gate-Zerfall (Kontext-Zerfall geschieht im Neuron.step)
        for n in self.exc + self.inh:
            n.dopa_trace *= h.tau_dopa

        for idx, s in enumerate(self.syn):
            if s.pre < 0: continue

            # Spuren-/Gewichtszerfall
            s.decay(h)

            pre_fired = s.pre in fired
            post_fired = s.post in fired

            if pre_fired:
                s.pre_trace += 1.0
                s.elig -= 0.5 * s.post_trace   # LTD-Anteil
            if post_fired:
                s.post_trace += 1.0
                s.elig += 1.0 * s.pre_trace    # LTP-Anteil

            # Postsynaptische Neuron-Parameter
            post_is_exc = (s.post < len(self.exc))
            post_neuron = self.exc[s.post] if post_is_exc else self.inh[s.post - len(self.exc)]

            # Lernrate lokal: Feuerraten-Homeostase
            lr_scale = max(h.lr_floor, 1.0 - 0.5 * max(0.0, post_neuron.rate_dev()))
            # Kontext-Gate (pro Layer über ctx_gain_local, pro Neuron über post_neuron.ctx_gate)
            ctx_gate_total = max(h.ctx_gate_min, min(h.ctx_gate_max, post_neuron.ctx_gate * self.ctx_gain_local))
            # Oszillator-Gate
            lr = lr_scale * h.base_lr * stdp_gate * ctx_gate_total

            # Dopamin-Gate (Dreifaktor)
            dopa_gain = 1.0 + h.dop_gain * post_neuron.dopa_trace

            delta = lr * dopa_gain * s.elig * bias_gate

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
        if idx in self.out_adj.get(s.pre, []):
            self.out_adj[s.pre].remove(idx)
        if idx in self.in_adj.get(s.post, []):
            self.in_adj[s.post].remove(idx)
        self.syn[idx] = Synapse(-1, -1, 0.0, inhibitory=False)

    def _synaptic_scaling(self) -> None:
        target = self.h.scaling_target_sum
        nE = len(self.exc)
        incoming: Dict[int, List[int]] = {i: [] for i in range(nE)}
        for idx, s in enumerate(self.syn):
            if s.pre < 0: continue
            if not s.inhibitory and s.post < nE:
                incoming[s.post].append(idx)
        for post, idxs in incoming.items():
            if not idxs: continue
            ssum = sum(max(0.0, self.syn[i].w) for i in idxs)
            if ssum <= 1e-6: continue
            scale = target / ssum
            scale = max(0.5, min(2.0, scale))
            for i in idxs:
                self.syn[i].w = min(self.h.w_max_exc, max(0.0, self.syn[i].w * scale))

    def _growth_rule_exc_exc(self, fired_exc: List[int]) -> None:
        h = self.h
        for pre in fired_exc:
            for post in fired_exc:
                if pre == post: continue
                # Kante vorhanden?
                exists = False
                for sidx in self.out_adj.get(pre, []):
                    s = self.syn[sidx]
                    if s.pre >= 0 and (not s.inhibitory) and s.post == post:
                        exists = True; break
                if exists: continue

                # Energie prüfen
                npre, npost = self.exc[pre], self.exc[post]
                if npre.energy < h.growth_energy_cost or npost.energy < h.growth_energy_cost:
                    continue

                # Overlap-Block
                in_pre = {self.syn[i].pre for i in self.in_adj.get(pre, []) if self.syn[i].pre >= 0 and not self.syn[i].inhibitory}
                in_post = {self.syn[i].pre for i in self.in_adj.get(post, []) if self.syn[i].pre >= 0 and not self.syn[i].inhibitory}
                denom = len(in_pre | in_post)
                jacc = (len(in_pre & in_post) / denom) if denom > 0 else 0.0
                if jacc > h.growth_overlap_block:
                    continue

                # anlegen
                w0 = random.uniform(*h.growth_init_range)
                idx = self._add_syn(pre, post, w0, inhibitory=False)
                if idx is not None:
                    npre.energy -= h.growth_energy_cost * 0.5
                    npost.energy -= h.growth_energy_cost * 0.5
                    self.grown_count += 1

    # ---- Metriken ----

    def active_exc_ids(self) -> List[int]:
        return [i for i, n in enumerate(self.exc) if n.fired]

    def reset_windows(self) -> None:
        for n in self.exc + self.inh:
            n.reset_window()

# ==============================
# 5) Kontext-Layer (eigene Population)
# ==============================

class ContextLayer:
    """Einfache EXC-Population, deren Aktivität die Plastizität in CA3/CA1 moduliert."""
    def __init__(self, size: int, h: Hyper, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.h = h
        self.neur: List[ComparatorNeuron] = [ComparatorNeuron(threshold=1.0, leak=0.95, refractory=2) for _ in range(size)]

    def stimulate(self, ids: List[int], drive: float) -> None:
        for i in ids:
            if 0 <= i < len(self.neur):
                self.neur[i].v += drive

    def step(self) -> List[int]:
        fired: List[int] = []
        for i, n in enumerate(self.neur):
            if n.step(0.0):  # nur Leck + vorherige Depolarisation
                fired.append(i)
        return fired

    def activity_level(self) -> float:
        # grober Aktivitätspegel 0..1 (Anteil der Feuenden im letzten Schritt)
        # hier approximiert über Membranladung: einfacher Mittelwert der Reizspannung
        # (in dieser simplen Variante zurückhaltend: 0..1 clampen)
        avg_v = sum(abs(n.v) for n in self.neur) / max(1, len(self.neur))
        return max(0.0, min(1.0, avg_v))

# ==============================
# 6) Hippocampales System (DG, CA3, CA1, Ctx, Comparator)
# ==============================

@dataclass
class Assembly:
    ids: List[int]
    tag: str = ""
    saliency: float = 1.0
    ts: float = field(default_factory=time.time)

class HippocampalSystem:
    """
    DG -> CA3 -> CA1 Pipeline + Kontext + Comparator:
    - DG: Pattern-Separation (EXC only).
    - CA3: rekurrenter Attraktor (EXC/INH) mit Wachstum, kontextmodulierter Plastizität.
    - CA1: Comparator-Layer (EXC/INH) + Comparator-Subpopulation (cmp) für explizites Mismatch.
    - Ctx: Kontextpopulation; moduliert Plastizität in CA3/CA1 über ctx_gate.
    - Oszillator: Theta/Gamma-Fenster steuern Plastizität zeitlich.
    """
    def __init__(self, h: Optional[Hyper] = None, seed: Optional[int] = 7):
        self.h = h or Hyper()
        if seed is not None:
            random.seed(seed)

        self.osc = Oscillator(self.h.theta_period, self.h.gamma_period)

        # Layers
        self.DG  = Layer(self.h.dg_size, 0,             self.h, self.osc, seed=seed)
        self.CA3 = Layer(self.h.ca3_exc, self.h.ca3_inh, self.h, self.osc, seed=seed+1)
        self.CA1 = Layer(self.h.ca1_exc, self.h.ca1_inh, self.h, self.osc, seed=seed+2)
        # Kontext
        self.Ctx = ContextLayer(self.h.ctx_size, self.h, seed=seed+3)

        # Comparator-Subpopulation in CA1 (dediziert)
        self.CA1_cmp_pos: List[ComparatorNeuron] = [ComparatorNeuron(threshold=1.0) for _ in range(self.h.cmp_size//2)]
        self.CA1_cmp_neg: List[ComparatorNeuron] = [ComparatorNeuron(threshold=1.0) for _ in range(self.h.cmp_size//2)]

        # Feedforward DG->CA3 / DG->CA1 / CA3->CA1
        self.ff_dg_ca3: List[Synapse] = []
        self.ff_dg_ca1: List[Synapse] = []
        self.ff_ca3_ca1: List[Synapse] = []
        self._connect_ff(self.DG,  self.CA3, prob=0.06, w_range=(0.05, 0.12), inhibitory=False, store="dg_ca3")
        self._connect_ff(self.DG,  self.CA1, prob=0.05, w_range=(0.03, 0.08), inhibitory=False, store="dg_ca1")
        self._connect_ff(self.CA3, self.CA1, prob=0.06, w_range=(0.04, 0.10), inhibitory=False, store="ca3_ca1")

        # Kontext-Kopplungsstärke an Layern setzen (wirkt in _plasticity_update)
        self.CA3.ctx_gain_local = self.h.ctx_to_ca3_gain
        self.CA1.ctx_gain_local = self.h.ctx_to_ca1_gain

        # Replay/Engramme
        self.engrams: List[Assembly] = []

        # Schlaf- und Diagnose
        self.sleeping: bool = False
        self.ca1_last_mismatch: List[float] = [0.0] * len(self.CA1.exc)

    # ---- Feedforward-Verbindungen ----

    def _connect_ff(
        self,
        src: Layer,
        dst: Layer,
        prob: float,
        w_range: Tuple[float, float],
        inhibitory: bool,
        store: str
    ) -> None:
        syns: List[Synapse] = []
        n_src = len(src.exc)
        n_dst = len(dst.exc)
        for i in range(n_src):
            for j in range(n_dst):
                if random.random() < prob:
                    w = random.uniform(*w_range)
                    syns.append(Synapse(pre=i, post=j, w=w if not inhibitory else -abs(w), inhibitory=inhibitory))
        if store == "dg_ca3": self.ff_dg_ca3 = syns
        elif store == "dg_ca1": self.ff_dg_ca1 = syns
        elif store == "ca3_ca1": self.ff_ca3_ca1 = syns

    def _ff_currents(self, syns: List[Synapse], src_active_exc: List[int], dst_exc_n: int) -> List[float]:
        currents = [0.0] * dst_exc_n
        fired_set = set(src_active_exc)
        for s in syns:
            if s.pre in fired_set:
                currents[s.post] += s.w
        return currents

    # ---- Kontext-Gating anwenden ----
    def _apply_context(self) -> None:
        """
        Kontextaktivität moduliert pro postsynaptischem Neuron die ctx_gate (Plastizitäts-Skalierung).
        Einfaches Schema: globaler Kontextpegel -> additiver Offset.
        (Erweiterbar: Ctx->Layer spezifische Projektionen/Matrizen)
        """
        level = self.Ctx.activity_level()  # 0..1
        # skalierter Offset um 1.0 herum
        k3 = 1.0 + (self.h.ctx_to_ca3_gain * (level - 0.5))
        k1 = 1.0 + (self.h.ctx_to_ca1_gain * (level - 0.5))
        k3 = max(self.h.ctx_gate_min, min(self.h.ctx_gate_max, k3))
        k1 = max(self.h.ctx_gate_min, min(self.h.ctx_gate_max, k1))
        for n in self.CA3.exc: n.ctx_gate = k3
        for n in self.CA1.exc: n.ctx_gate = k1

    # ---- Pipeline-Schritt ----
    def present_to_DG(self, stimulus_ids: List[int]) -> None:
        ids = [i for i in stimulus_ids if 0 <= i < len(self.DG.exc)]
        self.DG.stimulate_exc(ids, drive=1.2)

    def present_context(self, ctx_ids: List[int]) -> None:
        ids = [i for i in ctx_ids if 0 <= i < self.h.ctx_size]
        self.Ctx.stimulate(ids, drive=self.h.ctx_drive)

    def step(self) -> None:
        # Oszillator tickt (Theta/Gamma-Phase)
        self.osc.step()

        # Kontext-Schritt zuerst (erzeugt Aktivität/Level für Gates)
        self.Ctx.step()
        self._apply_context()

        # DG
        self.DG.sleeping = self.sleeping
        self.DG.step()

        # DG->CA3
        dg_active = self.DG.active_exc_ids()
        ca3_dg_ff = self._ff_currents(self.ff_dg_ca3, dg_active, len(self.CA3.exc))
        for j, I in enumerate(ca3_dg_ff):
            self.CA3.exc[j].v += I

        # CA3 rekurrent
        self.CA3.sleeping = self.sleeping
        self.CA3.step()

        # CA1: getrennte FF-Ströme
        ca1_from_dg  = self._ff_currents(self.ff_dg_ca1,  dg_active,                   len(self.CA1.exc))
        ca1_from_ca3 = self._ff_currents(self.ff_ca3_ca1, self.CA3.active_exc_ids(),   len(self.CA1.exc))

        # Mismatch-Profil (absolut)
        self.ca1_last_mismatch = [abs(a - b) for a, b in zip(ca1_from_dg, ca1_from_ca3)]

        # Ströme additiv auf CA1 legen
        for j, I in enumerate(ca1_from_dg):
            self.CA1.exc[j].v += I
        for j, I in enumerate(ca1_from_ca3):
            self.CA1.exc[j].v += I

        # CA1 rekurrent
        self.CA1.sleeping = self.sleeping
        self.CA1.step()

        # Comparator-Subpopulation (dediziert, ± Kanäle)
        # pos-Kanal bekommt (DG - CA3)+, neg-Kanal bekommt (CA3 - DG)+
        # Wir verteilen die Summenstromstärke gleichmäßig auf cmp-Neurone.
        diff_sum_pos = sum(max(0.0, d - c) for d, c in zip(ca1_from_dg, ca1_from_ca3))
        diff_sum_neg = sum(max(0.0, c - d) for d, c in zip(ca1_from_dg, ca1_from_ca3))
        if self.CA1_cmp_pos:
            per = diff_sum_pos / len(self.CA1_cmp_pos)
            for n in self.CA1_cmp_pos:
                n.step(per)
        if self.CA1_cmp_neg:
            per = diff_sum_neg / len(self.CA1_cmp_neg)
            for n in self.CA1_cmp_neg:
                n.step(per)

        # Fenster ggf. nach Schlafblöcken resetten (passiert im Schlaf)

    # ---- Belohnung / Dopamin (RPE) ----

    def reward(self, magnitude: float = 0.5) -> None:
        for n in self.CA3.exc + self.CA3.inh + self.CA1.exc + self.CA1.inh:
            n.dopa_trace = min(1.0, n.dopa_trace + magnitude)

    # ---- Replay (Schlaf) ----

    def store_assembly(self, ids: List[int], tag: str = "", saliency: float = 1.0) -> None:
        ids = [i for i in ids if 0 <= i < len(self.DG.exc)]
        self.engrams.append(Assembly(ids=ids, tag=tag, saliency=saliency))

    def sleep_replay(self, rounds: int = 2) -> None:
        if not self.engrams: return
        self.sleeping = True
        for _ in range(rounds):
            now = time.time()
            weights = []
            for a in self.engrams:
                rec = 1.0 / (1.0 + (now - a.ts))
                weights.append(max(1e-6, a.saliency * rec))
            total = sum(weights)
            probs = [w/total for w in weights]
            picks = random.choices(self.engrams, weights=probs, k=min(self.h.replay_batch, len(self.engrams)))
            for a in picks:
                for _ in range(self.h.replay_steps):
                    self.present_to_DG(a.ids)
                    self.step()
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
            case EventKind.CTX:
                ids = list(ev.data.get("ctx_ids", []))
                self.present_context(ids)
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
# 7) Hilfen & Demo
# ==============================

def sparse_ids(n: int, ratio: float, seed: Optional[int] = None) -> List[int]:
    if seed is not None: random.seed(seed)
    k = max(1, int(n * ratio))
    return random.sample(range(n), k=k)

def demo() -> None:
    h = Hyper()
    sys = HippocampalSystem(h, seed=17)

    # Drei DG-Assemblies
    A1 = sparse_ids(h.dg_size, h.dg_activity_ratio, seed=1)
    A2 = sparse_ids(h.dg_size, h.dg_activity_ratio, seed=2)
    A3 = sparse_ids(h.dg_size, h.dg_activity_ratio, seed=3)

    # Zwei Kontextmuster (z.B. "Ort" vs. "Aufgabe")
    Ctx1 = sparse_ids(h.ctx_size, 0.25, seed=21)
    Ctx2 = sparse_ids(h.ctx_size, 0.25, seed=22)

    # --- Lernen im Wachzustand mit Kontext ---
    for _ in range(6):
        sys.deliver(Event(EventKind.CTX,      data={"ctx_ids": Ctx1}))
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids":  A1}))
        for _ in range(5): sys.deliver(Event(EventKind.TICK))
    sys.store_assembly(A1, tag="A1", saliency=1.4)

    for _ in range(6):
        sys.deliver(Event(EventKind.CTX,      data={"ctx_ids": Ctx2}))
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids":  A2}))
        for _ in range(5): sys.deliver(Event(EventKind.TICK))
    sys.store_assembly(A2, tag="A2", saliency=1.0)

    for _ in range(6):
        sys.deliver(Event(EventKind.CTX,      data={"ctx_ids": Ctx1}))
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids":  A3}))
        for _ in range(5): sys.deliver(Event(EventKind.TICK))
    sys.store_assembly(A3, tag="A3", saliency=0.8)

    # Belohnung (RPE) -> Dopaminspuren ↑
    sys.deliver(Event(EventKind.REWARD, data={"magnitude": 0.6}))

    # Schlaf-Replay (mit Theta/Gamma-Gates weiterhin aktiv)
    sys.deliver(Event(EventKind.SLEEP, data={"rounds": 3}))

    # --- Pattern Completion unter Kontext-Cues ---
    def cue_and_complete(dg_ids: List[int], ctx_ids: List[int], frac: float = 0.4) -> Tuple[int, int, float, int]:
        k = max(1, int(len(dg_ids) * frac))
        cue = random.sample(dg_ids, k=k)
        sys.deliver(Event(EventKind.CTX,      data={"ctx_ids": ctx_ids}))
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids":  cue}))
        for _ in range(3):  sys.deliver(Event(EventKind.TICK))
        for _ in range(12): sys.deliver(Event(EventKind.TICK))
        ca3_act = len(sys.CA3.active_exc_ids())
        ca1_act = len(sys.CA1.active_exc_ids())
        mismatch_mean = sum(sys.ca1_last_mismatch) / max(1, len(sys.ca1_last_mismatch))
        cmp_active = sum(1 for n in [*sys.CA1_cmp_pos, *sys.CA1_cmp_neg] if n.fired)
        return ca3_act, ca1_act, mismatch_mean, cmp_active

    a1_ca3, a1_ca1, a1_mm, a1_cmp = cue_and_complete(A1, Ctx1, 0.4)
    a2_ca3, a2_ca1, a2_mm, a2_cmp = cue_and_complete(A2, Ctx2, 0.4)
    a3_ca3, a3_ca1, a3_mm, a3_cmp = cue_and_complete(A3, Ctx1, 0.4)

    print("CA3 act:", a1_ca3, a2_ca3, a3_ca3,
          "| CA1 act:", a1_ca1, a2_ca1, a3_ca1)
    print("CA1 Mismatch mean:", f"{a1_mm:.4f}", f"{a2_mm:.4f}", f"{a3_mm:.4f}",
          "| Comparator fired:", a1_cmp, a2_cmp, a3_cmp)
    print("Wachstum (neu): CA3 =", sys.CA3.grown_count, "| CA1 =", sys.CA1.grown_count)

if __name__ == "__main__":
    demo()
