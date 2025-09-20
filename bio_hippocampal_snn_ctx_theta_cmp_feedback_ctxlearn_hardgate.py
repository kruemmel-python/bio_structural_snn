from __future__ import annotations

# ------------------------------------------------------------
# Hippocampus-Stack (biologisch inspiriert, voll):
# DG (Separation) -> CA3 (Attraktor/Completion) -> CA1 (Comparator)
# - LIF-Spikes, EXC/INH, Energie-Budget
# - STDP × Eligibility × Dopamin (Dreifaktor)
# - Strukturelles Wachstum (EXC->EXC) + Overlap-Block + Pruning
# - Synaptische Skalierung (Homeostase)
# - Theta/Gamma-Oszillator mit LTP/LTD-Asymmetrie
# - Kontext-Gating per Ctx->(CA3,CA1) MATRIZEN + langsames Hebb-Lernen (L1-Normierung)
# - CA1-Comparator-Subpopulation (±(DG,CA3)) + Comparator-Feedback
# - Comparator-Feedback moduliert Plastizität & Replay-Temperatur
# - NEU: Phasen-spezifische Replay-Politiken (konsolidierend vs. explorativ)
# - NEU: Hard-Gate bei starkem Mismatch (temporärer Konsolidierungs-Block)
# ------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
from enum import Enum, auto
import random
import time
import math

# ==============================
# 0) Ereignisse & Utils
# ==============================

class EventKind(Enum):
    STIMULUS = auto()   # DG-Input
    CTX      = auto()   # Kontext-Input
    REWARD   = auto()   # Belohnung (Dopamin/RPE)
    SLEEP    = auto()   # Schlaf/Replay
    TICK     = auto()   # freier Zeitschritt

@dataclass
class Event:
    kind: EventKind
    data: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

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
    ctx_size: int = 60
    cmp_size: int = 80  # CA1 Comparator-Subpopulation

    # Konnektivität / Gewichte
    init_conn: float = 0.04
    w_init: Tuple[float, float] = (-0.03, 0.03)
    w_max_exc: float = 1.6
    w_max_inh: float = 1.2
    max_out_degree: int = 40
    max_in_degree: int = 70

    # STDP / Eligibility / Dopamin
    base_lr: float = 0.01
    tau_trace: float = 0.90
    tau_elig: float = 0.95
    tau_dopa: float = 0.98
    dop_gain: float = 0.8
    lr_floor: float = 0.003

    # Kontext-Gating (Matrix-basiert) + LERNEN
    ctx_drive: float = 1.2
    ctx_decay: float = 0.92
    ctx_gate_min: float = 0.5
    ctx_gate_max: float = 1.8
    ctx_gain_ca3: float = 1.0
    ctx_gain_ca1: float = 1.0
    ctx_init_w: Tuple[float, float] = (0.2, 1.0)
    ctx_lr: float = 0.002
    ctx_w_decay: float = 0.0005
    ctx_w_min: float = 0.05
    ctx_w_max: float = 2.0
    ctx_l1_target_ca3: float = 30.0
    ctx_l1_target_ca1: float = 25.0

    # Theta/Gamma + LTP/LTD-Asymmetrie
    theta_period: int = 12
    gamma_period: int = 3
    theta_potent_window: Tuple[float, float] = (0.2, 0.6)
    gamma_gate_strength: float = 0.6
    ltp_boost: float = 1.2
    ltd_boost: float = 1.2

    # Energie
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

    # Strukturelles Wachstum
    growth_overlap_block: float = 0.4
    growth_init_range: Tuple[float, float] = (0.01, 0.04)

    # Replay (Basis)
    replay_steps: int = 8
    replay_batch: int = 3
    replay_temp_base: float = 1.0
    replay_temp_max: float = 1.8
    replay_temp_min: float = 0.6
    feedback_to_temp: float = 0.8

    # Comparator-Feedback -> Plastizitäts-Gate
    feedback_plasticity_min: float = 0.6
    feedback_plasticity_max: float = 1.2
    feedback_to_plasticity: float = 0.6

    # NEU: Phasen-spezifische Replay-Politiken
    replay_potent_temp_factor: float = 0.8    # im Potenzfenster: kühler (mehr Konsolidierung)
    replay_anti_temp_factor: float = 1.2      # außerhalb: heißer (mehr Exploration)
    replay_potent_plasticity_boost: float = 1.1
    replay_anti_plasticity_drop: float = 0.85

    # NEU: Hard-Gate im Comparator
    hard_gate_mismatch_threshold: float = 0.05  # ab diesem normalisierten Level blocken
    hard_gate_cooldown_steps: int = 20          # Dauer des Blocks in Schritten
    hard_gate_min_plasticity: float = 0.2       # wie stark Lernen gedrosselt wird
    hard_gate_temp_boost: float = 1.15          # Replay-Temp-Erhöhung im Hard-Gate

# ==============================
# 2) Oszillator (Theta/Gamma) mit LTP/LTD-Bias
# ==============================

@dataclass
class Oscillator:
    theta_period: int
    gamma_period: int
    t: int = 0

    def step(self) -> None:
        self.t += 1

    def theta_phase(self) -> float:
        return (self.t % self.theta_period) / self.theta_period

    def gamma_phase(self) -> float:
        return (self.t % self.gamma_period) / self.gamma_period

    def in_potent_window(self, h: Hyper) -> bool:
        th = self.theta_phase()
        t0, t1 = h.theta_potent_window
        return t0 <= th <= t1

    def gates(self, h: Hyper) -> Tuple[float, float, float]:
        """
        (stdp_gate, ltp_bias, ltd_bias)
        """
        th = self.theta_phase()
        gm = self.gamma_phase()
        t0, t1 = h.theta_potent_window
        in_potent = (t0 <= th <= t1)
        ltp_bias = h.ltp_boost if in_potent else 1.0 / h.ltp_boost
        ltd_bias = 1.0 / h.ltd_boost if in_potent else h.ltd_boost
        theta_gate = 1.0 if in_potent else 0.4
        gamma_gate = 1.0 if gm < 0.5 else h.gamma_gate_strength
        stdp_gate = theta_gate * gamma_gate
        return stdp_gate, ltp_bias, ltd_bias

# ==============================
# 3) Neurone & Synapsen
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
    energy: float = 5.0
    energy_max: float = 5.0
    fire_count: int = 0
    win_count: int = 0
    target_rate: float = 0.08
    dopa_trace: float = 0.0
    ctx_gate: float = 1.0  # per-Neuron Plastizitäts-Gate

    def reset_spike(self) -> None:
        self.fired = False

    def step(self, input_current: float, h: Hyper, sleeping: bool) -> bool:
        self.energy = max(0.0, self.energy - h.idle_energy_cost)
        if self.ref_count > 0:
            self.ref_count -= 1
            self.v *= self.leak
            self.fired = False
        else:
            gate = 1.0 if self.energy >= h.energy_gate else 0.6
            self.v = self.v * self.leak + gate * input_current
            if self.v >= self.threshold and self.energy >= h.spike_energy_cost:
                self.fired = True
                self.v = 0.0
                self.ref_count = self.refractory
                self.energy -= h.spike_energy_cost
                self.fire_count += 1
            else:
                self.fired = False
        rec = h.energy_recover_sleep if sleeping else h.energy_recover_awake
        self.energy = min(self.energy_max, self.energy + rec)
        self.ctx_gate = 1.0 + (self.ctx_gate - 1.0) * h.ctx_decay
        self.win_count += 1
        return self.fired

    def rate_dev(self) -> float:
        if self.win_count == 0: return 0.0
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
    pre_trace: float = 0.0
    post_trace: float = 0.0
    elig: float = 0.0

    def decay(self, h: Hyper) -> None:
        self.pre_trace *= h.tau_trace
        self.post_trace *= h.tau_trace
        self.elig *= h.tau_elig
        if self.w > 0:
            self.w = max(0.0, self.w - h.weight_decay)
        else:
            self.w = min(0.0, self.w + h.weight_decay)

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
# 4) Layer (EXC/INH, Wachstum, Skalierung, Gates)
# ==============================

class Layer:
    def __init__(self, n_exc: int, n_inh: int, h: Hyper, osc: Oscillator, seed: Optional[int] = None):
        if seed is not None: random.seed(seed)
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
        # globale Gates (Comparator/Replay/Hard-Gate beeinflussen)
        self.feedback_plasticity_gate: float = 1.0
        self.growth_enabled: bool = True  # kann vom Hard-Gate disabled werden

        # Rekurrenz
        self._rand_connect_exc_exc(self.exc, self.h.init_conn)
        self._rand_connect_exc_inh(self.exc, self.inh, self.h.init_conn)
        self._rand_connect_inh_exc(self.inh, self.exc, self.h.init_conn)

        self._steps = 0
        self.grown_count: int = 0

    # --- Synapsenverwaltung ---
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
                    self._add_syn(i, nE + j, random.uniform(0.02, 0.08), inhibitory=False)

    def _rand_connect_inh_exc(self, inh: List[Neuron], exc: List[Neuron], p: float) -> None:
        nE, nI = len(exc), len(inh)
        for i in range(nI):
            for j in range(nE):
                if random.random() < p:
                    self._add_syn(nE + i, j, -abs(random.uniform(0.02, 0.08)), inhibitory=True)

    # --- Laufzeit ---
    def _collect_currents(self) -> List[float]:
        nE, nI = len(self.exc), len(self.inh)
        currents = [0.0] * (nE + nI)
        fired_mask = [False] * (nE + nI)
        for i, n in enumerate(self.exc): fired_mask[i] = n.fired
        for i, n in enumerate(self.inh): fired_mask[nE + i] = n.fired
        for s in self.syn:
            if s.pre < 0: continue
            if fired_mask[s.pre]: currents[s.post] += s.w
        return currents

    def stimulate_exc(self, ids: List[int], drive: float) -> None:
        for i in ids:
            if 0 <= i < len(self.exc): self.exc[i].v += drive

    def step(self) -> Tuple[List[int], List[int]]:
        nE, nI = len(self.exc), len(self.inh)
        for n in self.exc: n.reset_spike()
        for n in self.inh: n.reset_spike()
        currents = self._collect_currents()
        fired_exc, fired_inh = [], []
        for i, n in enumerate(self.exc):
            if n.step(currents[i], self.h, sleeping=self.sleeping): fired_exc.append(i)
        for i, n in enumerate(self.inh):
            if n.step(currents[nE + i], self.h, sleeping=self.sleeping): fired_inh.append(nE + i)

        self._plasticity_update(fired_exc, fired_inh)

        if not self.sleeping and len(fired_exc) >= 2 and self.growth_enabled:
            self._growth_rule_exc_exc(fired_exc)

        self._steps += 1
        if self._steps % self.h.scaling_interval == 0:
            self._synaptic_scaling()
        return fired_exc, fired_inh

    def _plasticity_update(self, fired_exc: List[int], fired_inh: List[int]) -> None:
        h = self.h
        stdp_gate, ltp_bias, ltd_bias = self.osc.gates(h)
        fired = set([*fired_exc, *fired_inh])

        for n in self.exc + self.inh:
            n.dopa_trace *= h.tau_dopa

        for idx, s in enumerate(self.syn):
            if s.pre < 0: continue
            s.decay(h)
            pre_fired = s.pre in fired
            post_fired = s.post in fired
            if pre_fired:
                s.pre_trace += 1.0
                s.elig -= 0.5 * s.post_trace
            if post_fired:
                s.post_trace += 1.0
                s.elig += 1.0 * s.pre_trace

            post_is_exc = (s.post < len(self.exc))
            post_neuron = self.exc[s.post] if post_is_exc else self.inh[s.post - len(self.exc)]

            lr = h.base_lr
            lr *= clamp(1.0 - 0.5 * max(0.0, post_neuron.rate_dev()), h.lr_floor, 1.5)
            lr *= stdp_gate
            lr *= clamp(post_neuron.ctx_gate, h.ctx_gate_min, h.ctx_gate_max)
            lr *= clamp(self.feedback_plasticity_gate, h.feedback_plasticity_min, h.feedback_plasticity_max)

            dopa_gain = 1.0 + h.dop_gain * post_neuron.dopa_trace
            bias = ltp_bias if s.elig >= 0.0 else ltd_bias
            delta = lr * dopa_gain * s.elig * bias

            if s.inhibitory:
                s.w = max(-h.w_max_inh, min(0.0, s.w + delta))
            else:
                s.w = min(h.w_max_exc, max(0.0, s.w + delta))

            if abs(s.w) < h.pruning_floor:
                self._kill_syn(idx)

    def _kill_syn(self, idx: int) -> None:
        s = self.syn[idx]
        if idx in self.out_adj.get(s.pre, []): self.out_adj[s.pre].remove(idx)
        if idx in self.in_adj.get(s.post, []): self.in_adj[s.post].remove(idx)
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
            scale = clamp(target / ssum, 0.5, 2.0)
            for i in idxs:
                self.syn[i].w = clamp(self.syn[i].w * scale, 0.0, self.h.w_max_exc)

    def _growth_rule_exc_exc(self, fired_exc: List[int]) -> None:
        h = self.h
        for pre in fired_exc:
            for post in fired_exc:
                if pre == post: continue
                exists = any(
                    (self.syn[idx].pre >= 0 and not self.syn[idx].inhibitory and self.syn[idx].post == post)
                    for idx in self.out_adj.get(pre, [])
                )
                if exists: continue
                npre, npost = self.exc[pre], self.exc[post]
                if npre.energy < h.growth_energy_cost or npost.energy < h.growth_energy_cost:
                    continue
                in_pre = {self.syn[i].pre for i in self.in_adj.get(pre, []) if self.syn[i].pre >= 0 and not self.syn[i].inhibitory}
                in_post = {self.syn[i].pre for i in self.in_adj.get(post, []) if self.syn[i].pre >= 0 and not self.syn[i].inhibitory}
                denom = len(in_pre | in_post)
                jacc = (len(in_pre & in_post) / denom) if denom > 0 else 0.0
                if jacc > h.growth_overlap_block: continue
                idx = self._add_syn(pre, post, random.uniform(*self.h.growth_init_range), inhibitory=False)
                if idx is not None:
                    npre.energy -= h.growth_energy_cost * 0.5
                    npost.energy -= h.growth_energy_cost * 0.5
                    self.grown_count += 1

    def active_exc_ids(self) -> List[int]:
        return [i for i, n in enumerate(self.exc) if n.fired]

    def reset_windows(self) -> None:
        for n in self.exc + self.inh: n.reset_window()

# ==============================
# 5) Kontext-Layer (Spike-Muster)
# ==============================

class ContextLayer:
    def __init__(self, size: int, h: Hyper, seed: Optional[int] = None):
        if seed is not None: random.seed(seed)
        self.h = h
        self.cells: List[ComparatorNeuron] = [ComparatorNeuron(threshold=1.0, leak=0.95, refractory=2) for _ in range(size)]
        self.last_fired_mask: List[int] = [0] * size

    def stimulate(self, ids: List[int], drive: float) -> None:
        for i in ids:
            if 0 <= i < len(self.cells): self.cells[i].v += drive

    def step(self) -> List[int]:
        fired: List[int] = []
        self.last_fired_mask = [0] * len(self.cells)
        for i, n in enumerate(self.cells):
            if n.step(0.0):
                fired.append(i)
                self.last_fired_mask[i] = 1
        return fired

# ==============================
# 6) HippocampalSystem (mit phasen-spezifischem Replay & Hard-Gate)
# ==============================

@dataclass
class Assembly:
    ids: List[int]
    tag: str = ""
    saliency: float = 1.0
    ts: float = field(default_factory=time.time)

class HippocampalSystem:
    def __init__(self, h: Optional[Hyper] = None, seed: Optional[int] = 7):
        self.h = h or Hyper()
        if seed is not None: random.seed(seed)
        self.osc = Oscillator(self.h.theta_period, self.h.gamma_period)

        # Schichten
        self.DG  = Layer(self.h.dg_size, 0,             self.h, self.osc, seed=seed)
        self.CA3 = Layer(self.h.ca3_exc, self.h.ca3_inh, self.h, self.osc, seed=seed+1)
        self.CA1 = Layer(self.h.ca1_exc, self.h.ca1_inh, self.h, self.osc, seed=seed+2)
        self.Ctx = ContextLayer(self.h.ctx_size, self.h, seed=seed+3)

        # CA1 Comparator-Subpopulation
        self.CA1_cmp_pos = [ComparatorNeuron(threshold=1.0) for _ in range(self.h.cmp_size // 2)]
        self.CA1_cmp_neg = [ComparatorNeuron(threshold=1.0) for _ in range(self.h.cmp_size // 2)]

        # Feedforward
        self.ff_dg_ca3: List[Synapse] = []
        self.ff_dg_ca1: List[Synapse] = []
        self.ff_ca3_ca1: List[Synapse] = []
        self._connect_ff(self.DG,  self.CA3, 0.06, (0.05, 0.12), False, "dg_ca3")
        self._connect_ff(self.DG,  self.CA1, 0.05, (0.03, 0.08), False, "dg_ca1")
        self._connect_ff(self.CA3, self.CA1, 0.06, (0.04, 0.10), False, "ca3_ca1")

        # Kontext-Gating-MATRIZEN
        self.ctxW_ca3: List[List[float]] = [
            [random.uniform(*self.h.ctx_init_w) for _ in range(self.h.ctx_size)]
            for _ in range(len(self.CA3.exc))
        ]
        self.ctxW_ca1: List[List[float]] = [
            [random.uniform(*self.h.ctx_init_w) for _ in range(self.h.ctx_size)]
            for _ in range(len(self.CA1.exc))
        ]

        # Replay/Engramme
        self.engrams: List[Assembly] = []

        # Diagnose & Steuerung
        self.sleeping = False
        self.ca1_last_mismatch: List[float] = [0.0] * len(self.CA1.exc)
        self.replay_temp: float = self.h.replay_temp_base
        self.hard_gate_countdown: int = 0  # >0 => Hard-Gate aktiv

    # --- Feedforward ---
    def _connect_ff(self, src: Layer, dst: Layer, prob: float, w_range: Tuple[float, float], inhibitory: bool, store: str) -> None:
        syns: List[Synapse] = []
        n_src, n_dst = len(src.exc), len(dst.exc)
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

    # --- Kontext-Gating per Matrix ---
    def _apply_context_matrix(self) -> None:
        a = self.Ctx.last_fired_mask
        eps = 1e-6
        for j, n in enumerate(self.CA3.exc):
            wj = self.ctxW_ca3[j]
            proj = sum(w * ai for w, ai in zip(wj, a))
            norm = sum(abs(w) for w in wj) + eps
            score = (proj / norm)
            gate = 1.0 + self.h.ctx_gain_ca3 * (score - 0.5)
            n.ctx_gate = clamp(gate, self.h.ctx_gate_min, self.h.ctx_gate_max)
        for j, n in enumerate(self.CA1.exc):
            wj = self.ctxW_ca1[j]
            proj = sum(w * ai for w, ai in zip(wj, a))
            norm = sum(abs(w) for w in wj) + eps
            score = (proj / norm)
            gate = 1.0 + self.h.ctx_gain_ca1 * (score - 0.5)
            n.ctx_gate = clamp(gate, self.h.ctx_gate_min, self.h.ctx_gate_max)

    # --- Lernen der Kontext-Matrizen ---
    def _learn_context_matrices(self) -> None:
        a = self.Ctx.last_fired_mask
        eta = self.h.ctx_lr
        dec = self.h.ctx_w_decay

        for j, n in enumerate(self.CA3.exc):
            if not n.fired:
                self.ctxW_ca3[j] = [clamp(w - dec, self.h.ctx_w_min, self.h.ctx_w_max) for w in self.ctxW_ca3[j]]
                continue
            wj = self.ctxW_ca3[j]
            for i in range(len(wj)):
                wj[i] += eta * (a[i] - 0.5)
                wj[i] = clamp(wj[i] - dec, self.h.ctx_w_min, self.h.ctx_w_max)
            s = sum(abs(x) for x in wj)
            if s > 1e-6:
                scale = self.h.ctx_l1_target_ca3 / s
                self.ctxW_ca3[j] = [clamp(x * scale, self.h.ctx_w_min, self.h.ctx_w_max) for x in wj]

        for j, n in enumerate(self.CA1.exc):
            if not n.fired:
                self.ctxW_ca1[j] = [clamp(w - dec, self.h.ctx_w_min, self.h.ctx_w_max) for w in self.ctxW_ca1[j]]
                continue
            wj = self.ctxW_ca1[j]
            for i in range(len(wj)):
                wj[i] += eta * (a[i] - 0.5)
                wj[i] = clamp(wj[i] - dec, self.h.ctx_w_min, self.h.ctx_w_max)
            s = sum(abs(x) for x in wj)
            if s > 1e-6:
                scale = self.h.ctx_l1_target_ca1 / s
                self.ctxW_ca1[j] = [clamp(x * scale, self.h.ctx_w_min, self.h.ctx_w_max) for x in wj]

    # --- Comparator-Feedback ---
    def _apply_comparator_feedback(self, diff_sum_pos: float, diff_sum_neg: float) -> None:
        total = diff_sum_pos + diff_sum_neg
        norm = len(self.CA1.exc) + 1e-6
        lvl = total / norm

        # Hard-Gate auslösen?
        if lvl >= self.h.hard_gate_mismatch_threshold:
            self.hard_gate_countdown = max(self.hard_gate_countdown, self.h.hard_gate_cooldown_steps)

        # Grund-Gate (ohne Hard-Gate)
        gate = 1.0 - self.h.feedback_to_plasticity * lvl
        gate = clamp(gate, self.h.feedback_plasticity_min, self.h.feedback_plasticity_max)

        # Hard-Gate aktiv?
        if self.hard_gate_countdown > 0:
            # Drossle Lernen deutlich, deaktiviere Wachstum, hebe Temperatur
            hard_gate = self.h.hard_gate_min_plasticity
            self.CA3.feedback_plasticity_gate = hard_gate
            self.CA1.feedback_plasticity_gate = hard_gate
            self.CA3.growth_enabled = False
            self.CA1.growth_enabled = False
            self.replay_temp = clamp(self.replay_temp * self.h.hard_gate_temp_boost,
                                     self.h.replay_temp_min, self.h.replay_temp_max)
            self.hard_gate_countdown -= 1
        else:
            # Normalbetrieb: Wachstum wieder erlauben
            self.CA3.growth_enabled = True
            self.CA1.growth_enabled = True
            self.CA3.feedback_plasticity_gate = gate
            self.CA1.feedback_plasticity_gate = gate

        # Replay-Temperatur Grundregel (ohne/mit Hard-Gate bereits angepasst)
        temp_base = self.h.replay_temp_base + self.h.feedback_to_temp * lvl
        self.replay_temp = clamp(max(self.replay_temp, temp_base), self.h.replay_temp_min, self.h.replay_temp_max)

    # --- Public I/O ---
    def present_to_DG(self, stimulus_ids: List[int]) -> None:
        ids = [i for i in stimulus_ids if 0 <= i < len(self.DG.exc)]
        self.DG.stimulate_exc(ids, drive=1.2)

    def present_context(self, ctx_ids: List[int]) -> None:
        ids = [i for i in ctx_ids if 0 <= i < self.h.ctx_size]
        self.Ctx.stimulate(ids, drive=self.h.ctx_drive)

    # --- Ein Schritt ---
    def step(self) -> None:
        self.osc.step()

        # Kontext
        self.Ctx.step()
        self._apply_context_matrix()

        # DG
        self.DG.sleeping = self.sleeping
        self.DG.step()

        # DG->CA3
        dg_active = self.DG.active_exc_ids()
        ca3_dg_ff = self._ff_currents(self.ff_dg_ca3, dg_active, len(self.CA3.exc))
        for j, I in enumerate(ca3_dg_ff): self.CA3.exc[j].v += I

        # CA3 rekurrent
        self.CA3.sleeping = self.sleeping
        self.CA3.step()

        # CA1: getrennte FF-Ströme
        ca1_from_dg  = self._ff_currents(self.ff_dg_ca1,  dg_active,                   len(self.CA1.exc))
        ca1_from_ca3 = self._ff_currents(self.ff_ca3_ca1, self.CA3.active_exc_ids(),   len(self.CA1.exc))
        self.ca1_last_mismatch = [abs(a - b) for a, b in zip(ca1_from_dg, ca1_from_ca3)]

        # CA1 depolarisieren
        for j, I in enumerate(ca1_from_dg):  self.CA1.exc[j].v += I
        for j, I in enumerate(ca1_from_ca3): self.CA1.exc[j].v += I

        # CA1 rekurrent
        self.CA1.sleeping = self.sleeping
        self.CA1.step()

        # Comparator-Subpopulation
        diff_sum_pos = sum(max(0.0, d - c) for d, c in zip(ca1_from_dg, ca1_from_ca3))
        diff_sum_neg = sum(max(0.0, c - d) for d, c in zip(ca1_from_dg, ca1_from_ca3))
        if self.CA1_cmp_pos:
            per = diff_sum_pos / len(self.CA1_cmp_pos)
            for n in self.CA1_cmp_pos: n.step(per)
        if self.CA1_cmp_neg:
            per = diff_sum_neg / len(self.CA1_cmp_neg)
            for n in self.CA1_cmp_neg: n.step(per)

        # Comparator-Feedback (inkl. Hard-Gate/Temp)
        self._apply_comparator_feedback(diff_sum_pos, diff_sum_neg)

        # Kontext-Matrizen lernen (langsam)
        self._learn_context_matrices()

    # --- Belohnung / Replay ---
    def reward(self, magnitude: float = 0.5) -> None:
        for n in self.CA3.exc + self.CA3.inh + self.CA1.exc + self.CA1.inh:
            n.dopa_trace = min(1.0, n.dopa_trace + magnitude)

    def store_assembly(self, ids: List[int], tag: str = "", saliency: float = 1.0) -> None:
        ids = [i for i in ids if 0 <= i < len(self.DG.exc)]
        self.engrams.append(Assembly(ids=ids, tag=tag, saliency=saliency))

    def _softmax_temp(self, xs: List[float], temp: float) -> List[float]:
        if not xs: return []
        x = [v / max(1e-9, temp) for v in xs]
        m = max(x)
        exps = [math.exp(v - m) for v in x]
        s = sum(exps)
        return [v / s for v in exps]

    def sleep_replay(self, rounds: int = 2) -> None:
        if not self.engrams: return
        self.sleeping = True
        for _ in range(rounds):
            # Auswahl mit Temperatur (diese kann im Lauf durch Hard-Gate/Comparator steigen)
            now = time.time()
            scores = []
            for a in self.engrams:
                rec = 1.0 / (1.0 + (now - a.ts))
                scores.append(max(1e-6, a.saliency * rec))
            # weiche Verteilung mit aktueller Temp
            probs = self._softmax_temp(scores, temp=self.replay_temp)
            k = min(self.h.replay_batch, len(self.engrams))
            picks = random.choices(self.engrams, weights=probs, k=k)

            for a in picks:
                for step_i in range(self.h.replay_steps):
                    # Phasen-spezifische Replay-Politik (vor JEDEM Schritt)
                    in_potent = self.osc.in_potent_window(self.h)
                    if in_potent:
                        # Konsolidierendes Replay: kühler, plastischer
                        self.replay_temp = clamp(self.replay_temp * self.h.replay_potent_temp_factor,
                                                 self.h.replay_temp_min, self.h.replay_temp_max)
                        boost = self.h.replay_potent_plasticity_boost
                        self.CA3.feedback_plasticity_gate = clamp(self.CA3.feedback_plasticity_gate * boost,
                                                                  self.h.feedback_plasticity_min, self.h.feedback_plasticity_max)
                        self.CA1.feedback_plasticity_gate = clamp(self.CA1.feedback_plasticity_gate * boost,
                                                                  self.h.feedback_plasticity_min, self.h.feedback_plasticity_max)
                    else:
                        # Exploratives Replay: heißer, Lernrate leicht drosseln
                        self.replay_temp = clamp(self.replay_temp * self.h.replay_anti_temp_factor,
                                                 self.h.replay_temp_min, self.h.replay_temp_max)
                        drop = self.h.replay_anti_plasticity_drop
                        self.CA3.feedback_plasticity_gate = clamp(self.CA3.feedback_plasticity_gate * drop,
                                                                  self.h.feedback_plasticity_min, self.h.feedback_plasticity_max)
                        self.CA1.feedback_plasticity_gate = clamp(self.CA1.feedback_plasticity_gate * drop,
                                                                  self.h.feedback_plasticity_min, self.h.feedback_plasticity_max)

                    # eigentliche Replay-Stimulation + Netzschritt
                    self.present_to_DG(a.ids)
                    self.step()

            # Fenster zurücksetzen für saubere Ratenmessung
            self.DG.reset_windows(); self.CA3.reset_windows(); self.CA1.reset_windows()
        self.sleeping = False

    # --- Ereignisse ---
    def deliver(self, ev: Event) -> None:
        match ev.kind:
            case EventKind.STIMULUS:
                self.present_to_DG(list(ev.data.get("dg_ids", [])))
                self.step()
            case EventKind.CTX:
                self.present_context(list(ev.data.get("ctx_ids", [])))
                self.step()
            case EventKind.REWARD:
                self.reward(float(ev.data.get("magnitude", 0.5)))
            case EventKind.SLEEP:
                self.sleep_replay(int(ev.data.get("rounds", 3)))
            case EventKind.TICK:
                self.step()
            case _:
                raise ValueError(f"Unbekanntes Event: {ev.kind!r}")

# ==============================
# 7) Demo / Tests
# ==============================

def sparse_ids(n: int, ratio: float, seed: Optional[int] = None) -> List[int]:
    if seed is not None: random.seed(seed)
    k = max(1, int(n * ratio))
    return random.sample(range(n), k=k)

def demo() -> None:
    h = Hyper()
    sys = HippocampalSystem(h, seed=31)

    # DG-Assemblies
    A1 = sparse_ids(h.dg_size, 0.04, seed=1)
    A2 = sparse_ids(h.dg_size, 0.04, seed=2)
    A3 = sparse_ids(h.dg_size, 0.04, seed=3)
    # Kontexte
    C1 = sparse_ids(h.ctx_size, 0.25, seed=11)
    C2 = sparse_ids(h.ctx_size, 0.25, seed=12)

    # Lernen (mit Kontext)
    for _ in range(6):
        sys.deliver(Event(EventKind.CTX,      data={"ctx_ids": C1}))
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids":  A1}))
        for _ in range(5): sys.deliver(Event(EventKind.TICK))
    sys.store_assembly(A1, "A1", saliency=1.4)

    for _ in range(6):
        sys.deliver(Event(EventKind.CTX,      data={"ctx_ids": C2}))
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids":  A2}))
        for _ in range(5): sys.deliver(Event(EventKind.TICK))
    sys.store_assembly(A2, "A2", saliency=1.0)

    for _ in range(6):
        sys.deliver(Event(EventKind.CTX,      data={"ctx_ids": C1}))
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids":  A3}))
        for _ in range(5): sys.deliver(Event(EventKind.TICK))
    sys.store_assembly(A3, "A3", saliency=0.8)

    # Belohnung
    sys.deliver(Event(EventKind.REWARD, data={"magnitude": 0.6}))

    # Schlaf (phasen-spezifische Politik & Hard-Gate aktiv)
    sys.deliver(Event(EventKind.SLEEP, data={"rounds": 3}))

    # Completion unter passendem Kontext
    def cue_and_complete(dg_ids: List[int], ctx_ids: List[int], frac: float = 0.4) -> Tuple[int, int, float, int, float, int]:
        k = max(1, int(len(dg_ids) * frac))
        cue = random.sample(dg_ids, k=k)
        sys.deliver(Event(EventKind.CTX,      data={"ctx_ids": ctx_ids}))
        sys.deliver(Event(EventKind.STIMULUS, data={"dg_ids":  cue}))
        for _ in range(3):  sys.deliver(Event(EventKind.TICK))
        for _ in range(12): sys.deliver(Event(EventKind.TICK))
        ca3_act = len(sys.CA3.active_exc_ids())
        ca1_act = len(sys.CA1.active_exc_ids())
        mismatch = sum(sys.ca1_last_mismatch) / max(1, len(sys.ca1_last_mismatch))
        cmp_active = sum(1 for n in [*sys.CA1_cmp_pos, *sys.CA1_cmp_neg] if n.fired)
        return ca3_act, ca1_act, mismatch, cmp_active, sys.replay_temp, sys.hard_gate_countdown

    a1 = cue_and_complete(A1, C1)
    a2 = cue_and_complete(A2, C2)
    a3 = cue_and_complete(A3, C1)

    print("CA3 act / CA1 act:", a1[0], a1[1], "|", a2[0], a2[1], "|", a3[0], a3[1])
    print("Mismatch mean:", f"{a1[2]:.4f}", f"{a2[2]:.4f}", f"{a3[2]:.4f}",
          "| Comparator fired:", a1[3], a2[3], a3[3])
    print("Replay-Temperatur (dyn.):", f"{a1[4]:.2f}", "| HardGate-Countdown:", a1[5])
    print("Wachstum neu: CA3 =", sys.CA3.grown_count, "| CA1 =", sys.CA1.grown_count)

if __name__ == "__main__":
    demo()
