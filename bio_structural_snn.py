from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
import random
import time
import math

# ============================================================
# 1) Ereignisse & Parameter
# ============================================================

class EventKind(Enum):
    STIMULUS = auto()
    REWARD = auto()
    SLEEP = auto()
    TICK = auto()

@dataclass
class Event:
    kind: EventKind
    data: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class Hyper:
    """Alle wichtigen Hyperparameter an einem Ort (leicht zu tweaken)."""
    # Neuron
    threshold: float = 1.0
    leak: float = 0.95
    refractory: int = 2

    # Energie/Metabolik
    spike_energy_cost: float = 1.0       # Energiekosten pro Spike
    idle_energy_cost: float = 0.02       # Grundumsatz pro Schritt
    energy_recover_awake: float = 0.05   # Regeneration im Wachzustand
    energy_recover_sleep: float = 0.25   # Regeneration pro Schlaf-Schritt
    energy_max: float = 5.0              # Kapazität (ATP/Glukose-Analog)
    energy_low_gate: float = 0.6         # unterhalb dessen: Plastizität drosseln

    # Synapsen / STDP / Wachstum
    w_init: Tuple[float, float] = (-0.03, 0.03)
    wmax: float = 1.6
    base_lr: float = 0.01
    tau_trace: float = 0.90             # Zerfall STDP-Spuren
    grow_threshold: float = 0.8          # Koaktivitätsschwelle für Neubildung
    growth_energy_cost: float = 0.6      # Energiegebühr für NEUE Synapse (Axonwachstum)
    pruning_floor: float = 0.02          # unterhalb: harte Löschung
    decay_per_step: float = 0.002        # passiver Gewichtszerfall (Vergessen)

    # Konnektivität & Kapazitäten
    initial_conn_prob: float = 0.06
    max_out_degree: int = 40             # axonales Budget (Anschlüsse pro Neuron)
    max_in_degree: int = 60              # dendritisches Budget
    overlap_block: float = 0.4           # hemmt Wachstum bei stark überlappenden Assemblies

    # Replay/Salienz
    replay_steps: int = 8
    replay_batch: int = 3

# ============================================================
# 2) Synapsen, Neuronen, Assemblies
# ============================================================

@dataclass
class Synapse:
    pre: int
    post: int
    w: float = 0.0
    pre_trace: float = 0.0
    post_trace: float = 0.0

    def decay(self, tau: float, decay_w: float) -> None:
        self.pre_trace *= tau
        self.post_trace *= tau
        # passiver Gewichtszerfall (Vergessen)
        if self.w > 0:
            self.w = max(0.0, self.w - decay_w)
        else:
            self.w = min(0.0, self.w + decay_w)  # auch inhibitorische Annäherung an 0

@dataclass
class Neuron:
    v: float = 0.0
    threshold: float = 1.0
    leak: float = 0.95
    refractory: int = 2
    ref_count: int = 0
    fired: bool = False

    # Energiehaushalt
    energy: float = 5.0
    energy_max: float = 5.0

    # Aktivitätsstatistik (für Feuerraten-Homeostase)
    fire_count: int = 0
    window_count: int = 0
    target_rate: float = 0.08  # Ziel-Feuerrate (pro Fenster)

    def reset_spike(self) -> None:
        self.fired = False

    def step(self, input_current: float, h: Hyper, sleeping: bool) -> bool:
        """Integriere Eingang; beachte Energie; feuere ggf.; aktualisiere Energie."""
        # Grundumsatz (ATP-Verbrauch)
        self.energy = max(0.0, self.energy - h.idle_energy_cost)

        if self.ref_count > 0:
            self.ref_count -= 1
            self.v *= self.leak
            self.fired = False
        else:
            # Energie-knappheit: excitability gedrosselt (soft gate)
            energy_gate = 1.0 if self.energy >= h.energy_low_gate else 0.6
            self.v = self.v * self.leak + energy_gate * input_current

            if self.v >= self.threshold and self.energy >= h.spike_energy_cost:
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

        # Feuerraten-Statistik
        self.window_count += 1
        return self.fired

    def rate_deviation(self) -> float:
        """Abweichung von der Zielrate (positiv = zu aktiv, negativ = zu ruhig)."""
        if self.window_count == 0:
            return 0.0
        rate = self.fire_count / self.window_count
        return rate - self.target_rate

    def reset_window(self) -> None:
        self.fire_count = 0
        self.window_count = 0

@dataclass
class Assembly:
    active_ids: List[int]
    tag: str = ""
    saliency: float = 1.0
    ts: float = field(default_factory=time.time)

# ============================================================
# 3) Netzwerk mit struktureller Plastizität
# ============================================================

class StructuralSNN:
    def __init__(self, n: int, h: Optional[Hyper] = None, seed: Optional[int] = 7):
        self.h = h or Hyper()
        if seed is not None:
            random.seed(seed)

        self.neurons: List[Neuron] = [
            Neuron(
                threshold=self.h.threshold,
                leak=self.h.leak,
                refractory=self.h.refractory,
                energy=self.h.energy_max,
                energy_max=self.h.energy_max,
            )
            for _ in range(n)
        ]
        self.synapses: List[Synapse] = []
        self.out_adj: Dict[int, List[int]] = {}
        self.in_adj: Dict[int, List[int]] = {}

        # spärliche Anfangskonnektivität
        for pre in range(n):
            for post in range(n):
                if pre == post:
                    continue
                if random.random() < self.h.initial_conn_prob:
                    self._add_synapse(pre, post, random.uniform(*self.h.w_init))

        # Gedächtnisspuren
        self.engrams: List[Assembly] = []
        self.sleeping: bool = False

    # ---------- Grundgraph-Operationen ----------

    def _add_synapse(self, pre: int, post: int, w: float) -> Optional[int]:
        """Fügt Synapse hinzu, wenn Budget (out/in-degree) vorhanden."""
        # Budget prüfen
        if len(self.out_adj.get(pre, [])) >= self.h.max_out_degree:
            return None
        if len(self.in_adj.get(post, [])) >= self.h.max_in_degree:
            return None
        idx = len(self.synapses)
        self.synapses.append(Synapse(pre=pre, post=post, w=w))
        self.out_adj.setdefault(pre, []).append(idx)
        self.in_adj.setdefault(post, []).append(idx)
        return idx

    def _remove_synapse(self, idx: int) -> None:
        """Synapse hart entfernen (Pruning)."""
        syn = self.synapses[idx]
        self.out_adj[syn.pre].remove(idx)
        self.in_adj[syn.post].remove(idx)
        # Platzhalter: wir markieren als „tote“ Synapse
        self.synapses[idx] = Synapse(pre=-1, post=-1, w=0.0)

    # ---------- Dynamik ----------

    def _collect_currents(self) -> List[float]:
        currents = [0.0] * len(self.neurons)
        for i, syn in enumerate(self.synapses):
            if syn.pre < 0:
                continue  # tote Synapse
            if self.neurons[syn.pre].fired:
                currents[syn.post] += syn.w
        return currents

    def _stdp_and_decay(self, fired: List[int]) -> None:
        """STDP + passiver Zerfall; Feuerraten-Homeostase skaliert LR lokal."""
        fired_set = set(fired)
        for i, syn in enumerate(self.synapses):
            if syn.pre < 0:
                continue
            # Zerfall (Spuren + Gewicht)
            syn.decay(self.h.tau_trace, self.h.decay_per_step)

            pre_fired = syn.pre in fired_set
            post_fired = syn.post in fired_set

            # Lernrate lokal modifiziert: zu aktive Postsynapsen -> etwas LTD-lastiger
            post_dev = self.neurons[syn.post].rate_deviation()  # >0 = zu aktiv
            lr_scale = 1.0 - 0.5 * max(0.0, post_dev)          # bremst Plastizität, wenn zu aktiv
            lr = max(0.2, lr_scale) * self.h.base_lr

            if pre_fired:
                syn.pre_trace += 1.0
                # Post vor Pre -> LTD
                syn.w -= lr * syn.post_trace
            if post_fired:
                syn.post_trace += 1.0
                # Pre vor Post -> LTP
                syn.w += lr * syn.pre_trace

            # Kappen
            syn.w = max(-self.h.wmax, min(self.h.wmax, syn.w))

            # Pruning: winzige Gewichte löschen
            if abs(syn.w) < self.h.pruning_floor:
                self._remove_synapse(i)

    def _growth_rule(self, coactive_pairs: List[Tuple[int, int]]) -> None:
        """Strukturelles Wachstum: Koaktive Paare bauen Synapsen, sofern Budget/Energie ok."""
        for pre, post in coactive_pairs:
            if pre == post:
                continue
            # Bestehende Kante?
            exists = False
            for idx in self.out_adj.get(pre, []):
                s = self.synapses[idx]
                if s.post == post:
                    exists = True
                    break
            if exists:
                continue

            # Energiegebühr für Wachstum beim prä- und postsynaptischen Neuron
            npre, npost = self.neurons[pre], self.neurons[post]
            if npre.energy < self.h.growth_energy_cost or npost.energy < self.h.growth_energy_cost:
                continue  # zu wenig Energie

            # Overlap-Block: wenn beide Neuronen sehr ähnliche Eingangsmuster haben,
            # hemmen wir redundantes Wachstum (einfacher Heuristik-Proxy)
            in_pre = {self.synapses[i].pre for i in self.in_adj.get(pre, []) if self.synapses[i].pre >= 0}
            in_post = {self.synapses[i].pre for i in self.in_adj.get(post, []) if self.synapses[i].pre >= 0}
            jacc = len(in_pre & in_post) / max(1, len(in_pre | in_post))
            if jacc > self.h.overlap_block:
                continue  # vermeidet „Kurzschlüsse“ in stark überlappenden Subnetzen

            # Budget prüfen und Synapse bauen
            idx = self._add_synapse(pre, post, w=random.uniform(*self.h.w_init))
            if idx is not None:
                npre.energy -= self.h.growth_energy_cost * 0.5
                npost.energy -= self.h.growth_energy_cost * 0.5

    # ---------- Öffentliche API ----------

    def stimulate(self, active_ids: List[int], drive: float = 1.2) -> None:
        """Externe Stimulation: depolarisiert die angegebenen Neuronen leicht."""
        for nid in active_ids:
            self.neurons[nid].v += drive

    def step(self) -> List[int]:
        """Ein Simulationsschritt (Wachzustand oder Schlaf je nach Flag)."""
        currents = self._collect_currents()
        fired: List[int] = []
        for i, n in enumerate(self.neurons):
            n.reset_spike()
        for i, n in enumerate(self.neurons):
            if n.step(currents[i], self.h, sleeping=self.sleeping):
                fired.append(i)

        # STDP, Zerfall, Pruning
        self._stdp_and_decay(fired)

        # Strukturelles Wachstum: koaktive Paare (einfach: alle Paare der Feuenden)
        # Nur im Wachzustand (im Schlaf konzentrieren wir uns auf Konsolidierung über STDP)
        if not self.sleeping and len(fired) >= 2:
            # Pairwise-Koaktivität absichern: Schwelle über „Koaktivitätsscore“
            # Wir verwenden hier einfach die aktuelle Aktivität als Proxy → Paar wird vorgeschlagen.
            self._growth_rule([(i, j) for i in fired for j in fired if i != j])

        return fired

    def store_assembly(self, asm: Assembly) -> None:
        self.engrams.append(asm)

    def sleep_replay(self, rounds: int = 2) -> None:
        """Schlaf: priorisierte Replays + schnelle Regeneration + kein Wachstum."""
        if not self.engrams:
            return
        self.sleeping = True
        for _ in range(rounds):
            # Priorisierung nach Salienz × Rezenz
            now = time.time()
            weights = []
            for asm in self.engrams:
                rec = 1.0 / (1.0 + (now - asm.ts))
                weights.append(max(1e-6, asm.saliency * rec))
            total = sum(weights)
            probs = [w / total for w in weights]
            chosen = random.choices(self.engrams, weights=probs, k=min(self.h.replay_batch, len(self.engrams)))

            for asm in chosen:
                # Leichtes Drive (niedriger als Wachlernen)
                for _ in range(self.h.replay_steps):
                    self.stimulate(asm.active_ids, drive=0.7)
                    self.step()

            # Nach jedem Replay-Fenster Feuerratenfenster zurücksetzen (Homeostase frisch)
            for n in self.neurons:
                n.reset_window()
        self.sleeping = False

    def deliver(self, ev: Event) -> None:
        match ev.kind:
            case EventKind.STIMULUS:
                ids = list(ev.data.get("active_ids", []))
                drive = float(ev.data.get("drive", 1.2))
                self.stimulate(ids, drive=drive)
                self.step()
            case EventKind.REWARD:
                # (Optional) Könnte Neuromodulator einbauen; hier: Energie-Boost als Proxy
                mag = float(ev.data.get("magnitude", 0.6))
                for n in self.neurons:
                    n.energy = min(n.energy_max, n.energy + mag)
                # leichte Erhöhung der Salienz der letzten Engramme
                if self.engrams:
                    self.engrams[-1].saliency *= 1.15
            case EventKind.SLEEP:
                rounds = int(ev.data.get("rounds", 3))
                self.sleep_replay(rounds=rounds)
            case EventKind.TICK:
                self.step()
            case _:
                raise ValueError(f"Unbekanntes Event: {ev.kind!r}")

# ============================================================
# 4) Kleine Demo / Experiment
# ============================================================

def make_sparse_ids(n: int, ratio: float, seed: Optional[int] = None) -> List[int]:
    if seed is not None:
        random.seed(seed)
    k = max(1, int(n * ratio))
    return random.sample(range(n), k=k)

def jaccard(a: List[int], b: List[int]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def run_demo() -> None:
    N = 320
    h = Hyper()
    net = StructuralSNN(N, h=h, seed=3)

    # Drei Assemblies lernen
    A1 = Assembly(make_sparse_ids(N, 0.06, seed=1), tag="A1", saliency=1.4)
    A2 = Assembly(make_sparse_ids(N, 0.06, seed=2), tag="A2", saliency=1.0)
    A3 = Assembly(make_sparse_ids(N, 0.06, seed=3), tag="A3", saliency=0.8)

    # wiederholte Präsentation (Wachzustand): strukturelles Wachstum + STDP
    for _ in range(6):
        net.stimulate(A1.active_ids, drive=1.2)
        for _ in range(6):
            net.step()
    net.store_assembly(A1)

    for _ in range(6):
        net.stimulate(A2.active_ids, drive=1.2)
        for _ in range(6):
            net.step()
    net.store_assembly(A2)

    for _ in range(6):
        net.stimulate(A3.active_ids, drive=1.2)
        for _ in range(6):
            net.step()
    net.store_assembly(A3)

    # Schlaf-/Replay-Konsolidierung
    net.deliver(Event(kind=EventKind.SLEEP, data={"rounds": 3}))

    # Pattern Completion: 40% Cues, freie Dynamik
    def cue_and_complete(ids: List[int], frac: float = 0.4) -> List[int]:
        k = max(1, int(len(ids) * frac))
        cue = random.sample(ids, k=k)
        # kurzer Cue
        net.stimulate(cue, drive=1.0)
        for _ in range(3):
            net.step()
        # freie Rekurrenz
        for _ in range(12):
            net.step()
        # letzter Schritt: aktive IDs lesen
        active = [i for i, n in enumerate(net.neurons) if n.fired]
        return active

    out1 = cue_and_complete(A1.active_ids, 0.4)
    out2 = cue_and_complete(A2.active_ids, 0.4)
    out3 = cue_and_complete(A3.active_ids, 0.4)

    print("Pattern-Completion (Jaccard):",
          f"A1={jaccard(A1.active_ids, out1):.3f}",
          f"A2={jaccard(A2.active_ids, out2):.3f}",
          f"A3={jaccard(A3.active_ids, out3):.3f}")

if __name__ == "__main__":
    run_demo()
