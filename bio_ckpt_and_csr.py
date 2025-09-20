from __future__ import annotations
# ------------------------------------------------------------
# Add-on: JSON-Checkpointing + CSR-Synapsen-Backend
# für das Hippocampus-Stack & den Book-Adapter (Plus/CLI).
# Nur Standardbibliothek, Python 3.12
# ------------------------------------------------------------

import json
import time
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Iterable

# --- Importiere deine bestehenden Module (ggf. Dateinamen anpassen)
import bio_hippocampal_snn_ctx_theta_cmp_feedback_ctxlearn_hardgate as hippo
# Optional: Adapter-Module, falls vorhanden (weich behandeln)
try:
    import book_ingestion_vsa_adapter_plus as adapter_plus
except Exception:
    adapter_plus = None

# ============================================================
# 1) JSON-Checkpointing
# ============================================================

# Hilfen: einfache (de-)serialisierbare Strukturen
def _neuron_to_dict(n: hippo.Neuron) -> Dict[str, Any]:
    return {
        "kind": n.kind.name,
        "threshold": n.threshold,
        "leak": n.leak,
        "refractory": n.refractory,
        "ref_count": n.ref_count,
        "fired": n.fired,
        "v": n.v,
        "energy": n.energy,
        "energy_max": n.energy_max,
        "fire_count": n.fire_count,
        "win_count": n.win_count,
        "target_rate": n.target_rate,
        "dopa_trace": n.dopa_trace,
        "ctx_gate": n.ctx_gate,
    }

def _neuron_from_dict(d: Dict[str, Any]) -> hippo.Neuron:
    k = hippo.CellType[d["kind"]]
    n = hippo.Neuron(
        kind=k, threshold=d["threshold"], leak=d["leak"], refractory=d["refractory"],
        ref_count=d["ref_count"], fired=d["fired"], v=d["v"],
        energy=d["energy"], energy_max=d["energy_max"], fire_count=d["fire_count"],
        win_count=d["win_count"], target_rate=d["target_rate"], dopa_trace=d["dopa_trace"],
        ctx_gate=d.get("ctx_gate", 1.0)
    )
    return n

def _syn_to_dict(s: hippo.Synapse) -> Dict[str, Any]:
    return {
        "pre": s.pre, "post": s.post, "w": s.w, "inhibitory": s.inhibitory,
        "pre_trace": s.pre_trace, "post_trace": s.post_trace, "elig": s.elig
    }

def _syn_from_dict(d: Dict[str, Any]) -> hippo.Synapse:
    return hippo.Synapse(
        pre=d["pre"], post=d["post"], w=d["w"], inhibitory=d["inhibitory"],
        pre_trace=d["pre_trace"], post_trace=d["post_trace"], elig=d["elig"]
    )

def _layer_to_dict(L: hippo.Layer) -> Dict[str, Any]:
    return {
        "exc": [_neuron_to_dict(n) for n in L.exc],
        "inh": [_neuron_to_dict(n) for n in L.inh],
        "syn": [_syn_to_dict(s) for s in L.syn],
        "out_adj": {str(k): v[:] for k, v in L.out_adj.items()},
        "in_adj": {str(k): v[:] for k, v in L.in_adj.items()},
        "sleeping": L.sleeping,
        "feedback_plasticity_gate": L.feedback_plasticity_gate,
        "growth_enabled": L.growth_enabled,
        "steps": getattr(L, "_steps", 0),
        "grown_count": getattr(L, "grown_count", 0),
    }

def _layer_from_dict(h: hippo.Hyper, osc: hippo.Oscillator, d: Dict[str, Any]) -> hippo.Layer:
    # Basiskonstruktor aufrufen, dann überschreiben
    L = hippo.Layer(n_exc=len(d["exc"]), n_inh=len(d["inh"]), h=h, osc=osc, seed=None)
    # ersetze Neuronen
    L.exc = [_neuron_from_dict(x) for x in d["exc"]]
    L.inh = [_neuron_from_dict(x) for x in d["inh"]]
    # Synapsen/Adjazenz
    L.syn = [_syn_from_dict(x) for x in d["syn"]]
    L.out_adj = {int(k): v[:] for k, v in d["out_adj"].items()}
    L.in_adj  = {int(k): v[:] for k, v in d["in_adj"].items()}
    L.sleeping = d["sleeping"]
    L.feedback_plasticity_gate = d["feedback_plasticity_gate"]
    L.growth_enabled = d["growth_enabled"]
    L._steps = d.get("steps", 0)
    L.grown_count = d.get("grown_count", 0)
    return L

def _cmp_neuron_to_dict(c: hippo.ComparatorNeuron) -> Dict[str, Any]:
    return {
        "threshold": c.threshold, "leak": c.leak, "refractory": c.refractory,
        "v": c.v, "ref_count": c.ref_count, "fired": c.fired
    }

def _cmp_neuron_from_dict(d: Dict[str, Any]) -> hippo.ComparatorNeuron:
    c = hippo.ComparatorNeuron(d["threshold"], d["leak"], d["refractory"])
    c.v = d["v"]; c.ref_count = d["ref_count"]; c.fired = d["fired"]
    return c

def save_hippo(system: hippo.HippocampalSystem, path: str) -> None:
    """
    Speichere komplettes Hippocampus-System als JSON.
    """
    h = system.h
    osc = system.osc
    payload = {
        "hyper": asdict(h),
        "osc": {"theta_period": osc.theta_period, "gamma_period": osc.gamma_period, "t": osc.t},
        "DG": _layer_to_dict(system.DG),
        "CA3": _layer_to_dict(system.CA3),
        "CA1": _layer_to_dict(system.CA1),
        "Ctx": {
            "cells": [_cmp_neuron_to_dict(c) for c in system.Ctx.cells],
            "last_fired_mask": system.Ctx.last_fired_mask[:],
        },
        "cmp_pos": [_cmp_neuron_to_dict(c) for c in system.CA1_cmp_pos],
        "cmp_neg": [_cmp_neuron_to_dict(c) for c in system.CA1_cmp_neg],
        "ff_dg_ca3": [_syn_to_dict(s) for s in system.ff_dg_ca3],
        "ff_dg_ca1": [_syn_to_dict(s) for s in system.ff_dg_ca1],
        "ff_ca3_ca1": [_syn_to_dict(s) for s in system.ff_ca3_ca1],
        "ctxW_ca3": system.ctxW_ca3,
        "ctxW_ca1": system.ctxW_ca1,
        "engrams": [{"ids": e.ids, "tag": e.tag, "saliency": e.saliency, "ts": e.ts} for e in system.engrams],
        "sleeping": system.sleeping,
        "ca1_last_mismatch": system.ca1_last_mismatch[:],
        "replay_temp": system.replay_temp,
        "hard_gate_countdown": system.hard_gate_countdown,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_hippo(path: str) -> hippo.HippocampalSystem:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    h = hippo.Hyper(**d["hyper"])
    osc = hippo.Oscillator(d["osc"]["theta_period"], d["osc"]["gamma_period"], t=d["osc"]["t"])
    sys = hippo.HippocampalSystem(h, seed=None)
    sys.osc = osc

    # Layers
    sys.DG  = _layer_from_dict(h, osc, d["DG"])
    sys.CA3 = _layer_from_dict(h, osc, d["CA3"])
    sys.CA1 = _layer_from_dict(h, osc, d["CA1"])

    # Context
    sys.Ctx.cells = [_cmp_neuron_from_dict(x) for x in d["Ctx"]["cells"]]
    sys.Ctx.last_fired_mask = d["Ctx"]["last_fired_mask"]

    # Comparator pop
    sys.CA1_cmp_pos = [_cmp_neuron_from_dict(x) for x in d["cmp_pos"]]
    sys.CA1_cmp_neg = [_cmp_neuron_from_dict(x) for x in d["cmp_neg"]]

    # FF syns
    sys.ff_dg_ca3  = [_syn_from_dict(x) for x in d["ff_dg_ca3"]]
    sys.ff_dg_ca1  = [_syn_from_dict(x) for x in d["ff_dg_ca1"]]
    sys.ff_ca3_ca1 = [_syn_from_dict(x) for x in d["ff_ca3_ca1"]]

    # Context-gating matrices
    sys.ctxW_ca3 = d["ctxW_ca3"]
    sys.ctxW_ca1 = d["ctxW_ca1"]

    # Engrams
    sys.engrams = [hippo.Assembly(ids=e["ids"], tag=e["tag"], saliency=e["saliency"], ts=e["ts"]) for e in d["engrams"]]

    sys.sleeping = d["sleeping"]
    sys.ca1_last_mismatch = d["ca1_last_mismatch"]
    sys.replay_temp = d["replay_temp"]
    sys.hard_gate_countdown = d["hard_gate_countdown"]
    return sys

# --- Optional: Adapter-Checkpointing (nur, wenn Adapter-Modul importiert) ---
def save_adapter(adapter: "adapter_plus.BookAdapter", path: str) -> None:
    if adapter_plus is None:
        raise RuntimeError("Adapter-Modul nicht verfügbar.")
    vsa = adapter.vsa
    rl = adapter.role_learner
    cortex = adapter.cortex
    proj = adapter.projector
    seq = adapter.seqctl

    payload = {
        "hippo": None,  # separat speichern (eigene Datei), hier nur Adapterzustände
        "vsa": {
            "dim": vsa.dim, "rng_seed": vsa.rng_seed,
            "lexicon": vsa.lexicon,
            "roles": vsa.roles,
        },
        "role_learner": {
            "lr": rl.lr, "decay": rl.decay, "pos_bins": rl.pos_bins,
            "ctx_role": rl.ctx_role, "pos_role": rl.pos_role,
        },
        "cortex": {
            "store": cortex.store,
            "relations": cortex.relations,
            "negations": cortex.negations,
            "uses": cortex.uses,
            "ts": cortex.ts,
        },
        "projector": {
            "dg_size": proj.dg_size, "ctx_size": proj.ctx_size,
            "k_active": proj.k_active, "rng_seed": proj.rng_seed,
            "perm_dg": proj.perm_dg, "perm_ctx": proj.perm_ctx,
        },
        "seq": {"present": True},
        "adapter_hparams": {
            "compress_every_paras": adapter.compress_every_paras,
            "compress_keep_top": adapter.compress_keep_top,
            "compress_archive_tag": adapter.compress_archive_tag,
            "mismatch_trigger": adapter.mismatch_trigger,
            "coh_low_trigger": adapter.coh_low_trigger,
            "target_replay_rounds": adapter.target_replay_rounds,
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_adapter(system: hippo.HippocampalSystem, path: str) -> "adapter_plus.BookAdapter":
    if adapter_plus is None:
        raise RuntimeError("Adapter-Modul nicht verfügbar.")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    vsa = adapter_plus.VSA(dim=d["vsa"]["dim"], rng_seed=d["vsa"]["rng_seed"])
    vsa.lexicon = {k: v[:] for k, v in d["vsa"]["lexicon"].items()}
    vsa.roles = {k: v[:] for k, v in d["vsa"]["roles"].items()}

    rl = adapter_plus.RoleLearner(lr=d["role_learner"]["lr"], decay=d["role_learner"]["decay"], pos_bins=d["role_learner"]["pos_bins"])
    rl.ctx_role = {tuple(map(int, k.strip("()").split(", "))): v for k, v in d["role_learner"]["ctx_role"].items()} if isinstance(d["role_learner"]["ctx_role"], dict) else d["role_learner"]["ctx_role"]
    # Falls JSON Keys bereits Tupel sind (neuere Dumps)
    if isinstance(rl.ctx_role, dict):
        # Nichts weiter
        pass
    rl.pos_role = {int(k): v for k, v in d["role_learner"]["pos_role"].items()}

    cortex = adapter_plus.CortexMemory(vsa)
    cortex.store = [(t, v[:]) for (t, v) in d["cortex"]["store"]]
    cortex.relations = {tuple(k.split("|")) if isinstance(k, str) else tuple(k): v for k, v in d["cortex"]["relations"].items()}
    cortex.negations = {tuple(k.split("|")) if isinstance(k, str) else tuple(k): v for k, v in d["cortex"]["negations"].items()}
    cortex.uses = {k: int(v) for k, v in d["cortex"]["uses"].items()}
    cortex.ts = {k: float(v) for k, v in d["cortex"]["ts"].items()}

    proj = adapter_plus.SparseProjector(dg_size=d["projector"]["dg_size"], ctx_size=d["projector"]["ctx_size"],
                                        k_active=d["projector"]["k_active"], rng_seed=d["projector"]["rng_seed"])
    proj.perm_dg = d["projector"]["perm_dg"][:]
    proj.perm_ctx = d["projector"]["perm_ctx"][:]

    adapter = adapter_plus.BookAdapter(system=system, vsa=vsa, role_learner=rl, projector=proj)
    # Hparams
    ap = d["adapter_hparams"]
    adapter.compress_every_paras = ap["compress_every_paras"]
    adapter.compress_keep_top = ap["compress_keep_top"]
    adapter.compress_archive_tag = ap["compress_archive_tag"]
    adapter.mismatch_trigger = ap["mismatch_trigger"]
    adapter.coh_low_trigger = ap["coh_low_trigger"]
    adapter.target_replay_rounds = ap["target_replay_rounds"]
    # Cortex/SeqCtl setzen
    adapter.cortex = cortex
    adapter.seqctl = adapter_plus.SequenceController(adapter.vsa, adapter.cortex)
    return adapter

# ============================================================
# 2) CSR-Backend
# ============================================================

@dataclass
class CSR:
    """
    Minimaler CSR-Container für gerichtete gewichtete Kanten.
    data:   Gewicht (float)
    indices: Zielindex (int)
    indptr:  Zeiger in data/indices (Länge: N_rows+1)
    inh_mask: paralleles bool-Array (inhibitorisch?); Länge = len(data)
    Hinweis: "Row" = präsynaptischer Index
    """
    data: List[float]
    indices: List[int]
    indptr: List[int]
    inh_mask: List[bool]

    @classmethod
    def from_synapses(cls, syn: List[hippo.Synapse], n_rows: int) -> "CSR":
        # Sammle ausgehende Kanten pro pre
        rows: List[List[Tuple[int, float, bool]]] = [[] for _ in range(n_rows)]
        for s in syn:
            if s.pre < 0:
                continue
            rows[s.pre].append((s.post, s.w, s.inhibitory))
        data: List[float] = []
        indices: List[int] = []
        inh_mask: List[bool] = []
        indptr: List[int] = [0]
        for r in rows:
            # stabile Reihenfolge (optional)
            r.sort(key=lambda x: x[0])
            for post, w, inh in r:
                data.append(w); indices.append(post); inh_mask.append(inh)
            indptr.append(len(data))
        return cls(data=data, indices=indices, indptr=indptr, inh_mask=inh_mask)

    def to_synapses(self) -> List[hippo.Synapse]:
        syn: List[hippo.Synapse] = []
        for pre in range(len(self.indptr)-1):
            start, end = self.indptr[pre], self.indptr[pre+1]
            for k in range(start, end):
                post = self.indices[k]
                w = self.data[k]
                inh = self.inh_mask[k]
                syn.append(hippo.Synapse(pre=pre, post=post, w=w, inhibitory=inh))
        return syn

    def forward_add(self, fired_mask: List[bool], out_len: int, acc: Optional[List[float]] = None) -> List[float]:
        """
        akkumuliert Beiträge data[k] auf acc[post], für alle pre mit fired_mask[pre]==True
        """
        if acc is None:
            acc = [0.0] * out_len
        N = len(self.indptr) - 1
        for pre in range(N):
            if not fired_mask[pre]:
                continue
            start, end = self.indptr[pre], self.indptr[pre+1]
            # addiere alle Ausgänge
            for k in range(start, end):
                acc[self.indices[k]] += self.data[k]
        return acc

class CSRLayer(hippo.Layer):
    """
    Drop-in-Ersatz für Layer mit CSR-basierten rekurrenten Synapsen.
    Feedforward bleibt im Hippocampus-System (separate Listen).
    """
    def __init__(self, n_exc: int, n_inh: int, h: hippo.Hyper, osc: hippo.Oscillator, seed: Optional[int] = None):
        super().__init__(n_exc, n_inh, h, osc, seed)
        # ersetze rekurrente Synapsen durch CSR
        self._rebuild_csr()

    def _rebuild_csr(self) -> None:
        # Wir nehmen ALLE Synapsen (exc+inh) die in self.syn liegen; "Row" ist der präindex (exc+inh als gemeinsame Indizierung)
        n_rows = len(self.exc) + len(self.inh)
        self._csr = CSR.from_synapses(self.syn, n_rows)

    def step(self) -> Tuple[List[int], List[int]]:
        """
        Überschreibt die rekurrente Vorwärtsausbreitung via CSR; Plastizität bleibt in Elternklasse,
        aber wir ersetzen nur _collect_currents durch CSR-Fassung.
        """
        nE, nI = len(self.exc), len(self.inh)
        # Spike-Flag zurücksetzen
        for n in self.exc: n.reset_spike()
        for n in self.inh: n.reset_spike()

        # fired_mask aus Vorzustand (letzter Schritt), ABER wir brauchen sie frisch:
        fired_mask = [False] * (nE + nI)
        # Wir nutzen die Membranpotenziale erst NACH Eingangssummation, deshalb hier 2-Phasen:
        # 1) Erst rekurrente Ströme basierend auf "fired" Flags aus letztem Schritt zu addieren ist falsch.
        #    Korrekt: Wir müssen "fired" aus der aktuellen Schrittentscheidung nehmen; dazu iterieren wir:
        #    a) Currents aus CSR basierend auf "fired" des VORIGEN Schrittes? – Nein.
        #    b) Lösung: Wir sammeln Currents aus CSR basierend auf "fired" aus diesem Schritt ist zirkulär.
        # -> Vereinfachung: Wir verwenden das gleiche Timing wie die ursprüngliche Implementation:
        #    Dort wurde _collect_currents() VOR step() aufgerufen und hat "fired" vom Vor-Schritt genutzt.
        #    Um Konsistenz zu halten, speichern wir bei jedem step() das fired_mask_prev.
        fired_prev = getattr(self, "_fired_prev_mask", [False] * (nE + nI))
        currents = self._csr.forward_add(fired_prev, out_len=(nE + nI), acc=[0.0]*(nE + nI))

        fired_exc: List[int] = []
        fired_inh: List[int] = []

        # Neuron-Updates (mit bereits akkumulierten Strömen)
        for i, n in enumerate(self.exc):
            if n.step(currents[i], self.h, sleeping=self.sleeping):
                fired_exc.append(i)
        for i, n in enumerate(self.inh):
            if n.step(currents[nE + i], self.h, sleeping=self.sleeping):
                fired_inh.append(nE + i)

        # Plastizität + Wachstum/Skalierung wie gewohnt:
        self._plasticity_update(fired_exc, fired_inh)
        if not self.sleeping and len(fired_exc) >= 2 and self.growth_enabled:
            self._growth_rule_exc_exc(fired_exc)
            # CSR neu aufbauen, da sich Struktur geändert haben kann
            self._rebuild_csr()

        self._steps = getattr(self, "_steps", 0) + 1
        if self._steps % self.h.scaling_interval == 0:
            self._synaptic_scaling()
            self._rebuild_csr()
        # fired_mask für nächsten Schritt puffern
        fired_mask_now = [False]*(nE+nI)
        for i in fired_exc: fired_mask_now[i] = True
        for i in fired_inh: fired_mask_now[i] = True
        self._fired_prev_mask = fired_mask_now
        return fired_exc, fired_inh

# ------- Hilfen: Migration eines Hippocampus-Systems auf CSRLayer -------

def to_csr_system(system: hippo.HippocampalSystem) -> hippo.HippocampalSystem:
    """
    Erzeuge ein neues HippocampusSystem mit CSR-basierten Layern.
    Alle Zustände (Neuronen, Synapsen, Gates) werden übernommen.
    """
    h = system.h
    osc = system.osc

    def _clone_to_csr(src: hippo.Layer) -> CSRLayer:
        L = CSRLayer(n_exc=len(src.exc), n_inh=len(src.inh), h=h, osc=osc, seed=None)
        # kopiere Neuronen 1:1
        L.exc = [hippo.Neuron(**_neuron_to_dict(n)) for n in src.exc]  # nutzt gleichen ctor
        for j,n in enumerate(L.exc):
            # rekonstruiere korrekt: __init__ erwartet Positionen namensgleich
            L.exc[j] = _neuron_from_dict(_neuron_to_dict(src.exc[j]))
        L.inh = [_neuron_from_dict(_neuron_to_dict(n)) for n in src.inh]
        # Synapsen kopieren
        L.syn = [_syn_from_dict(_syn_to_dict(s)) for s in src.syn]
        L.out_adj = {k: v[:] for k, v in src.out_adj.items()}
        L.in_adj = {k: v[:] for k, v in src.in_adj.items()}
        L.sleeping = src.sleeping
        L.feedback_plasticity_gate = src.feedback_plasticity_gate
        L.growth_enabled = src.growth_enabled
        L._steps = getattr(src, "_steps", 0)
        L.grown_count = getattr(src, "grown_count", 0)
        # CSR aufbauen
        L._rebuild_csr()
        # fired_prev setzen (leer)
        L._fired_prev_mask = [False] * (len(L.exc) + len(L.inh))
        return L

    new = hippo.HippocampalSystem(h, seed=None)
    new.osc = osc
    new.DG  = _clone_to_csr(system.DG)
    new.CA3 = _clone_to_csr(system.CA3)
    new.CA1 = _clone_to_csr(system.CA1)

    # Kontexte & Comparator
    new.Ctx.cells = [hippo.ComparatorNeuron(**_cmp_neuron_to_dict(c)) for c in system.Ctx.cells]
    for i,c in enumerate(new.Ctx.cells):
        cc = system.Ctx.cells[i]
        c.v = cc.v; c.ref_count = cc.ref_count; c.fired = cc.fired
    new.Ctx.last_fired_mask = system.Ctx.last_fired_mask[:]

    new.CA1_cmp_pos = [hippo.ComparatorNeuron(**_cmp_neuron_to_dict(c)) for c in system.CA1_cmp_pos]
    new.CA1_cmp_neg = [hippo.ComparatorNeuron(**_cmp_neuron_to_dict(c)) for c in system.CA1_cmp_neg]

    # Feedforward, Gating-Matrizen, Engramme, Statusvariablen
    new.ff_dg_ca3  = [_syn_from_dict(_syn_to_dict(s)) for s in system.ff_dg_ca3]
    new.ff_dg_ca1  = [_syn_from_dict(_syn_to_dict(s)) for s in system.ff_dg_ca1]
    new.ff_ca3_ca1 = [_syn_from_dict(_syn_to_dict(s)) for s in system.ff_ca3_ca1]
    new.ctxW_ca3 = [row[:] for row in system.ctxW_ca3]
    new.ctxW_ca1 = [row[:] for row in system.ctxW_ca1]
    new.engrams = [hippo.Assembly(ids=e.ids[:], tag=e.tag, saliency=e.saliency, ts=e.ts) for e in system.engrams]
    new.sleeping = system.sleeping
    new.ca1_last_mismatch = system.ca1_last_mismatch[:]
    new.replay_temp = system.replay_temp
    new.hard_gate_countdown = system.hard_gate_countdown
    return new

# ------------------------------------------------------------
# Convenience-Funktionen (Public API)
# ------------------------------------------------------------

def save_checkpoint_all(system: hippo.HippocampalSystem,
                        ckpt_hippo_path: str,
                        adapter: Optional["adapter_plus.BookAdapter"] = None,
                        ckpt_adapter_path: Optional[str] = None) -> None:
    """
    Speichere System und (optional) Adapter getrennt.
    """
    save_hippo(system, ckpt_hippo_path)
    if adapter is not None:
        if ckpt_adapter_path is None:
            raise ValueError("Pfad für Adapter-Checkpoint fehlt.")
        save_adapter(adapter, ckpt_adapter_path)

def load_checkpoint_all(ckpt_hippo_path: str,
                        ckpt_adapter_path: Optional[str] = None) -> Tuple[hippo.HippocampalSystem, Optional["adapter_plus.BookAdapter"]]:
    """
    Lade System und (optional) Adapter.
    """
    sys = load_hippo(ckpt_hippo_path)
    ad = None
    if ckpt_adapter_path is not None:
        if adapter_plus is None:
            raise RuntimeError("Adapter-Modul nicht verfügbar, aber Adapter-Checkpoint angegeben.")
        ad = load_adapter(sys, ckpt_adapter_path)
    return sys, ad

def migrate_to_csr(ckpt_hippo_path_in: Optional[str] = None,
                   ckpt_hippo_path_out: Optional[str] = None,
                   system: Optional[hippo.HippocampalSystem] = None) -> hippo.HippocampalSystem:
    """
    Migriere ein existierendes System (aus Datei oder RAM) in CSR-Layer und (optional) speichere es.
    """
    if system is None:
        if ckpt_hippo_path_in is None:
            raise ValueError("Entweder 'system' oder 'ckpt_hippo_path_in' angeben.")
        system = load_hippo(ckpt_hippo_path_in)
    csr_sys = to_csr_system(system)
    if ckpt_hippo_path_out:
        save_hippo(csr_sys, ckpt_hippo_path_out)
    return csr_sys

# ============================================================
# 3) Mini-Demo (optional)
# ============================================================

def _demo():
    print(">> Lade Demo-Hippocampus (neu)…")
    h = hippo.Hyper()
    sys0 = hippo.HippocampalSystem(h, seed=777)

    # Kurzes „Lernen“
    ids = list(range(0, 20, 2))
    sys0.present_to_DG(ids)
    for _ in range(10): sys0.step()
    sys0.store_assembly(ids, tag="DEMO", saliency=1.0)
    sys0.reward(0.5)
    sys0.sleep_replay(rounds=1)

    print(">> Speichere Checkpoint…")
    save_hippo(sys0, "demo_hippo.json")

    print(">> Lade Checkpoint…")
    sys1 = load_hippo("demo_hippo.json")

    print(">> Migriere zu CSR…")
    sys2 = to_csr_system(sys1)
    save_hippo(sys2, "demo_hippo_csr.json")
    print("Fertig. Engramme:", len(sys2.engrams))

if __name__ == "__main__":
    _demo()
