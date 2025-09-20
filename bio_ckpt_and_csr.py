from __future__ import annotations
# ============================================================
# JSON-Checkpointing + CSR-Backend (V2)
# - GZip-Unterstützung (.gz)
# - Schema-Versionierung + Migration
# - Teil-Checkpoints (Gates/Engramme/Synapsen/…)
# - Mini-Tests (Doctest + Smoke)
# Nur Standardbibliothek, Python 3.12
# ============================================================

import json
import gzip
import time
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any, Iterable

# -- ggf. anpassen: Modulnamen deines Hippocampus-Stacks und (optional) des Adapters
import bio_hippocampal_snn_ctx_theta_cmp_feedback_ctxlearn_hardgate as hippo
try:
    import book_ingestion_vsa_adapter_plus_cli as adapter_plus
except Exception:
    adapter_plus = None

# ----------------------------- Meta/Schema ------------------------------

SCHEMA_VERSION = "1.0.0"  # diese Datei erzeugt Dumps mit 1.0.0
SCHEMA_TYPE_HIPPO = "hippo_system"
SCHEMA_TYPE_ADAPTER = "adapter_state"

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _open_any(path: str, mode: str):
    """
    Öffnet transparent gzip oder plain.
    mode: 'rt'/'wt' (Text), 'rb'/'wb' (Binär). Wir nutzen Text.
    """
    if path.endswith(".gz"):
        # gzip im Textmodus mit UTF-8
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")

# ------------------------- (De)Serialisierer ----------------------------

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
    return hippo.Neuron(
        kind=k, threshold=d["threshold"], leak=d["leak"], refractory=d["refractory"],
        ref_count=d["ref_count"], fired=d["fired"], v=d["v"],
        energy=d["energy"], energy_max=d["energy_max"],
        fire_count=d["fire_count"], win_count=d["win_count"],
        target_rate=d["target_rate"], dopa_trace=d["dopa_trace"],
        ctx_gate=d.get("ctx_gate", 1.0),
    )

def _syn_to_dict(s: hippo.Synapse) -> Dict[str, Any]:
    return {
        "pre": s.pre, "post": s.post, "w": s.w, "inhibitory": s.inhibitory,
        "pre_trace": getattr(s, "pre_trace", 0.0),
        "post_trace": getattr(s, "post_trace", 0.0),
        "elig": getattr(s, "elig", 0.0),
    }

def _syn_from_dict(d: Dict[str, Any]) -> hippo.Synapse:
    return hippo.Synapse(
        pre=d["pre"], post=d["post"], w=d["w"], inhibitory=d["inhibitory"],
        pre_trace=d.get("pre_trace", 0.0), post_trace=d.get("post_trace", 0.0), elig=d.get("elig", 0.0),
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
    L = hippo.Layer(n_exc=len(d["exc"]), n_inh=len(d["inh"]), h=h, osc=osc, seed=None)
    L.exc = [_neuron_from_dict(x) for x in d["exc"]]
    L.inh = [_neuron_from_dict(x) for x in d["inh"]]
    L.syn = [_syn_from_dict(x) for x in d["syn"]]
    L.out_adj = {int(k): v[:] for k, v in d["out_adj"].items()}
    L.in_adj = {int(k): v[:] for k, v in d["in_adj"].items()}
    L.sleeping = d["sleeping"]
    L.feedback_plasticity_gate = d["feedback_plasticity_gate"]
    L.growth_enabled = d["growth_enabled"]
    L._steps = d.get("steps", 0)
    L.grown_count = d.get("grown_count", 0)
    return L

def _cmp_to_dict(c: hippo.ComparatorNeuron) -> Dict[str, Any]:
    return {"threshold": c.threshold, "leak": c.leak, "refractory": c.refractory,
            "v": c.v, "ref_count": c.ref_count, "fired": c.fired}

def _cmp_from_dict(d: Dict[str, Any]) -> hippo.ComparatorNeuron:
    c = hippo.ComparatorNeuron(d["threshold"], d["leak"], d["refractory"])
    c.v = d["v"]; c.ref_count = d["ref_count"]; c.fired = d["fired"]
    return c

# ------------------------- Hippo Save/Load (voll) -----------------------

def save_hippo(system: hippo.HippocampalSystem, path: str, gzip_force: bool = False) -> None:
    """
    Speichere komplettes Hippocampus-System als JSON (optional gzip).
    >>> import bio_hippocampal_snn_ctx_theta_cmp_feedback_ctxlearn_hardgate as hippo
    >>> h = hippo.Hyper(); s = hippo.HippocampalSystem(h, seed=1)
    >>> save_hippo(s, "tmp_hippo.json")
    """
    meta = {
        "schema_version": SCHEMA_VERSION,
        "schema_type": SCHEMA_TYPE_HIPPO,
        "created": _now_iso(),
    }
    h = system.h
    osc = system.osc
    payload = {
        "_meta": meta,
        "hyper": asdict(h),
        "osc": {"theta_period": osc.theta_period, "gamma_period": osc.gamma_period, "t": osc.t},
        "DG": _layer_to_dict(system.DG),
        "CA3": _layer_to_dict(system.CA3),
        "CA1": _layer_to_dict(system.CA1),
        "Ctx": {
            "cells": [_cmp_to_dict(c) for c in system.Ctx.cells],
            "last_fired_mask": system.Ctx.last_fired_mask[:],
        },
        "cmp_pos": [_cmp_to_dict(c) for c in system.CA1_cmp_pos],
        "cmp_neg": [_cmp_to_dict(c) for c in system.CA1_cmp_neg],
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
    use_gz = gzip_force or path.endswith(".gz")
    with _open_any(path, "wt") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _migrate_hippo_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migriert ältere Dumps auf aktuelles Schema.
    Regeln (Beispiele):
    - Wenn '_meta' fehlt -> füge Standard ein (vor v1.0.0).
    - Fehlt 'ctx_gate' bei Neuronen -> default 1.0.
    """
    if "_meta" not in d:
        d["_meta"] = {"schema_version": "0.0.0", "schema_type": SCHEMA_TYPE_HIPPO, "created": _now_iso()}
    # Neuron-Defaults sichern: werden ohnehin in _neuron_from_dict mit defaults gefüllt
    return d

def load_hippo(path: str) -> hippo.HippocampalSystem:
    """
    Lade komplettes Hippocampus-System (JSON/gz).
    >>> s = load_hippo("tmp_hippo.json")
    >>> isinstance(s, hippo.HippocampalSystem)
    True
    """
    with _open_any(path, "rt") as f:
        d = json.load(f)
    d = _migrate_hippo_payload(d)
    h = hippo.Hyper(**d["hyper"])
    osc = hippo.Oscillator(d["osc"]["theta_period"], d["osc"]["gamma_period"], t=d["osc"]["t"])
    sys = hippo.HippocampalSystem(h, seed=None)
    sys.osc = osc
    sys.DG  = _layer_from_dict(h, osc, d["DG"])
    sys.CA3 = _layer_from_dict(h, osc, d["CA3"])
    sys.CA1 = _layer_from_dict(h, osc, d["CA1"])
    sys.Ctx.cells = [_cmp_from_dict(x) for x in d["Ctx"]["cells"]]
    sys.Ctx.last_fired_mask = d["Ctx"]["last_fired_mask"]
    sys.CA1_cmp_pos = [_cmp_from_dict(x) for x in d["cmp_pos"]]
    sys.CA1_cmp_neg = [_cmp_from_dict(x) for x in d["cmp_neg"]]
    sys.ff_dg_ca3  = [_syn_from_dict(x) for x in d["ff_dg_ca3"]]
    sys.ff_dg_ca1  = [_syn_from_dict(x) for x in d["ff_dg_ca1"]]
    sys.ff_ca3_ca1 = [_syn_from_dict(x) for x in d["ff_ca3_ca1"]]
    sys.ctxW_ca3 = d["ctxW_ca3"]; sys.ctxW_ca1 = d["ctxW_ca1"]
    sys.engrams = [hippo.Assembly(ids=e["ids"], tag=e["tag"], saliency=e["saliency"], ts=e["ts"]) for e in d["engrams"]]
    sys.sleeping = d["sleeping"]; sys.ca1_last_mismatch = d["ca1_last_mismatch"]
    sys.replay_temp = d["replay_temp"]; sys.hard_gate_countdown = d["hard_gate_countdown"]
    return sys

# -------------------- Adapter Save/Load (optional) ----------------------

def save_adapter(adapter: "adapter_plus.BookAdapter", path: str, gzip_force: bool = False) -> None:
    if adapter_plus is None:
        raise RuntimeError("Adapter-Modul nicht verfügbar.")
    vsa = adapter.vsa; rl = adapter.role_learner; cortex = adapter.cortex; proj = adapter.projector
    meta = {"schema_version": SCHEMA_VERSION, "schema_type": SCHEMA_TYPE_ADAPTER, "created": _now_iso()}
    payload = {
        "_meta": meta,
        "vsa": {"dim": vsa.dim, "rng_seed": vsa.rng_seed, "lexicon": vsa.lexicon, "roles": vsa.roles},
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
        "adapter_hparams": {
            "compress_every_paras": adapter.compress_every_paras,
            "compress_keep_top": adapter.compress_keep_top,
            "compress_archive_tag": adapter.compress_archive_tag,
            "mismatch_trigger": adapter.mismatch_trigger,
            "coh_low_trigger": adapter.coh_low_trigger,
            "target_replay_rounds": adapter.target_replay_rounds,
        }
    }
    with _open_any(path, "wt") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _migrate_adapter_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    if "_meta" not in d:
        d["_meta"] = {"schema_version": "0.0.0", "schema_type": SCHEMA_TYPE_ADAPTER, "created": _now_iso()}
    return d

def load_adapter(system: hippo.HippocampalSystem, path: str) -> "adapter_plus.BookAdapter":
    if adapter_plus is None:
        raise RuntimeError("Adapter-Modul nicht verfügbar.")
    with _open_any(path, "rt") as f:
        d = json.load(f)
    d = _migrate_adapter_payload(d)
    vsa = adapter_plus.VSA(dim=d["vsa"]["dim"], rng_seed=d["vsa"]["rng_seed"])
    vsa.lexicon = {k: v[:] for k, v in d["vsa"]["lexicon"].items()}
    vsa.roles = {k: v[:] for k, v in d["vsa"]["roles"].items()}
    rl = adapter_plus.RoleLearner(lr=d["role_learner"]["lr"], decay=d["role_learner"]["decay"], pos_bins=d["role_learner"]["pos_bins"])
    rl.ctx_role = d["role_learner"]["ctx_role"]
    rl.pos_role = {int(k): v for k, v in d["role_learner"]["pos_role"].items()}
    cortex = adapter_plus.CortexMemory(vsa)
    cortex.store = [(t, v[:]) for (t, v) in d["cortex"]["store"]]
    cortex.relations = {tuple(k) if isinstance(k, list) else tuple(k.split("|")) if isinstance(k, str) else tuple(k): v
                        for k, v in d["cortex"]["relations"].items()}
    cortex.negations = {tuple(k) if isinstance(k, list) else tuple(k.split("|")) if isinstance(k, str) else tuple(k): v
                        for k, v in d["cortex"]["negations"].items()}
    cortex.uses = {k: int(v) for k, v in d["cortex"]["uses"].items()}
    cortex.ts = {k: float(v) for k, v in d["cortex"]["ts"].items()}
    proj = adapter_plus.SparseProjector(dg_size=d["projector"]["dg_size"], ctx_size=d["projector"]["ctx_size"],
                                        k_active=d["projector"]["k_active"], rng_seed=d["projector"]["rng_seed"])
    proj.perm_dg = d["projector"]["perm_dg"][:]; proj.perm_ctx = d["projector"]["perm_ctx"][:]
    adapter = adapter_plus.BookAdapter(system=system, vsa=vsa, role_learner=rl, projector=proj)
    # Hyperparameter
    hp = d["adapter_hparams"]
    adapter.compress_every_paras = hp["compress_every_paras"]
    adapter.compress_keep_top = hp["compress_keep_top"]
    adapter.compress_archive_tag = hp["compress_archive_tag"]
    adapter.mismatch_trigger = hp["mismatch_trigger"]
    adapter.coh_low_trigger = hp["coh_low_trigger"]
    adapter.target_replay_rounds = hp["target_replay_rounds"]
    adapter.cortex = cortex
    adapter.seqctl = adapter_plus.SequenceController(adapter.vsa, adapter.cortex)
    return adapter

# ---------------------- Teil-Checkpoints (Hippo) ------------------------

def save_gates(system: hippo.HippocampalSystem, path: str) -> None:
    """
    Speichert NUR Kontext-Gating-Matrizen (CA3/CA1) + Meta.
    """
    payload = {"_meta": {"schema_version": SCHEMA_VERSION, "schema_type": "gates", "created": _now_iso()},
               "ctxW_ca3": system.ctxW_ca3, "ctxW_ca1": system.ctxW_ca1}
    with _open_any(path, "wt") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_gates(system: hippo.HippocampalSystem, path: str) -> None:
    with _open_any(path, "rt") as f:
        d = json.load(f)
    system.ctxW_ca3 = d["ctxW_ca3"]; system.ctxW_ca1 = d["ctxW_ca1"]

def save_engrams(system: hippo.HippocampalSystem, path: str) -> None:
    payload = {"_meta": {"schema_version": SCHEMA_VERSION, "schema_type": "engrams", "created": _now_iso()},
               "engrams": [{"ids": e.ids, "tag": e.tag, "saliency": e.saliency, "ts": e.ts} for e in system.engrams]}
    with _open_any(path, "wt") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_engrams(system: hippo.HippocampalSystem, path: str) -> None:
    with _open_any(path, "rt") as f:
        d = json.load(f)
    system.engrams = [hippo.Assembly(ids=e["ids"], tag=e["tag"], saliency=e["saliency"], ts=e["ts"]) for e in d["engrams"]]

def save_synapses_layers(system: hippo.HippocampalSystem, path: str) -> None:
    """
    Speichert NUR rekurrente Layer-Synapsen (DG/CA3/CA1) + FF.
    """
    payload = {"_meta": {"schema_version": SCHEMA_VERSION, "schema_type": "synapses", "created": _now_iso()},
               "DG": {"syn": [_syn_to_dict(s) for s in system.DG.syn],
                      "out_adj": {str(k): v[:] for k,v in system.DG.out_adj.items()},
                      "in_adj": {str(k): v[:] for k,v in system.DG.in_adj.items()}},
               "CA3": {"syn": [_syn_to_dict(s) for s in system.CA3.syn],
                       "out_adj": {str(k): v[:] for k,v in system.CA3.out_adj.items()},
                       "in_adj": {str(k): v[:] for k,v in system.CA3.in_adj.items()}},
               "CA1": {"syn": [_syn_to_dict(s) for s in system.CA1.syn],
                       "out_adj": {str(k): v[:] for k,v in system.CA1.out_adj.items()},
                       "in_adj": {str(k): v[:] for k,v in system.CA1.in_adj.items()}},
               "ff_dg_ca3": [_syn_to_dict(s) for s in system.ff_dg_ca3],
               "ff_dg_ca1": [_syn_to_dict(s) for s in system.ff_dg_ca1],
               "ff_ca3_ca1": [_syn_to_dict(s) for s in system.ff_ca3_ca1] }
    with _open_any(path, "wt") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_synapses_layers(system: hippo.HippocampalSystem, path: str) -> None:
    with _open_any(path, "rt") as f:
        d = json.load(f)
    for name in ("DG","CA3","CA1"):
        layer = getattr(system, name)
        sub = d[name]
        layer.syn = [_syn_from_dict(x) for x in sub["syn"]]
        layer.out_adj = {int(k): v[:] for k, v in sub["out_adj"].items()}
        layer.in_adj  = {int(k): v[:] for k, v in sub["in_adj"].items()}
    system.ff_dg_ca3  = [_syn_from_dict(x) for x in d["ff_dg_ca3"]]
    system.ff_dg_ca1  = [_syn_from_dict(x) for x in d["ff_dg_ca1"]]
    system.ff_ca3_ca1 = [_syn_from_dict(x) for x in d["ff_ca3_ca1"]]

# ------------------------------- CSR ------------------------------------

@dataclass
class CSR:
    data: List[float]
    indices: List[int]
    indptr: List[int]
    inh_mask: List[bool]

    @classmethod
    def from_synapses(cls, syn: List[hippo.Synapse], n_rows: int) -> "CSR":
        rows: List[List[Tuple[int, float, bool]]] = [[] for _ in range(n_rows)]
        for s in syn:
            if s.pre < 0: continue
            rows[s.pre].append((s.post, s.w, s.inhibitory))
        data: List[float] = []; indices: List[int] = []; inh_mask: List[bool] = []; indptr: List[int] = [0]
        for r in rows:
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
                syn.append(hippo.Synapse(pre=pre, post=self.indices[k], w=self.data[k], inhibitory=self.inh_mask[k]))
        return syn

    def forward_add(self, fired_mask: List[bool], out_len: int, acc: Optional[List[float]] = None) -> List[float]:
        if acc is None:
            acc = [0.0] * out_len
        N = len(self.indptr) - 1
        for pre in range(N):
            if not fired_mask[pre]:
                continue
            start, end = self.indptr[pre], self.indptr[pre+1]
            for k in range(start, end):
                acc[self.indices[k]] += self.data[k]
        return acc

class CSRLayer(hippo.Layer):
    """
    Drop-in Ersatz mit CSR für die rekurrente Vorwärtsausbreitung.
    """
    def __init__(self, n_exc: int, n_inh: int, h: hippo.Hyper, osc: hippo.Oscillator, seed: Optional[int] = None):
        super().__init__(n_exc, n_inh, h, osc, seed)
        self._rebuild_csr()

    def _rebuild_csr(self) -> None:
        n_rows = len(self.exc) + len(self.inh)
        self._csr = CSR.from_synapses(self.syn, n_rows)

    def step(self) -> Tuple[List[int], List[int]]:
        nE, nI = len(self.exc), len(self.inh)
        for n in self.exc: n.reset_spike()
        for n in self.inh: n.reset_spike()
        fired_prev = getattr(self, "_fired_prev_mask", [False] * (nE + nI))
        currents = self._csr.forward_add(fired_prev, out_len=(nE + nI), acc=[0.0]*(nE + nI))
        fired_exc: List[int] = []; fired_inh: List[int] = []
        for i, n in enumerate(self.exc):
            if n.step(currents[i], self.h, sleeping=self.sleeping):
                fired_exc.append(i)
        for i, n in enumerate(self.inh):
            if n.step(currents[nE + i], self.h, sleeping=self.sleeping):
                fired_inh.append(nE + i)
        self._plasticity_update(fired_exc, fired_inh)
        if not self.sleeping and len(fired_exc) >= 2 and self.growth_enabled:
            self._growth_rule_exc_exc(fired_exc)
            self._rebuild_csr()
        self._steps = getattr(self, "_steps", 0) + 1
        if self._steps % self.h.scaling_interval == 0:
            self._synaptic_scaling()
            self._rebuild_csr()
        fired_now = [False]*(nE+nI)
        for i in fired_exc: fired_now[i] = True
        for i in fired_inh: fired_now[i] = True
        self._fired_prev_mask = fired_now
        return fired_exc, fired_inh

def to_csr_system(system: hippo.HippocampalSystem) -> hippo.HippocampalSystem:
    """
    Migriert ein System auf CSRLayer (Zustände bleiben erhalten).
    """
    h = system.h; osc = system.osc
    def _clone(src: hippo.Layer) -> CSRLayer:
        L = CSRLayer(n_exc=len(src.exc), n_inh=len(src.inh), h=h, osc=osc, seed=None)
        L.exc = [_neuron_from_dict(_neuron_to_dict(n)) for n in src.exc]
        L.inh = [_neuron_from_dict(_neuron_to_dict(n)) for n in src.inh]
        L.syn = [_syn_from_dict(_syn_to_dict(s)) for s in src.syn]
        L.out_adj = {k: v[:] for k, v in src.out_adj.items()}
        L.in_adj  = {k: v[:] for k, v in src.in_adj.items()}
        L.sleeping = src.sleeping
        L.feedback_plasticity_gate = src.feedback_plasticity_gate
        L.growth_enabled = src.growth_enabled
        L._steps = getattr(src, "_steps", 0)
        L.grown_count = getattr(src, "grown_count", 0)
        L._rebuild_csr()
        L._fired_prev_mask = [False] * (len(L.exc) + len(L.inh))
        return L
    new = hippo.HippocampalSystem(h, seed=None)
    new.osc = osc
    new.DG  = _clone(system.DG)
    new.CA3 = _clone(system.CA3)
    new.CA1 = _clone(system.CA1)
    new.Ctx.cells = [_cmp_from_dict(_cmp_to_dict(c)) for c in system.Ctx.cells]
    new.Ctx.last_fired_mask = system.Ctx.last_fired_mask[:]
    new.CA1_cmp_pos = [_cmp_from_dict(_cmp_to_dict(c)) for c in system.CA1_cmp_pos]
    new.CA1_cmp_neg = [_cmp_from_dict(_cmp_to_dict(c)) for c in system.CA1_cmp_neg]
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

# -------------------------- Convenience-API -----------------------------

def save_checkpoint_all(system: hippo.HippocampalSystem,
                        ckpt_hippo_path: str,
                        adapter: Optional["adapter_plus.BookAdapter"] = None,
                        ckpt_adapter_path: Optional[str] = None) -> None:
    save_hippo(system, ckpt_hippo_path)
    if adapter is not None:
        if ckpt_adapter_path is None:
            raise ValueError("Pfad für Adapter-Checkpoint fehlt.")
        save_adapter(adapter, ckpt_adapter_path)

def load_checkpoint_all(ckpt_hippo_path: str,
                        ckpt_adapter_path: Optional[str] = None) -> Tuple[hippo.HippocampalSystem, Optional["adapter_plus.BookAdapter"]]:
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
    if system is None:
        if ckpt_hippo_path_in is None:
            raise ValueError("Entweder 'system' oder 'ckpt_hippo_path_in' angeben.")
        system = load_hippo(ckpt_hippo_path_in)
    csr_sys = to_csr_system(system)
    if ckpt_hippo_path_out:
        save_hippo(csr_sys, ckpt_hippo_path_out)
    return csr_sys

# ------------------------------- Tests ----------------------------------

TEST_DOC = r"""
Doctest: Save/Load Hippo minimal
>>> import bio_hippocampal_snn_ctx_theta_cmp_feedback_ctxlearn_hardgate as hippo
>>> h = hippo.Hyper(); s = hippo.HippocampalSystem(h, seed=123)
>>> save_hippo(s, "ck_hippo.json")
>>> s2 = load_hippo("ck_hippo.json")
>>> isinstance(s2, hippo.HippocampalSystem)
True

Doctest: Gates/Engrams Teil-Checkpoints
>>> save_gates(s2, "ck_gates.json")
>>> save_engrams(s2, "ck_engrams.json")
>>> load_gates(s2, "ck_gates.json")  # smoke
>>> load_engrams(s2, "ck_engrams.json")  # smoke
>>> True
True

Doctest: Synapsen partial
>>> save_synapses_layers(s2, "ck_syn.json")
>>> load_synapses_layers(s2, "ck_syn.json")
>>> True
True
"""

def _run_doctests() -> None:
    import doctest
    result = doctest.testmod(report=True, extraglobs={})
    print(f"[Doctest] failed={result.failed} attempted={result.attempted}")

def _smoke_demo() -> None:
    print("[Smoke] Erzeuge System, speichere/lad…")
    h = hippo.Hyper(); sys0 = hippo.HippocampalSystem(h, seed=777)
    sys0.present_to_DG([0,2,4,6,8]); [sys0.step() for _ in range(5)]
    sys0.store_assembly([0,2,4,6,8], tag="DEMO", saliency=1.0)
    save_hippo(sys0, "smoke_hippo.json")
    sys1 = load_hippo("smoke_hippo.json")
    print("[Smoke] Engramme nach Load:", len(sys1.engrams))
    print("[Smoke] CSR-Migration…")
    csr = to_csr_system(sys1)
    [csr.step() for _ in range(3)]
    save_hippo(csr, "smoke_hippo_csr.json")
    print("[Smoke] OK.")

if __name__ == "__main__":
    _run_doctests()
    _smoke_demo()
