"""
discovery_agent.py
========================
Implements an automated feature-discovery search loop for predictive maintenance.

This module orchestrates an iterative search over a set of signal channels and 
feature transformations to find an optimal feature specification that maximizes 
predictive performance (ROC-AUC) against a baseline. It supports both 
deterministic (scripted) and LLM-driven proposers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

try:
    from . import feature_transforms as ft
    from . import featurize_spec as fs
    from .evaluate_features import evaluate_features
except ImportError:
    import feature_transforms as ft
    import featurize_spec as fs
    from evaluate_features import evaluate_features


# --------------------------------------------------------------------------
# agent-facing tools
# --------------------------------------------------------------------------

def describe_channels(flight_iter: fs.FlightIter, downsample: int = 200) -> dict:
    try:
        from .feature_transforms import _sanitize_series
    except ImportError:
        from feature_transforms import _sanitize_series
    rec = next(iter(flight_iter()))
    out = {"sample_filename": rec.filename, "n_channels": len(rec.channels),
           "channels": {}}
    for ch, arr in rec.channels.items():
        arr = _sanitize_series(arr)
        step = max(1, arr.size // downsample)
        out["channels"][ch] = {
            "n_steps": int(arr.size), "mean": round(float(arr.mean()), 3),
            "std": round(float(arr.std()), 3), "min": round(float(arr.min()), 3),
            "max": round(float(arr.max()), 3),
            "preview": [round(float(v), 2) for v in arr[::step][:downsample]]}
    return out


def list_transforms() -> list[dict]:
    return ft.menu()


TOOL_SCHEMAS = [
    {"name": "describe_channels",
     "description": "Channel names, basic stats, and a downsampled preview of one "
                    "sample flight, to decide which transforms suit which channels.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "list_transforms",
     "description": "The menu of available feature transforms with tunable params.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "evaluate_features",
     "description": "Build a feature table from your spec and return its honest "
                    "tail-disjoint holdout AUC. Call repeatedly to search.",
     "input_schema": {"type": "object", "properties": {"features": {
         "type": "array", "items": {"type": "object", "properties": {
             "transform": {"type": "string"},
             "channels": {"type": "array", "items": {"type": "string"}},
             "params": {"type": "object"}},
             "required": ["transform", "channels"]}}}, "required": ["features"]}},
]


# --------------------------------------------------------------------------
# stopping criteria + session
# --------------------------------------------------------------------------

@dataclass
class StopConfig:
    max_iters: int = 20
    patience: int = 5
    min_improvement: float = 0.005
    time_budget_s: float | None = None
    max_features: int | None = None       # absolute cap; if None, resolved from feature_budget
    feature_budget: int | None = 150       # cap = baseline_n_features + this (resolved after baseline)


@dataclass
class DiscoverySession:
    flight_iter: fs.FlightIter
    holdout_fold: object = 0
    stop: StopConfig = field(default_factory=StopConfig)
    log_path: str | Path = "discovery_log.jsonl"
    history: list[dict] = field(default_factory=list)
    best: dict | None = None
    _iters_since_improve: int = 0
    _t_start: float | None = None

    def baseline(self) -> dict:
        sample = next(iter(self.flight_iter()))
        spec = {"spec_id": "baseline_summary_stats",
                "features": [{"transform": "summary_stats",
                              "channels": list(sample.channels.keys())}]}
        res = self._evaluate(spec, note="baseline (bridge-1 parity)")
        self.best = res
        return res

    def submit(self, features: list[dict], note: str = "") -> dict:
        import time
        if self._t_start is None:
            self._t_start = time.time()
        if self.stop.max_features is not None:
            est = _estimate_n_features(features, self.flight_iter)
            if est > self.stop.max_features:
                return {"rejected": True,
                        "reason": f"estimated {est} features > cap {self.stop.max_features}"}
        res = self._evaluate({"features": features}, note=note)
        improved = (res["roc_auc"] is not None and self.best["roc_auc"] is not None  #type: ignore
                    and res["roc_auc"] >= self.best["roc_auc"] + self.stop.min_improvement)  #type: ignore
        if improved:
            self.best = res
            self._iters_since_improve = 0
        else:
            self._iters_since_improve += 1
        return res

    def should_stop(self) -> tuple[bool, str]:
        import time
        n = len([h for h in self.history if h["spec_id"] != "baseline_summary_stats"])
        if n >= self.stop.max_iters:
            return True, f"hit max_iters ({self.stop.max_iters})"
        if self._iters_since_improve >= self.stop.patience:
            return True, f"no improvement in {self.stop.patience} iters"
        if self.stop.time_budget_s and self._t_start and \
                time.time() - self._t_start > self.stop.time_budget_s:
            return True, "time budget exhausted"
        return False, ""

    def _evaluate(self, spec, note):
        res = evaluate_features(spec, self.flight_iter, holdout_fold=self.holdout_fold,  #type: ignore
                                log_path=self.log_path, note=note)
        self.history.append(res)
        return res

    def summary(self) -> dict:
        base = next((h for h in self.history
                     if h["spec_id"] == "baseline_summary_stats"), None)
        lift = (self.best["roc_auc"] - base["roc_auc"]) if (base and self.best) else None
        return {"n_evaluations": len(self.history),
                "baseline_auc": base["roc_auc"] if base else None,
                "best_auc": self.best["roc_auc"] if self.best else None,
                "lift_over_baseline": round(lift, 4) if lift is not None else None,
                "best_spec_id": self.best["spec_id"] if self.best else None}

    def best_spec(self) -> dict | None:
        """Reconstruct the winning canonical spec from the log (for writing
        best_spec.json)."""
        if not self.best:
            return None
        sid = self.best["spec_id"]
        for line in Path(self.log_path).read_text().splitlines():
            e = json.loads(line)
            if e["spec_id"] == sid:
                return {"spec_id": sid, "features": e["spec"]}
        return None


def _estimate_n_features(features, flight_iter) -> int:
    sample = next(iter(flight_iter()))
    total = 0
    for entry in features:
        t = ft.get(entry["transform"])
        params = {**t.defaults, **entry.get("params", {})}
        for ch in entry["channels"]:
            if ch in sample.channels:
                total += len(ft.apply(entry["transform"], sample.channels[ch], params))
    return total


# --------------------------------------------------------------------------
# proposers
# --------------------------------------------------------------------------

Proposer = Callable[["DiscoverySession", dict], "list[dict] | None"]


def _rank_signal_channels(describe: dict, k: int = 4) -> list[str]:
    """Pick the k highest-std channels as candidate signal channels. Cheap,
    dataset-agnostic heuristic so the scripted proposer works on real sensor
    names without hardcoding which sensors carry the fault."""
    chans = describe["channels"]
    return sorted(chans, key=lambda c: chans[c]["std"], reverse=True)[:k]


def scripted_proposer(session: DiscoverySession, context: dict) -> "list[dict] | None":
    """Deterministic stand-in for the LLM. Walks a small curriculum on the
    variance-ranked signal channels. Returns None when exhausted."""
    all_ch = context["channels"]
    sig = context.setdefault("_sig", _rank_signal_channels(context["describe"]))
    script = [
        [{"transform": "summary_stats", "channels": all_ch},
         {"transform": "fft_band_energy", "channels": sig, "params": {"n_bands": 8}}],
        [{"transform": "summary_stats", "channels": all_ch},
         {"transform": "fft_band_energy", "channels": sig, "params": {"n_bands": 8}},
         {"transform": "spectral_shape", "channels": sig}],
        [{"transform": "summary_stats", "channels": all_ch},
         {"transform": "fft_band_energy", "channels": sig, "params": {"n_bands": 8}},
         {"transform": "autocorr_lags", "channels": sig, "params": {"lags": [1, 5, 20, 50]}}],
        [{"transform": "summary_stats", "channels": all_ch},
         {"transform": "crossing_rate", "channels": sig}],   # cheap spectral proxy
    ]
    idx = context.setdefault("_idx", 0)
    if idx >= len(script):
        return None
    context["_idx"] = idx + 1
    return script[idx]


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model reply that may wrap it in prose or fences."""
    import re
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    return json.loads(t[start:end + 1])


def _menu_compact(menu: list[dict]) -> str:
    lines = []
    for e in menu:
        d = e.get("defaults") or {}
        params = (" params=" + json.dumps(d)) if d else ""
        lines.append(f"- {e['transform']}: {e.get('doc', '').strip()}{params}")
    return "\n".join(lines)


def make_llm_proposer(chat_fn: Callable[[str, str], str], max_retries: int = 5,
                      max_features_hint: int | None = None, verbose: bool = False) -> Proposer:
    """Turn a chat function into a discovery proposer.

    chat_fn(system, user) -> model text. The model sees the channels, the
    transform menu, the running AUC history, AND the live feature budget, then
    returns ONE spec as JSON: {"features": [{"transform", "channels", "params"}]}.
    Output {"done": true} to stop. This is the seam where the LLM does real,
    unscripted search -- no correct answer to grade against.

    The proposer self-checks each spec's feature count against the live cap
    (session.stop.max_features) and asks the model to shrink (fewer channels /
    fewer transforms) when it's over, so oversized specs are corrected in-loop
    instead of being silently rejected by the search.
    """
    SYSTEM_BASE = (
        "You are a feature-engineering search agent for an aviation predictive-"
        "maintenance classifier (predicts whether a flight precedes maintenance, "
        "from per-sensor-channel features). Propose ONE feature spec that beats the "
        "current best held-out ROC-AUC. Use transform names exactly as listed; signal "
        "usually lives in higher-variance channels. Each transform emits SEVERAL "
        "features per channel, so applying transforms across all channels explodes the "
        "count fast -- you do NOT have to use every channel. A small spec that beats the "
        "baseline is far more credible than a large one. Think briefly, then output ONLY "
        'a JSON object: {"features": [{"transform": <name>, "channels": [<names>], '
        '"params": {..}}]}. Do not repeat a spec already tried. To stop, output '
        '{"done": true}.')

    def _proposer(session: "DiscoverySession", context: dict) -> "list[dict] | None":
        chans = context["describe"]["channels"]
        ranked = sorted(chans, key=lambda c: chans[c]["std"], reverse=True)
        ch_summary = "\n".join(f"  {c}: std={chans[c]['std']}" for c in ranked)
        menu_txt = _menu_compact(context["menu"])
        base = next((h for h in session.history
                     if h["spec_id"] == "baseline_summary_stats"), None)
        base_n = context.get("baseline_n_features")
        cap = session.stop.max_features
        hist = "\n".join(f"  {h['spec_id']}: AUC={h['roc_auc']} ({h['n_features']} feats)"
                         for h in session.history[-10:]) or "  (none yet)"
        budget_line = ""
        if cap is not None:
            budget_line = (f"FEATURE BUDGET: the baseline is summary_stats on all "
                           f"{len(chans)} channels = {base_n} features. Your spec must total "
                           f"<= {cap} features. To stay under it, either keep summary_stats and "
                           f"add a FEW targeted transforms on only the top-variance channels, OR "
                           f"drop channels. Count before you answer.\n\n")
        user = (
            f"CHANNELS (name: std):\n{ch_summary}\n\n"
            f"TRANSFORMS (features each emits per channel shown in params/doc):\n{menu_txt}\n\n"
            f"{budget_line}"
            f"BASELINE summary_stats AUC = {base['roc_auc'] if base else '?'}.\n"
            f"HISTORY (recent last):\n{hist}\n"
            f"CURRENT BEST AUC = {session.best['roc_auc'] if session.best else '?'}.\n\n"
            "Propose the next spec. JSON only."
        )
        tried_ids = {h["spec_id"] for h in session.history}
        for attempt in range(max_retries + 1):
            try:
                txt = chat_fn(SYSTEM_BASE, user)
            except Exception as e:
                print(f"  [llm proposer] chat error: {e}")
                return None
            if verbose:
                print(f"  [llm raw] {txt[:200]}")
            try:
                obj = _extract_json(txt)
                if isinstance(obj, dict) and obj.get("done") is True:
                    return None
                feats = obj["features"] if isinstance(obj, dict) and "features" in obj else obj
                canon = fs.canonical_spec({"features": feats})
                if canon["spec_id"] in tried_ids:
                    user += "\n\nThat spec was already tried; propose a DIFFERENT one."
                    continue
                if cap is not None:
                    est = _estimate_n_features(feats, session.flight_iter)
                    if est > cap:
                        user += (f"\n\nThat spec is ~{est} features, over the budget of {cap}. "
                                 f"Use FEWER channels or fewer transforms and try again.")
                        continue
                return feats  #type: ignore
            except Exception as e:
                user += f"\n\nYour previous reply was invalid ({e}). Output ONLY the JSON object."
        return None  # couldn't get a valid, in-budget, novel spec within retries

    return _proposer


def run_discovery(flight_iter: fs.FlightIter, proposer: Proposer = scripted_proposer,
                  holdout_fold=0, stop: StopConfig | None = None,
                  log_path: str | Path = "discovery_log.jsonl") -> dict:
    session = DiscoverySession(flight_iter, holdout_fold, stop or StopConfig(), log_path)
    base = session.baseline()
    # Resolve the feature cap relative to the baseline so it's never below it.
    if session.stop.max_features is None and session.stop.feature_budget is not None:
        session.stop.max_features = base["n_features"] + session.stop.feature_budget
        print(f"[discovery] feature cap = {session.stop.max_features} "
              f"(baseline {base['n_features']} + budget {session.stop.feature_budget})")
    desc = describe_channels(flight_iter)
    context = {"channels": list(desc["channels"].keys()), "describe": desc,
               "menu": list_transforms(), "history": session.history,
               "baseline_n_features": base["n_features"],
               "feature_cap": session.stop.max_features}
    print(f"baseline (summary_stats): AUC={base['roc_auc']}  [{base['n_features']} features]")
    while True:
        stop_now, why = session.should_stop()
        if stop_now:
            print(f"stopping: {why}"); break
        features = proposer(session, context)
        if features is None:
            print("stopping: proposer exhausted"); break
        res = session.submit(features, note=f"best so far {session.best['roc_auc']}")  #type: ignore
        if res.get("rejected"):
            print(f"  rejected: {res['reason']}")
            context.setdefault("rejections", []).append(res["reason"])
            continue
        flag = "  <-- new best" if res["spec_id"] == session.best["spec_id"] else ""  #type: ignore
        print(f"  {res['spec_id']}: AUC={res['roc_auc']} "
              f"[{res['n_features']} feats, overlap={res['plane_overlap']}, "
              f"{res['seconds']}s]{flag}")
    summ = session.summary()
    print(f"\nbest: {summ['best_spec_id']}  AUC={summ['best_auc']}  "
          f"(baseline {summ['baseline_auc']}, lift {summ['lift_over_baseline']:+})")
    return {"summary": summ, "best": session.best,
            "best_spec": session.best_spec(), "history": session.history}


if __name__ == "__main__":
    it = fs.synthetic_flight_iter(seed=1) if hasattr(fs, "synthetic_flight_iter") else None  #type: ignore
    if it:
        out = run_discovery(it, stop=StopConfig(max_iters=10, patience=4),
                            log_path="/tmp/discovery_log.jsonl")
        print("\nsummary:", json.dumps(out["summary"], indent=2))