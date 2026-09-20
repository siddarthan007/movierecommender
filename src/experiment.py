"""Autonomous experiment harness: snapshot baselines, compare candidates,
apply promotion gates, emit Recommendation Strength Reports.

Usage:
  python -m src.experiment snapshot <name>     # full eval -> experiments/<name>/
  python -m src.experiment compare <parent> <cand>
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
import torch

from .config import CFG
from .evaluate import run_eval
from .exp_metrics import bootstrap_delta_ci

EXP_DIR = Path("experiments")
FINAL = "ranker_mmr"          # production method compared by gates
STACK_METHODS = ["ranker", "ranker_mmr", "ranker_mmr_div"]

GATES = dict(
    ndcg10_min_delta=0.0,       # must improve
    recall10_max_reg=-0.002,    # tolerable recall regression
    coverage_max_reg=-0.10,     # relative -10% allowed
    ild_max_reg=-0.05,          # absolute ILD drop allowed
    head_exp_max_up=0.05,       # popularity-bias headroom
    ci_must_exclude_zero=True,  # bootstrap significance
)


def _hash(path: Path) -> str:
    if not path.exists():
        return "none"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def snapshot(name: str, cfg=CFG, n_users: int | None = None,
             overrides: dict | None = None) -> dict:
    """Run full eval, persist summary + per-user details + artifact hashes.

    overrides: {"ranker": path, "covis": path} swaps artifacts without
    mutating the production files in models/.
    """
    from .train import _load_stack, _load_sasrec
    from .index import load_index
    from .split import temporal_split
    from .dedup import build_entities, dedupe_recs

    ov = overrides or {}
    out = EXP_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    movies_pd, emb, fac, bias, ratings = _load_stack(cfg)
    train, val, test, t1, t2 = temporal_split(ratings, cfg)
    content_index = load_index(cfg.models_path / "content_hnsw.faiss", cfg)
    cf_index = load_index(cfg.models_path / "cf_hnsw.faiss", cfg)
    ranker = joblib.load(ov.get("ranker", cfg.models_path / "ranker.joblib"))
    pop_items = np.load(cfg.models_path / "pop_items.npy")
    covis_p = Path(ov.get("covis", cfg.models_path / "covis.npz"))
    covis_mat = sp.load_npz(covis_p) if covis_p.exists() else None
    sasrec = _load_sasrec(cfg)
    sasrec_model = sasrec[0] if sasrec else None
    if sasrec_model is not None and torch.cuda.is_available():
        sasrec_model = sasrec_model.to("cuda")
    from .train import _load_tt
    tt = _load_tt(cfg, ov)

    entity_key, dstats = build_entities(movies_pd)
    dedupe_fn = (lambda recs: dedupe_recs(recs, entity_key, dstats["rep_item"]))

    t0 = time.time()
    table, details = run_eval(movies_pd, train, val, test, emb, fac, bias,
                              content_index, cf_index, ranker, pop_items, cfg,
                              covis_mat=covis_mat, sasrec_model=sasrec_model,
                              n_users=n_users, dedupe_fn=dedupe_fn, tt=tt)
    # artifact hashes
    hashes = {p.name: _hash(p) for p in sorted(cfg.models_path.iterdir())}
    summary = {
        "name": name, "ts": time.time(), "eval_s": time.time() - t0,
        "overrides": {k: str(v) for k, v in ov.items()},
        "n_users": details["funnel"]["n_users"],
        "dedup": {k: v for k, v in dstats.items() if k != "entity_key"},
        "funnel": details["funnel"],
        "cand_recall": details.get("cand_recall", {}),
        "table": table, "segments": details["segments"],
        "artifact_hashes": hashes,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(
        out / "per_user.npz",
        **{f"{m}__ndcg10": np.array([u["ndcg@10"] for u in details["per_user"][m]])
           for m in details["per_user"]},
        **{f"{m}__recall10": np.array([u["recall@10"] for u in details["per_user"][m]])
           for m in details["per_user"]},
        **{f"{m}__recall100_proxy": np.zeros(len(details["per_user"].get(FINAL, [])))
           for m in [FINAL]})
    pd.DataFrame(table).T.to_csv(out / "table.csv")
    print(f"[exp] snapshot '{name}' -> {out}  ({time.time()-t0:.0f}s)")
    return summary


def compare(parent_name: str, cand_name: str) -> dict:
    """Apply promotion gates + bootstrap -> report dict."""
    P = json.loads((EXP_DIR / parent_name / "summary.json").read_text())
    C = json.loads((EXP_DIR / cand_name / "summary.json").read_text())
    Pp = np.load(EXP_DIR / parent_name / "per_user.npz")
    Cp = np.load(EXP_DIR / cand_name / "per_user.npz")

    pt, ct = P["table"], C["table"]
    pf, cf_ = P["table"].get(FINAL, {}), C["table"].get(FINAL, {})
    delta, lo, hi = bootstrap_delta_ci(
        Pp[f"{FINAL}__ndcg10"], Cp[f"{FINAL}__ndcg10"])
    dr, lr_, hr = bootstrap_delta_ci(
        Pp[f"{FINAL}__recall10"], Cp[f"{FINAL}__recall10"])

    gates = []
    gates.append(("NDCG@10 improves", delta > GATES["ndcg10_min_delta"]))
    gates.append(("Recall@10 no regress", dr > GATES["recall10_max_reg"]))
    cov_p, cov_c = pf.get("coverage", 0), cf_.get("coverage", 0)
    cov_rel = (cov_c - cov_p) / max(cov_p, 1e-9)
    gates.append((f"coverage (rel {cov_rel:+.1%})", cov_rel > GATES["coverage_max_reg"]))
    ild_d = cf_.get("ild", 0) - pf.get("ild", 0)
    gates.append((f"diversity (ILD {ild_d:+.3f})", ild_d > GATES["ild_max_reg"]))
    he_d = cf_.get("head_exposure", 0) - pf.get("head_exposure", 0)
    gates.append((f"pop-bias (head {he_d:+.3f})", he_d < GATES["head_exp_max_up"]))
    gates.append((f"bootstrap NDCG CI [{lo:+.4f},{hi:+.4f}]",
                  lo > 0 if GATES["ci_must_exclude_zero"] else True))

    passed = all(g[1] for g in gates)
    decision = "PROMOTE" if passed else "REJECT"

    # funnel diagnosis
    cand_r = C["funnel"]["cand_recall_union"]
    final_r = cf_.get("recall@10", 0)
    diag = []
    if cand_r < 0.35:
        diag.append("retrieval-limited (union cand recall low)")
    if final_r < 0.4 * cand_r and cand_r > 0:
        diag.append("ranking-limited (ranker underuses candidates)")
    mmr_nd = ct.get("ranker_mmr", {}).get("ndcg@10", 0)
    rnk_nd = ct.get("ranker", {}).get("ndcg@10", 0)
    if rnk_nd - mmr_nd > 0.01:
        diag.append(f"rerank-limited (MMR cost {-rnk_nd + mmr_nd:+.4f} NDCG)")

    report = {
        "parent": parent_name, "candidate": cand_name,
        "delta_ndcg10": delta, "ci95": [lo, hi],
        "delta_recall10": dr, "ci95_recall": [lr_, hr],
        "gates": [{"gate": g, "pass": p} for g, p in gates],
        "diagnosis": diag or ["no dominant bottleneck flagged"],
        "decision": decision,
        "table_parent": pf, "table_cand": cf_,
        "segments_cand": C["segments"].get(FINAL, {}),
        "segments_parent": P["segments"].get(FINAL, {}),
    }
    (EXP_DIR / cand_name / "decision.json").write_text(json.dumps(report, indent=2))
    return report


def format_report(rep: dict) -> str:
    g = "\n".join(f"  [{'x' if gg['pass'] else ' '}] {gg['gate']}"
                  for gg in rep["gates"])
    seg = "\n".join(f"  {s:7s} n={v['n']:5d}  ndcg@10 {v['ndcg@10']:.4f}  "
                    f"recall@10 {v['recall@10']:.4f}"
                    for s, v in sorted(rep["segments_cand"].items()))
    tp, tc = rep["table_parent"], rep["table_cand"]
    rows = []
    for m in ["ndcg@10", "recall@10", "hitrate@10", "coverage", "ild",
              "novelty", "head_exposure", "personalization", "mrr"]:
        if m in tp or m in tc:
            rows.append(f"  {m:15s} {tp.get(m, float('nan')):.4f} -> "
                        f"{tc.get(m, float('nan')):.4f}")
    return f"""
==================== EXPERIMENT REPORT ====================
parent    : {rep['parent']}
candidate : {rep['candidate']}
decision  : {rep['decision']}
----------------------------------------------------------
dNDCG@10  : {rep['delta_ndcg10']:+.4f}  95% CI [{rep['ci95'][0]:+.4f}, {rep['ci95'][1]:+.4f}]
dRecall@10: {rep['delta_recall10']:+.4f}  95% CI [{rep['ci95_recall'][0]:+.4f}, {rep['ci95_recall'][1]:+.4f}]
------------------- final stack metrics -------------------
{chr(10).join(rows)}
---------------------- gates -----------------------------
{g}
------------------- candidate segments --------------------
{seg}
---------------------- diagnosis --------------------------
{chr(10).join('  - ' + d for d in rep['diagnosis'])}
===========================================================
"""


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "snapshot":
        rest = sys.argv[3:]
        n_u = int(rest[0]) if rest and rest[0].isdigit() else None
        kv = dict(a.split("=", 1) for a in rest if "=" in a)
        ov = {k: Path(v) for k, v in kv.items()
              if k in ("ranker", "covis", "tt_model", "tt_emb")}
        s = snapshot(sys.argv[2], n_users=n_u, overrides=ov or None)
        df = pd.DataFrame(s["table"]).T
        pd.set_option("display.width", 250)
        print(df.round(4).to_string())
    elif cmd == "compare":
        rep = compare(sys.argv[2], sys.argv[3])
        print(format_report(rep))
    else:
        print("usage: python -m src.experiment <snapshot NAME [users] | compare P C>")


if __name__ == "__main__":
    main()
