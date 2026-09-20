# Movie Recommender — Agent Notes

## Architecture (v2)

```
MovieLens-32M (32M ratings, 87,585 movies) + TMDB (1.24M)
  -> polars ETL -> parquet (movies.parquet 43MB, ratings.parquet 178MB)
  -> field-weighted canonical text (signature head line: title — genres — dir — year)
Retrievers (temporal-split, all items = item_idx):
  content  MiniLM 384d embeddings (GPU) -> FAISS HNSW (IP=cosine)
  cf       BPR MF d=128 (GPU) -> FAISS HNSW      [val R@20 .0358 — primary]
  lightgcn d=64 L=3 + mixed hard negs (GPU)      [val R@20 .0245 — kept as artifact, cf row used BPR]
  covis    item-item co-visitation graph (user-weighted, cosine, top-100)  [BEST retriever R@10 .0367]
  sasrec   causal Transformer seq model d=64 maxlen=50 (GPU) [val R@20 .0331]
  popularity  bayes_rating x log(rating_count) top-200
Union ~600 candidates -> cand_meta provenance (score+rank per source)
  -> LightGBM LambdaRank (41 features, graded labels 0/1/2)
  -> mood genre boost -> MMR (adaptive lambda 0.85-0.4*exploration)
  -> deterministic explanations (anchor movie + evidence signals)
Serving: src/recommender.py Recommender.load() -> .recommend(ids, weights, disliked_ids, mood, exploration)
UI: app.py (native Streamlit: segmented_control, containers+border; light editorial theme
  in .streamlit/config.toml; search = rapidfuzz fuzzy + year-aware + per-token coverage;
  technical inspector in sidebar via Recommender.diagnose(); main page hides all ML terms)
```

## Commands

- Pipeline: `python -m src.train --steps etl,embed,cf,lightgcn,covis,sasrec,index,ranker,eval`
- Python: `C:\Users\LENOVO\miniconda3\python.exe` (3.13, torch 2.9.1+cu130, CUDA ok, RTX 5060 8GB)
- App: `streamlit run app.py`  •  Tests: `python -m pytest tests/ -q` (38 tests)
- Heavy steps: embed ~2min, bpr ~12min, lightgcn ~6min, covis ~1min, sasrec ~3min, ranker ~2.5min, eval ~3min

## Honest metrics (temporal test, 4k users, R@10 / NDCG@10)

```
popularity   .028 / .105        union_naive  .028 / .116
content      .003 / .013        ranker       .063 / .216
cf (lgcn)    .023 / .099        ranker+MMR   .064 / .215  <- stack
covis        .037 / .156   <- best single retriever
sasrec       .025 / .094
v1 stack was .059/.191 -> v2 +8% recall, +13% ndcg, coverage .0074->.0125
```

## Experiment loop (autonomous)

- Policy: `RECOMMENDER_AGENT.md`. Runner: `python -m src.experiment snapshot NAME [n_users] [covis=path ranker=path]` / `compare PARENT CAND`. Overrides let a snapshot eval variant artifacts without mutating models/.
- Snapshots in `experiments/<name>/` (summary.json, table.csv, per_user.npz, decision.json).
- Gates: NDCG@10 up, Recall@10 no regress, coverage/ILD no collapse, pop-bias bounded, bootstrap CI excludes 0 -> else REJECT.
- Funnel diag in compare: retrieval/ranking/rerank-limited.

## Experiment log (4k fixed users, seed 0)

| exp | change | NDCG@10 | decision |
|---|---|---|---|
| v2_baseline | parent | .2168 | — |
| v3_covis_feats | +5 covis-per-item features (max/mean/top3/recent/wavg) | .2195 | PROMOTE |
| v4_bigunion | union ~1000 (all K doubled) | .2162 | REJECT |
| v5_covis_heavy | covis_k=300, others shrunk | .2213 | REJECT (CI noise) |
| v6_covis_rated | rating-weighted covis edges | .2190 | REJECT |
| v7_fine_grades | LambdaRank grades 0-4 | .2222 | PROMOTE (current) |
| v8_covis_recency | recency-rank edge weight (floor .5) | .2179 | REJECT (-.0044, CI<0) |
| v9_covis_jaccard | Jaccard edge norm vs cosine | .2170 | REJECT (-.0052, CI<0) |
| v10_poshist | hist-sim/covis feats on positives only | .2201 | REJECT (-.0021, CI<0) |
| v11_multisnap | ranker multi-snapshot cuts .4/.7/1.0 | .2115 | REJECT (-.0107, CI<0) |
| v12_twotower | two-tower retriever (MiniLM+BPR towers, InfoNCE) 3ep | .2169 | REJECT (-.0053, CI<0); tt standalone r@20 .005 = undertrained |
| v12b (aborted) | same, 8ep + hard_neg 16 | — | INVESTIGATE closed: loss plateaued 6.40->6.245 by ep4; retrieval gap too large to close |

- LightGCN neg-mix grid: unif .0239 / pop .0244 / mid .0242 / hard .0246 -> all < BPR .0358 -> REJECT LightGCN, keep BPR.
- Dedup: catalog clean (0 merges; all items keyed by imdb/tmdb).
- Funnel: union cand recall 26.3% -> retrieval-limited AND ranking-limited.

## Gotchas learned

- LightGCN: propagate PER BATCH (once/epoch = stale grads, won't learn).
- Hard negatives: content-hard 35% hurt on this data (false negatives on similar-liked items). Mix used: 30% uniform / 35% popularity / 35% content-hard; SeenIndex = encoded (u*n_items+j) keys + torch.isin.
- SASRec: RIGHT-pad seqs; causal mask alone (pad positions in future, never attended) — pad-mask+causal produces NaN via masked V. Sampled-softmax chunked (einsum M*neg*D OOMs 8GB).
- Covis: user-blocked C=R'R OOMs (293M nnz); item-blocked rows + per-row top-K + cap user history 300 items -> 3.7M nnz.
- Covis edge variants all rejected: rating-weight (v6), recency-rank (v8), Jaccard norm (v9). Binary+cosine+user-damping is the local optimum; edge quality is NOT the bottleneck — candidate recall 26% + ranker discrimination are.
- hist-sim features must keep disliked items (v10 reject): ranker learns negative signal from them; filtering to positives loses it.
- Multi-snapshot ranker training (v11) rejected: prefix-state queries shift feature distribution (covis_score gain x65) away from full-history serving state. If retried, keep profile = full history and only vary label window, or add snapshot-depth feature.
- step_ranker/step_covis take variant params (snapshots=, recency_floor=, norm=, covis_path=, out_name=, tt=); defaults reproduce production artifacts.
- Two-tower (src/twotower.py): item tower concat(MiniLM384,BPR128)->256d, user tower dual-head attention pooling (pos/neg history). InfoNCE: in-batch + shared popularity negs (top-5k pool) + content-hard negs (content_nn), FN-guard resamples negs colliding with user seen set. Serving = dense u@item_emb (no ANN needed at 87k). ranker predict slices X to booster_.num_feature() for 46/49-compat.
- Per-source candidate recall @20/50/100/250/500 now in summary.json["cand_recall"]. v12: union cand recall .263->.280 but tt standalone weak (r@20 .005) — needs more epochs/harder negs.
- eval METHODS gained "tt" row; per-source recall needs `seen` defined in run_eval scope.
- ETL: production_companies kept for company_match feature; canonical text has signature head line for field weighting.
- ItemStore.num() tolerates missing columns (toy tests).
- ranker feature order locked in FEATURES list — app/tests depend on indices.
