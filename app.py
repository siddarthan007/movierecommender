"""Movie Discovery — editorial discovery UI + technical inspector sidebar.

Main page helps you choose. Sidebar helps you understand why.
"""
import json
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import CFG
from src.features import FEATURES
from src.recommender import Recommender, MOOD_GENRES

IMG = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER = ("https://upload.wikimedia.org/wikipedia/commons/6/65/"
               "No-Image-Placeholder.svg")

st.set_page_config(page_title="Movie Discovery", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding-top:2.6rem;max-width:1180px}
[data-testid="stImage"] img{border-radius:6px}
h1{font-weight:700;letter-spacing:-.02em}
h2,h3{letter-spacing:-.01em}
section[data-testid="stSidebar"][aria-expanded="true"]{width:400px!important}
section[data-testid="stSidebar"][aria-expanded="true"]>div{width:400px!important}
section[data-testid="stSidebar"] .stCaption{color:#6b6963}
button[kind="tertiary"]{font-size:.82rem;padding-left:.35rem;padding-right:.35rem;
  white-space:nowrap;color:#2f4f7f}
button[kind="tertiary"] p{color:#2f4f7f}
button[kind="tertiary"]:hover{color:#1d3457;text-decoration:underline}
button[kind="tertiary"]:hover p{color:#1d3457}
/* uniform card height: flex column, last element pinned to bottom */
[data-testid="stVerticalBlockBorderWrapper"]>div{
  display:flex;flex-direction:column;height:100%}
[data-testid="stVerticalBlockBorderWrapper"]>div>div:last-child{
  margin-top:auto}
/* clamp card text so long titles never change height */
[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown p{
  overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;
  -webkit-box-orient:vertical}
/* segmented control stays on one line */
div[data-testid="stSegmentedControl"]{flex-wrap:nowrap;min-width:0}
div[data-testid="stSegmentedControl"] label{
  flex:1;min-width:0;padding:.15rem 0}
div[data-testid="stSegmentedControl"] label p{font-size:.7rem}
</style>""", unsafe_allow_html=True)

ACCENT = "#2f4f7f"
MUTED = "#6b6963"


@st.cache_resource
def get_rec() -> Recommender:
    return Recommender.load(CFG)


@st.cache_data(ttl=3600)
def search(q: str) -> pd.DataFrame:
    return get_rec().search(q, limit=8)


@st.cache_data
def eval_summary() -> dict | None:
    p = Path("experiments/v7_fine_grades/summary.json")
    return json.loads(p.read_text()) if p.exists() else None


def poster_url(row) -> str:
    p = row.get("poster_path")
    return f"{IMG}{p}" if isinstance(p, str) and p else PLACEHOLDER


SKIP_GENRES = {"IMAX", "(no genres listed)"}
POPULAR = ["Interstellar", "Arrival", "The Dark Knight", "Dune"]


def disp_title(t: str) -> str:
    m = re.match(r"^(.*),\s*(The|A|An)(\s*\(\d{4}\)\s*)?$", t)
    if m:
        return f"{m.group(2)} {m.group(1)}{m.group(3) or ''}"
    return t


def genre_list(row, n: int = 3) -> list:
    g = row.get("genres")
    if not isinstance(g, (list, np.ndarray)):
        return []
    return [x for x in g if x not in SKIP_GENRES][:n]


def meta_line(row) -> str:
    parts = []
    if pd.notna(row.get("year")):
        parts.append(str(int(row["year"])))
    g = genre_list(row)
    if g:
        parts.append(" · ".join(g))
    return "  ·  ".join(parts)


def explain_text(e: dict) -> str:
    s = f"Because you liked {disp_title(e['anchor'])}."
    if e.get("signals"):
        s += " " + " · ".join(e["signals"]) + "."
    return s


def feedback(key_prefix: str, mid: int):
    with st.container():
        if st.button("Relevant", key=f"{key_prefix}_ok_{mid}",
                     type="tertiary"):
            st.session_state["disliked"].discard(mid)
            st.session_state["feedback"].add(mid)
            st.session_state["dirty"] = True
        if st.button("Not relevant", key=f"{key_prefix}_no_{mid}",
                     type="tertiary"):
            st.session_state["feedback"].discard(mid)
            st.session_state["disliked"].add(mid)
            st.session_state["dirty"] = True


def trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n].rstrip() + "…"


def rec_card(row, e: dict, key_prefix: str):
    with st.container(border=True, height=560):
        st.image(poster_url(row), width="stretch")
        st.markdown(f"**{trunc(disp_title(row['title']), 44)}**")
        st.caption(trunc(meta_line(row), 40))
        if e:
            st.caption(trunc(explain_text(e), 120))
        feedback(key_prefix, int(row["movieId"]))


# ---------- init ----------

try:
    rec = get_rec()
except Exception as e:
    st.error(f"Model artifacts not loaded: {e}")
    st.stop()

MOVIES = rec.a.movies
BY_ID = MOVIES.set_index("movieId")

st.session_state.setdefault("selected", [])   # [{mid, w}]
st.session_state.setdefault("disliked", set())
st.session_state.setdefault("feedback", set())
st.session_state.setdefault("diag", None)
st.session_state.setdefault("dirty", False)

sel_ids = {s["mid"] for s in st.session_state["selected"]}


def row_of(mid: int):
    try:
        return BY_ID.loc[mid]
    except KeyError:
        return None


def run_recommend():
    ids = [s["mid"] for s in st.session_state["selected"]]
    wts = np.array([s["w"] for s in st.session_state["selected"]],
                   dtype=np.float64)
    mood = st.session_state.get("mood")
    with st.spinner("Generating recommendations"):
        st.session_state["diag"] = rec.diagnose(
            ids, weights=wts,
            disliked_ids=list(st.session_state["disliked"]),
            mood=None if mood in (None, "Any") else [mood],
            exploration=st.session_state.get("explore", 0.35), k=25)


# ---------- header ----------

st.caption("CINEMA DISCOVERY")
st.title("Find your next movie.")
st.write("Choose a few films you like and we'll build recommendations "
         "around your taste.")

# ---------- search (fragment: Enter/click reruns only this block) ----------

@st.fragment
def search_block():
    q = st.text_input("Search movies", placeholder="Search by title",
                      key="q", label_visibility="collapsed")
    cols = st.columns([0.55, 1.0, 0.75, 1.4, 0.6, 5])
    cols[0].caption("Try:")
    for i, t in enumerate(POPULAR):
        cols[i + 1].button(t, key=f"pop_{i}", type="tertiary",
                           on_click=lambda t=t: st.session_state.update(q=t))
    if not q.strip():
        return
    hits = search(q)
    if len(hits) == 0:
        st.caption(f"No matches for “{q.strip()}”.")
        return
    for _, h in hits.iterrows():
        mid = int(h["movieId"])
        c1, c2, c3 = st.columns([1, 7, 1])
        c1.image(poster_url(h), width=64)
        yr = "" if re.search(r"\(\d{4}\)\s*$", h["title"]) else \
            (f" ({int(h['year'])})" if pd.notna(h.get("year")) else "")
        c2.markdown(f"**{disp_title(h['title'])}**{yr}")
        sub = []
        if isinstance(h.get("director"), str) and h["director"]:
            sub.append(h["director"])
        sub += genre_list(h, 3)
        if sub:
            c2.caption(" · ".join(sub))
        if mid not in sel_ids and c3.button("Add", key=f"add_{mid}"):
            st.session_state["selected"].append({"mid": mid, "w": 1.0})
            st.session_state["dirty"] = True
            st.rerun()


search_block()

# ---------- taste shelf ----------

selected = [s for s in st.session_state["selected"] if row_of(s["mid"]) is not None]
if selected:
    st.write("")
    st.subheader("Your taste")
    cols = st.columns(4)
    for i, s in enumerate(selected):
        row = row_of(s["mid"])
        mid = s["mid"]
        with cols[i % 4]:
            with st.container(border=True, height=500):
                st.image(poster_url(row), width="stretch")
                st.markdown(f"**{trunc(disp_title(row['title']), 36)}**")
                w = st.segmented_control(
                    "Preference", ["Dislike", "Like", "Love"],
                    default={-0.8: "Dislike", 1.0: "Like", 1.8: "Love"}[s["w"]],
                    key=f"w_{mid}", label_visibility="collapsed")
                wv = {"Dislike": -0.8, "Like": 1.0, "Love": 1.8}[w]
                if wv != s["w"]:
                    s["w"] = wv
                    st.session_state["dirty"] = True
                if st.button("Remove", key=f"rm_{mid}", type="tertiary",
                             width="stretch"):
                    st.session_state["selected"] = [
                        x for x in st.session_state["selected"]
                        if x["mid"] != mid]
                    st.session_state["dirty"] = True
                    st.rerun()

# ---------- controls + action ----------

if selected:
    st.write("")
    st.slider("How adventurous?", 0.0, 1.0, 0.35, 0.05, key="explore",
              help="Left: more like what you already watch. "
                   "Right: more unexpected discoveries.")
    f1, f2 = st.columns(2)
    f1.caption("Familiar")
    f2.markdown("<div style='text-align:right;color:#6b6963;"
                "font-size:.8rem'>Discover</div>", unsafe_allow_html=True)
    moods = ["Any"] + list(MOOD_GENRES.keys())
    st.selectbox("What are you in the mood for?", moods, index=0, key="mood")
    st.write("")
    if st.button("Recommend", type="primary") or st.session_state["dirty"]:
        st.session_state["dirty"] = False
        run_recommend()
else:
    st.write("")
    st.info("Add two or three films you like to get recommendations.")

# ---------- recommendations ----------

diag = st.session_state["diag"]
recs = diag["res"] if diag else None
expl = diag["expl"] if diag else {}

if recs is not None and len(recs):
    st.divider()
    st.subheader("Recommended for you")
    st.caption("Based on your selections and viewing preferences.")

    top = recs.iloc[0]
    e = expl.get(int(top["movieId"]), {})
    c1, c2 = st.columns([1, 4])
    c1.image(poster_url(top))
    c2.markdown(f"### {disp_title(top['title'])}")
    c2.caption(meta_line(top))
    ov = top.get("overview")
    if isinstance(ov, str) and ov:
        c2.markdown(ov[:400])
    if e:
        c2.markdown(explain_text(e))
    with c2:
        feedback("hero", int(top["movieId"]))

    def section(title, rows: pd.DataFrame, cols_n: int, key_prefix: str):
        if len(rows) == 0:
            return
        st.write("")
        st.subheader(title)
        cols = st.columns(cols_n)
        for j, (_, row) in enumerate(rows.iterrows()):
            with cols[j % cols_n]:
                rec_card(row, expl.get(int(row["movieId"]), {}),
                         f"{key_prefix}_{j}")

    section("More like this", recs.iloc[1:9], 4, "more")
    section("Outside your usual picks", recs.iloc[9:25], 4, "wild")


# ================= inspector sidebar =================

def hbars(df: pd.DataFrame, x: str, y: str, color=ACCENT, height=None):
    c = (alt.Chart(df).mark_bar(color=color)
         .encode(x=alt.X(x, title=None),
                 y=alt.Y(y, sort="-x", title=None))
         .properties(height=height or max(120, 34 * len(df))))
    st.altair_chart(c, width="stretch")


def heat(df: pd.DataFrame, x: str, y: str, v: str, height=260):
    base = (alt.Chart(df)
            .encode(x=alt.X(x, title=None), y=alt.Y(y, title=None)))
    rects = base.mark_rect().encode(
        color=alt.Color(v, scale=alt.Scale(
            range=["#f0efe9", "#5f77a6", ACCENT]), title=None, legend=None))
    text = base.mark_text(fontSize=13).encode(
        text=alt.Text(v, format=".2f"),
        color=alt.condition(alt.datum[v] > 0.55, alt.value("white"),
                            alt.value("#1c1b18")))
    st.altair_chart(rects + text, width="stretch")


def source_strength(item_idx: int) -> pd.DataFrame:
    meta = diag["cand_meta"].get(int(item_idx), {})
    cfg = rec.cfg
    rows = []
    if "content_rank" in meta:
        rows.append(("Content", 1 - meta["content_rank"] / cfg.content_k))
    if "cf_rank" in meta:
        rows.append(("Collaborative", 1 - meta["cf_rank"] / cfg.cf_k))
    if "covis_rank" in meta:
        rows.append(("Co-visitation", 1 - meta["covis_rank"] / cfg.covis_k))
    if "sasrec_rank" in meta:
        rows.append(("Sequence", 1 - meta["sasrec_rank"] / cfg.sasrec_k))
    if meta.get("in_pop"):
        rows.append(("Popularity", 0.15))
    return pd.DataFrame(rows, columns=["signal", "strength"])


FRIENDLY = {
    "content_sim": "Content similarity", "cf_sim": "Collaborative",
    "covis_score": "Co-visitation", "covis_max": "Co-watch (best pick)",
    "covis_mean": "Co-watch (average)", "covis_wavg": "Co-watch (weighted)",
    "sasrec_score": "Sequence model", "hist_cos_max": "History match",
    "hist_cos_mean": "History (average)", "hist_cos_wavg": "History (weighted)",
    "hist_cf_max": "Collaborative match", "genre_affinity": "Genre affinity",
    "director_match": "Director", "cast_overlap": "Cast overlap",
    "keyword_overlap": "Keywords", "company_match": "Studio",
    "bayes_rating": "Quality score", "in_pop": "Popularity prior",
    "year_diff": "Era distance", "decade_affinity": "Era affinity",
}


def feature_breakdown(item_idx: int) -> pd.DataFrame:
    loc = np.where(diag["cand_idx"] == int(item_idx))[0]
    if len(loc) == 0:
        return pd.DataFrame(columns=["feature", "value"])
    x = diag["X"][loc[0]]
    rows = [(FRIENDLY.get(n, n.replace("_", " ")), float(v))
            for n, v in zip(FEATURES, x) if v > 0 and n in FRIENDLY]
    rows.sort(key=lambda r: -r[1])
    return pd.DataFrame(rows[:10], columns=["feature", "value"])


def pca_scatter(sel_items, rec_items):
    """2D PCA of content embeddings — selected, recommended, context."""
    emb = rec.a.content_emb
    rng = np.random.default_rng(0)
    ctx = rng.choice(rec.a.pop_items[:2000], size=120, replace=False)
    all_idx = np.concatenate([sel_items, rec_items, ctx])
    X = emb[all_idx]
    X = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    P = X @ vt[:2].T
    labels = (["Your pick"] * len(sel_items)
              + ["Recommended"] * len(rec_items) + ["Catalog"] * len(ctx))
    order = ["Your pick", "Recommended", "Catalog"]
    df = pd.DataFrame({"x": P[:, 0], "y": P[:, 1], "kind": labels})
    c = (alt.Chart(df).mark_circle().encode(
        x=alt.X("x", title=None, axis=None),
        y=alt.Y("y", title=None, axis=None),
        color=alt.Color("kind", title=None, sort=order, scale=alt.Scale(
            domain=order, range=["#a33b32", ACCENT, "#c9c6bd"])),
        size=alt.Size("kind", title=None, sort=order, scale=alt.Scale(
            domain=order, range=[140, 90, 30])),
        opacity=alt.value(0.85))
        .properties(height=300))
    st.altair_chart(c, width="stretch")


with st.sidebar:
    st.caption("RECOMMENDATION INSPECTOR")
    if diag is None:
        st.write("Generate recommendations to inspect how the system "
                 "arrived at them.")
    else:
        shown = recs["movieId"].astype(int).tolist()
        labels = {int(r["movieId"]): disp_title(r["title"])
                  for _, r in recs.iterrows()}
        cur = st.selectbox("Inspecting", shown,
                           format_func=lambda m: labels.get(m, str(m)),
                           key="inspect_mid")
        cur_idx = rec.a.movie_id_to_idx.get(int(cur))
        cur_row = row_of(int(cur))
        e = expl.get(int(cur), {})

        with st.expander("Why this movie", expanded=True):
            st.markdown(f"**{disp_title(cur_row['title'])}**")
            if e:
                st.write(explain_text(e))
            sc = recs.loc[recs["movieId"] == int(cur), "score"]
            if len(sc):
                st.caption(f"Ranker score  {float(sc.iloc[0]):.3f}")
            sig = source_strength(cur_idx)
            if len(sig):
                hbars(sig, "strength", "signal")

        with st.expander("Relationships"):
            sel_items = np.array([rec.a.movie_id_to_idx[s["mid"]]
                                  for s in selected])
            rec_items = np.array([rec.a.movie_id_to_idx[m]
                                  for m in shown[:6]])
            sel_t = [disp_title(row_of(s["mid"])["title"]) for s in selected]
            rec_t = [labels[m] for m in shown[:6]]
            rows = []
            emb = rec.a.content_emb
            for i, si in enumerate(sel_items):
                for j, ri in enumerate(rec_items):
                    rows.append({"sel": sel_t[i], "rec": rec_t[j],
                                 "sim": float(emb[si] @ emb[ri])})
            st.caption("Selected × recommendation similarity")
            heat(pd.DataFrame(rows), "rec", "sel", "sim")

        with st.expander("Retrieval"):
            cm = diag["cand_meta"]
            counts = {
                "Co-visitation": sum("covis_rank" in m for m in cm.values()),
                "Collaborative": sum("cf_rank" in m for m in cm.values()),
                "Content": sum("content_rank" in m for m in cm.values()),
                "Sequence": sum("sasrec_rank" in m for m in cm.values()),
                "Popularity": sum(1 for m in cm.values() if m.get("in_pop")),
            }
            st.caption("Recommendation funnel")
            st.metric("Candidates generated", len(diag["cand_idx"]))
            st.metric("Ranked", len(diag["top"]))
            st.metric("Displayed", len(diag["sel"]))
            st.caption("Candidates per source")
            hbars(pd.DataFrame(counts.items(),
                               columns=["source", "count"]),
                  "count", "source")

        with st.expander("Ranking"):
            st.caption(f"Strongest signals for "
                       f"{disp_title(cur_row['title'])}")
            fb = feature_breakdown(cur_idx)
            if len(fb):
                hbars(fb, "value", "feature")

        with st.expander("Vector space"):
            st.caption("2D projection of the content embedding space")
            pca_scatter(sel_items, diag["sel"][:10])

        es = eval_summary()
        if es:
            with st.expander("Evaluation"):
                st.caption(f"Temporal test · {es['n_users']:,} users")
                row = es["table"]["ranker_mmr"]
                a1, a2 = st.columns(2)
                a1.metric("NDCG@10", f"{row['ndcg@10']:.4f}")
                a2.metric("Recall@10", f"{row['recall@10']:.4f}")
                a1.metric("Hit rate", f"{row['hitrate@10']:.3f}")
                a2.metric("Coverage", f"{row['coverage']:.4f}")
                st.caption("Per-retriever NDCG@10")
                tab = es["table"]
                names = {"content": "Content", "cf": "Collaborative",
                         "covis": "Co-visitation", "sasrec": "Sequence",
                         "popularity": "Popularity"}
                df = pd.DataFrame(
                    [(names[k], tab[k]["ndcg@10"]) for k in names
                     if k in tab], columns=["retriever", "ndcg"])
                hbars(df, "ndcg", "retriever")


