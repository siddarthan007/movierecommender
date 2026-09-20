# Autonomous Recommendation Evaluation Policy

The recommender agent must never declare an experiment successful
based on a single metric.

Every experiment must be compared against the immutable parent
experiment using the same temporal test users and random seeds.

## PRIMARY METRICS

NDCG@10, Recall@10, HitRate@10

## SECONDARY METRICS

NDCG@20, Recall@20, MRR, catalog coverage, long-tail coverage, ILD,
novelty, personalization, popularity exposure, latency

## RETRIEVAL METRICS

For every retriever independently report Recall@20/50/100/250/500 and
candidate-source overlap. The candidate union must report these too.
A ranking model cannot compensate for missing candidates.

## PROMOTION RULE

An experiment may be promoted only if:

1. NDCG@10 improves materially over parent
2. Recall@10 does not regress materially
3. Recall@100 does not regress materially
4. coverage does not collapse
5. diversity does not collapse
6. popularity bias does not materially worsen
7. latency remains within budget
8. improvement survives bootstrap confidence testing

- Accuracy up + retrieval recall down -> REJECT.
- Coverage up + NDCG materially down -> REJECT.
- NDCG up only via popularity exposure up -> FLAG.
- Improvement within noise -> REJECT.
- Conflicting metrics -> RUN MORE SEEDS / MORE USERS.
- Aggregate up but a user segment seriously harmed -> FLAG + investigate.

Never select a model based on intuition. The metrics decide promotion.

## EXPERIMENT RECORD

After every experiment produce: EXPERIMENT, PARENT, CHANGE, METRICS,
SEGMENT METRICS, RETRIEVAL DIAGNOSIS, RANKING DIAGNOSIS, DIVERSITY
DIAGNOSIS, LATENCY, STATISTICAL CONFIDENCE, DECISION
(one of PROMOTE / REJECT / INVESTIGATE).

Never modify the parent model directly. Every promoted experiment
becomes the new parent. Stop when improvements plateau.

## Streamlit product/UI quality policy

The Streamlit interface is part of the product and must be evaluated
and improved alongside the recommendation engine.

OBJECTIVE: a professional, restrained movie-discovery interface —
a polished software product, not an AI demo, dashboard, feed,
component showcase, or "cute" recommendation app.

NON-NEGOTIABLE:
- No emojis anywhere in the UI.
- No decorative symbols used as pseudo-icons.
- No excessive rounded cards, gradients, glow, animated gimmicks.
- No "AI-powered" marketing language, fake conversational copy,
  generic phrases ("Discover your next obsession").
- No excessive uppercase text or custom HTML/CSS.
- Prefer native Streamlit primitives whenever they provide the UI:
  st.container(border=True), st.columns, st.form, st.segmented_control,
  st.selectbox, st.multiselect, st.slider, st.popover, st.tabs,
  st.expander, st.button, st.image, st.caption, st.metric,
  st.session_state.
- Custom CSS only when native Streamlit cannot achieve a clearly
  defined visual requirement. Never rebuild controls with raw HTML.

INFORMATION ARCHITECTURE (main screen):
1. Header: title + one restrained sentence.
2. Search: catalog search, ranked normalized matches.
3. Taste selection: compact shelf of selected movies.
4. Preference controls: preference strength, exploration,
   optional mood/context.
5. One primary recommendation action.
6. Recommendations in meaningful sections:
   Recommended for you / Because of your selections /
   More like this / Explore beyond your usual choices.

MOVIE RESULT: poster, title, year, one or two metadata fields,
deterministic recommendation reason, lightweight text feedback.
Never fabricate explanations.

FEEDBACK: professional text controls only — "Like" / "Not relevant"
or "Like" / "Dislike". No hearts, crosses, thumbs, stars, emoji.

LAYOUT: hierarchy over uniform grids — a primary recommendation,
then supporting rows. Avoid five equal columns everywhere.
Use whitespace intentionally.

STATES to handle factually: no movies selected, empty search,
loading, results, generation error. No playful error copy.

PERFORMANCE: keep @st.cache_resource for artifacts,
@st.cache_data for deterministic work, session_state for recs.
Presentation-only widget changes must not recompute recommendations.

UI EVALUATION: after any UI change, audit hierarchy, density,
consistency, readability, search/taste/recs usability, feedback
clarity, states, desktop + narrow layout, accessibility, clutter.
Verdict: PASS / NEEDS WORK. The interface must contain zero emoji
characters and zero decorative pseudo-icons.

VERIFICATION (run before declaring UI clean):
  rg -n "[\p{Emoji_Presentation}\p{Extended_Pictographic}]" app.py src tests
  rg -n "unsafe_allow_html|<div|<button|<style" app.py src
Second check isn't automatically a failure — inspect each hit and
remove it when a native primitive can do the job.

No experiment is complete until: metric gates pass, tests pass,
serving smoke passes, UI audit passes, zero emoji.
