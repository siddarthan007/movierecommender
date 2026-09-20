"""ETL: raw CSV -> processed parquet.

Builds the unified movie universe:
  MovieLens movies.csv + links.csv <- tmdbId -> TMDB_all_movies.csv
  + per-movie rating aggregates (count, mean, Bayesian average)
  + canonical text document per movie (for embeddings)

Outputs:
  data/processed/movies.parquet   one row per movie (item_idx ordering)
  data/processed/ratings.parquet  userId, item_idx, rating, timestamp
  data/processed/tags.parquet     userId, item_idx, tag, timestamp
"""
from __future__ import annotations

import re
import numpy as np
import polars as pl

from .config import CFG, Config

TMDB_COLS = [
    "id", "vote_average", "vote_count", "release_date", "revenue", "runtime",
    "budget", "imdb_id", "original_language", "original_title", "overview",
    "popularity", "tagline", "genres", "production_companies",
    "production_countries", "cast", "director", "writers", "keywords",
    "imdb_rating", "imdb_votes", "poster_path", "certification_us",
]


def _split_field(s, sep=",", n=None):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    parts = [p.strip() for p in str(s).split(sep) if p.strip()]
    return parts[:n] if n else parts


def _split_pipe(s, n=None):
    return _split_field(s, sep="|", n=n)


def _year_from_date(s) -> float:
    if not s or not isinstance(s, str):
        return np.nan
    m = re.match(r"(\d{4})", s)
    return float(m.group(1)) if m else np.nan


def _ml_year(title: str) -> float:
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return float(m.group(1)) if m else np.nan


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def build_canonical_text(row: dict, cast_top_n: int = 5, max_chars: int = 1200) -> str:
    """Canonical movie document, field-weighted via a signature head line.

    The head line repeats title/genres/director/year up front so the
    encoder weights them more strongly (early + repeated tokens).
    """
    title = _clean(row.get("title")) or ""
    genres = row.get("genres") or []
    director = row.get("director") or []
    head_bits = [title]
    if genres:
        head_bits.append(", ".join(genres))
    if director:
        head_bits.append("dir. " + ", ".join(director[:2]))
    if row.get("year") and not np.isnan(row["year"]):
        head_bits.append(str(int(row["year"])))
    parts = [" — ".join(head_bits), f"Title: {title}"]
    if genres:
        parts.append("Genres: " + ", ".join(genres))
    overview = _clean(row.get("overview"))
    if overview:
        parts.append("Overview: " + str(overview).strip())
    tagline = _clean(row.get("tagline"))
    if tagline:
        parts.append("Tagline: " + str(tagline).strip())
    if director:
        parts.append("Director: " + ", ".join(director[:3]))
    if row.get("cast"):
        parts.append("Cast: " + ", ".join(row["cast"][:cast_top_n]))
    if row.get("keywords"):
        parts.append("Keywords: " + ", ".join(row["keywords"][:20]))
    if row.get("year") and not np.isnan(row["year"]):
        parts.append(f"Year: {int(row['year'])}")
    text = "\n".join(p for p in parts if p.split(": ", 1)[-1].strip())
    return text[:max_chars]


def run_etl(cfg: Config = CFG) -> tuple[pl.DataFrame, pl.DataFrame]:
    cfg.ensure_dirs()
    ml = cfg.ml_path

    print("[etl] loading movielens movies/links")
    movies = pl.read_csv(ml / "movies.csv")
    links = pl.read_csv(ml / "links.csv").with_columns(
        pl.col("tmdbId").cast(pl.Float64, strict=False)
    )

    print("[etl] scanning TMDB (lazy)")
    tmdb = (
        pl.scan_csv(cfg.tmdb_path, infer_schema_length=1000)
        .select(TMDB_COLS)
        .with_columns(pl.col("id").cast(pl.Float64, strict=False))
        .filter(pl.col("id").is_not_null())
    )

    joined = (
        movies.lazy()
        .join(links.lazy().select(["movieId", "imdbId", "tmdbId"]), on="movieId", how="left")
        .join(tmdb, left_on="tmdbId", right_on="id", how="left")
        .collect(engine="streaming")
    )
    print(f"[etl] joined universe: {joined.height} movies "
          f"({joined.filter(pl.col('overview').is_not_null()).height} with TMDB match)")

    print("[etl] rating aggregates (lazy, 32M rows)")
    rate_stats = (
        pl.scan_csv(ml / "ratings.csv")
        .group_by("movieId")
        .agg(
            pl.col("rating").mean().alias("rating_mean"),
            pl.col("rating").count().alias("rating_count"),
        )
        .collect(engine="streaming")
    )
    global_mean = (
        pl.scan_csv(ml / "ratings.csv").select(pl.col("rating").mean()).collect(engine="streaming").item()
    )

    df = joined.join(rate_stats, on="movieId", how="left")
    df = df.with_columns([
        pl.col("rating_count").fill_null(0),
        pl.col("rating_mean").fill_null(global_mean),
    ])
    df = df.with_columns(
        ((pl.col("rating_count") * pl.col("rating_mean") + cfg.bayes_m * global_mean)
         / (pl.col("rating_count") + cfg.bayes_m)).alias("bayes_rating")
    )
    df = df.sort("movieId").with_row_index("item_idx")

    print("[etl] writing ratings.parquet with item_idx mapping")
    id_map = df.select(["movieId", "item_idx"])
    ratings = (
        pl.scan_csv(ml / "ratings.csv")
        .join(id_map.lazy(), on="movieId", how="inner")
        .select(["userId", "item_idx", "rating", "timestamp"])
        .collect(engine="streaming")
    )
    ratings.write_parquet(cfg.processed_path / "ratings.parquet", compression="zstd")

    print("[etl] building canonical text + list fields (python rows)")
    pdf = df.to_pandas()
    pdf["genres_ml"] = pdf["genres"].apply(lambda s: _split_pipe(s))
    pdf["cast_list"] = pdf["cast"].apply(lambda s: _split_field(s, ",", cfg.cast_top_n))
    pdf["director_list"] = pdf["director"].apply(lambda s: _split_field(s, ",", 3))
    pdf["keywords_list"] = pdf["keywords"].apply(lambda s: _split_pipe(s, 25))
    pdf["companies_list"] = pdf["production_companies"].apply(lambda s: _split_field(s, ",", 5))
    pdf["year"] = pdf.apply(
        lambda r: _year_from_date(r["release_date"]) if not np.isnan(_year_from_date(r["release_date"]))
        else _ml_year(r["title"]), axis=1)
    pdf["text"] = pdf.apply(
        lambda r: build_canonical_text(
            {"title": r["title"], "genres": r["genres_ml"], "overview": r["overview"],
             "tagline": r["tagline"], "director": r["director_list"],
             "cast": r["cast_list"], "keywords": r["keywords_list"], "year": r["year"]},
            cfg.cast_top_n, cfg.emb_max_chars),
        axis=1)

    keep = pdf[[
        "item_idx", "movieId", "imdbId", "tmdbId", "title", "year",
        "genres_ml", "overview", "tagline", "director_list", "cast_list",
        "keywords_list", "companies_list", "original_language", "runtime", "popularity",
        "vote_average", "vote_count", "imdb_rating", "imdb_votes",
        "rating_mean", "rating_count", "bayes_rating", "poster_path",
        "certification_us", "text",
    ]].rename(columns={"genres_ml": "genres", "director_list": "director",
                       "cast_list": "cast", "keywords_list": "keywords",
                       "companies_list": "production_companies"})
    out = pl.from_pandas(keep)
    out.write_parquet(cfg.processed_path / "movies.parquet", compression="zstd")

    print(f"[etl] done. movies={out.height}, ratings={ratings.height}, "
          f"global_mean={global_mean:.3f}")
    return out, ratings


if __name__ == "__main__":
    run_etl()
