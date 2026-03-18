"""
model_builder.py
────────────────
Reads  Dataset/movie_dataset.csv, drops ALL rows that have any null value,
then builds content-based similarity vectors and saves three pickle files:

    df_popular.pkl    — for the "Top Popular Movies" page
    df_recommend.pkl  — for recommendations & movie detail page
    similarity.pkl    — cosine-similarity matrix (n_movies × n_movies)

Run directly:
    python model_builder.py

Or import and call build_pipeline() from anywhere (e.g. the Streamlit app).
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Fixed paths ───────────────────────────────────────────────────────────────
DATASET_PATH = Path("Dataset") / "movie_dataset.csv"
OUTPUT_DIR   = Path(".")      # pkl files land in the same folder as app.py
MAX_FEATURES = 5000           # TF-IDF vocabulary size


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _tolist(text) -> list:
    """'Action, Drama'  →  ['Action', 'Drama']"""
    if isinstance(text, str):
        return [i.strip() for i in text.split(",") if i.strip()]
    return []


def _collapse(lst: list) -> list:
    """['Tom Hanks', 'Meg Ryan']  →  ['TomHanks', 'MegRyan']"""
    return [i.replace(" ", "") for i in lst]


# ── Core pipeline ─────────────────────────────────────────────────────────────

def build_pipeline(
    dataset_path: Path = DATASET_PATH,
    output_dir:   Path = OUTPUT_DIR,
    max_features: int  = MAX_FEATURES,
    log: Callable[[str], None] = print,
) -> dict:
    """
    Full pipeline: load → dropna → feature engineering → TF-IDF → cosine sim → save.

    Parameters
    ----------
    dataset_path : Path to movie_dataset.csv  (default: Dataset/movie_dataset.csv)
    output_dir   : Where to write .pkl files  (default: current directory)
    max_features : TF-IDF vocabulary cap      (default: 5000)
    log          : Callable for progress strings (default: print)

    Returns
    -------
    dict with keys: n_raw, n_clean, n_popular, n_recommend, output_dir, artefacts
    """
    dataset_path = Path(dataset_path)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'.\n"
            "Make sure the file exists at  Dataset/movie_dataset.csv  "
            "relative to the directory where you run the app."
        )

    # ── 1. Load ───────────────────────────────────────────────────────────
    log("📂  Loading dataset…")
    df = pd.read_csv(dataset_path)
    n_raw = len(df)
    log(f"    Loaded {n_raw:,} raw rows.")

    # ── 2. Drop rows with ANY null value ──────────────────────────────────
    log("🧹  Dropping rows with null values (df.dropna)…")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_clean = len(df)
    log(f"    Retained {n_clean:,} rows  ({n_raw - n_clean:,} dropped).")

    if n_clean == 0:
        raise ValueError(
            "Dataset is empty after dropna(). "
            "Check that movie_dataset.csv contains valid rows."
        )

    # ── 3. Build df_popular (for Top Movies page) ─────────────────────────
    log("🏗️   Building df_popular…")
    df_popular = df[[
        "id", "title", "averageRating", "numVotes", "release_date",
        "runtime", "adult", "popularity", "poster_path",
        "genres", "production_countries", "overview",
    ]].copy()
    df_popular["genres"]               = df_popular["genres"].apply(_tolist)
    df_popular["production_countries"] = df_popular["production_countries"].apply(_tolist)
    df_popular["release_date"]         = pd.to_datetime(
        df_popular["release_date"], errors="coerce"
    )
    log(f"    df_popular  →  {len(df_popular):,} movies.")

    # ── 4. Build df_recommend (for Recommend & Details pages) ────────────
    log("🏗️   Building df_recommend…")
    df_recommend = df[[
        "id", "title", "overview", "poster_path", "genres",
        "keywords", "directors", "writers", "cast",
        "release_date", "runtime", "averageRating",
    ]].copy()

    for col in ["genres", "keywords", "directors", "writers", "cast"]:
        df_recommend[col] = df_recommend[col].apply(_tolist).apply(_collapse)

    df_recommend["tags"] = (
        df_recommend["overview"].apply(lambda x: str(x).split())
        + df_recommend["genres"]
        + df_recommend["keywords"]
        + df_recommend["cast"].apply(lambda x: x[:5])
        + df_recommend["directors"].apply(lambda x: x[:1])
        + df_recommend["writers"].apply(lambda x: x[:1])
    )
    df_recommend["tags"] = df_recommend["tags"].apply(lambda x: " ".join(x).lower())
    log(f"    df_recommend  →  {len(df_recommend):,} movies.")

    # ── 5. TF-IDF + cosine similarity ────────────────────────────────────
    log(f"🔢  TF-IDF vectorisation  (max_features={max_features:,})…")
    tfidf  = TfidfVectorizer(max_features=max_features, stop_words="english")
    vector = tfidf.fit_transform(df_recommend["tags"]).toarray()
    log(f"    Matrix shape: {vector.shape}")

    log("📐  Computing cosine similarity…")
    t0         = time.perf_counter()
    similarity = cosine_similarity(vector)
    log(f"    Done in {time.perf_counter() - t0:.1f}s  |  shape: {similarity.shape}")

    # ── 6. Save pickles ───────────────────────────────────────────────────
    log("💾  Saving pickle files…")
    saves = {
        "df_popular.pkl":   df_popular.to_dict(),
        "df_recommend.pkl": df_recommend.to_dict(),
        "similarity.pkl":   similarity,
    }
    for fname, obj in saves.items():
        dest = output_dir / fname
        with open(dest, "wb") as f:
            pickle.dump(obj, f)
        log(f"    ✓  {dest}")

    log("✅  Pipeline complete!")

    return {
        "n_raw":       n_raw,
        "n_clean":     n_clean,
        "n_popular":   len(df_popular),
        "n_recommend": len(df_recommend),
        "output_dir":  str(output_dir),
        "artefacts":   list(saves.keys()),
    }


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        stats = build_pipeline()
        print("\n── Summary ─────────────────────────────────────")
        print(f"  Raw rows            : {stats['n_raw']:,}")
        print(f"  After dropna        : {stats['n_clean']:,}")
        print(f"  df_popular  movies  : {stats['n_popular']:,}")
        print(f"  df_recommend movies : {stats['n_recommend']:,}")
        print(f"  Output directory    : {stats['output_dir']}")
        print("────────────────────────────────────────────────")
    except Exception as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        sys.exit(1)
