# train.py
"""
Train models for Week 10 coffee exercise.

Writes:
 - models/model_1.pickle  AND repo-root model_1.pickle
 - models/model_2.pickle  AND repo-root model_2.pickle
 - models/roast_map.pickle AND repo-root roast_map.pickle
 - models/model_3.pickle (optional, only in models/)
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

# Root-level filenames expected by autograder
ROOT_MODEL_1 = Path("model_1.pickle")
ROOT_MODEL_2 = Path("model_2.pickle")
ROOT_ROAST_MAP = Path("roast_map.pickle")


def _dump_both(obj, filename_in_models: Path, root_filename: Path):
    """Helper to pickle to models/ and also root path for autograder compatibility."""
    # write to models/
    with open(filename_in_models, "wb") as f:
        pickle.dump(obj, f)
    # also write at repo root (autograder expects root-level files)
    with open(root_filename, "wb") as f:
        pickle.dump(obj, f)


def train_model_1(df: pd.DataFrame):
    # Model 1: LinearRegression on 100g_USD -> rating
    X = df[["100g_USD"]].astype(float).values.reshape(-1, 1)
    y = df["rating"].astype(float).values
    model = LinearRegression()
    model.fit(X, y)

    _dump_both(model, OUT_DIR / "model_1.pickle", ROOT_MODEL_1)
    print("Saved model_1.pickle (models/ and repo root)")


def train_model_2(df: pd.DataFrame):
    # Model 2: DecisionTreeRegressor on 100g_USD + roast_cat -> rating
    # Build roast mapping
    roast_vals = pd.Categorical(df["roast"].fillna("UNKNOWN"))
    roast_categories = list(roast_vals.categories)
    roast_map = {r: i for i, r in enumerate(roast_categories)}
    df["_roast_cat"] = df["roast"].fillna("UNKNOWN").map(roast_map)

    X = df[["100g_USD", "_roast_cat"]].astype(float).values
    y = df["rating"].astype(float).values

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    _dump_both(model, OUT_DIR / "model_2.pickle", ROOT_MODEL_2)
    _dump_both(roast_map, OUT_DIR / "roast_map.pickle", ROOT_ROAST_MAP)
    print("Saved model_2.pickle and roast_map.pickle (models/ and repo root)")


def train_model_3(df: pd.DataFrame):
    # Optional: TF-IDF on desc_3 -> LinearRegression; kept only in models/
    if "desc_3" not in df.columns:
        print("desc_3 column not present; skipping model_3")
        return
    texts = df["desc_3"].fillna("").astype(str).values
    y = df["rating"].astype(float).values
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    lr = LinearRegression()
    pipeline = make_pipeline(vec, lr)
    pipeline.fit(texts, y)

    # pipeline contains vectorizer; save only to models/
    with open(OUT_DIR / "model_3.pickle", "wb") as f:
        pickle.dump(pipeline, f)
    print("Saved model_3.pickle (models/ only)")


def main():
    print("Downloading data...")
    df = pd.read_csv(DATA_URL)

    # require the basic columns
    if "100g_USD" not in df.columns or "rating" not in df.columns:
        raise RuntimeError("Expected '100g_USD' and 'rating' columns in dataset")

    # Basic hygiene
    df["100g_USD"] = pd.to_numeric(df["100g_USD"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["100g_USD", "rating"]).reset_index(drop=True)

    train_model_1(df)
    train_model_2(df)
    train_model_3(df)


if __name__ == "__main__":
    main()