# train.py
"""
Train 3 simple models on the Coffee Analysis data and persist artifacts:

- model_1.pickle : LinearRegression on 100g_USD -> rating
- model_2.pickle : DecisionTreeRegressor on (100g_USD, roast_cat) -> rating
- roast_map.pickle: dict mapping roast str -> integer category
- model_3.pickle & model_3_vec.pickle: TF-IDF + LinearRegression for text-only model
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

# Data URL
DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

def train_model_1(df: pd.DataFrame):
    # Model 1: LinearRegression on 100g_USD -> rating
    X = df[["100g_USD"]].astype(float).values.reshape(-1, 1)
    y = df["rating"].astype(float).values
    model = LinearRegression()
    model.fit(X, y)
    with open(OUT_DIR / "model_1.pickle", "wb") as f:
        pickle.dump(model, f)
    print("Saved model_1.pickle")

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

    with open(OUT_DIR / "model_2.pickle", "wb") as f:
        pickle.dump(model, f)
    with open(OUT_DIR / "roast_map.pickle", "wb") as f:
        pickle.dump(roast_map, f)

    print("Saved model_2.pickle and roast_map.pickle")

def train_model_3(df: pd.DataFrame):
    # Optional: TF-IDF on desc_3 -> LinearRegression
    if "desc_3" not in df.columns:
        print("desc_3 column not present; skipping model_3")
        return
    texts = df["desc_3"].fillna("").astype(str).values
    y = df["rating"].astype(float).values
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    lr = LinearRegression()
    pipeline = make_pipeline(vec, lr)
    pipeline.fit(texts, y)

    with open(OUT_DIR / "model_3.pickle", "wb") as f:
        pickle.dump(pipeline, f)
    # pipeline includes vectorizer internally; but for explicit use you can dump vectorizer separately:
    print("Saved model_3.pickle (TF-IDF + LinearRegression)")

def main():
    print("Downloading data...")
    df = pd.read_csv(DATA_URL)
    # Basic hygiene
    df = df.dropna(subset=["100g_USD", "rating"], how="any")  # must have numeric target and price
    df["100g_USD"] = pd.to_numeric(df["100g_USD"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["100g_USD", "rating"])
    df.reset_index(drop=True, inplace=True)

    train_model_1(df)
    train_model_2(df)
    train_model_3(df)

if __name__ == "__main__":
    main()