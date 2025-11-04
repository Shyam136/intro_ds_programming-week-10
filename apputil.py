# apputil.py additions (imports)
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Sequence

# paths for artifacts
_MODEL_DIR = Path("models")
_MODEL_1 = _MODEL_DIR / "model_1.pickle"
_MODEL_2 = _MODEL_DIR / "model_2.pickle"
_ROAST_MAP = _MODEL_DIR / "roast_map.pickle"
_MODEL_3 = _MODEL_DIR / "model_3.pickle"

def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_rating(df_X: pd.DataFrame, text: bool = False) -> np.ndarray:
    """
    Predict coffee `rating` for a DataFrame df_X.

    - If text=True: df_X should be a single-column DataFrame/Series of text strings.
      Uses model_3.pickle (TF-IDF pipeline).
    - Else: df_X is expected to have columns ["100g_USD", "roast"].
      If 'roast' values are unseen by model_2, fallback to model_1 using only 100g_USD.

    Returns a numpy array of predicted ratings.
    """
    if text:
        # use model_3 (pipeline)
        model3 = _load_pickle(_MODEL_3)
        # df_X might be DataFrame with column "text" or a Series
        if isinstance(df_X, pd.DataFrame):
            # pick the first column if named differently
            col = df_X.columns[0]
            texts = df_X[col].astype(str).fillna("").values
        else:
            texts = pd.Series(df_X).astype(str).fillna("").values
        preds = model3.predict(texts)
        return preds

    # non-text path
    required = ["100g_USD", "roast"]
    for col in required:
        if col not in df_X.columns:
            raise ValueError(f"df_X must contain columns {required}")

    # load artifacts (may raise FileNotFoundError if not trained)
    model1 = _load_pickle(_MODEL_1)
    model2 = _load_pickle(_MODEL_2)
    roast_map = _load_pickle(_ROAST_MAP)

    # Prepare arrays
    usd = pd.to_numeric(df_X["100g_USD"], errors="coerce").values.reshape(-1, 1)
    roast_series = df_X["roast"].fillna("UNKNOWN").astype(str)

    preds = np.full(len(df_X), np.nan, dtype=float)
    # decide rows where roast known vs unknown
    mapped = roast_series.map(roast_map)
    known_mask = mapped.notna()
    unknown_mask = ~known_mask

    # for known roasts, use model_2 (both features)
    if known_mask.any():
        X_known = np.column_stack([usd[known_mask].ravel(), mapped[known_mask].astype(float).values])
        preds[known_mask] = model2.predict(X_known)

    # for unknown roasts, fallback to model_1 using USD
    if unknown_mask.any():
        preds[unknown_mask] = model1.predict(usd[unknown_mask])

    return preds