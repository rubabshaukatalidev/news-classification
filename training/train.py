"""Train the Urdu news topic classifier.

Usage
-----
    python -m training.train                          # train on bundled seed data
    python -m training.train --data my_news.csv       # CSV with `text,label`
    python -m training.train --data news.xlsx --text-col headline --label-col topic

Outputs `models/model.joblib` and `models/metrics.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from training.text import normalize_urdu

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "seed_urdu_news.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(preprocessor=normalize_urdu, analyzer="word",
                                     ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(preprocessor=normalize_urdu, analyzer="char_wb",
                                     ngram_range=(2, 5), sublinear_tf=True, min_df=1)),
        ]
    )
    clf = LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def load_data(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise SystemExit(f"Columns not found in {path.name}: {sorted(missing)}. Found: {list(df.columns)}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna().drop_duplicates(subset="text")
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_data(args.data, args.text_col, args.label_col)
    print(f"Loaded {len(df)} rows, {df['label'].nunique()} classes from {args.data.name}")
    print(df["label"].value_counts().to_string(), "\n")

    # Cross-validated estimate (more honest than one tiny split on small data)
    min_class = int(df["label"].value_counts().min())
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    cv_pred = cross_val_predict(build_pipeline(), df["text"], df["label"], cv=cv)
    cv_macro_f1 = f1_score(df["label"], cv_pred, average="macro")
    print(f"{n_splits}-fold CV macro-F1: {cv_macro_f1:.3f}")

    # Held-out report for per-class detail
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=args.seed
    )
    held = build_pipeline().fit(X_tr, y_tr)
    report = classification_report(y_te, held.predict(X_te), output_dict=True, zero_division=0)
    print(classification_report(y_te, held.predict(X_te), zero_division=0))

    # Final model on all data
    model = build_pipeline().fit(df["text"], df["label"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(
            {
                "n_samples": int(len(df)),
                "classes": sorted(df["label"].unique().tolist()),
                "cv_folds": n_splits,
                "cv_macro_f1": round(float(cv_macro_f1), 4),
                "heldout_report": report,
                "data_source": args.data.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
