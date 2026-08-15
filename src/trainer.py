"""
ML Training Pipeline - ICMI v2.0

Trains multiple candidate algorithms (CatBoost, XGBoost, RandomForest,
MLP) per brand, evaluates them on a common holdout split + cross-
validation, and keeps only the best performer for each brand. Brands
with too little data fall back to one GLOBAL model trained on every
brand combined.

No target leakage: categorical features are either handled natively
(CatBoost) or one-hot encoded (everyone else) - never encoded using
the price column.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


CAT_FEATURES: List[str] = [
    "brand_slug",
    "name",
    "trim",
    "fuel",
    "transmission",
    "body_status",
]

NUM_FEATURES: List[str] = [
    "year",
    "mileage",
    "mileage_unknown",
    "body_status_ordinal",
    "age",
]

FEATURES: List[str] = CAT_FEATURES + NUM_FEATURES
TARGET: str = "price"


def _load_training_config(config_path: str = "config/brands.yaml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("training", {})


def _build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """One-hot encode categoricals; optionally scale numerics (for the NN)."""
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_FEATURES,
            ),
            ("num", numeric_step, NUM_FEATURES),
        ]
    )


def _build_candidates(random_seed: int = 42) -> Dict[str, Pipeline]:
    """
    Build one fresh, unfitted sklearn Pipeline per candidate algorithm.

    CatBoost gets raw categoricals (native handling - no leakage-prone
    encoding). The rest go through one-hot encoding via ColumnTransformer.
    """
    candidates: Dict[str, Pipeline] = {}

    cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]
    candidates["catboost"] = Pipeline(
        [
            (
                "model",
                CatBoostRegressor(
                    iterations=800,
                    learning_rate=0.05,
                    depth=8,
                    l2_leaf_reg=3,
                    random_seed=random_seed,
                    cat_features=cat_indices,
                    verbose=False,
                ),
            ),
        ]
    )

    candidates["xgboost"] = Pipeline(
        [
            ("prep", _build_preprocessor(scale_numeric=False)),
            (
                "model",
                XGBRegressor(
                    n_estimators=600,
                    learning_rate=0.05,
                    max_depth=7,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    candidates["random_forest"] = Pipeline(
        [
            ("prep", _build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    candidates["mlp"] = Pipeline(
        [
            ("prep", _build_preprocessor(scale_numeric=True)),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=800,
                    early_stopping=True,
                    random_state=random_seed,
                ),
            ),
        ]
    )

    return candidates


def evaluate_candidates(
    df: pd.DataFrame,
    candidates: Dict[str, Pipeline],
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> Tuple[str, Pipeline, Dict]:
    """
    Fit + evaluate every candidate on the same split, return the winner.

    Selection is based on cross-validated R^2 (more robust than a
    single holdout split), with the holdout MAE/R^2 kept for reporting.

    Returns:
        (best_name, best_fitted_pipeline, comparison_dict)
    """
    X = df[FEATURES].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].astype(str)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    # Keep folds sane on small (per-brand) datasets.
    n_splits = min(cv_folds, max(2, len(df) // 20))

    comparison: Dict[str, Dict] = {}
    fitted: Dict[str, Pipeline] = {}

    for name, pipeline in candidates.items():
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Manual K-fold instead of sklearn's cross_val_score: the
            # latter calls clone() on the estimator, and CatBoost's
            # cat_features constructor param breaks sklearn's clone
            # contract (get_params() doesn't round-trip cleanly), which
            # silently drops CatBoost from every comparison. Refitting
            # the same pipeline instance per fold sidesteps clone()
            # entirely and works identically for every candidate.
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            fold_scores = []
            for train_idx, val_idx in kfold.split(X):
                pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                fold_pred = pipeline.predict(X.iloc[val_idx])
                fold_scores.append(r2_score(y.iloc[val_idx], fold_pred))
            cv_scores = np.array(fold_scores)

            # Re-fit on the holdout train split so the saved pipeline
            # matches the test_mae/test_r2 reported above.
            pipeline.fit(X_train, y_train)

            comparison[name] = {
                "test_mae": float(mae),
                "test_mae_million": round(float(mae / 1e6), 2),
                "test_r2": round(float(r2), 4),
                "cv_r2_mean": round(float(cv_scores.mean()), 4),
                "cv_r2_std": round(float(cv_scores.std()), 4),
            }
            fitted[name] = pipeline

            logger.info(
                "  {:<14} | test_R2={:.4f} | cv_R2={:.4f} (+/-{:.4f}) | MAE={:.2f}M",
                name,
                r2,
                cv_scores.mean(),
                cv_scores.std(),
                mae / 1e6,
            )
        except Exception as e:
            logger.warning("  {:<14} FAILED: {}", name, e)
            comparison[name] = {"error": str(e)}

    if not fitted:
        raise RuntimeError("All candidate models failed to train.")

    best_name = max(fitted, key=lambda n: comparison[n]["cv_r2_mean"])
    return best_name, fitted[best_name], comparison


class ModelTrainer:
    """Orchestrates per-brand + global model training and artifact saving."""

    def __init__(self, config_path: str = "config/brands.yaml"):
        self.cfg = _load_training_config(config_path)
        self.min_samples = self.cfg.get("min_samples_per_brand", 60)
        self.test_size = self.cfg.get("test_size", 0.2)
        self.cv_folds = self.cfg.get("cv_folds", 5)
        self.seed = self.cfg.get("random_seed", 42)

    def load_data(
        self, path: str = "data/processed/processed_latest.parquet"
    ) -> pd.DataFrame:
        """
        Load and prepare training data.

        Args:
            path: Path to processed Parquet file.

        Returns:
            Prepared DataFrame.
        """
        df = pd.read_parquet(path)

        for col in CAT_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df = df.dropna(subset=FEATURES + [TARGET])

        logger.info(
            "Loaded data | {} rows | {} brands | features: {}",
            len(df),
            df["brand_slug"].nunique(),
            len(FEATURES),
        )
        return df

    def train_all(self, df: pd.DataFrame, output_dir: str = "artifacts") -> Dict:
        """
        Train a GLOBAL fallback model (all brands) plus a dedicated best
        model per brand that has enough data.

        Args:
            df: Full processed dataset (all brands).
            output_dir: Base directory for model artifacts.

        Returns:
            Summary dict of what was trained and how it performed.
        """
        Path(output_dir, "models").mkdir(parents=True, exist_ok=True)
        Path(output_dir, "metadata").mkdir(parents=True, exist_ok=True)

        summary: Dict = {"trained_at": datetime.now().isoformat(), "brands": {}}

        # --- Global fallback model (covers brands with too little data) ---
        logger.info("Training GLOBAL model | {} rows", len(df))
        candidates = _build_candidates(self.seed)
        best_name, best_pipeline, comparison = evaluate_candidates(
            df, candidates, self.test_size, self.cv_folds, self.seed
        )
        self._save_model("global", best_name, best_pipeline, comparison, df, output_dir)
        summary["global"] = {
            "best_model": best_name,
            "n_samples": len(df),
            "test_r2": comparison[best_name]["test_r2"],
        }

        # --- Per-brand models ---
        for brand, brand_df in df.groupby("brand_slug"):
            if len(brand_df) < self.min_samples:
                logger.info(
                    "Skipping per-brand model for '{}' ({} rows < {} minimum) "
                    "-> will use global fallback",
                    brand,
                    len(brand_df),
                    self.min_samples,
                )
                continue

            logger.info(
                "Training models for brand '{}' | {} rows", brand, len(brand_df)
            )
            candidates = _build_candidates(self.seed)
            try:
                best_name, best_pipeline, comparison = evaluate_candidates(
                    brand_df, candidates, self.test_size, self.cv_folds, self.seed
                )
            except Exception as e:
                logger.error("Training failed for brand '{}': {}", brand, e)
                continue

            self._save_model(
                brand, best_name, best_pipeline, comparison, brand_df, output_dir
            )
            summary["brands"][brand] = {
                "best_model": best_name,
                "n_samples": len(brand_df),
                "test_r2": comparison[best_name]["test_r2"],
            }

        summary_path = Path(output_dir, "metadata", "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(
            "Training complete | {} dedicated brand model(s) + 1 global fallback",
            len(summary["brands"]),
        )
        return summary

    def _save_model(
        self,
        key: str,
        best_name: str,
        pipeline: Pipeline,
        comparison: Dict,
        df: pd.DataFrame,
        output_dir: str,
    ) -> None:
        models_dir = Path(output_dir) / "models"
        meta_dir = Path(output_dir) / "metadata"

        model_path = models_dir / f"{key}.joblib"
        joblib.dump(pipeline, model_path)

        metadata = {
            "brand": key,
            "best_model": best_name,
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(df),
            "features": FEATURES,
            "categorical_features": CAT_FEATURES,
            "numerical_features": NUM_FEATURES,
            "candidates_compared": comparison,
        }
        meta_path = meta_dir / f"{key}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("Saved '{}' model ({}) -> {}", key, best_name, model_path)


def train_pipeline(
    data_path: str = "data/processed/processed_latest.parquet",
) -> Dict:
    """
    End-to-end training pipeline: per-brand best model + global fallback.

    Args:
        data_path: Path to cleaned data.

    Returns:
        Training summary (which algorithm won per brand, sample counts, R^2).
    """
    trainer = ModelTrainer()
    df = trainer.load_data(data_path)

    if len(df) < 100:
        raise ValueError(f"Insufficient data: {len(df)} rows (need >= 100)")

    summary = trainer.train_all(df)

    logger.info("Training pipeline complete!")
    return summary
