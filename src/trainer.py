"""
ML Training Pipeline - ICMI v2.0
Unified CatBoost model for all brands.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


class ModelTrainer:
    """
    Train a unified CatBoost model across all brands.

    CatBoost handles categorical features natively — no encoding needed.
    This eliminates the target leakage from price-based ordinal encoding.
    """

    # Features used for training
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

    def __init__(self):
        self.model: CatBoostRegressor | None = None
        self.metadata: Dict = {}

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

        # Ensure categoricals are strings
        for col in self.CAT_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Drop rows with missing features or target
        df = df.dropna(subset=self.FEATURES + [self.TARGET])

        logger.info(
            "Loaded data | {} rows | {} brands | features: {}",
            len(df),
            df["brand_slug"].nunique(),
            len(self.FEATURES),
        )
        return df

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Tuple[CatBoostRegressor, Dict]:
        """
        Train model with holdout test set and cross-validation.

        Args:
            df: Training DataFrame.
            test_size: Fraction for test set.
            cv_folds: Number of CV folds.

        Returns:
            Tuple of (trained model, metadata dict).
        """
        X = df[self.FEATURES]
        y = df[self.TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        logger.info(
            "Train split | train={} | test={}",
            len(X_train),
            len(X_test),
        )

        # CatBoost categorical feature indices
        cat_indices = [
            i
            for i, col in enumerate(self.FEATURES)
            if col in self.CAT_FEATURES
        ]

        self.model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            early_stopping_rounds=100,
            verbose=100,
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            cat_features=cat_indices,
            use_best_model=True,
        )

        # Evaluate on holdout
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        logger.info("Running {}-fold cross-validation...", cv_folds)
        cv_scores = cross_val_score(
            self.model,
            X,
            y,
            cv=cv_folds,
            scoring="r2",
            fit_params={"cat_features": cat_indices},
        )

        # Feature importance
        importance = dict(
            zip(
                self.FEATURES,
                self.model.get_feature_importance().tolist(),
            )
        )

        self.metadata = {
            "model_type": "CatBoostRegressor",
            "trained_at": datetime.now().isoformat(),
            "features": self.FEATURES,
            "categorical_features": self.CAT_FEATURES,
            "numerical_features": self.NUM_FEATURES,
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_brands": df["brand_slug"].nunique(),
            "brands": df["brand_slug"].value_counts().to_dict(),
            "performance": {
                "test_mae": float(mae),
                "test_mae_million": round(float(mae / 1e6), 2),
                "test_r2": round(float(r2), 4),
                "cv_r2_mean": round(float(cv_scores.mean()), 4),
                "cv_r2_std": round(float(cv_scores.std()), 4),
            },
            "feature_importance": {
                k: round(v, 4)
                for k, v in sorted(
                    importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            },
        }

        logger.info(
            "Test MAE: {:.2f}M Toman | Test R²: {:.4f}", mae / 1e6, r2
        )
        logger.info(
            "CV R²: {:.4f} (+/- {:.4f})",
            cv_scores.mean(),
            cv_scores.std(),
        )

        return self.model, self.metadata

    def save(self, output_dir: str = "artifacts") -> Dict[str, str]:
        """
        Save model, metadata, and feature list.

        Args:
            output_dir: Base directory for model artifacts.

        Returns:
            Dictionary of saved file paths.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")

        models_dir = Path(output_dir) / "models"
        scalers_dir = Path(output_dir) / "scalers"
        meta_dir = Path(output_dir) / "metadata"

        for d in [models_dir, scalers_dir, meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}

        # Save model
        model_path = models_dir / f"catboost_{timestamp}.cbm"
        self.model.save_model(str(model_path))
        paths["model"] = str(model_path)

        # Save metadata
        meta_path = meta_dir / f"metadata_{timestamp}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        paths["metadata"] = str(meta_path)

        # Save feature list
        feat_path = meta_dir / f"features_{timestamp}.joblib"
        joblib.dump(
            {
                "features": self.FEATURES,
                "cat_features": self.CAT_FEATURES,
                "num_features": self.NUM_FEATURES,
            },
            feat_path,
        )
        paths["features"] = str(feat_path)

        # Update "latest" symlinks (or copy for Windows compatibility)
        for suffix, path in [
            ("model", model_path),
            ("metadata", meta_path),
            ("features", feat_path),
        ]:
            latest = models_dir / f"latest_{suffix}"
            if suffix in ("metadata", "features"):
                latest = meta_dir / f"latest_{suffix}"

            if latest.exists() or latest.is_symlink():
                latest.unlink()

            try:
                latest.symlink_to(path.name)
            except OSError:
                import shutil

                shutil.copy2(path, latest)

        logger.info("Saved model artifacts to {}", output_dir)
        return paths


def train_pipeline(
    data_path: str = "data/processed/processed_latest.parquet",
) -> Dict[str, str]:
    """
    End-to-end training pipeline.

    Args:
        data_path: Path to cleaned data.

    Returns:
        Paths to saved artifacts.
    """
    trainer = ModelTrainer()
    df = trainer.load_data(data_path)

    if len(df) < 100:
        raise ValueError(f"Insufficient data: {len(df)} rows (need >= 100)")

    trainer.train(df)
    paths = trainer.save()

    logger.info("Training pipeline complete!")
    return paths
