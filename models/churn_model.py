"""
models/churn_model.py
=====================
Churn Prediction using XGBoost with:
  - Optuna hyperparameter optimization
  - SHAP explainability
  - MLflow experiment tracking
  - SMOTE for class imbalance
  - Threshold optimization for F1 score
"""

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
import mlflow
import mlflow.xgboost
import optuna
from typing import Dict, Any, Optional
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, average_precision_score
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# Features used for churn prediction
CHURN_FEATURES = [
    "recency_days", "frequency", "monetary", "avg_order_value",
    "avg_review_score", "pct_late_deliveries", "avg_delivery_days",
    "n_product_categories", "avg_installments", "weekend_order_ratio",
    "tenure_days", "avg_inter_purchase_days", "r_score", "f_score",
    "m_score", "rfm_total", "n_reviews_with_text"
]


class ChurnPredictor:
    """
    XGBoost-based customer churn predictor with MLflow tracking.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment for tracking.
    n_trials : int
        Number of Optuna trials for hyperparameter search.
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        experiment_name: str = "churn_prediction",
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model: Optional[XGBClassifier] = None
        self.best_threshold: float = 0.5
        self.scaler: Optional[StandardScaler] = None
        self._eval_results: Dict[str, Any] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> "ChurnPredictor":
        """
        Full training pipeline:
          1. Prepare features & labels
          2. Handle class imbalance with SMOTE
          3. Hyperparameter optimization with Optuna
          4. Final model training with best params
          5. Threshold optimization
          6. Log everything to MLflow

        Parameters
        ----------
        df : pd.DataFrame
            Master features table containing 'churn' column.
        """
        logger.info("Training ChurnPredictor...")

        # ── 1. Prepare data ───────────────────────────────────────────────
        X, y = self._prepare_data(df)
        logger.info(f"  Features: {X.shape[1]} | Samples: {X.shape[0]:,} | "
                    f"Churn rate: {y.mean():.1%}")

        # ── 2. Scale features ─────────────────────────────────────────────
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # ── 3. Handle class imbalance with SMOTE ──────────────────────────
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_scaled, y)
        logger.info(f"  After SMOTE: {X_res.shape[0]:,} samples "
                    f"(balanced from {y.mean():.1%} churn rate)")

        # ── 4. Hyperparameter optimization ────────────────────────────────
        logger.info(f"  Running Optuna ({self.n_trials} trials)...")
        best_params = self._optimize_hyperparameters(X_res, y_res)
        logger.info(f"  Best params: {best_params}")

        # ── 5. Train final model ──────────────────────────────────────────
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name="xgboost_churn_final"):
            self.model = XGBClassifier(
                **best_params,
                random_state=self.random_state,
                eval_metric="auc",
                use_label_encoder=False
            )
            self.model.fit(X_res, y_res)

            # ── 6. Threshold optimization on original (unbalanced) data ──
            probas = self.model.predict_proba(X_scaled)[:, 1]
            self.best_threshold = self._optimize_threshold(probas, y)
            logger.info(f"  Optimal classification threshold: {self.best_threshold:.3f}")

            # ── 7. Evaluate ───────────────────────────────────────────────
            self._eval_results = self._compute_metrics(probas, y, self.best_threshold)

            # ── 8. Log to MLflow ──────────────────────────────────────────
            mlflow.log_params(best_params)
            mlflow.log_params({"threshold": self.best_threshold, "n_features": X.shape[1]})
            mlflow.log_metrics(self._eval_results)
            mlflow.xgboost.log_model(self.model, "churn_model")
            logger.info(f"  MLflow run logged: ROC-AUC = {self._eval_results['roc_auc']:.4f}")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary churn predictions (0 or 1)."""
        probas = self.predict_proba(df)
        return (probas >= self.best_threshold).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return churn probability for each customer (0.0 to 1.0)."""
        self._check_fitted()
        X = df[CHURN_FEATURES].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def evaluate(self) -> Dict[str, float]:
        """Return the evaluation metrics dict from the last training run."""
        return self._eval_results

    def explain(self, df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate SHAP values for feature importance explanation.

        SHAP (SHapley Additive exPlanations) tells us how much each
        feature contributed to a specific prediction. Critical for
        explaining model decisions to non-technical stakeholders.

        Returns
        -------
        pd.DataFrame
            Mean absolute SHAP values per feature, sorted descending.
        """
        self._check_fitted()
        X = df[CHURN_FEATURES].fillna(0).head(n_samples)
        X_scaled = self.scaler.transform(X)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)

        importance = pd.DataFrame({
            "feature": CHURN_FEATURES,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        return importance

    def save(self, path: str):
        """Serialize model + scaler + threshold to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "threshold": self.best_threshold,
                "features": CHURN_FEATURES,
                "eval_results": self._eval_results
            }, f)
        logger.info(f"  ChurnPredictor saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "ChurnPredictor":
        """Load a previously saved ChurnPredictor."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        predictor = cls()
        predictor.model = data["model"]
        predictor.scaler = data["scaler"]
        predictor.best_threshold = data["threshold"]
        predictor._eval_results = data["eval_results"]
        return predictor

    # ── Private helpers ────────────────────────────────────────────────────────

    def _prepare_data(self, df: pd.DataFrame):
        available_features = [f for f in CHURN_FEATURES if f in df.columns]
        missing = set(CHURN_FEATURES) - set(available_features)
        if missing:
            logger.warning(f"  Missing features (will use 0): {missing}")

        X = df[available_features].fillna(0)
        y = df["churn"].values
        return X, y

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optuna-based hyperparameter search maximizing cross-validated AUC."""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            }
            clf = XGBClassifier(
                **params,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="auc"
            )
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def _optimize_threshold(self, probas: np.ndarray, y: np.ndarray) -> float:
        """Find probability threshold that maximizes F1 score."""
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.2, 0.8, 0.01):
            preds = (probas >= thresh).astype(int)
            f1 = f1_score(y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        return best_thresh

    def _compute_metrics(self, probas: np.ndarray, y: np.ndarray, threshold: float) -> Dict[str, float]:
        preds = (probas >= threshold).astype(int)
        return {
            "roc_auc":          round(roc_auc_score(y, probas), 4),
            "avg_precision":    round(average_precision_score(y, probas), 4),
            "precision":        round(precision_score(y, preds, zero_division=0), 4),
            "recall":           round(recall_score(y, preds, zero_division=0), 4),
            "f1":               round(f1_score(y, preds, zero_division=0), 4),
        }

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")
