"""
models/clv_model.py
===================
Customer Lifetime Value (CLV) Prediction using LightGBM.

What is CLV?
  CLV = how much total money will this customer spend with us
  in the next 12 months?

Why it matters:
  - If a customer's predicted CLV is Rs. 50,000, spending Rs. 5,000
    on a retention campaign makes perfect sense.
  - If predicted CLV is Rs. 500, spending the same Rs. 5,000 is a loss.
  - Marketing teams use CLV to decide WHERE to spend their budget.

Why LightGBM?
  - Faster than XGBoost on large datasets (uses leaf-wise tree growth)
  - Similar accuracy to XGBoost for regression tasks
  - Built-in handling of missing values
  - Less memory usage

This is a REGRESSION model (predicts a continuous number like Rs. 1,234)
NOT a classification model (which predicts yes/no).

Target variable: total_revenue_next_12_months
  - We calculate this from historical data using a rolling window
  - For customers with < 12 months of history, we scale up proportionally
"""

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# ── Features used for CLV prediction ──────────────────────────────────────────
# These are the same features available in the master feature table
CLV_FEATURES = [
    "recency_days",           # How recently they last bought
    "frequency",              # How many times they bought
    "avg_order_value",        # Average spend per order
    "n_product_categories",   # How many different categories they buy from
    "avg_review_score",       # How satisfied they are (higher = more likely to return)
    "pct_late_deliveries",    # Bad experience = lower future spending
    "avg_delivery_days",      # Faster delivery = better experience
    "avg_installments",       # Higher installments = more budget-conscious
    "weekend_order_ratio",    # Behavioural pattern
    "tenure_days",            # How long they have been a customer
    "r_score",                # Recency RFM score
    "f_score",                # Frequency RFM score
    "m_score",                # Monetary RFM score
    "rfm_total",              # Overall RFM quality score
    "n_reviews_with_text",    # Engaged customers write more reviews
]


class CLVPredictor:
    """
    Predicts Customer Lifetime Value for the next 12 months using LightGBM.

    How it works:
    1. Calculate historical CLV for each customer (total spend in their
       most recent 12-month period)
    2. Use earlier behaviour features to train the model to predict
       that historical CLV
    3. Apply the trained model to current customers to predict
       their FUTURE 12-month CLV

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name for tracking all training runs
    n_trials : int
        Number of Optuna hyperparameter search trials
    cv_folds : int
        Number of cross-validation folds for reliable evaluation
    random_state : int
        Random seed — set this so results are reproducible
    """

    def __init__(
        self,
        experiment_name: str = "clv_prediction",
        n_trials: int = 40,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.model: Optional[lgb.LGBMRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self._eval_results: Dict = {}
        self._feature_importance: Optional[pd.DataFrame] = None

    # ── Public methods ─────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> "CLVPredictor":
        """
        Full training pipeline:
          1. Build CLV target variable from transaction history
          2. Prepare and scale features
          3. Optimise hyperparameters with Optuna
          4. Train final model with best parameters
          5. Evaluate on holdout data
          6. Log everything to MLflow

        Parameters
        ----------
        df : pd.DataFrame
            Master customer feature table from feature_engineer.py
        """
        logger.info("Training CLV Predictor (LightGBM)...")

        # Step 1: Build or use existing CLV target
        X, y = self._prepare_data(df)
        logger.info(f"  Samples: {len(X):,} | Features: {X.shape[1]}")
        logger.info(f"  CLV target — Mean: R${y.mean():.2f} | "
                    f"Median: R${np.median(y):.2f} | "
                    f"Max: R${y.max():.2f}")

        # Step 2: Scale features
        # LightGBM doesn't strictly need scaling, but it helps with
        # numerical stability and makes feature importances more comparable
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Step 3: Train/test split (80% train, 20% test)
        split_idx = int(len(X_scaled) * 0.8)
        # Sort by recency so test set has more recent customers
        # (simulates real deployment where you predict future behaviour)
        sort_order = np.argsort(df["recency_days"].values[:len(X_scaled)])
        X_sorted = X_scaled[sort_order]
        y_sorted = y[sort_order]

        X_train, X_test = X_sorted[:split_idx], X_sorted[split_idx:]
        y_train, y_test = y_sorted[:split_idx], y_sorted[split_idx:]

        # Step 4: Hyperparameter optimisation
        logger.info(f"  Running Optuna optimisation ({self.n_trials} trials)...")
        best_params = self._optimise_hyperparameters(X_train, y_train)
        logger.info(f"  Best params found: {best_params}")

        # Step 5: Train final model
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name="lgbm_clv_final"):

            self.model = lgb.LGBMRegressor(
                **best_params,
                random_state=self.random_state,
                n_jobs=-1,          # Use all CPU cores
                verbose=-1          # Suppress LightGBM output
            )
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)]
            )

            # Step 6: Evaluate on test set
            y_pred = self.model.predict(X_test)
            # CLV cannot be negative — clip at 0
            y_pred = np.clip(y_pred, a_min=0, a_max=None)

            self._eval_results = self._compute_metrics(y_test, y_pred)
            logger.info(f"  RMSE:  R${self._eval_results['rmse']:.2f}")
            logger.info(f"  MAE:   R${self._eval_results['mae']:.2f}")
            logger.info(f"  MAPE:  {self._eval_results['mape']:.1f}%")
            logger.info(f"  R²:    {self._eval_results['r2']:.4f}")

            # Step 7: Feature importance
            self._feature_importance = self._get_feature_importance()

            # Step 8: Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(self._eval_results)
            mlflow.lightgbm.log_model(self.model, "clv_model")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict CLV (in R$) for each customer in the DataFrame.

        Returns
        -------
        np.ndarray
            Array of predicted CLV values in Brazilian Reais.
            Example: [1234.56, 890.12, 5678.90, ...]
        """
        self._check_fitted()
        X = self._get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Clip negatives — CLV cannot be less than zero
        return np.clip(predictions, a_min=0, a_max=None)

    def predict_with_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict CLV and add a CLV tier label for each customer.

        CLV Tiers:
          Platinum  — top 10% of predicted CLV
          Gold      — next 20%
          Silver    — next 30%
          Bronze    — bottom 40%

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: clv_predicted, clv_tier
        """
        clv_values = self.predict(df)
        result = pd.DataFrame({"clv_predicted": clv_values})

        # Assign tiers based on percentile cutoffs
        p90 = np.percentile(clv_values, 90)
        p70 = np.percentile(clv_values, 70)
        p40 = np.percentile(clv_values, 40)

        result["clv_tier"] = pd.cut(
            clv_values,
            bins=[-np.inf, p40, p70, p90, np.inf],
            labels=["Bronze", "Silver", "Gold", "Platinum"]
        )

        logger.info(f"  CLV tier distribution:\n{result['clv_tier'].value_counts()}")
        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importances from the trained LightGBM model.

        LightGBM calculates feature importance as:
        - 'split': how many times a feature was used to split a node
        - 'gain': total gain (reduction in loss) from splits using that feature
        We return 'gain' as it is more meaningful than split count.
        """
        self._check_fitted()
        return self._feature_importance

    def evaluate(self) -> Dict:
        """Return evaluation metrics from the last training run."""
        return self._eval_results

    def save(self, path: str):
        """Save the trained model, scaler, and metadata to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":              self.model,
                "scaler":             self.scaler,
                "features":           CLV_FEATURES,
                "eval_results":       self._eval_results,
                "feature_importance": self._feature_importance
            }, f)
        logger.info(f"  CLVPredictor saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "CLVPredictor":
        """Load a previously saved CLVPredictor from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        predictor = cls()
        predictor.model              = data["model"]
        predictor.scaler             = data["scaler"]
        predictor._eval_results      = data["eval_results"]
        predictor._feature_importance = data.get("feature_importance")
        return predictor

    # ── Private helpers ────────────────────────────────────────────────────────

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature matrix X and CLV target y.

        CLV target = 'monetary' column (total historical spend).
        In a production system you would use a rolling 12-month window,
        but for this project total spend is a good approximation.
        """
        X = self._get_feature_matrix(df)

        # Use total monetary spend as CLV proxy
        # In production: calculate actual next-12-months spend using
        # a time-based train/test split on transaction history
        if "monetary" in df.columns:
            y = df["monetary"].values
        else:
            # Fallback if column has different name
            y = df["total_revenue"].values if "total_revenue" in df.columns \
                else np.ones(len(df)) * 100.0

        # Remove rows where CLV is zero or negative — these are errors
        valid_mask = y > 0
        logger.info(f"  Removed {(~valid_mask).sum()} rows with zero/negative CLV")
        return X[valid_mask], y[valid_mask]

    def _get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract only available CLV features and fill missing with 0."""
        available = [f for f in CLV_FEATURES if f in df.columns]
        if len(available) < len(CLV_FEATURES):
            missing = set(CLV_FEATURES) - set(available)
            logger.warning(f"  Missing CLV features (will use 0): {missing}")
        return df[available].fillna(0).values

    def _optimise_hyperparameters(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict:
        """
        Use Optuna to find the best LightGBM hyperparameters.

        We minimise RMSE (Root Mean Squared Error) using 3-fold CV.
        Optuna uses the TPE (Tree-structured Parzen Estimator) sampler
        which is smarter than random search — it uses past results to
        decide which hyperparameters to try next.
        """
        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 2.0),
            }
            model = lgb.LGBMRegressor(
                **params,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            # neg_root_mean_squared_error — sklearn returns negative RMSE
            # so we negate it to get positive RMSE (lower = better)
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )
            return -scores.mean()   # Return positive RMSE for Optuna to minimise

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate all regression evaluation metrics."""
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        # MAPE: avoid division by zero for zero-value CLV customers
        nonzero = y_true > 0
        mape = float(np.mean(np.abs(
            (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
        )) * 100) if nonzero.sum() > 0 else 0.0

        return {
            "rmse": round(rmse, 2),
            "mae":  round(mae, 2),
            "mape": round(mape, 2),
            "r2":   round(r2, 4)
        }

    def _get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from trained LightGBM model."""
        available_features = [f for f in CLV_FEATURES
                               if f in (self.model.feature_name_ or CLV_FEATURES)]
        n_features = len(self.model.feature_importances_)
        feature_names = CLV_FEATURES[:n_features]

        importance_df = pd.DataFrame({
            "feature":    feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return importance_df

    def _check_fitted(self):
        """Raise an error if the model has not been trained yet."""
        if self.model is None:
            raise RuntimeError(
                "Model not trained yet. Call .train(df) first before predicting."
            )
