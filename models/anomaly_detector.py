"""
models/anomaly_detector.py
==========================
Transaction Anomaly Detection using Isolation Forest + Z-Score.

What is an anomaly?
  An anomaly is a data point that does NOT follow the normal pattern.
  In e-commerce, anomalies can mean:

  1. FRAUD          — someone using a stolen credit card places 20 orders
                      in one minute at midnight
  2. BILLING ERROR  — freight cost is higher than the product price itself
  3. SYSTEM GLITCH  — same order charged twice (duplicate transaction)
  4. BULK BUYER     — a wholesaler buying 500 units of one product
                      (not fraud, but unusual — needs review)

How Isolation Forest works (simple explanation):
  Imagine a forest of decision trees, each randomly splitting data.
  Normal points are surrounded by many similar points — they need
  MANY random splits before they are "isolated" (alone in a region).
  Anomalies are rare and different — they get isolated in VERY FEW splits.
  Short path to isolation = anomaly.
  Long path to isolation = normal.

Why Isolation Forest?
  - Does NOT need labelled fraud data to train (unsupervised)
  - Scales well to large datasets
  - Works well in high-dimensional feature spaces
  - Fast and memory-efficient
  - Industry standard for unsupervised anomaly detection

We also add Z-Score detection as a second layer:
  Z-Score measures how many standard deviations a value is from the mean.
  If Z-Score > 3 (more than 3 std devs away), it is flagged as anomalous.
"""

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ── Features used for anomaly detection ───────────────────────────────────────
# These capture the financial and behavioural dimensions of each transaction
ANOMALY_FEATURES = [
    "monetary",            # Total spend — very high = suspicious
    "avg_order_value",     # Average order value — unusually high = suspicious
    "total_freight",       # Total freight paid — should be proportional to order value
    "frequency",           # Order frequency — very high in short time = suspicious
    "recency_days",        # Very new customer with very high spending = suspicious
    "avg_installments",    # Very high installments = financial strain (not necessarily fraud but unusual)
    "n_product_categories",# Buying from every category at once = unusual
    "pct_late_deliveries", # 100% late deliveries = might be a quality/fraud issue
]


class AnomalyDetector:
    """
    Detects anomalous transactions and customers using:
      1. Isolation Forest — the primary detection algorithm
      2. Z-Score detection — secondary layer for obvious statistical outliers

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in the dataset.
        0.02 means we expect about 2% of records to be anomalous.
        This affects where the decision boundary is drawn.
        For fraud detection, typically 0.01 to 0.05.
    z_score_threshold : float
        Z-score above which a feature value is flagged as anomalous.
        Standard is 3.0 (3 standard deviations from mean).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.02,
        z_score_threshold: float = 3.0,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.z_score_threshold = z_score_threshold
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self._eval_results: Dict = {}
        self._anomaly_reasons: Optional[pd.DataFrame] = None

    # ── Public methods ─────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """
        Train the Isolation Forest on the customer feature data.

        Parameters
        ----------
        df : pd.DataFrame
            Master customer feature table.
        """
        logger.info("Training Anomaly Detector (Isolation Forest)...")

        X = self._get_feature_matrix(df)
        logger.info(f"  Training on {X.shape[0]:,} records, {X.shape[1]} features")

        # Scale features — important so no single feature dominates
        # (e.g., monetary in hundreds vs frequency in single digits)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Isolation Forest
        # n_estimators=200: use 200 trees (more = more stable results)
        # max_samples='auto': each tree uses min(256, n_samples) samples
        # contamination: expected fraction of anomalies
        # n_jobs=-1: use all CPU cores for speed
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled)

        # Get anomaly scores for evaluation
        scores = self.model.decision_function(X_scaled)
        labels = self.model.predict(X_scaled)  # -1 = anomaly, 1 = normal

        n_anomalies = (labels == -1).sum()
        logger.info(f"  Anomalies detected: {n_anomalies:,} "
                    f"({n_anomalies/len(labels):.1%} of all records)")
        logger.info(f"  Anomaly score range: {scores.min():.3f} to {scores.max():.3f}")

        self._eval_results = {
            "n_anomalies":          int(n_anomalies),
            "anomaly_rate":         round(n_anomalies / len(labels), 4),
            "avg_anomaly_score":    round(float(scores[labels == -1].mean()), 4),
            "contamination_used":   self.contamination
        }

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies and return the DataFrame with anomaly columns added.

        Columns added:
          is_anomaly          — 1 if anomalous, 0 if normal
          anomaly_score       — continuous score (more negative = more anomalous)
          anomaly_severity    — 'High', 'Medium', 'Low', or 'Normal'
          anomaly_reason      — which feature triggered the flag
          z_score_flag        — 1 if Z-Score method also flagged this record

        Returns
        -------
        pd.DataFrame
            Input DataFrame with anomaly columns appended.
        """
        self._check_fitted()

        X = self._get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)

        # Isolation Forest predictions
        # predict() returns: 1 = normal, -1 = anomaly
        # Convert to: 0 = normal, 1 = anomaly
        raw_labels = self.model.predict(X_scaled)
        scores     = self.model.decision_function(X_scaled)

        result = df.copy()
        result["is_anomaly"]    = (raw_labels == -1).astype(int)
        result["anomaly_score"] = scores

        # Severity levels based on score
        # More negative score = more anomalous = higher severity
        score_25th = np.percentile(scores[raw_labels == -1], 25) if (raw_labels == -1).any() else -0.1
        score_50th = np.percentile(scores[raw_labels == -1], 50) if (raw_labels == -1).any() else -0.05

        def get_severity(row):
            if row["is_anomaly"] == 0:
                return "Normal"
            elif row["anomaly_score"] < score_25th:
                return "High"
            elif row["anomaly_score"] < score_50th:
                return "Medium"
            else:
                return "Low"

        result["anomaly_severity"] = result.apply(get_severity, axis=1)

        # Z-Score secondary detection
        z_flags = self._z_score_detection(df)
        result["z_score_flag"] = z_flags

        # Anomaly reason (which feature is most extreme)
        available_features = [f for f in ANOMALY_FEATURES if f in df.columns]
        result["anomaly_reason"] = self._explain_anomaly(df, raw_labels, available_features)

        n_high   = (result["anomaly_severity"] == "High").sum()
        n_medium = (result["anomaly_severity"] == "Medium").sum()
        n_low    = (result["anomaly_severity"] == "Low").sum()
        logger.info(f"  Anomaly breakdown — High: {n_high} | Medium: {n_medium} | Low: {n_low}")

        return result

    def get_anomaly_summary(self, df_with_anomalies: pd.DataFrame) -> pd.DataFrame:
        """
        Return a clean summary table of only the flagged anomalies.
        Sorted from most severe to least severe.
        """
        cols = ["is_anomaly", "anomaly_score", "anomaly_severity",
                "anomaly_reason", "z_score_flag"]
        available_cols = [c for c in cols if c in df_with_anomalies.columns]

        # Add useful context columns if they exist
        context_cols = ["customer_unique_id", "monetary", "avg_order_value",
                        "frequency", "recency_days"]
        for c in context_cols:
            if c in df_with_anomalies.columns:
                available_cols = [c] + available_cols

        anomalies = df_with_anomalies[
            df_with_anomalies["is_anomaly"] == 1
        ][available_cols].sort_values("anomaly_score")

        return anomalies.reset_index(drop=True)

    def evaluate(self) -> Dict:
        """Return detection statistics from the last fit() call."""
        return self._eval_results

    def save(self, path: str):
        """Save model and scaler to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":               self.model,
                "scaler":              self.scaler,
                "contamination":       self.contamination,
                "z_score_threshold":   self.z_score_threshold,
                "eval_results":        self._eval_results
            }, f)
        logger.info(f"  AnomalyDetector saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        """Load a previously saved AnomalyDetector from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        detector = cls(
            contamination=data["contamination"],
            z_score_threshold=data["z_score_threshold"]
        )
        detector.model          = data["model"]
        detector.scaler         = data["scaler"]
        detector._eval_results  = data["eval_results"]
        return detector

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract available anomaly features and fill missing values with 0."""
        available = [f for f in ANOMALY_FEATURES if f in df.columns]
        if len(available) < len(ANOMALY_FEATURES):
            missing = set(ANOMALY_FEATURES) - set(available)
            logger.warning(f"  Missing anomaly features (using 0): {missing}")
        return df[available].fillna(0).values

    def _z_score_detection(self, df: pd.DataFrame) -> np.ndarray:
        """
        Secondary anomaly detection using Z-Score method.

        Z-Score = (value - mean) / standard_deviation
        If |Z-Score| > threshold (default 3.0) for ANY feature,
        that record is flagged.

        This catches obvious statistical outliers that are easy to explain
        to non-technical stakeholders ('this order value is 4 standard
        deviations above average').
        """
        available = [f for f in ANOMALY_FEATURES if f in df.columns]
        X = df[available].fillna(0)

        # Calculate Z-score for each feature
        z_scores = np.abs((X - X.mean()) / (X.std() + 1e-8))

        # A record is flagged if ANY feature has Z-Score above threshold
        flagged = (z_scores > self.z_score_threshold).any(axis=1)
        return flagged.astype(int).values

    def _explain_anomaly(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        features: List[str]
    ) -> List[str]:
        """
        For each anomaly, identify which feature is most extreme
        compared to the normal population.
        This gives the human-readable reason for the flag.

        Example output:
          'monetary is 4.2 std devs above normal'
          'frequency is 3.8 std devs above normal'
        """
        reasons = []
        X = df[features].fillna(0)
        normal_mask = labels == 1  # 1 = normal in Isolation Forest

        # Calculate mean and std of NORMAL records for each feature
        if normal_mask.sum() > 0:
            normal_mean = X[normal_mask].mean()
            normal_std  = X[normal_mask].std() + 1e-8
        else:
            normal_mean = X.mean()
            normal_std  = X.std() + 1e-8

        for i, (_, row) in enumerate(X.iterrows()):
            if labels[i] == 1:  # Normal
                reasons.append("Normal")
            else:
                # Find which feature deviates most from the normal mean
                z = abs((row - normal_mean) / normal_std)
                top_feature = z.idxmax()
                top_z = z[top_feature]
                direction = "above" if row[top_feature] > normal_mean[top_feature] else "below"
                reasons.append(
                    f"{top_feature} is {top_z:.1f} std devs {direction} normal"
                )

        return reasons

    def _check_fitted(self):
        """Raise error if model not trained yet."""
        if self.model is None:
            raise RuntimeError(
                "Model not fitted yet. Call .fit(df) before calling .predict()."
            )
