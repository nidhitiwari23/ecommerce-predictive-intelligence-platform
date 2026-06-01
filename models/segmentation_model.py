"""
models/segmentation_model.py
============================
Customer Segmentation using:
  - RFM-based features
  - K-Means clustering with optimal k via Elbow + Silhouette
  - Hierarchical clustering for dendrogram visualization
  - Business-friendly segment labeling
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster

logger = logging.getLogger(__name__)


# ── Segment label mapping ──────────────────────────────────────────────────────
# Based on RFM score: high rfm_total → Champions, etc.
SEGMENT_RULES = {
    "Champions":         {"rfm_min": 11, "rfm_max": 15},
    "Potential Loyalists": {"rfm_min": 8,  "rfm_max": 10},
    "At-Risk":           {"rfm_min": 4,  "rfm_max": 7},
    "Bargain Hunters":   {"rfm_min": 0,  "rfm_max": 3},
}

CLUSTER_FEATURES = ["recency_days", "frequency", "monetary",
                     "avg_order_value", "n_product_categories",
                     "avg_review_score", "pct_late_deliveries"]


class CustomerSegmenter:
    """
    Segments customers into behavioral groups using K-Means clustering.

    The segments map to real business actions:
      Champions         → VIP treatment, early access, referral programs
      Potential Loyalists → Loyalty rewards, personalized recommendations
      At-Risk           → Win-back campaigns, special discounts
      Bargain Hunters   → Sale alerts only, low marketing spend

    Parameters
    ----------
    n_clusters : int
        Number of clusters (use find_optimal_k() to determine this).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self._cluster_profiles: Optional[pd.DataFrame] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def find_optimal_k(self, df: pd.DataFrame, k_range: range = range(2, 11)) -> Dict:
        """
        Use Elbow method + Silhouette score to find the optimal number of clusters.

        The Elbow method plots inertia (within-cluster sum of squares) vs k.
        The 'elbow' — where the rate of improvement slows — is the optimal k.

        Silhouette score measures how similar a point is to its own cluster
        compared to other clusters. Range: -1 to +1. Higher is better.

        Returns
        -------
        dict
            {'k_range': list, 'inertia': list, 'silhouette': list,
             'suggested_k': int}
        """
        X = self._prepare_features(df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        inertias, silhouettes = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))
            logger.info(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={silhouettes[-1]:.3f}")

        # Suggested k = argmax silhouette
        best_k = list(k_range)[np.argmax(silhouettes)]
        logger.info(f"  Suggested optimal k: {best_k} (highest silhouette = {max(silhouettes):.3f})")

        return {
            "k_range": list(k_range),
            "inertia": inertias,
            "silhouette": silhouettes,
            "suggested_k": best_k
        }

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit K-Means and assign cluster labels + business segment names.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with added columns:
              cluster_id      (0-based integer cluster)
              segment_label   (human-readable business name)
        """
        X = self._prepare_features(df)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # K-Means clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20,
            max_iter=500
        )
        cluster_ids = self.kmeans.fit_predict(X_scaled)

        result = df.copy()
        result["cluster_id"] = cluster_ids

        # Evaluate clustering quality
        sil = silhouette_score(X_scaled, cluster_ids)
        db  = davies_bouldin_score(X_scaled, cluster_ids)
        logger.info(f"  K-Means quality: silhouette={sil:.3f}, "
                    f"davies-bouldin={db:.3f} (lower DB is better)")

        # Assign business segment labels based on RFM total score
        result = self._assign_segment_labels(result)

        # Store cluster profiles for reporting
        self._cluster_profiles = self.get_segment_profile(result)
        logger.info(f"  Segment distribution:\n{result['segment_label'].value_counts()}")

        return result

    def get_segment_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary profile for each segment.
        Shows the average behavior of customers in each group.
        """
        if "segment_label" not in df.columns:
            raise ValueError("Run fit_predict() first to generate segment labels.")

        profile = (
            df.groupby("segment_label")
            .agg(
                n_customers=("customer_unique_id" if "customer_unique_id" in df.columns else "recency_days",
                              "count"),
                avg_recency=("recency_days", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                avg_review=("avg_review_score", "mean"),
                churn_rate=("churn", "mean") if "churn" in df.columns else ("recency_days", lambda x: np.nan),
            )
            .reset_index()
            .round(2)
        )
        return profile

    def run_hierarchical(self, df: pd.DataFrame, n_clusters: int = 4) -> np.ndarray:
        """
        Alternative: Hierarchical/Agglomerative Clustering.

        Unlike K-Means which needs k upfront, hierarchical clustering
        builds a tree (dendrogram) showing how clusters merge. Useful for
        understanding the data structure and validating K-Means results.
        """
        X = self._prepare_features(df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X_scaled)
        logger.info(f"  Hierarchical clustering done: {n_clusters} clusters")
        return labels

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "kmeans": self.kmeans,
                "scaler": self.scaler,
                "n_clusters": self.n_clusters,
                "cluster_profiles": self._cluster_profiles
            }, f)
        logger.info(f"  Segmenter saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "CustomerSegmenter":
        with open(path, "rb") as f:
            data = pickle.load(f)
        seg = cls(n_clusters=data["n_clusters"])
        seg.kmeans = data["kmeans"]
        seg.scaler = data["scaler"]
        seg._cluster_profiles = data["cluster_profiles"]
        return seg

    # ── Private helpers ────────────────────────────────────────────────────────

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [f for f in CLUSTER_FEATURES if f in df.columns]
        return df[available].fillna(0)

    def _assign_segment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map RFM total score to business segment names.
        Higher rfm_total → better customer.
        """
        def label_customer(rfm_total):
            if rfm_total >= 11:
                return "Champions"
            elif rfm_total >= 8:
                return "Potential Loyalists"
            elif rfm_total >= 4:
                return "At-Risk"
            else:
                return "Bargain Hunters"

        if "rfm_total" in df.columns:
            df["segment_label"] = df["rfm_total"].apply(label_customer)
        else:
            # Fallback: label by cluster centroid rank (highest monetary = Champions)
            if self.kmeans is not None:
                # Find which cluster has highest average monetary value
                cluster_monetary = (
                    df.groupby("cluster_id")["monetary"].mean()
                    .rank(ascending=False).astype(int)
                )
                rank_to_label = {1: "Champions", 2: "Potential Loyalists",
                                  3: "At-Risk", 4: "Bargain Hunters"}
                df["segment_label"] = df["cluster_id"].map(
                    {k: rank_to_label.get(v, "Other") for k, v in cluster_monetary.items()}
                )
        return df
