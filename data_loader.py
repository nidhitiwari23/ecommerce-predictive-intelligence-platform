"""
data_loader.py
==============
Handles loading all raw datasets with schema validation.

Datasets expected in data/raw/:
  olist_orders_dataset.csv
  olist_order_items_dataset.csv
  olist_customers_dataset.csv
  olist_products_dataset.csv
  olist_order_reviews_dataset.csv
  olist_order_payments_dataset.csv
  olist_sellers_dataset.csv
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# ── Expected columns for each file (used for validation) ──────────────────────
SCHEMAS: Dict[str, list] = {
    "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp",
                "order_approved_at", "order_delivered_carrier_date",
                "order_delivered_customer_date", "order_estimated_delivery_date"],
    "order_items": ["order_id", "order_item_id", "product_id", "seller_id",
                    "shipping_limit_date", "price", "freight_value"],
    "customers": ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                  "customer_city", "customer_state"],
    "products": ["product_id", "product_category_name", "product_name_length",
                 "product_description_length", "product_photos_qty",
                 "product_weight_g", "product_length_cm", "product_height_cm",
                 "product_width_cm"],
    "reviews": ["review_id", "order_id", "review_score", "review_comment_title",
                "review_comment_message", "review_creation_date",
                "review_answer_timestamp"],
    "payments": ["order_id", "payment_sequential", "payment_type",
                 "payment_installments", "payment_value"],
    "sellers": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
}


class DataLoader:
    """
    Loads all raw CSV datasets with validation and logging.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing raw CSV files.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self._validate_dir()

    def _validate_dir(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                "Please create it and place the Olist CSV files inside."
            )

    def _load_csv(self, filename: str, schema_key: str) -> pd.DataFrame:
        """Load a single CSV file and validate its columns."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}. Returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"  Loaded {filename}: {df.shape[0]:,} rows, {df.shape[1]} columns")

        # Column validation
        expected = set(SCHEMAS[schema_key])
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            logger.warning(f"  Missing expected columns in {filename}: {missing}")

        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets and return as a dictionary.

        Returns
        -------
        dict
            Keys: 'orders', 'order_items', 'customers', 'products',
                  'reviews', 'payments', 'sellers'
        """
        logger.info(f"Loading datasets from: {self.data_dir}")
        datasets = {
            "orders":      self._load_csv("olist_orders_dataset.csv", "orders"),
            "order_items": self._load_csv("olist_order_items_dataset.csv", "order_items"),
            "customers":   self._load_csv("olist_customers_dataset.csv", "customers"),
            "products":    self._load_csv("olist_products_dataset.csv", "products"),
            "reviews":     self._load_csv("olist_order_reviews_dataset.csv", "reviews"),
            "payments":    self._load_csv("olist_order_payments_dataset.csv", "payments"),
            "sellers":     self._load_csv("olist_sellers_dataset.csv", "sellers"),
        }

        # Log total memory usage
        total_mb = sum(df.memory_usage(deep=True).sum() / 1e6
                       for df in datasets.values() if not df.empty)
        logger.info(f"Total memory used by raw data: {total_mb:.1f} MB")

        return datasets
