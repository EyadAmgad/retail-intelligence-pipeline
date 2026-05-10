"""
clean_orders.py
---------------
Cleans and enriches the Olist orders dataset by:
  1. Parsing all timestamp columns to datetime
  2. Computing delivery metrics (actual days, estimated days, delay, is_late)
  3. Extracting date parts for the dim_date join
  4. Flagging non-delivered orders without dropping them

Input files:
    data/raw/olist_orders_dataset.csv

Output:
    data/processed/clean_orders.parquet

Usage:
    python transformation/clean_orders.py
"""

import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

VALID_STATUSES = {
    "delivered", "shipped", "canceled", "unavailable",
    "invoiced", "processing", "created", "approved",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load and parse timestamps
# ---------------------------------------------------------------------------

def load_orders(path: Path) -> pd.DataFrame:
    log.info("Loading orders from %s", path)
    orders = pd.read_csv(path, dtype={"order_id": str, "customer_id": str})

    log.info("Raw shape: %s", orders.shape)

    for col in TIMESTAMP_COLS:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
        n_null = orders[col].isnull().sum()
        if n_null > 0:
            log.info("  %-40s → %d nulls (expected for non-delivered orders)", col, n_null)

    return orders


# ---------------------------------------------------------------------------
# Step 2 — Validate order status
# ---------------------------------------------------------------------------

def validate_status(orders: pd.DataFrame) -> pd.DataFrame:
    invalid = ~orders["order_status"].isin(VALID_STATUSES)
    if invalid.any():
        log.warning(
            "%d rows have unexpected order_status: %s",
            invalid.sum(),
            orders.loc[invalid, "order_status"].unique().tolist(),
        )

    log.info(
        "Order status distribution:\n%s",
        orders["order_status"].value_counts().to_string(),
    )
    return orders


# ---------------------------------------------------------------------------
# Step 3 — Compute delivery metrics
# ---------------------------------------------------------------------------

def compute_delivery_metrics(orders: pd.DataFrame) -> pd.DataFrame:
    """
    All metrics are computed only for rows where the required timestamps
    exist. For non-delivered orders the columns remain NaN — they are NOT
    dropped. This preserves cancelled/processing orders for status analysis
    in the notebooks.
    """

    # Time from purchase to carrier handoff
    orders["approval_time_hours"] = (
        (orders["order_approved_at"] - orders["order_purchase_timestamp"])
        .dt.total_seconds() / 3600
    ).round(2)

    # Actual end-to-end delivery time in days
    orders["delivery_days_actual"] = (
        (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"])
        .dt.total_seconds() / 86400
    ).round(2)

    # What the platform promised at purchase time
    orders["delivery_days_estimated"] = (
        (orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"])
        .dt.total_seconds() / 86400
    ).round(2)

    # Positive = late, negative = early delivery
    orders["delay_days"] = (
        orders["delivery_days_actual"] - orders["delivery_days_estimated"]
    ).round(2)

    orders["is_late"] = orders["delay_days"] > 0

    # Sanity check: negative delivery times indicate data issues
    n_negative = (orders["delivery_days_actual"] < 0).sum()
    if n_negative > 0:
        log.warning("%d rows have negative delivery_days_actual — check timestamps", n_negative)

    delivered = orders["order_status"] == "delivered"
    log.info(
        "Delivered orders: %d | Late: %d (%.1f%% of delivered)",
        delivered.sum(),
        orders.loc[delivered, "is_late"].sum(),
        100 * orders.loc[delivered, "is_late"].mean(),
    )

    return orders


# ---------------------------------------------------------------------------
# Step 4 — Extract date parts
# ---------------------------------------------------------------------------

def extract_date_parts(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar attributes from order_purchase_timestamp.
    These are used to join with dim_date in the star schema.
    """
    ts = orders["order_purchase_timestamp"]

    orders["order_year"]      = ts.dt.year
    orders["order_month"]     = ts.dt.month
    orders["order_quarter"]   = ts.dt.quarter
    orders["order_dayofweek"] = ts.dt.dayofweek        # 0=Mon, 6=Sun
    orders["order_dayofweek_name"] = ts.dt.day_name()
    orders["is_weekend"]      = orders["order_dayofweek"].isin([5, 6])
    orders["order_hour"]      = ts.dt.hour             # useful for time-of-day analysis

    return orders


# ---------------------------------------------------------------------------
# Step 5 — Final column selection and quality report
# ---------------------------------------------------------------------------

def finalize(orders: pd.DataFrame) -> pd.DataFrame:
    col_order = [
        # Keys
        "order_id", "customer_id", "order_status",
        # Raw timestamps
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
        # Derived metrics
        "approval_time_hours",
        "delivery_days_actual", "delivery_days_estimated",
        "delay_days", "is_late",
        # Date parts
        "order_year", "order_month", "order_quarter",
        "order_dayofweek", "order_dayofweek_name",
        "is_weekend", "order_hour",
    ]
    orders = orders[col_order]

    log.info("Final shape: %s", orders.shape)
    log.info(
        "Null summary:\n%s",
        orders.isnull().sum()[orders.isnull().sum() > 0].to_string(),
    )

    return orders


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    orders = load_orders(RAW_DIR / "olist_orders_dataset.csv")
    orders = validate_status(orders)
    orders = compute_delivery_metrics(orders)
    orders = extract_date_parts(orders)
    orders = finalize(orders)

    out = PROCESSED_DIR / "clean_orders.parquet"
    orders.to_parquet(out, index=False)
    log.info("Saved clean_orders → %s", out)


if __name__ == "__main__":
    main()