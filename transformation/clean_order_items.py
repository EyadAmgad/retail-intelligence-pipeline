"""
clean_order_items.py
--------------------
Cleans and enriches the Olist order items dataset by:
  1. Parsing shipping_limit_date as datetime
  2. Computing item-level revenue columns
  3. Flagging price outliers
  4. Building an order-level aggregation table

Input files:
    data/raw/olist_order_items_dataset.csv

Output:
    data/processed/clean_order_items.parquet   (item grain)
    data/processed/order_items_agg.parquet     (order grain aggregation)

Usage:
    python transformation/clean_order_items.py
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

# Anything above this price is flagged — not dropped
PRICE_OUTLIER_THRESHOLD = 5000.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------

def load_items(path: Path) -> pd.DataFrame:
    log.info("Loading order items from %s", path)
    items = pd.read_csv(
        path,
        dtype={
            "order_id":      str,
            "product_id":    str,
            "seller_id":     str,
            "order_item_id": int,
        },
    )
    log.info("Raw shape: %s", items.shape)

    items["shipping_limit_date"] = pd.to_datetime(
        items["shipping_limit_date"], errors="coerce"
    )

    null_dates = items["shipping_limit_date"].isnull().sum()
    if null_dates > 0:
        log.warning("%d null shipping_limit_date values", null_dates)

    return items


# ---------------------------------------------------------------------------
# Step 2 — Validate and flag
# ---------------------------------------------------------------------------

def validate_prices(items: pd.DataFrame) -> pd.DataFrame:
    # Negative prices are data errors
    n_neg_price   = (items["price"] < 0).sum()
    n_neg_freight = (items["freight_value"] < 0).sum()
    if n_neg_price > 0:
        log.warning("%d rows with negative price", n_neg_price)
    if n_neg_freight > 0:
        log.warning("%d rows with negative freight_value", n_neg_freight)

    # Outlier flag — keep rows, just label them
    items["is_price_outlier"] = items["price"] > PRICE_OUTLIER_THRESHOLD
    n_outliers = items["is_price_outlier"].sum()
    log.info(
        "Price outliers (> R$%.0f): %d rows — keeping with flag",
        PRICE_OUTLIER_THRESHOLD,
        n_outliers,
    )

    log.info(
        "Price stats:\n  min=%.2f | median=%.2f | mean=%.2f | max=%.2f",
        items["price"].min(),
        items["price"].median(),
        items["price"].mean(),
        items["price"].max(),
    )

    return items


# ---------------------------------------------------------------------------
# Step 3 — Compute item-level revenue columns
# ---------------------------------------------------------------------------

def compute_item_revenue(items: pd.DataFrame) -> pd.DataFrame:
    """
    Each row is one item within an order. An order can have multiple items
    from different sellers (order_item_id sequences within order_id).
    """
    items["total_item_value"] = (items["price"] + items["freight_value"]).round(2)

    log.info(
        "Multi-item orders: %d orders contain more than one item",
        (items.groupby("order_id")["order_item_id"].max() > 1).sum(),
    )

    return items


# ---------------------------------------------------------------------------
# Step 4 — Build order-level aggregation
# ---------------------------------------------------------------------------

def aggregate_to_order(items: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse item rows to one row per order_id.
    This aggregation is joined onto fact_orders in build_star_schema.py.
    """
    agg = (
        items.groupby("order_id", as_index=False)
        .agg(
            n_items=("order_item_id", "max"),           # max item sequence = item count
            order_total_price=("price", "sum"),
            order_total_freight=("freight_value", "sum"),
            order_total_value=("total_item_value", "sum"),
            n_unique_sellers=("seller_id", "nunique"),
            n_unique_products=("product_id", "nunique"),
            has_price_outlier=("is_price_outlier", "any"),
        )
    )

    agg["order_total_price"]   = agg["order_total_price"].round(2)
    agg["order_total_freight"] = agg["order_total_freight"].round(2)
    agg["order_total_value"]   = agg["order_total_value"].round(2)

    log.info("Order-level aggregation shape: %s", agg.shape)
    log.info(
        "GMV total: R$ {:,.2f}".format(agg["order_total_value"].sum())
    )

    return agg


# ---------------------------------------------------------------------------
# Step 5 — Finalize item-level output
# ---------------------------------------------------------------------------

def finalize_items(items: pd.DataFrame) -> pd.DataFrame:
    col_order = [
        "order_id", "order_item_id", "product_id", "seller_id",
        "shipping_limit_date", "price", "freight_value",
        "total_item_value", "is_price_outlier",
    ]
    return items[col_order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    items = load_items(RAW_DIR / "olist_order_items_dataset.csv")
    items = validate_prices(items)
    items = compute_item_revenue(items)

    agg = aggregate_to_order(items)
    items = finalize_items(items)

    item_out = PROCESSED_DIR / "clean_order_items.parquet"
    agg_out  = PROCESSED_DIR / "order_items_agg.parquet"

    items.to_parquet(item_out, index=False)
    agg.to_parquet(agg_out, index=False)

    log.info("Saved clean_order_items → %s", item_out)
    log.info("Saved order_items_agg   → %s", agg_out)


if __name__ == "__main__":
    main()