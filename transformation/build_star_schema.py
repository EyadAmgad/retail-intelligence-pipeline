"""
build_star_schema.py
--------------------
Assembles the analytics-ready star schema from all cleaned parquet files.

Reads from data/processed/:
    clean_orders.parquet
    clean_order_items.parquet
    order_items_agg.parquet
    clean_payments.parquet
    clean_reviews.parquet
    dim_customers.parquet
    dim_products.parquet
    dim_sellers.parquet

Writes to data/processed/:
    fact_orders.parquet      ← one row per order-item (the fact grain)
    dim_customers.parquet    ← already exists, untouched
    dim_products.parquet     ← already exists, untouched
    dim_sellers.parquet      ← already exists, untouched
    dim_date.parquet         ← generated from order purchase timestamps

Star schema shape:

                    dim_date
                       │
    dim_customers ─── fact_orders ─── dim_products
                       │
                    dim_sellers

Usage:
    python transformation/build_star_schema.py
"""

import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Make sure all clean_*.py scripts have been run first."
        )
    df = pd.read_parquet(path)
    log.info("Loaded %-35s → %s", name, df.shape)
    return df


def report_join_loss(before: int, after: int, step: str) -> None:
    lost = before - after
    if lost > 0:
        log.warning(
            "%s: lost %d rows (%.2f%%) after join",
            step, lost, 100 * lost / before,
        )
    else:
        log.info("%s: no rows lost ✓", step)


# ---------------------------------------------------------------------------
# Step 1 — Build dim_date
# ---------------------------------------------------------------------------

def build_dim_date(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a calendar dimension from the range of purchase timestamps.
    One row per calendar date — covers the full data date range plus buffer.
    """
    log.info("Building dim_date...")

    min_date = orders["order_purchase_timestamp"].min().date()
    max_date = orders["order_purchase_timestamp"].max().date()
    log.info("Date range in data: %s → %s", min_date, max_date)

    dates = pd.date_range(start=min_date, end=max_date, freq="D")
    dim_date = pd.DataFrame({"full_date": dates})

    dim_date["date_id"]        = dim_date["full_date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"]           = dim_date["full_date"].dt.year
    dim_date["month"]          = dim_date["full_date"].dt.month
    dim_date["month_name"]     = dim_date["full_date"].dt.month_name()
    dim_date["quarter"]        = dim_date["full_date"].dt.quarter
    dim_date["week_of_year"]   = dim_date["full_date"].dt.isocalendar().week.astype(int)
    dim_date["day_of_month"]   = dim_date["full_date"].dt.day
    dim_date["day_of_week"]    = dim_date["full_date"].dt.dayofweek       # 0=Mon
    dim_date["day_name"]       = dim_date["full_date"].dt.day_name()
    dim_date["is_weekend"]     = dim_date["day_of_week"].isin([5, 6])
    dim_date["year_month"]     = dim_date["full_date"].dt.to_period("M").astype(str)
    dim_date["year_quarter"]   = (
        dim_date["year"].astype(str) + "-Q" + dim_date["quarter"].astype(str)
    )

    log.info("dim_date shape: %s  (%d days)", dim_date.shape, len(dim_date))
    return dim_date


# ---------------------------------------------------------------------------
# Step 2 — Build fact_orders
# ---------------------------------------------------------------------------

def build_fact_orders(
    orders:     pd.DataFrame,
    items:      pd.DataFrame,
    items_agg:  pd.DataFrame,
    payments:   pd.DataFrame,
    reviews:    pd.DataFrame,
    dim_date:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Fact grain: one row per order-item (order_id + order_item_id + product_id + seller_id).

    This is the most granular level that supports:
      - per-item revenue analysis
      - per-seller performance
      - per-product analysis
    while still carrying order-level measures (payment, delivery, review).

    Denormalised measures on every row (repeated per item within an order)
    are clearly suffixed _order_ to avoid confusion.
    """
    log.info("Building fact_orders...")

    # ------------------------------------------------------------------ #
    # 2a. Start from order items (item grain)
    # ------------------------------------------------------------------ #
    fact = items.copy()
    n_start = len(fact)
    log.info("Starting from order_items: %d rows", n_start)

    # ------------------------------------------------------------------ #
    # 2b. Join core order attributes
    # ------------------------------------------------------------------ #
    order_cols = [
        "order_id", "customer_id", "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "approval_time_hours",
        "delivery_days_actual",
        "delivery_days_estimated",
        "delay_days",
        "is_late",
        "order_year", "order_month", "order_quarter",
        "order_dayofweek", "order_dayofweek_name",
        "is_weekend", "order_hour",
    ]
    fact = fact.merge(orders[order_cols], on="order_id", how="left")
    report_join_loss(n_start, len(fact), "items ← orders")

    # ------------------------------------------------------------------ #
    # 2c. Join order-level item aggregates (totals, n_items)
    # ------------------------------------------------------------------ #
    fact = fact.merge(items_agg, on="order_id", how="left")
    report_join_loss(n_start, len(fact), "items ← items_agg")

    # ------------------------------------------------------------------ #
    # 2d. Join payments
    # ------------------------------------------------------------------ #
    fact = fact.merge(payments, on="order_id", how="left")
    n_missing_payment = fact["total_payment_value"].isnull().sum()
    if n_missing_payment > 0:
        log.warning(
            "%d order-items have no payment record (%.2f%%)",
            n_missing_payment,
            100 * n_missing_payment / len(fact),
        )
    report_join_loss(n_start, len(fact), "items ← payments")

    # ------------------------------------------------------------------ #
    # 2e. Join reviews (left join — not every order has a review)
    # ------------------------------------------------------------------ #
    fact = fact.merge(reviews[[
        "order_id", "review_score", "sentiment_bucket",
        "review_response_time_hours", "has_comment_message",
    ]], on="order_id", how="left")

    n_no_review = fact["review_score"].isnull().sum()
    log.info(
        "Order-items without a review: %d (%.1f%%)",
        n_no_review,
        100 * n_no_review / len(fact),
    )

    # ------------------------------------------------------------------ #
    # 2f. Join date_id for dim_date FK
    # ------------------------------------------------------------------ #
    fact["purchase_date"] = fact["order_purchase_timestamp"].dt.normalize()
    date_lookup = dim_date[["full_date", "date_id"]].copy()
    date_lookup["full_date"] = pd.to_datetime(date_lookup["full_date"])
    fact = fact.merge(
        date_lookup,
        left_on="purchase_date",
        right_on="full_date",
        how="left",
    ).drop(columns=["full_date", "purchase_date"])

    # ------------------------------------------------------------------ #
    # 2g. Final column ordering
    # ------------------------------------------------------------------ #
    col_order = [
        # --- surrogate / natural keys ---
        "order_id",
        "order_item_id",
        "customer_id",       # FK → dim_customers
        "product_id",        # FK → dim_products
        "seller_id",         # FK → dim_sellers
        "date_id",           # FK → dim_date

        # --- order status ---
        "order_status",

        # --- timestamps ---
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date",

        # --- item-level measures ---
        "price",
        "freight_value",
        "total_item_value",
        "is_price_outlier",

        # --- order-level measures (denormalised) ---
        "n_items",
        "order_total_price",
        "order_total_freight",
        "order_total_value",
        "n_unique_sellers",
        "n_unique_products",
        "has_price_outlier",

        # --- delivery measures ---
        "approval_time_hours",
        "delivery_days_actual",
        "delivery_days_estimated",
        "delay_days",
        "is_late",

        # --- calendar attributes ---
        "order_year",
        "order_month",
        "order_quarter",
        "order_dayofweek",
        "order_dayofweek_name",
        "is_weekend",
        "order_hour",

        # --- payment measures ---
        "total_payment_value",
        "primary_payment_type",
        "n_payment_rows",
        "n_payment_types",
        "is_split_payment",
        "max_installments",
        "installment_bin",

        # --- review measures ---
        "review_score",
        "sentiment_bucket",
        "review_response_time_hours",
        "has_comment_message",
    ]

    # Only keep columns that exist (guard for optional fields)
    col_order = [c for c in col_order if c in fact.columns]
    fact = fact[col_order]

    log.info("fact_orders final shape: %s", fact.shape)
    return fact


# ---------------------------------------------------------------------------
# Step 3 — Validate the star schema
# ---------------------------------------------------------------------------

def validate_star_schema(
    fact:          pd.DataFrame,
    dim_customers: pd.DataFrame,
    dim_products:  pd.DataFrame,
    dim_sellers:   pd.DataFrame,
    dim_date:      pd.DataFrame,
) -> None:
    """
    Referential integrity checks: every FK in fact_orders should
    resolve to a row in the corresponding dimension table.
    """
    log.info("Running referential integrity checks...")

    checks = [
        ("customer_id",  dim_customers, "customer_id",  "dim_customers"),
        ("product_id",   dim_products,  "product_id",   "dim_products"),
        ("seller_id",    dim_sellers,   "seller_id",    "dim_sellers"),
        ("date_id",      dim_date,      "date_id",      "dim_date"),
    ]

    all_ok = True
    for fact_col, dim_df, dim_col, dim_name in checks:
        dim_keys    = set(dim_df[dim_col].unique())
        fact_keys   = set(fact[fact_col].dropna().unique())
        unmatched   = fact_keys - dim_keys
        match_rate  = 100 * (1 - len(unmatched) / max(len(fact_keys), 1))

        if unmatched:
            log.warning(
                "  %-15s → %-15s: %d unmatched keys (%.1f%% match rate)",
                fact_col, dim_name, len(unmatched), match_rate,
            )
            all_ok = False
        else:
            log.info(
                "  %-15s → %-15s: all keys matched ✓",
                fact_col, dim_name,
            )

    if all_ok:
        log.info("All referential integrity checks passed ✓")
    else:
        log.warning("Some FK mismatches found — review before loading to Snowflake")


# ---------------------------------------------------------------------------
# Step 4 — Business summary
# ---------------------------------------------------------------------------

def print_business_summary(fact: pd.DataFrame) -> None:
    """
    Print a quick sanity-check business summary.
    These numbers should match your EDA notebook findings.
    """
    delivered = fact[fact["order_status"] == "delivered"]

    log.info("=" * 60)
    log.info("STAR SCHEMA BUSINESS SUMMARY")
    log.info("=" * 60)
    log.info("Total order-item rows      : %d",   len(fact))
    log.info("Unique orders              : %d",   fact["order_id"].nunique())
    log.info("Unique customers (orders)  : %d",   fact["customer_id"].nunique())
    log.info("Unique products            : %d",   fact["product_id"].nunique())
    log.info("Unique sellers             : %d",   fact["seller_id"].nunique())
    log.info("Date range                 : %s → %s",
             fact["order_purchase_timestamp"].min().date(),
             fact["order_purchase_timestamp"].max().date())
    log.info("Total GMV (all items)      : R$ {:>12,.2f}".format(
             fact["total_item_value"].sum()))
    log.info("Total GMV (delivered only) : R$ {:>12,.2f}".format(
             delivered["total_item_value"].sum()))
    log.info("Avg order value            : R$ {:>8,.2f}".format(
             fact.groupby("order_id")["order_total_value"].first().mean()))
    log.info("Late delivery rate         : {:.1f}%".format(
             100 * delivered["is_late"].mean()))
    log.info("Avg review score           : {:.3f}".format(
             fact["review_score"].mean()))
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load all processed files
    orders      = load("clean_orders.parquet")
    items       = load("clean_order_items.parquet")
    items_agg   = load("order_items_agg.parquet")
    payments    = load("clean_payments.parquet")
    reviews     = load("clean_reviews.parquet")
    dim_customers = load("dim_customers.parquet")
    dim_products  = load("dim_products.parquet")
    dim_sellers   = load("dim_sellers.parquet")

    # Build dim_date from order timestamps
    dim_date = build_dim_date(orders)

    # Build fact table
    fact = build_fact_orders(
        orders, items, items_agg, payments, reviews, dim_date
    )

    # Validate referential integrity
    validate_star_schema(fact, dim_customers, dim_products, dim_sellers, dim_date)

    # Print business summary
    print_business_summary(fact)

    # Save outputs
    outputs = {
        "fact_orders.parquet": fact,
        "dim_date.parquet":    dim_date,
    }
    for filename, df in outputs.items():
        out = PROCESSED_DIR / filename
        df.to_parquet(out, index=False)
        log.info("Saved %s → %s", filename, out)

    log.info("Star schema build complete.")
    log.info(
        "Processed files ready for Snowflake load:\n  %s",
        "\n  ".join([
            "fact_orders.parquet",
            "dim_customers.parquet",
            "dim_products.parquet",
            "dim_sellers.parquet",
            "dim_date.parquet",
        ])
    )


if __name__ == "__main__":
    main()