"""
clean_payments.py
-----------------
Cleans and enriches the Olist order payments dataset by:
  1. Validating payment types and installment values
  2. Aggregating multi-row orders to one row per order_id
  3. Identifying the primary payment method per order
  4. Binning installment counts for analysis

Input files:
    data/raw/olist_order_payments_dataset.csv

Output:
    data/processed/clean_payments.parquet   (order grain, one row per order)

Usage:
    python transformation/clean_payments.py
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

VALID_PAYMENT_TYPES = {"credit_card", "boleto", "voucher", "debit_card", "not_defined"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load and validate
# ---------------------------------------------------------------------------

def load_payments(path: Path) -> pd.DataFrame:
    log.info("Loading payments from %s", path)
    payments = pd.read_csv(path, dtype={"order_id": str})
    log.info("Raw shape: %s", payments.shape)
    log.info("Unique orders in payments: %d", payments["order_id"].nunique())

    # Validate payment types
    unknown_types = set(payments["payment_type"].unique()) - VALID_PAYMENT_TYPES
    if unknown_types:
        log.warning("Unknown payment types found: %s", unknown_types)

    log.info(
        "Payment type distribution:\n%s",
        payments["payment_type"].value_counts().to_string(),
    )

    # Validate installments — should be >= 1 for credit cards
    n_zero_installments = (payments["payment_installments"] == 0).sum()
    if n_zero_installments > 0:
        log.warning(
            "%d rows have payment_installments = 0 — likely 'not_defined' type",
            n_zero_installments,
        )

    return payments


# ---------------------------------------------------------------------------
# Step 2 — Identify primary payment type per order
# ---------------------------------------------------------------------------

def get_primary_payment_type(payments: pd.DataFrame) -> pd.Series:
    """
    For orders with multiple payment rows (e.g. voucher + credit card),
    the primary type is the one with the highest payment_value.
    Ties are broken alphabetically (deterministic).
    """
    idx = (
        payments.groupby("order_id")["payment_value"]
        .idxmax()
    )
    return payments.loc[idx, ["order_id", "payment_type"]].set_index("order_id")["payment_type"]


# ---------------------------------------------------------------------------
# Step 3 — Aggregate to order grain
# ---------------------------------------------------------------------------

def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """
    One order can have multiple payment rows (payment_sequential tracks this).
    Collapse to one row per order with summed value and enriched fields.
    """
    primary_type = get_primary_payment_type(payments)

    agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            total_payment_value=("payment_value", "sum"),
            n_payment_rows=("payment_sequential", "count"),
            n_payment_types=("payment_type", "nunique"),
            max_installments=("payment_installments", "max"),
        )
    )

    agg["total_payment_value"] = agg["total_payment_value"].round(2)
    agg["primary_payment_type"] = agg["order_id"].map(primary_type)
    agg["is_split_payment"] = agg["n_payment_types"] > 1

    log.info(
        "Orders with split payment (multiple methods): %d (%.1f%%)",
        agg["is_split_payment"].sum(),
        100 * agg["is_split_payment"].mean(),
    )

    return agg


# ---------------------------------------------------------------------------
# Step 4 — Bin installments
# ---------------------------------------------------------------------------

def bin_installments(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Installment usage is a key behavioral signal in Brazilian e-commerce.
    Brazilians commonly split purchases into 6-12 monthly payments.
    """
    bins   = [0, 1, 3, 6, 12, float("inf")]
    labels = ["1", "2-3", "4-6", "7-12", "13+"]

    agg["installment_bin"] = pd.cut(
        agg["max_installments"],
        bins=bins,
        labels=labels,
        right=True,
    )

    log.info(
        "Installment bin distribution:\n%s",
        agg["installment_bin"].value_counts().sort_index().to_string(),
    )

    return agg


# ---------------------------------------------------------------------------
# Step 5 — Finalize
# ---------------------------------------------------------------------------

def finalize(agg: pd.DataFrame) -> pd.DataFrame:
    col_order = [
        "order_id",
        "total_payment_value",
        "primary_payment_type",
        "n_payment_rows",
        "n_payment_types",
        "is_split_payment",
        "max_installments",
        "installment_bin",
    ]
    agg = agg[col_order]
    log.info("Final payments shape: %s", agg.shape)
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    payments = load_payments(RAW_DIR / "olist_order_payments_dataset.csv")
    agg      = aggregate_payments(payments)
    agg      = bin_installments(agg)
    agg      = finalize(agg)

    out = PROCESSED_DIR / "clean_payments.parquet"
    agg.to_parquet(out, index=False)
    log.info("Saved clean_payments → %s", out)


if __name__ == "__main__":
    main()