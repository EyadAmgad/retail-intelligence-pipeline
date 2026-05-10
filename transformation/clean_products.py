"""
clean_products.py
-----------------
Cleans and enriches the Olist products dataset by:
  1. Fixing column name typos (lenght → length)
  2. Joining English category translations
  3. Filling missing category names
  4. Computing volume and flagging missing dimension data
  5. Binning products by size

Input files:
    data/raw/olist_products_dataset.csv
    data/raw/product_category_name_translation.csv

Output:
    data/processed/dim_products.parquet

Usage:
    python transformation/clean_products.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load and fix column names
# ---------------------------------------------------------------------------

def load_products(path: Path) -> pd.DataFrame:
    log.info("Loading products from %s", path)
    products = pd.read_csv(path, dtype={"product_id": str})
    log.info("Raw shape: %s", products.shape)

    # Fix the typos that exist in the original dataset
    products = products.rename(columns={
        "product_name_lenght":        "product_name_length",
        "product_description_lenght": "product_description_length",
    })
    log.info("Renamed typo columns: product_name_lenght → product_name_length, "
             "product_description_lenght → product_description_length")

    return products


# ---------------------------------------------------------------------------
# Step 2 — Load and join English translations
# ---------------------------------------------------------------------------

def apply_translations(products: pd.DataFrame, translation_path: Path) -> pd.DataFrame:
    log.info("Loading category translations from %s", translation_path)
    translations = pd.read_csv(translation_path)

    # Fill missing Portuguese category names BEFORE joining
    # so they map to 'unknown' in English as well
    n_missing_cat = products["product_category_name"].isnull().sum()
    log.info("Products with missing category_name: %d — filling with 'unknown'", n_missing_cat)
    products["product_category_name"] = products["product_category_name"].fillna("unknown")

    # Inject 'unknown' into translation table if not already there
    if "unknown" not in translations["product_category_name"].values:
        unknown_row = pd.DataFrame([{
            "product_category_name":         "unknown",
            "product_category_name_english": "unknown",
        }])
        translations = pd.concat([translations, unknown_row], ignore_index=True)

    products = products.merge(translations, on="product_category_name", how="left")

    n_unmapped = products["product_category_name_english"].isnull().sum()
    if n_unmapped > 0:
        log.warning(
            "%d products could not be mapped to an English category — filling with 'unknown'",
            n_unmapped,
        )
        products["product_category_name_english"] = \
            products["product_category_name_english"].fillna("unknown")

    log.info(
        "Top 10 English categories:\n%s",
        products["product_category_name_english"].value_counts().head(10).to_string(),
    )

    return products


# ---------------------------------------------------------------------------
# Step 3 — Validate physical dimensions
# ---------------------------------------------------------------------------

def validate_dimensions(products: pd.DataFrame) -> pd.DataFrame:
    dim_cols = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    null_summary = products[dim_cols].isnull().sum()
    log.info("Null counts in dimension columns:\n%s", null_summary.to_string())

    # Flag rows missing ANY physical dimension
    products["has_missing_dimensions"] = products[dim_cols].isnull().any(axis=1)
    log.info(
        "Products missing at least one dimension: %d (%.1f%%)",
        products["has_missing_dimensions"].sum(),
        100 * products["has_missing_dimensions"].mean(),
    )

    return products


# ---------------------------------------------------------------------------
# Step 4 — Compute volume and size bin
# ---------------------------------------------------------------------------

def compute_size_features(products: pd.DataFrame) -> pd.DataFrame:
    """
    product_volume_cm3: used for freight cost analysis
    size_category: coarse grouping for notebook segmentation
    """
    products["product_volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    ).round(2)

    # Weight-based size bins (grams)
    bins   = [0, 300, 1000, 5000, float("inf")]
    labels = ["small", "medium", "large", "extra_large"]

    products["size_category"] = pd.cut(
        products["product_weight_g"],
        bins=bins,
        labels=labels,
        right=True,
    )

    log.info(
        "Size category distribution:\n%s",
        products["size_category"].value_counts().sort_index().to_string(),
    )

    return products


# ---------------------------------------------------------------------------
# Step 5 — Finalize
# ---------------------------------------------------------------------------

def finalize(products: pd.DataFrame) -> pd.DataFrame:
    col_order = [
        "product_id",
        "product_category_name",
        "product_category_name_english",
        "product_name_length",
        "product_description_length",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_volume_cm3",
        "size_category",
        "has_missing_dimensions",
    ]
    products = products[col_order]
    log.info("Final dim_products shape: %s", products.shape)
    return products


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    products = load_products(RAW_DIR / "olist_products_dataset.csv")
    products = apply_translations(products, RAW_DIR / "product_category_name_translation.csv")
    products = validate_dimensions(products)
    products = compute_size_features(products)
    products = finalize(products)

    out = PROCESSED_DIR / "dim_products.parquet"
    products.to_parquet(out, index=False)
    log.info("Saved dim_products → %s", out)


if __name__ == "__main__":
    main()