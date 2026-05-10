"""
clean_sellers.py
----------------
Cleans and enriches the Olist sellers dataset by:
  1. Normalizing city names and validating state codes
  2. Joining geo coordinates via zip code prefix
     (reuses dim_geolocation.parquet produced by clean_geolocation.py)

Input files:
    data/raw/olist_sellers_dataset.csv
    data/processed/dim_geolocation.parquet   (must run clean_geolocation.py first)

Output:
    data/processed/dim_sellers.parquet

Usage:
    python transformation/clean_sellers.py
"""

import logging
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

VALID_BR_STATES = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
    "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
    "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (mirrors clean_customers.py)
# ---------------------------------------------------------------------------

def normalize_city(city: str) -> str:
    if not isinstance(city, str):
        return "unknown"
    city = city.strip()
    try:
        city = city.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    city = unicodedata.normalize("NFD", city)
    city = "".join(c for c in city if unicodedata.category(c) != "Mn")
    return city.title()


def validate_states(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    invalid_mask = ~df[col].isin(VALID_BR_STATES)
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        log.warning(
            "%s: %d rows have unrecognised state code in '%s': %s",
            label, n_invalid, col,
            df.loc[invalid_mask, col].unique().tolist(),
        )
    df[f"{col}_valid"] = ~invalid_mask
    return df


# ---------------------------------------------------------------------------
# Step 1 — Load and clean sellers
# ---------------------------------------------------------------------------

def load_sellers(path: Path) -> pd.DataFrame:
    log.info("Loading sellers from %s", path)
    sellers = pd.read_csv(
        path,
        dtype={
            "seller_id":              str,
            "seller_zip_code_prefix": str,
            "seller_city":            str,
            "seller_state":           str,
        },
    )
    log.info("Raw shape: %s", sellers.shape)

    # Sanity: seller_id should be unique
    n_dup = sellers.duplicated(subset="seller_id").sum()
    if n_dup > 0:
        log.warning("%d duplicate seller_id rows found", n_dup)
    else:
        log.info("seller_id is unique — no duplicates")

    sellers["seller_city"] = sellers["seller_city"].apply(normalize_city)
    sellers = validate_states(sellers, "seller_state", "sellers")

    log.info(
        "Sellers by state (top 10):\n%s",
        sellers["seller_state"].value_counts().head(10).to_string(),
    )

    return sellers


# ---------------------------------------------------------------------------
# Step 2 — Load pre-built geolocation lookup
# ---------------------------------------------------------------------------

def load_geolocation(path: Path) -> pd.DataFrame:
    """
    dim_geolocation.parquet was produced by clean_geolocation.py.
    It has one row per zip_code_prefix with mean lat/lng.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run clean_geolocation.py first."
        )
    log.info("Loading geolocation lookup from %s", path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Step 3 — Join geo coordinates
# ---------------------------------------------------------------------------

def enrich_with_geo(sellers: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    geo_lookup = geo.rename(columns={
        "geolocation_zip_code_prefix": "seller_zip_code_prefix",
        "lat": "seller_lat",
        "lng": "seller_lng",
    })[["seller_zip_code_prefix", "seller_lat", "seller_lng"]]

    sellers = sellers.merge(geo_lookup, on="seller_zip_code_prefix", how="left")

    sellers["has_geo"] = sellers["seller_lat"].notna()
    n_missing = (~sellers["has_geo"]).sum()
    if n_missing > 0:
        log.warning(
            "%d sellers (%.1f%%) have no geolocation match",
            n_missing, 100 * n_missing / len(sellers),
        )
    else:
        log.info("All sellers matched to geolocation")

    return sellers


# ---------------------------------------------------------------------------
# Step 4 — Finalize
# ---------------------------------------------------------------------------

def finalize(sellers: pd.DataFrame) -> pd.DataFrame:
    # Drop the _valid flag if all states are valid (constant column)
    if sellers["seller_state_valid"].all():
        log.info("All seller_state values are valid — dropping seller_state_valid column")
        sellers = sellers.drop(columns=["seller_state_valid"])
        col_order = [
            "seller_id", "seller_zip_code_prefix",
            "seller_city", "seller_state",
            "seller_lat", "seller_lng", "has_geo",
        ]
    else:
        col_order = [
            "seller_id", "seller_zip_code_prefix",
            "seller_city", "seller_state", "seller_state_valid",
            "seller_lat", "seller_lng", "has_geo",
        ]

    sellers = sellers[col_order]
    log.info("Final dim_sellers shape: %s", sellers.shape)
    return sellers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sellers = load_sellers(RAW_DIR / "olist_sellers_dataset.csv")
    geo     = load_geolocation(PROCESSED_DIR / "dim_geolocation.parquet")
    sellers = enrich_with_geo(sellers, geo)
    sellers = finalize(sellers)

    out = PROCESSED_DIR / "dim_sellers.parquet"
    sellers.to_parquet(out, index=False)
    log.info("Saved dim_sellers → %s", out)


if __name__ == "__main__":
    main()