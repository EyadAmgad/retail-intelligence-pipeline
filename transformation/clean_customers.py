"""
clean_customers.py
------------------
Cleans and enriches the Olist customers dataset by:
  1. Standardizing city names and validating state codes
  2. Deduplicating and bounding-box-filtering the geolocation dataset
  3. Joining geo coordinates onto customers via zip code prefix

Input files:
    data/raw/olist_customers_dataset.csv
    data/raw/olist_geolocation_dataset.csv

Output:
    data/processed/dim_customers.parquet
    data/processed/dim_geolocation.parquet  (reusable by clean_sellers.py)

Usage:
    python transformation/clean_customers.py
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

# Valid Brazilian state abbreviations (26 states + DF)
VALID_BR_STATES = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
    "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
    "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

# Brazil's approximate bounding box (lat, lng)
BR_LAT_MIN, BR_LAT_MAX = -35.0, 5.5
BR_LNG_MIN, BR_LNG_MAX = -74.0, -28.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_city(city: str) -> str:
    """
    Normalize a Brazilian city name:
      - Strip leading/trailing whitespace
      - Decode garbled UTF-8 (e.g. 'sÃo paulo' → 'são paulo')
      - Remove accents so that joins are accent-insensitive
      - Title-case for display

    Example:
        'sÃo paulo'  → 'Sao Paulo'
        'BELO HORIZONTE' → 'Belo Horizonte'
        'rio de janeiro ' → 'Rio De Janeiro'
    """
    if not isinstance(city, str):
        return "unknown"

    city = city.strip()

    # Attempt to fix mojibake (latin-1 bytes decoded as utf-8)
    try:
        city = city.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass  # already valid utf-8 — carry on

    # Strip accents (NFD decomposition → remove combining marks)
    city = unicodedata.normalize("NFD", city)
    city = "".join(c for c in city if unicodedata.category(c) != "Mn")

    return city.title()


def validate_states(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    """
    Flag rows whose state code is not in the official Brazilian state list.
    Logs a warning summary; does NOT drop the rows so the analyst can
    investigate them in the notebook.
    """
    invalid_mask = ~df[col].isin(VALID_BR_STATES)
    n_invalid = invalid_mask.sum()

    if n_invalid > 0:
        log.warning(
            "%s: %d rows have an unrecognised state code in '%s': %s",
            label,
            n_invalid,
            col,
            df.loc[invalid_mask, col].unique().tolist(),
        )

    df[f"{col}_valid"] = ~invalid_mask
    return df


# ---------------------------------------------------------------------------
# Step 1 — Clean geolocation
# ---------------------------------------------------------------------------

def clean_geolocation(path: Path) -> pd.DataFrame:
    """
    The raw geolocation file has many rows per zip code prefix (multiple
    lat/lng samples collected at different times).

    Strategy:
      1. Drop rows outside Brazil's bounding box (data-entry errors).
      2. Average lat/lng per zip prefix to produce a single centroid.
      3. Take the most-frequent city and state per zip prefix.

    Returns a DataFrame with one row per zip_code_prefix.
    """
    log.info("Loading geolocation data from %s", path)
    geo = pd.read_csv(path, dtype={"geolocation_zip_code_prefix": str})

    log.info("Geolocation raw shape: %s", geo.shape)

    # --- bounding-box filter ---
    before = len(geo)
    geo = geo[
        geo["geolocation_lat"].between(BR_LAT_MIN, BR_LAT_MAX)
        & geo["geolocation_lng"].between(BR_LNG_MIN, BR_LNG_MAX)
    ]
    log.info(
        "Dropped %d rows outside Brazil bounding box (%d remaining)",
        before - len(geo),
        len(geo),
    )

    # --- normalize city names before aggregation ---
    geo["geolocation_city"] = geo["geolocation_city"].apply(normalize_city)

    # --- aggregate: mean coords, modal city & state per zip ---
    geo_agg = (
        geo.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
            city=("geolocation_city", lambda x: x.mode().iloc[0]),
            state=("geolocation_state", lambda x: x.mode().iloc[0]),
        )
    )

    log.info(
        "Geolocation after dedup: %d unique zip prefixes (from %d raw rows)",
        len(geo_agg),
        before,
    )

    return geo_agg


# ---------------------------------------------------------------------------
# Step 2 — Clean customers
# ---------------------------------------------------------------------------

def clean_customers(path: Path) -> pd.DataFrame:
    """
    Clean the raw customers file:
      - Enforce correct dtypes
      - Normalize city names
      - Validate state codes
      - Document the customer_id vs customer_unique_id relationship

    Returns the cleaned DataFrame (still at the raw grain: one row per
    customer_id, which is one row per order).
    """
    log.info("Loading customers data from %s", path)
    customers = pd.read_csv(
        path,
        dtype={
            "customer_id": str,
            "customer_unique_id": str,
            "customer_zip_code_prefix": str,
            "customer_city": str,
            "customer_state": str,
        },
    )

    log.info("Customers raw shape: %s", customers.shape)

    # --- sanity: no nulls expected in key columns ---
    key_cols = ["customer_id", "customer_unique_id", "customer_zip_code_prefix"]
    nulls = customers[key_cols].isnull().sum()
    if nulls.any():
        log.warning("Unexpected nulls in key columns:\n%s", nulls[nulls > 0])

    # --- document the many-to-one relationship ---
    n_orders = len(customers)
    n_unique = customers["customer_unique_id"].nunique()
    n_repeat = n_unique - (
        customers.groupby("customer_unique_id")["customer_id"]
        .count()
        .eq(1)
        .sum()
    )
    log.info(
        "customer_id rows (= orders): %d | unique customers: %d | repeat buyers: %d",
        n_orders,
        n_unique,
        n_repeat,
    )

    # --- normalize city name ---
    customers["customer_city"] = customers["customer_city"].apply(normalize_city)

    # --- validate states ---
    customers = validate_states(customers, "customer_state", "customers")

    return customers


# ---------------------------------------------------------------------------
# Step 3 — Join and produce dim_customers
# ---------------------------------------------------------------------------

def build_dim_customers(
    customers: pd.DataFrame,
    geo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich customers with geo coordinates via zip code prefix.

    Join type: left — we keep ALL customers even if their zip prefix
    is not in the geolocation table (small % of rows). Missing coords
    remain as NaN and are flagged with has_geo=False.
    """
    log.info("Joining customers with geolocation on zip code prefix...")

    dim = customers.merge(
        geo.rename(columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "lat": "customer_lat",
            "lng": "customer_lng",
            # city/state from geo used only as fallback; keep customer's own values
        })[["customer_zip_code_prefix", "customer_lat", "customer_lng"]],
        on="customer_zip_code_prefix",
        how="left",
    )

    # --- flag rows where geo lookup failed ---
    dim["has_geo"] = dim["customer_lat"].notna()
    n_missing_geo = (~dim["has_geo"]).sum()
    if n_missing_geo > 0:
        log.warning(
            "%d customers (%.1f%%) have no geolocation match",
            n_missing_geo,
            100 * n_missing_geo / len(dim),
        )

    # --- final column order ---
    dim = dim[[
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
        "customer_state_valid",
        "customer_lat",
        "customer_lng",
        "has_geo",
    ]]

    log.info("dim_customers shape: %s", dim.shape)
    return dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Clean geolocation (also saved separately for reuse in clean_sellers.py)
    geo = clean_geolocation(RAW_DIR / "olist_geolocation_dataset.csv")
    geo_out = PROCESSED_DIR / "dim_geolocation.parquet"
    geo.to_parquet(geo_out, index=False)
    log.info("Saved dim_geolocation → %s", geo_out)

    # 2. Clean customers
    customers = clean_customers(RAW_DIR / "olist_customers_dataset.csv")

    # 3. Join and build final dim
    dim_customers = build_dim_customers(customers, geo)

    # 4. Save
    out = PROCESSED_DIR / "dim_customers.parquet"
    dim_customers.to_parquet(out, index=False)
    log.info("Saved dim_customers → %s", out)

    # 5. Quick sanity summary
    log.info("\n%s", dim_customers.describe(include="all").T[["count", "unique", "top"]].to_string())


if __name__ == "__main__":
    main()