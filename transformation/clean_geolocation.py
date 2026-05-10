"""
clean_geolocation.py
--------------------
Standalone cleaning of the Olist geolocation dataset:
  1. Filters coordinates outside Brazil's bounding box
  2. Deduplicates to one row per zip code prefix (mean lat/lng, modal city/state)
  3. Normalizes city names

This output (dim_geolocation.parquet) is consumed by:
  - clean_customers.py
  - clean_sellers.py

Input files:
    data/raw/olist_geolocation_dataset.csv

Output:
    data/processed/dim_geolocation.parquet

Usage:
    python transformation/clean_geolocation.py
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

# Brazil's approximate geographic bounding box
BR_LAT_MIN, BR_LAT_MAX = -35.0,  5.5
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


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------

def load_geolocation(path: Path) -> pd.DataFrame:
    log.info("Loading geolocation from %s", path)
    geo = pd.read_csv(
        path,
        dtype={"geolocation_zip_code_prefix": str},
    )
    log.info("Raw shape: %s  (one row per GPS sample, not per zip)", geo.shape)
    log.info("Unique zip prefixes (raw): %d", geo["geolocation_zip_code_prefix"].nunique())
    return geo


# ---------------------------------------------------------------------------
# Step 2 — Bounding box filter
# ---------------------------------------------------------------------------

def filter_bounding_box(geo: pd.DataFrame) -> pd.DataFrame:
    before = len(geo)

    in_bbox = (
        geo["geolocation_lat"].between(BR_LAT_MIN, BR_LAT_MAX)
        & geo["geolocation_lng"].between(BR_LNG_MIN, BR_LNG_MAX)
    )
    outliers = geo[~in_bbox]

    if not outliers.empty:
        log.warning(
            "Dropping %d rows outside Brazil bounding box "
            "(lat [%.1f, %.1f], lng [%.1f, %.1f])",
            len(outliers), BR_LAT_MIN, BR_LAT_MAX, BR_LNG_MIN, BR_LNG_MAX,
        )
        log.warning(
            "Sample outlier coordinates:\n%s",
            outliers[["geolocation_lat", "geolocation_lng",
                       "geolocation_city", "geolocation_state"]]
            .head(5).to_string(index=False),
        )

    geo = geo[in_bbox]
    log.info(
        "After bounding box filter: %d rows (dropped %d)", len(geo), before - len(geo)
    )
    return geo


# ---------------------------------------------------------------------------
# Step 3 — Normalize city names
# ---------------------------------------------------------------------------

def normalize_cities(geo: pd.DataFrame) -> pd.DataFrame:
    geo["geolocation_city"] = geo["geolocation_city"].apply(normalize_city)
    log.info(
        "Unique cities after normalization: %d",
        geo["geolocation_city"].nunique(),
    )
    return geo


# ---------------------------------------------------------------------------
# Step 4 — Deduplicate to one row per zip prefix
# ---------------------------------------------------------------------------

def deduplicate(geo: pd.DataFrame) -> pd.DataFrame:
    """
    Each zip prefix has many GPS samples. Strategy:
      - lat/lng: take the mean (centroid approximation)
      - city, state: take the mode (most frequent label)
    """
    geo_agg = (
        geo.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            lat=("geolocation_lat",  "mean"),
            lng=("geolocation_lng",  "mean"),
            city=("geolocation_city",  lambda x: x.mode().iloc[0]),
            state=("geolocation_state", lambda x: x.mode().iloc[0]),
            n_samples=("geolocation_lat", "count"),  # how many raw rows contributed
        )
    )

    geo_agg["lat"] = geo_agg["lat"].round(6)
    geo_agg["lng"] = geo_agg["lng"].round(6)

    log.info(
        "After dedup: %d unique zip prefixes (avg %.1f samples per prefix)",
        len(geo_agg),
        geo_agg["n_samples"].mean(),
    )
    log.info(
        "Sample stats:\n  min_samples=%d | median_samples=%.0f | max_samples=%d",
        geo_agg["n_samples"].min(),
        geo_agg["n_samples"].median(),
        geo_agg["n_samples"].max(),
    )

    return geo_agg


# ---------------------------------------------------------------------------
# Step 5 — Finalize
# ---------------------------------------------------------------------------

def finalize(geo: pd.DataFrame) -> pd.DataFrame:
    # Rename to align with join keys used by clean_customers and clean_sellers
    geo = geo.rename(columns={
        "geolocation_zip_code_prefix": "geolocation_zip_code_prefix",  # keep as-is
    })
    col_order = [
        "geolocation_zip_code_prefix",
        "lat", "lng",
        "city", "state",
        "n_samples",
    ]
    geo = geo[col_order]
    log.info("Final dim_geolocation shape: %s", geo.shape)

    log.info(
        "State coverage (top 10):\n%s",
        geo["state"].value_counts().head(10).to_string(),
    )
    return geo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    geo = load_geolocation(RAW_DIR / "olist_geolocation_dataset.csv")
    geo = filter_bounding_box(geo)
    geo = normalize_cities(geo)
    geo = deduplicate(geo)
    geo = finalize(geo)

    out = PROCESSED_DIR / "dim_geolocation.parquet"
    geo.to_parquet(out, index=False)
    log.info("Saved dim_geolocation → %s", out)


if __name__ == "__main__":
    main()