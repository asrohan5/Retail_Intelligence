"""Raw data ingestion."""

import logging
from pathlib import Path

import pandas as pd

from src import config

from logger import logging as logger

def _read_source() -> pd.DataFrame:
    """Read the raw Excel sheet, using a pickle cache when it is newer than the source.

    Parsing the Excel file takes ~1 minute; the cache makes repeat runs near-instant.
    """
    source, cache = Path(config.RAW_DATA_PATH), Path(config.CACHE_PATH)

    if cache.exists() and source.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
        logger.info("Loading cached data from %s", cache)
        return pd.read_pickle(cache)

    logger.info("Loading data from %s ...", source)
    df = pd.read_excel(source, sheet_name=config.SHEET_NAME)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def sample_customers(df: pd.DataFrame, frac: float, random_state: int) -> pd.DataFrame:
    """Sample whole customers, keeping every transaction of each sampled customer.

    Row-level sampling would randomly delete orders from each customer's history,
    corrupting recency, frequency, monetary value and (critically) the churn label.
    Anonymous rows (no Customer ID) are kept as-is; they are excluded downstream.
    """
    if frac >= 1.0:
        return df
    ids = pd.Series(df["Customer ID"].dropna().unique()).sample(frac=frac, random_state=random_state)
    keep = df["Customer ID"].isna() | df["Customer ID"].isin(ids)
    return df[keep]


def load_and_optimize_data() -> pd.DataFrame | None:
    """Load the raw transactions (optionally sampling customers) and downcast Country.

    Returns:
        The loaded DataFrame, or None if the source file is missing or unreadable.
    """
    try:
        df_full = _read_source()
        df = sample_customers(df_full, config.SAMPLE_FRAC, config.RANDOM_STATE).copy()
        df["Country"] = df["Country"].astype("category")

        logger.info(
            "Loaded %d rows, %d customers (%.1f MB in memory)",
            len(df),
            df["Customer ID"].nunique(),
            df.memory_usage(deep=True).sum() / 1024**2,
        )
        return df.reset_index(drop=True)

    except FileNotFoundError:
        logger.error("File not found! Please check the data directory: %s", config.RAW_DATA_PATH)
        return None
    except Exception:
        logger.exception("Unexpected error while loading data")
        return None


if __name__ == "__main__":
    from src.logging_config import setup_logging

    setup_logging()
    data = load_and_optimize_data()
    if data is not None:
        print(data.head())
