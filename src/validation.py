"""Exploratory data integrity (EDI) checks.

Run standalone (`python -m src.validation`) to audit a freshly ingested
DataFrame before it enters the feature pipeline.
"""

import logging

import pandas as pd


from logger import logging as logger


def check_uniques(df: pd.DataFrame) -> dict:
    """Audit unique entity counts to spot duplication or sparsity."""
    stats = {
        "unique_invoices": df["Invoice"].nunique(),
        "unique_stock_codes": df["StockCode"].nunique(),
        "unique_customers": df["Customer ID"].nunique(),
    }
    logger.info("Unique entity audit: %s", stats)
    return stats


def check_zero_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Identify potential freebies or data-entry errors (Price <= 0)."""
    zero_price_df = df[df["Price"] <= 0]
    pct_zero = (len(zero_price_df) / len(df)) * 100 if len(df) else 0.0
    affected_quantity = zero_price_df["Quantity"].sum()
    logger.info("Zero-price rows: %d (%.2f%%)", len(zero_price_df), pct_zero)
    logger.info("Volume moved for free: %d", affected_quantity)
    return zero_price_df


def check_description_consistency(df: pd.DataFrame) -> pd.Series:
    """Find StockCodes mapped to more than one Description."""
    description_counts = df.groupby("StockCode")["Description"].nunique()
    inconsistent = description_counts[description_counts > 1]

    if not inconsistent.empty:
        top_examples = inconsistent.sort_values(ascending=False).head(3)
        logger.warning("Red flag: %d StockCodes have multiple descriptions", len(inconsistent))
        logger.warning("Top offenders:\n%s", top_examples)
    else:
        logger.info("Data cleanliness: all StockCodes have consistent descriptions")

    return inconsistent


def check_temporal_gaps(df: pd.DataFrame) -> pd.Timedelta:
    """Check for the longest gap between consecutive transactions."""
    df_sorted = df.sort_values("InvoiceDate")
    time_diffs = df_sorted["InvoiceDate"].diff()
    max_gap = time_diffs.max()
    logger.info("Longest gap between transactions: %s", max_gap)
    return max_gap


def check_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Flag cancellation invoices (prefixed 'C') that have a positive quantity."""
    invoice_str = df["Invoice"].astype(str)
    bad_cancellations = df[invoice_str.str.startswith("C", na=False) & (df["Quantity"] > 0)]

    if not bad_cancellations.empty:
        logger.error("Critical: found %d cancellations with positive quantity", len(bad_cancellations))
    else:
        logger.info("Logic check: all cancellations have negative quantities")

    return bad_cancellations


def check_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag Quantity outliers using the 1.5*IQR rule."""
    q1 = df["Quantity"].quantile(0.25)
    q3 = df["Quantity"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df["Quantity"] < lower_bound) | (df["Quantity"] > upper_bound)]
    logger.info("Outlier detection: found %d rows outside typical Quantity range", len(outliers))
    logger.info("Extreme threshold: > %.2f units", upper_bound)
    return outliers


def check_geographic_breadth(df: pd.DataFrame) -> pd.Series:
    """Rank countries by price variance."""
    price_variance = df.groupby("Country")["Price"].var().sort_values(ascending=False)
    logger.info("Top countries by price variance:\n%s", price_variance.head())
    return price_variance


def check_ghost_customers(df: pd.DataFrame) -> None:
    """Compare spend between identified customers and anonymous ('ghost') orders."""
    null_customers = df["Customer ID"].isnull().sum()
    pct_null = (null_customers / len(df)) * 100 if len(df) else 0.0
    logger.info("Ghost customers: %d rows (%.1f%%) have no ID", null_customers, pct_null)

    temp_revenue = df["Quantity"] * df["Price"]
    avg_registered = temp_revenue[df["Customer ID"].notnull()].mean()
    avg_ghost = temp_revenue[df["Customer ID"].isnull()].mean()
    logger.info("Avg spend — registered: %.2f, guest: %.2f", avg_registered, avg_ghost)


def run_all_checks(df: pd.DataFrame) -> None:
    """Run the full exploratory data integrity suite."""
    logger.info("--- Starting Exploratory Data Integrity (EDI) ---")
    check_uniques(df)
    check_zero_prices(df)
    check_description_consistency(df)
    check_temporal_gaps(df)
    check_cancellations(df)
    check_outliers(df)
    check_geographic_breadth(df)
    check_ghost_customers(df)
    logger.info("--- EDI complete ---")


if __name__ == "__main__":
    from src.ingestion import load_and_optimize_data
    from src.logging_config import setup_logging

    setup_logging()

    data = load_and_optimize_data()
    if data is not None:
        run_all_checks(data)
