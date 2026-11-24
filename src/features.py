"""Customer-level feature engineering for the churn model."""

import numpy as np
import pandas as pd

# The exact columns (and order) the model is trained on and the API serves.
# Selecting explicitly (instead of dropping "bad" columns) makes the contract obvious.
MODEL_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "spend_velocity",
    "purchase_interval_std",
    "unique_products",
    "unique_categories",
    "weekend_ratio",
    "avg_unit_price",
    "avg_hour",
    "whale_score",
    "return_rate",
]


def engineer_features(
    df_clean: pd.DataFrame, df_raw: pd.DataFrame, ref_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Build the customer-level feature matrix used for churn modeling.

    Args:
        df_clean: Cleaned transactions (no cancellations / zero-price rows).
        df_raw: Unfiltered transactions covering the SAME time frame as `df_clean`;
            used only for return rate, since cancellations were removed from `df_clean`.
        ref_date: The "as of" date for recency and the spend windows. Pass the training
            cutoff so features describe the world exactly as of that date. Defaults to
            the last transaction date in `df_clean`.
    """
    ref_date = pd.Timestamp(ref_date) if ref_date is not None else df_clean["InvoiceDate"].max()

    # One row per invoice. Line items of one invoice share a timestamp, so purchase-level
    # features (gaps, weekday, hour) must be computed per invoice, not per line item.
    invoices = (
        df_clean.groupby(["Customer ID", "Invoice"], as_index=False)
        .agg(InvoiceDate=("InvoiceDate", "min"), TotalAmount=("TotalAmount", "sum"))
        .sort_values(["Customer ID", "InvoiceDate"])
    )
    by_customer = invoices.groupby("Customer ID")

    features = pd.DataFrame(
        {
            "recency": (ref_date - by_customer["InvoiceDate"].max()).dt.days,
            "frequency": by_customer["Invoice"].nunique(),
            "monetary": by_customer["TotalAmount"].sum(),
        }
    )

    # Spend momentum: last 30 days vs. the 30 days before that, as a LOG ratio.
    # A plain ratio blows up when prior spend is 0 (we used to divide by 1, which silently
    # turned the ratio into raw currency). log1p(a) - log1p(b) is defined everywhere,
    # symmetric (growth and decline are mirror images) and tame for large spenders.
    window = pd.Timedelta(days=30)
    recent = invoices[invoices["InvoiceDate"] > ref_date - window]
    prior = invoices[
        (invoices["InvoiceDate"] <= ref_date - window) & (invoices["InvoiceDate"] > ref_date - 2 * window)
    ]
    features["spend_last_30d"] = recent.groupby("Customer ID")["TotalAmount"].sum().reindex(features.index, fill_value=0.0)
    features["spend_prior_30d"] = prior.groupby("Customer ID")["TotalAmount"].sum().reindex(features.index, fill_value=0.0)
    features["spend_velocity"] = np.log1p(features["spend_last_30d"].clip(lower=0)) - np.log1p(
        features["spend_prior_30d"].clip(lower=0)
    )

    # Purchase-interval regularity: std of days between consecutive INVOICES.
    # Undefined (NaN) for customers with < 3 invoices; filled with 0 here. `frequency`
    # already tells the model which customers have too little history for this to mean anything.
    invoices["gap_days"] = invoices.groupby("Customer ID")["InvoiceDate"].diff().dt.days
    features["purchase_interval_std"] = (
        invoices.groupby("Customer ID")["gap_days"].std().reindex(features.index).fillna(0.0)
    )

    # Shopping-time behaviour (per invoice)
    invoices["is_weekend"] = (invoices["InvoiceDate"].dt.dayofweek >= 5).astype(float)
    invoices["hour"] = invoices["InvoiceDate"].dt.hour
    features["weekend_ratio"] = invoices.groupby("Customer ID")["is_weekend"].mean()
    features["avg_hour"] = invoices.groupby("Customer ID")["hour"].mean()

    # Product breadth and basket characteristics (per line item).
    # NOTE: `unique_categories` uses the first 3 characters of StockCode as a crude category
    # proxy. It is a heuristic, not a validated product taxonomy.
    lines = df_clean.assign(category_proxy=df_clean["StockCode"].astype(str).str[:3])
    by_lines = lines.groupby("Customer ID")
    features["unique_products"] = by_lines["StockCode"].nunique()
    features["unique_categories"] = by_lines["category_proxy"].nunique()
    features["avg_unit_price"] = by_lines["Price"].mean()
    # "Whale" score: a customer's average line quantity relative to the population average
    features["whale_score"] = by_lines["Quantity"].mean() / lines["Quantity"].mean()

    # Return rate: share of a customer's invoices that were cancellations. Uses df_raw because
    # cleaning removed cancellations from df_clean.
    raw = df_raw[df_raw["Customer ID"].isin(features.index)]
    total_invoices = raw.groupby("Customer ID")["Invoice"].nunique()
    cancelled_invoices = (
        raw[raw["Invoice"].astype(str).str.startswith("C", na=False)].groupby("Customer ID")["Invoice"].nunique()
    )
    features["return_rate"] = (
        cancelled_invoices.reindex(features.index, fill_value=0)
        / total_invoices.reindex(features.index, fill_value=1)
    ).fillna(0.0)

    return features
