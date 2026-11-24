"""Assembles the labeled churn dataset via a time-based split."""

import logging

import pandas as pd

from src import config
from src.features import engineer_features


from logger import logging as logger

def create_churn_dataset(
    df_clean: pd.DataFrame,
    df_raw: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
    horizon_days: int = config.CHURN_WINDOW_DAYS,
) -> pd.DataFrame:
    """Build a labeled churn dataset using a time-based (not random) split.

    Features are engineered only from transactions up to `cutoff_date` ("past").
    A customer is labeled churned (1) if they placed no order in the `horizon_days`
    after the cutoff ("future window"). Using a fixed-length window keeps the label
    definition identical across different cutoffs.
    """
    data_end = df_clean["InvoiceDate"].max()
    if cutoff_date is None:
        cutoff_date = data_end - pd.Timedelta(days=horizon_days)
    cutoff_date = pd.Timestamp(cutoff_date)
    window_end = cutoff_date + pd.Timedelta(days=horizon_days)

    if window_end > data_end:
        logger.warning(
            "Label window ends %s but data ends %s: customers' later orders are not observable, "
            "so churn may be overstated.",
            window_end.date(),
            data_end.date(),
        )

    logger.info("Cutoff: %s | label window: (%s, %s]", cutoff_date.date(), cutoff_date.date(), window_end.date())

    raw_dates = pd.to_datetime(df_raw["InvoiceDate"])
    df_past = df_clean[df_clean["InvoiceDate"] <= cutoff_date]
    df_raw_past = df_raw[raw_dates <= cutoff_date]
    df_future = df_clean[(df_clean["InvoiceDate"] > cutoff_date) & (df_clean["InvoiceDate"] <= window_end)]

    features = engineer_features(df_past, df_raw_past, ref_date=cutoff_date)

    active_in_window = df_future["Customer ID"].unique()
    features["is_churned"] = (~features.index.isin(active_in_window)).astype(int)

    logger.info(
        "Dataset: %d customers | churn rate %.2f%%", len(features), features["is_churned"].mean() * 100
    )
    return features
