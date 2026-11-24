"""RFM scoring, cohort/retention analysis, and executive-summary metrics."""

import logging

import numpy as np
import pandas as pd

from logger import logging as logger

RFM_LABELS = [1, 2, 3, 4, 5]


def clean_data_for_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Drop anonymous orders/cancellations/free items and derive line totals."""
    initial_count = len(df)

    df_clean = df.dropna(subset=["Customer ID"]).copy()
    df_clean = df_clean[~df_clean["Invoice"].astype(str).str.startswith("C")]
    df_clean = df_clean[df_clean["Price"] > 0]

    df_clean["TotalAmount"] = df_clean["Quantity"] * df_clean["Price"]
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

    logger.info("Data cleaning: %d -> %d rows", initial_count, len(df_clean))
    return df_clean


def calculate_rfm_values(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw Recency, Frequency, and Monetary values per customer."""
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer ID")
        .agg(
            {
                "InvoiceDate": lambda x: (reference_date - x.max()).days,
                "Invoice": "nunique",
                "TotalAmount": "sum",
            }
        )
        .reset_index()
    )
    rfm.rename(
        columns={"InvoiceDate": "Recency", "Invoice": "Frequency", "TotalAmount": "Monetary"},
        inplace=True,
    )
    return rfm


def assign_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Bucket R/F/M into quintile scores (1-5) and build the RFM segment code.

    Values are rank-transformed (ties broken by order) before qcut so that heavily
    tied columns such as Frequency cannot produce duplicate bin edges.
    """
    rfm_df["R_score"] = pd.qcut(
        rfm_df["Recency"].rank(method="first"), q=5, labels=list(reversed(RFM_LABELS))
    )
    rfm_df["F_score"] = pd.qcut(rfm_df["Frequency"].rank(method="first"), q=5, labels=RFM_LABELS)
    rfm_df["M_score"] = pd.qcut(rfm_df["Monetary"].rank(method="first"), q=5, labels=RFM_LABELS)

    rfm_df["Segment"] = (
        rfm_df["R_score"].astype(str) + rfm_df["F_score"].astype(str) + rfm_df["M_score"].astype(str)
    )
    return rfm_df


def segment_for_scores(r: int, f: int, m: int) -> str:
    """Map one customer's R/F/M scores to a business segment.

    Rule ORDER matters: the first matching rule wins, so the more specific rules
    ("At Risk": low recency AND high frequency) must come before the broader
    "Loyalists" rule, otherwise they can never be reached.
    """
    if r >= 5 and f >= 5 and m >= 5:
        return "Champions"
    if r <= 2 and f >= 4:
        return "At Risk"
    if f >= 4:
        return "Loyalists"
    if r >= 4 and f == 1:
        return "New Customers"
    if r <= 2 and f <= 2:
        return "Hibernating"
    return "Potential Loyalist"


def define_customer_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Add a human-readable business segment to each customer."""
    rfm_df["Segment"] = [
        segment_for_scores(int(r), int(f), int(m))
        for r, f, m in zip(rfm_df["R_score"], rfm_df["F_score"], rfm_df["M_score"])
    ]
    return rfm_df


def calculate_cohort_index(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each transaction a month-based cohort index relative to first purchase."""
    df["InvoiceMonth"] = df["InvoiceDate"].apply(lambda x: x.replace(day=1).date())
    df["CohortMonth"] = df.groupby("Customer ID")["InvoiceMonth"].transform("min")

    invoice_year = pd.to_datetime(df["InvoiceMonth"]).dt.year
    invoice_month = pd.to_datetime(df["InvoiceMonth"]).dt.month
    cohort_year = pd.to_datetime(df["CohortMonth"]).dt.year
    cohort_month = pd.to_datetime(df["CohortMonth"]).dt.month

    years_diff = invoice_year - cohort_year
    month_diff = invoice_month - cohort_month
    df["CohortIndex"] = years_diff * 12 + month_diff + 1
    return df


def get_retention_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot cohort data into a month-0-indexed retention rate matrix."""
    cohort_data = df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index="CohortMonth", columns="CohortIndex", values="Customer ID")
    cohort_size = cohort_pivot.iloc[:, 0]
    return cohort_pivot.divide(cohort_size, axis=0)


def calculate_monthly_aov(df: pd.DataFrame) -> pd.Series:
    """Average order value per month."""
    monthly = df.groupby("InvoiceMonth").agg({"TotalAmount": "sum", "Invoice": "nunique"})
    monthly["AOV"] = monthly["TotalAmount"] / monthly["Invoice"]
    return monthly["AOV"]


def calculate_new_vs_repeat(df: pd.DataFrame) -> pd.DataFrame:
    """Split monthly active customers into New vs. Repeat buyers."""
    first_purchase = df.groupby("Customer ID")["InvoiceDate"].min().reset_index()
    first_purchase.columns = ["Customer ID", "FirstPurchaseDate"]

    df = df.merge(first_purchase, on="Customer ID")
    df["UserType"] = np.where(df["InvoiceDate"] == df["FirstPurchaseDate"], "New", "Repeat")

    return df.groupby(["InvoiceMonth", "UserType"])["Customer ID"].nunique().unstack()


def calculate_churn_risk(df: pd.DataFrame, days_threshold: int = 90) -> pd.DataFrame:
    """Flag customers dormant for longer than `days_threshold` and the revenue at stake."""
    max_date = df["InvoiceDate"].max()
    last_purchase = (
        df.groupby("Customer ID").agg({"InvoiceDate": "max", "TotalAmount": "sum"}).reset_index()
    )
    last_purchase["DaysDormant"] = (max_date - last_purchase["InvoiceDate"]).dt.days

    churn_risk = last_purchase[last_purchase["DaysDormant"] > days_threshold]
    logger.warning("Churn risk: %d users dormant > %d days", len(churn_risk), days_threshold)
    logger.warning("Potential revenue at risk: %.2f", churn_risk["TotalAmount"].sum())
    return churn_risk


def build_executive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble a monthly revenue / growth / active-users / AOV summary table."""
    aov = calculate_monthly_aov(df)
    monthly_revenue = df.groupby("InvoiceMonth")["TotalAmount"].sum()
    growth_pct = monthly_revenue.pct_change() * 100
    active_users = df.groupby("InvoiceMonth")["Customer ID"].nunique()

    summary = pd.DataFrame(
        {
            "Revenue": monthly_revenue,
            "Growth_Pct": growth_pct,
            "Active_Users": active_users,
            "AOV": aov,
        }
    )
    return summary.round(2)
