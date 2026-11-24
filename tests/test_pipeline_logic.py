"""Regression tests for the data/feature logic: each test guards a bug that was found and fixed."""

import numpy as np
import pandas as pd

from src.data_factory import create_churn_dataset
from src.features import MODEL_FEATURES, engineer_features
from src.ingestion import sample_customers
from src.metrics import clean_data_for_metrics, segment_for_scores

CUTOFF = pd.Timestamp("2011-09-01")


def _frames(transactions):
    df_raw = transactions.copy()
    return df_raw, clean_data_for_metrics(df_raw)


def test_at_risk_segment_is_reachable():
    # Low recency score + high frequency used to be swallowed by the broader "Loyalists" rule
    assert segment_for_scores(r=1, f=5, m=3) == "At Risk"
    assert segment_for_scores(r=5, f=5, m=5) == "Champions"
    assert segment_for_scores(r=4, f=4, m=2) == "Loyalists"


def test_sampling_keeps_whole_customers(transactions):
    sampled = sample_customers(transactions, frac=0.3, random_state=1)
    full_counts = transactions.dropna(subset=["Customer ID"]).groupby("Customer ID").size()
    kept_counts = sampled.dropna(subset=["Customer ID"]).groupby("Customer ID").size()
    assert 0 < len(kept_counts) < len(full_counts)
    # every sampled customer retains ALL of their rows (row-level sampling would not)
    assert (kept_counts == full_counts.loc[kept_counts.index]).all()


def test_features_have_no_nans_and_stay_bounded(transactions):
    df_raw, df_clean = _frames(transactions)
    feats = engineer_features(df_clean, df_raw, ref_date=CUTOFF)
    assert not feats[MODEL_FEATURES].isna().any().any()
    assert feats["spend_velocity"].abs().max() < 20  # log-ratio, not raw currency


def test_purchase_interval_uses_invoices_not_line_items():
    # One customer, 3 invoices 10 days apart, each with 4 line items (same timestamp).
    dates = pd.to_datetime(["2011-01-01", "2011-01-11", "2011-01-21"])
    rows = [
        {"Invoice": f"{i}", "StockCode": f"A{j}", "Quantity": 1, "Price": 1.0, "TotalAmount": 1.0,
         "InvoiceDate": d, "Customer ID": 1.0}
        for i, d in enumerate(dates) for j in range(4)
    ]
    df = pd.DataFrame(rows)
    feats = engineer_features(df, df, ref_date=pd.Timestamp("2011-02-01"))
    # gaps are [10, 10] -> std 0. Line-item gaps would be [0,0,0,10,0,0,0,10,...] -> std > 0.
    assert feats.loc[1.0, "purchase_interval_std"] == 0.0
    assert feats.loc[1.0, "frequency"] == 3


def test_features_never_see_the_future(transactions):
    """Leakage guard: changing post-cutoff data must not change any pre-cutoff feature."""
    df_raw, df_clean = _frames(transactions)
    base = create_churn_dataset(df_clean, df_raw, cutoff_date=CUTOFF)

    tampered_clean = df_clean.copy()
    future = tampered_clean["InvoiceDate"] > CUTOFF
    tampered_clean.loc[future, "Quantity"] *= 50
    tampered_clean.loc[future, "TotalAmount"] *= 50
    tampered = create_churn_dataset(tampered_clean, df_raw, cutoff_date=CUTOFF)

    pd.testing.assert_frame_equal(base[MODEL_FEATURES], tampered[MODEL_FEATURES])


def test_churn_label_follows_the_window():
    def order(cid, day, inv):
        return {"Invoice": inv, "StockCode": "A1", "Quantity": 1, "Price": 1.0, "TotalAmount": 1.0,
                "InvoiceDate": pd.Timestamp(day), "Customer ID": cid}

    df = pd.DataFrame([
        order(1.0, "2011-08-01", "a"), order(1.0, "2011-09-15", "b"),  # buys inside window -> stays
        order(2.0, "2011-08-01", "c"),                                  # silent after cutoff -> churned
        order(3.0, "2011-08-01", "d"), order(3.0, "2011-12-20", "e"),  # buys only AFTER the window -> churned
        order(4.0, "2011-12-30", "f"),                                  # extends the data's end date
    ])
    out = create_churn_dataset(df, df, cutoff_date=CUTOFF, horizon_days=90)
    assert out.loc[1.0, "is_churned"] == 0
    assert out.loc[2.0, "is_churned"] == 1
    assert out.loc[3.0, "is_churned"] == 1
    assert 4.0 not in out.index  # first purchase after cutoff: not a customer yet at the cutoff
