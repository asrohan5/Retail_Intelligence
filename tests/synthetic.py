"""Synthetic retail transactions with the same schema as Online Retail II.

Customers have a random 'lifetime', so later activity genuinely depends on earlier
behaviour and there is real churn signal for the model to find.
"""

import numpy as np
import pandas as pd


def make_transactions(n_customers: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2010-12-01"), pd.Timestamp("2011-12-09")
    rows, inv_no = [], 100000

    def line(inv, cid, when, qty=None):
        return {
            "Invoice": str(inv),
            "StockCode": str(int(rng.integers(10000, 10150))),
            "Description": "ITEM",
            "Quantity": int(rng.integers(1, 20)) if qty is None else qty,
            "InvoiceDate": when,
            "Price": float(rng.uniform(0.5, 15)),
            "Customer ID": cid,
            "Country": "United Kingdom",
        }

    for cid in range(12000, 12000 + n_customers):
        first = start + pd.Timedelta(days=int(rng.integers(0, 260)))
        last = min(first + pd.Timedelta(days=int(rng.integers(15, 360))), end)
        span = max(1, int((last - first).total_seconds()))
        n_orders = max(1, int(rng.poisson(max(1.0, (last - first).days / 35))))
        for _ in range(n_orders):
            when = first + pd.Timedelta(seconds=int(rng.integers(0, span)))
            inv_no += 1
            for _ in range(int(rng.integers(1, 6))):
                rows.append(line(inv_no, float(cid), when))
            if rng.random() < 0.05:
                inv_no += 1
                rows.append(line(f"C{inv_no}", float(cid), when, qty=-int(rng.integers(1, 5))))

    for _ in range(80):  # guest checkouts (no Customer ID)
        inv_no += 1
        rows.append(line(inv_no, np.nan, start + pd.Timedelta(days=int(rng.integers(0, 365)))))

    return pd.DataFrame(rows)
