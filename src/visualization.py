"""All plotting functions for the retail intelligence pipeline.

Consolidated from the former eda.py + visualization.py — every chart the
pipeline produces lives here, saved under PLOT_PATH.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PLOT_PATH = "plots"
os.makedirs(PLOT_PATH, exist_ok=True)


def plot_churn_separation(df_labeled: pd.DataFrame) -> None:
    """Compare spend velocity between churned and retained customers, and show
    each feature's correlation with the churn target."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_labeled, x="is_churned", y="spend_velocity", showfliers=False)
    plt.title("Impact of Spend Velocity on Churn (Outliers Hidden)")
    plt.xlabel("Churn Status (0=Stayed, 1=Left)")
    plt.ylabel("Spend Velocity (Ratio)")
    plt.savefig(os.path.join(PLOT_PATH, "0_churn_separation_spend_velocity.png"))
    plt.close()

    plt.figure(figsize=(12, 10))
    corr = df_labeled.corr(numeric_only=True)
    sns.heatmap(
        corr[["is_churned"]].sort_values(by="is_churned", ascending=False),
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.title("Feature Correlation with Churn Target")
    plt.savefig(os.path.join(PLOT_PATH, "0_churn_feature_correlation.png"))
    plt.close()


def plot_retention_heatmap(retention_matrix: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 8))
    plt.title("Customer Retention Rate (Monthly Cohorts)", fontsize=16)
    sns.heatmap(
        retention_matrix, annot=True, fmt=".0%", cmap="Blues", cbar_kws={"label": "Retention Rate"}
    )
    plt.xlabel("Cohort Index (Months)")
    plt.ylabel("Cohort Month")
    plt.savefig(os.path.join(PLOT_PATH, "1_customer_retention_rate.png"))
    plt.close()


def plot_segment_distribution(rfm_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 8))
    plt.title("Customer Segmentation Distribution", fontsize=14)

    segment_data = (
        rfm_df.groupby("Segment")
        .agg({"Customer ID": "count", "Monetary": "mean"})
        .sort_values("Customer ID", ascending=False)
        .reset_index()
    )

    sns.barplot(data=segment_data, x="Customer ID", y="Segment", hue="Segment", palette="viridis", legend=False)
    plt.xlabel("No. of Customers")
    plt.ylabel("Market Segment")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "2_customer_segmentation_distribution.png"))
    plt.close()


def plot_revenue_growth(exec_summary: pd.DataFrame) -> None:
    plot_df = exec_summary.iloc[:-1].copy()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("Monthly Revenue and Growth Velocity")
    sns.barplot(x=plot_df.index, y=plot_df["Revenue"], ax=ax1, color="skyblue", alpha=0.7)
    ax1.set_ylabel("Total Revenue")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    sns.lineplot(x=np.arange(len(plot_df)), y=plot_df["Growth_Pct"], ax=ax2, color="red", marker="o", linewidth=2)
    ax2.set_ylabel("Month-over-month growth (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "3_monthly_revenue_growth_velocity.png"))
    plt.close()


def plot_product_performance(df: pd.DataFrame) -> None:
    prod_perf = df.groupby("Description").agg({"Quantity": "sum", "Price": "mean"}).reset_index()
    prod_perf = prod_perf.sort_values("Quantity", ascending=False).head(100)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=prod_perf, x="Quantity", y="Price", alpha=0.6, size="Quantity", sizes=(20, 200))
    plt.title("Top 100 Products: Volume vs. Unit Price")
    plt.xscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(PLOT_PATH, "4_volume_vs_unit_price.png"))
    plt.close()
