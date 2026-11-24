"""Pipeline orchestrator. Run from the project root:  python -m src.main

Toggle the RUN_* flags to control which stages execute. Stages depend on files written by
earlier stages (the first full run needs everything on).
"""

import logging
import os

import pandas as pd

from src import config, features
from src import evaluation as ev
from src import model as ml_engine
from src import serialization
from src import visualization as viz
from src.data_factory import create_churn_dataset
from src.features import MODEL_FEATURES
from src.ingestion import load_and_optimize_data
from src.logging_config import setup_logging
from src.metrics import (
    assign_rfm_scores,
    build_executive_summary,
    calculate_cohort_index,
    calculate_rfm_values,
    clean_data_for_metrics,
    define_customer_segments,
    get_retention_matrix,
)

RUN_VIZ_AND_REPORTING = True   # RFM segments, cohort retention, revenue charts
RUN_FEATURE_ENGG = True        # sanity-check feature matrix at the data's last date
RUN_TARGET_GEN = True          # build labelled churn datasets (main + earlier cutoff for out-of-time test)
RUN_MODEL_TRAINING = True      # tune, compare, select, refit, save model
RUN_DIAGNOSTICS = True         # learning/validation curves, calibration, lift, permutation importance
RUN_OOT_VALIDATION = True      # train on earlier cutoff, test on later one

MODELING_DATA = os.path.join(config.OUTPUT_DIR, "modeling_data.csv")
OOT_TRAIN_DATA = os.path.join(config.OUTPUT_DIR, "modeling_data_oot_train.csv")

from logger import logging as logger


def run_pipeline() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    df_raw = df_clean = None
    if RUN_VIZ_AND_REPORTING or RUN_FEATURE_ENGG or RUN_TARGET_GEN:
        df_raw = load_and_optimize_data()
        if df_raw is None:
            logger.error("Pipeline failed: could not load data")
            return
        df_clean = clean_data_for_metrics(df_raw)
    else:
        logger.info("Skipping raw data load")

    if RUN_VIZ_AND_REPORTING:
        logger.info("::: RUNNING REPORTING LAYER :::")
        rfm = define_customer_segments(assign_rfm_scores(calculate_rfm_values(df_clean)))
        logger.info("Segment sizes:\n%s", rfm["Segment"].value_counts().to_string())

        retention = get_retention_matrix(calculate_cohort_index(df_clean))
        exec_summary = build_executive_summary(df_clean)

        viz.plot_retention_heatmap(retention)
        viz.plot_segment_distribution(rfm)
        viz.plot_revenue_growth(exec_summary)

    if RUN_FEATURE_ENGG:
        logger.info("::: RUNNING FEATURE ENGINEERING :::")
        feature_matrix = features.engineer_features(df_clean, df_raw)
        print(feature_matrix[["spend_velocity", "return_rate", "whale_score"]].head())
        feature_matrix.to_csv(os.path.join(config.OUTPUT_DIR, "feature_matrix.csv"))

    if RUN_TARGET_GEN:
        logger.info("::: CREATING TRAINING SETS :::")
        train_df = create_churn_dataset(df_clean, df_raw, cutoff_date=pd.Timestamp(config.TRAIN_CUTOFF))
        train_df.to_csv(MODELING_DATA)
        viz.plot_churn_separation(train_df)

        oot_df = create_churn_dataset(df_clean, df_raw, cutoff_date=pd.Timestamp(config.OOT_TRAIN_CUTOFF))
        oot_df.to_csv(OOT_TRAIN_DATA)

    if RUN_MODEL_TRAINING:
        try:
            df_model = pd.read_csv(MODELING_DATA, index_col=0)
        except FileNotFoundError:
            logger.error("CRITICAL: %s not found. Set RUN_TARGET_GEN = True first.", MODELING_DATA)
            return

        result = ml_engine.train_and_select(df_model, run_diagnostics=RUN_DIAGNOSTICS)

        if RUN_OOT_VALIDATION:
            try:
                df_oot_train = pd.read_csv(OOT_TRAIN_DATA, index_col=0)
                ev.out_of_time_validation(result["tuned"], df_oot_train, df_model)
            except FileNotFoundError:
                logger.error("%s not found. Set RUN_TARGET_GEN = True to enable out-of-time validation.", OOT_TRAIN_DATA)

        serialization.save_production_artifacts(
            result["final_model"], MODEL_FEATURES, metadata={"model_name": result["name"]}
        )

    logger.info("+++++ Pipeline execution completed successfully +++++")


if __name__ == "__main__":
    setup_logging()
    run_pipeline()
