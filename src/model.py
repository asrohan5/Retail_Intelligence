"""Candidate churn models: tuning, comparison, selection and final fit."""

import logging

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config
from src import evaluation as ev
from src.features import MODEL_FEATURES
from src.transforms import signed_log1p


from logging import logging as logger

# Heavy right-tailed features get a log transform before scaling (linear model only;
# trees are invariant to monotonic transforms so they use the raw features).
SKEWED_FEATURES = ["monetary", "frequency", "whale_score", "avg_unit_price", "unique_products", "unique_categories"]


def split_data(df: pd.DataFrame):
    """Stratified train/test split on customers. The test set is touched only for final reporting."""
    X, y = df[MODEL_FEATURES], df["is_churned"]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)


def build_logreg_pipeline() -> Pipeline:
    """log-transform skewed features -> standardize -> logistic regression.

    Everything lives inside ONE Pipeline, so within cross-validation the transforms are
    re-fit on each training fold (no leakage), and the exact same object is serialized
    and served (no train/serve skew).
    """
    pre = ColumnTransformer(
        [("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one"), SKEWED_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline(
        [("pre", pre), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
    )


def candidate_specs() -> dict:
    seed = config.RANDOM_STATE
    return {
        "logistic_regression": (
            build_logreg_pipeline(),
            {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            # n_estimators is fixed: more trees never hurt and are not a regularizer, so tuning it wastes budget.
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
            {"max_depth": [2, 3, 4, 6], "min_samples_leaf": [2, 5, 10], "max_features": ["sqrt", 0.5, 0.8]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=seed),
            {"learning_rate": [0.01, 0.03, 0.1], "max_depth": [1, 2, 3], "max_iter": [50, 100, 300], "min_samples_leaf": [10, 20, 50]},
        ),
    }


def _warn_on_grid_edge(name: str, best: dict, grid: dict) -> None:
    """A winner on the edge of its grid means the true optimum may lie outside the search."""
    for param, values in grid.items():
        numeric = [v for v in values if isinstance(v, (int, float))]
        if len(numeric) >= 3 and best[param] in (min(numeric), max(numeric)):
            logger.warning("%s: best %s=%s sits on the edge of the grid %s; consider widening it.", name, param, best[param], values)


def tune_candidates(X_train, y_train) -> dict:
    """Grid-search every candidate with stratified 5-fold CV on ROC-AUC (training data only)."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fitted = {}
    for name, (estimator, grid) in candidate_specs().items():
        logger.info("Tuning %s ...", name)
        search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        logger.info("%s best params: %s | CV AUC %.4f", name, search.best_params_, search.best_score_)
        _warn_on_grid_edge(name, search.best_params_, grid)
        fitted[name] = search
    return fitted


def logistic_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Standardized coefficients of the logistic pipeline (positive = pushes toward churn)."""
    names = [n.split("__")[-1] for n in pipeline.named_steps["pre"].get_feature_names_out()]
    coefs = pipeline.named_steps["clf"].coef_[0]
    return pd.DataFrame({"feature": names, "coefficient": coefs}).sort_values("coefficient", ascending=False)


def train_and_select(df_model: pd.DataFrame, run_diagnostics: bool = True) -> dict:
    """Tune all candidates, compare them honestly, pick one, and refit it on all data.

    Returns a dict with the final (all-data) model, the chosen name, the tuned estimator
    (fit on the training split only, needed for diagnostics / out-of-time checks) and the summary table.
    """
    X_train, X_test, y_train, y_test = split_data(df_model)
    logger.info("Train: %d customers | Test: %d customers | churn rate %.1f%%", len(X_train), len(X_test), y_train.mean() * 100)

    fitted = tune_candidates(X_train, y_train)
    summary = ev.compare_models(fitted, X_train, y_train, X_test, y_test)
    logger.info("--- Model comparison ---\n%s", summary.drop(columns="best_params").round(4).to_string(index=False))

    best_name = ev.select_simplest_within_noise(summary)
    tuned = fitted[best_name].best_estimator_
    logger.info("Selected model: %s (simplest within one std of the best CV AUC)", best_name)

    if best_name == "logistic_regression":
        logger.info("--- Logistic coefficients ---\n%s", logistic_coefficients(tuned).round(3).to_string(index=False))

    if run_diagnostics:
        ev.run_diagnostics(best_name, tuned, X_train, y_train, X_test, y_test)

    # Production model: same hyperparameters, refit on ALL labelled customers.
    X_all, y_all = df_model[MODEL_FEATURES], df_model["is_churned"]
    final_model = clone(tuned).fit(X_all, y_all)

    return {"final_model": final_model, "name": best_name, "tuned": tuned, "summary": summary}
