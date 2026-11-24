"""Model evaluation: baselines, cross-validation, uncertainty, diagnostics, out-of-time test.

Evaluation protocol used by this project
----------------------------------------
1. One stratified 80/20 split. The 20% test set is never used for tuning or model selection.
2. Every candidate is tuned with stratified 5-fold CV on the 80% training part only.
3. Candidates are compared on repeated-CV AUC (training part). Test AUC (with a bootstrap CI)
   is reported but NOT used to pick the model.
4. The chosen model is then stress-tested out-of-time (train on an earlier cutoff, test on a later one).
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, learning_curve, validation_curve

from src import config
from src.features import MODEL_FEATURES

logger = logging.getLogger(__name__)

# Simplest -> most complex. Used by the one-standard-error selection rule.
COMPLEXITY_ORDER = ["logistic_regression", "random_forest", "hist_gradient_boosting"]

VALIDATION_PARAMS = {
    "logistic_regression": ("clf__C", [0.001, 0.01, 0.1, 1, 10, 100]),
    "random_forest": ("max_depth", [1, 2, 3, 5, 8, 12, 16]),
    "hist_gradient_boosting": ("max_depth", [1, 2, 3, 4, 6, 8]),
}


def _plt():
    import matplotlib.pyplot as plt

    os.makedirs(config.PLOT_DIR, exist_ok=True)
    return plt


def _save(plt, name: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, name))
    plt.close()


# ----------------------------------------------------------------- uncertainty & comparison
def bootstrap_auc_ci(y_true, y_score, n_boot: int = 1000, seed: int = config.RANDOM_STATE) -> tuple[float, float]:
    """95% percentile-bootstrap confidence interval for ROC-AUC."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def baseline_report(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """What do trivial rules score? A model has to beat these to be worth anything."""
    majority = max(y_test.mean(), 1 - y_test.mean())
    rows = [
        {"baseline": "predict majority class (accuracy)", "score": majority},
        {"baseline": "random ranking (ROC-AUC)", "score": 0.5},
        {"baseline": "recency alone, higher = riskier (ROC-AUC)", "score": roc_auc_score(y_test, X_test["recency"])},
        {"baseline": "frequency alone, lower = riskier (ROC-AUC)", "score": roc_auc_score(y_test, -X_test["frequency"])},
    ]
    return pd.DataFrame(rows)


def compare_models(fitted: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Repeated-CV AUC on train + bootstrap-CI test AUC for each tuned candidate."""
    cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS, random_state=config.RANDOM_STATE)
    rows = []
    for name, search in fitted.items():
        est = search.best_estimator_
        cv_scores = cross_val_score(clone(est), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        proba = est.predict_proba(X_test)[:, 1]
        low, high = bootstrap_auc_ci(y_test, proba)
        rows.append(
            {
                "model": name,
                "cv_auc_mean": cv_scores.mean(),
                "cv_auc_std": cv_scores.std(),
                "test_auc": roc_auc_score(y_test, proba),
                "test_auc_ci_low": low,
                "test_auc_ci_high": high,
                "test_brier": brier_score_loss(y_test, proba),
                "best_params": search.best_params_,
            }
        )
    return pd.DataFrame(rows)


def select_simplest_within_noise(summary: pd.DataFrame) -> str:
    """One-standard-error rule: pick the SIMPLEST model whose CV AUC is within one
    standard deviation of the best. If extra complexity does not buy a difference larger
    than the noise, it is not worth shipping."""
    best = summary.loc[summary["cv_auc_mean"].idxmax()]
    threshold = best["cv_auc_mean"] - best["cv_auc_std"]
    eligible = summary[summary["cv_auc_mean"] >= threshold]
    for name in COMPLEXITY_ORDER:
        if name in set(eligible["model"]):
            return name
    return best["model"]


# ----------------------------------------------------------------- business-facing tables
def lift_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift: if we contact the riskiest X% of customers, what share of churners do we reach?"""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)}).sort_values("p", ascending=False)
    df["decile"] = pd.qcut(np.arange(len(df)), n_bins, labels=range(1, n_bins + 1))
    base_rate = df["y"].mean()
    table = df.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"))
    table["churn_rate"] = table["churners"] / table["customers"]
    table["lift"] = table["churn_rate"] / base_rate
    table["cum_pct_of_churners_captured"] = table["churners"].cumsum() / table["churners"].sum()
    return table.reset_index()


def threshold_summary(y_true, y_score, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    """Precision / recall / share flagged at several probability cut-offs."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rows = []
    for t in thresholds:
        flagged = y_score >= t
        tp = int((flagged & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": t,
                "share_flagged": flagged.mean(),
                "precision": tp / flagged.sum() if flagged.sum() else np.nan,
                "recall": tp / (y_true == 1).sum(),
            }
        )
    return pd.DataFrame(rows)


def subgroup_auc(X_test: pd.DataFrame, y_test, y_score) -> pd.DataFrame:
    """Error analysis: does ranking quality hold across customer types?"""
    freq = X_test["frequency"].to_numpy()
    groups = {"1 order": freq == 1, "2-3 orders": (freq >= 2) & (freq <= 3), "4+ orders": freq >= 4}
    y_test, y_score = np.asarray(y_test), np.asarray(y_score)
    rows = []
    for label, mask in groups.items():
        ok = mask.sum() > 0 and len(np.unique(y_test[mask])) == 2
        rows.append(
            {
                "group": label,
                "customers": int(mask.sum()),
                "churn_rate": y_test[mask].mean() if mask.sum() else np.nan,
                "auc": roc_auc_score(y_test[mask], y_score[mask]) if ok else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- plots
def plot_learning_curve(name: str, estimator, X, y) -> None:
    """Gap between train and validation AUC => variance; both low => bias; still rising => more data helps."""
    plt = _plt()
    cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=1, random_state=config.RANDOM_STATE)
    sizes, train_scores, val_scores = learning_curve(
        clone(estimator), X, y, cv=cv, scoring="roc_auc", train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1
    )
    plt.figure(figsize=(8, 5))
    for scores, label in ((train_scores, "Training AUC"), (val_scores, "Validation AUC")):
        plt.plot(sizes, scores.mean(axis=1), marker="o", label=label)
        plt.fill_between(sizes, scores.mean(axis=1) - scores.std(axis=1), scores.mean(axis=1) + scores.std(axis=1), alpha=0.15)
    plt.xlabel("Training set size")
    plt.ylabel("ROC-AUC")
    plt.title(f"Learning curve: {name}")
    plt.legend()
    _save(plt, "7_learning_curve.png")


def plot_validation_curve(name: str, estimator, X, y) -> None:
    """Sweep the main complexity knob; shows where the model starts overfitting."""
    if name not in VALIDATION_PARAMS:
        return
    plt = _plt()
    param, values = VALIDATION_PARAMS[name]
    cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=1, random_state=config.RANDOM_STATE)
    train_scores, val_scores = validation_curve(
        clone(estimator), X, y, param_name=param, param_range=values, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    plt.figure(figsize=(8, 5))
    for scores, label in ((train_scores, "Training AUC"), (val_scores, "Validation AUC")):
        plt.plot(values, scores.mean(axis=1), marker="o", label=label)
        plt.fill_between(values, scores.mean(axis=1) - scores.std(axis=1), scores.mean(axis=1) + scores.std(axis=1), alpha=0.15)
    if param.endswith("C"):
        plt.xscale("log")
    plt.xlabel(param)
    plt.ylabel("ROC-AUC")
    plt.title(f"Validation curve: {name}")
    plt.legend()
    _save(plt, "8_validation_curve.png")


def plot_calibration(y_true, y_score) -> None:
    """If the model says 0.7, do ~70% of those customers actually churn?"""
    plt = _plt()
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.xlabel("Mean predicted churn probability")
    plt.ylabel("Observed churn rate")
    plt.title("Calibration (test set)")
    plt.legend()
    _save(plt, "9_calibration.png")


def plot_pr_curve(y_true, y_score) -> None:
    plt = _plt()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.axhline(np.mean(y_true), color="k", linestyle="--", label="No-skill (base churn rate)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall (AP = {average_precision_score(y_true, y_score):.3f})")
    plt.legend()
    _save(plt, "10_precision_recall.png")


def plot_confusion(y_true, y_score, threshold: float = 0.5) -> None:
    import seaborn as sns

    plt = _plt()
    cm = confusion_matrix(y_true, (np.asarray(y_score) >= threshold).astype(int))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted (1 = churn)")
    plt.ylabel("Actual (1 = churn)")
    plt.title(f"Confusion matrix @ threshold {threshold}")
    _save(plt, "5_confusion_matrix.png")


def plot_permutation_importance(estimator, X_test, y_test) -> pd.DataFrame:
    """Model-agnostic importance: how much does test AUC drop when one feature is shuffled?
    (Unlike impurity-based importances, this is not biased toward high-cardinality features.)"""
    plt = _plt()
    result = permutation_importance(
        estimator, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    imp = pd.DataFrame(
        {"feature": X_test.columns, "auc_drop": result.importances_mean, "std": result.importances_std}
    ).sort_values("auc_drop")
    plt.figure(figsize=(8, 6))
    plt.barh(imp["feature"], imp["auc_drop"], xerr=imp["std"], color="forestgreen")
    plt.xlabel("Drop in test ROC-AUC when feature is shuffled")
    plt.title("Permutation importance")
    _save(plt, "6_permutation_importance.png")
    return imp.sort_values("auc_drop", ascending=False)


# ----------------------------------------------------------------- out-of-time & orchestration
def out_of_time_validation(estimator, df_train_period: pd.DataFrame, df_test_period: pd.DataFrame) -> dict:
    """Train on an EARLIER cutoff, score a LATER one. The honest test of 'will this still work next quarter?'"""
    model = clone(estimator).fit(df_train_period[MODEL_FEATURES], df_train_period["is_churned"])
    proba = model.predict_proba(df_test_period[MODEL_FEATURES])[:, 1]
    y = df_test_period["is_churned"]
    low, high = bootstrap_auc_ci(y, proba)
    result = {"auc": roc_auc_score(y, proba), "ci_low": low, "ci_high": high, "n_test": len(y)}
    logger.info(
        "Out-of-time AUC: %.3f (95%% CI %.3f-%.3f) | train churn %.1f%% -> test churn %.1f%%",
        result["auc"], low, high, df_train_period["is_churned"].mean() * 100, y.mean() * 100,
    )
    return result


def run_diagnostics(name: str, estimator, X_train, y_train, X_test, y_test) -> None:
    """Run and log every diagnostic for the chosen model."""
    proba = estimator.predict_proba(X_test)[:, 1]

    logger.info("--- Baselines ---\n%s", baseline_report(X_test, y_test).round(3).to_string(index=False))
    logger.info("--- Lift by decile (test) ---\n%s", lift_table(y_test, proba).round(3).to_string(index=False))
    logger.info("--- Threshold trade-offs (test) ---\n%s", threshold_summary(y_test, proba).round(3).to_string(index=False))
    logger.info("--- AUC by customer type (test) ---\n%s", subgroup_auc(X_test, y_test, proba).round(3).to_string(index=False))

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    lift_table(y_test, proba).to_csv(os.path.join(config.OUTPUT_DIR, "lift_table.csv"), index=False)

    plot_learning_curve(name, estimator, X_train, y_train)
    plot_validation_curve(name, estimator, X_train, y_train)
    plot_calibration(y_test, proba)
    plot_pr_curve(y_test, proba)
    plot_confusion(y_test, proba)
    imp = plot_permutation_importance(estimator, X_test, y_test)
    logger.info("--- Permutation importance ---\n%s", imp.round(4).to_string(index=False))
