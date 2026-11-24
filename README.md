# Retail Intelligence Engine

Customer analytics for an online retailer: **RFM segmentation and cohort reporting**, a **90-day churn model** built with strictly time-based validation, and a **FastAPI service** with automated tests and **GitHub Actions CI**.

The goal of this project is engineering and methodology: a leak-free pipeline, honest evaluation, a tested API, and CI. It is not a claim of a state-of-the-art churn model. The results section says plainly what the model can and cannot do.

## What it does

1. **Reporting layer** – RFM scores and business segments (Champions, Loyalists, At Risk, Hibernating, ...), monthly cohort retention, revenue growth, product performance.
2. **Churn model** – predicts whether a customer will place *no* order in the 90 days after a cutoff date. Features are built only from data before the cutoff; labels come only from the window after it.
3. **Serving** – a FastAPI app that loads the saved model once at startup and exposes `/predict_churn` and `/health`.
4. **CI** – GitHub Actions runs the test suite on Python 3.11 and 3.13 for every push and pull request.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Place the dataset at `data/online_retail_II.xlsx` (sheet `Year 2010-2011`; Online Retail transactions from the UCI Machine Learning Repository). The data is not committed to the repo.

```bash
python -m src.main               # full pipeline: reports, features, training, diagnostics, saves model
pytest tests/ -v                 # test suite (no trained model or data needed)
uvicorn src.api:app --reload     # serve the model; docs at http://127.0.0.1:8000/docs
```

The first run parses the Excel file (about a minute) and caches it; the model search takes several minutes. Each stage can be switched off with the `RUN_*` flags at the top of `src/main.py`.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict_churn -H "Content-Type: application/json" -d '{
  "recency": 20, "frequency": 6, "monetary": 400, "spend_velocity": 0.3,
  "purchase_interval_std": 12, "unique_products": 15, "unique_categories": 8,
  "weekend_ratio": 0.1, "avg_unit_price": 6, "avg_hour": 12,
  "whale_score": 1.0, "return_rate": 0.0
}'
```

Out-of-range inputs (for example `weekend_ratio: 1.5`) return HTTP 422; a missing model returns HTTP 503.

## Project layout

```
src/
  config.py          paths, cutoffs, churn window, evaluation settings
  ingestion.py       Excel load + cache, customer-level sampling
  validation.py      data integrity checks (outliers, cancellations, zero prices, ...)
  metrics.py         RFM scores, segments, cohorts, retention, executive summary
  features.py        customer-level features (computed as of a reference date)
  data_factory.py    time-based churn dataset (features before cutoff, label after)
  model.py           candidate models, tuning, selection, final fit
  evaluation.py      baselines, CV, bootstrap CIs, diagnostics, out-of-time test
  serialization.py   save/load model artifact
  api.py             FastAPI service
  main.py            pipeline orchestrator
tests/               API tests (mocked model) + regression tests for data/feature logic
.github/workflows/   CI
```

## Key design decisions

| Decision | Why |
|---|---|
| Time-based split, not random | Churn is defined by *future* behaviour. A random split leaks the future into the features. |
| Fixed 90-day label window | Same churn definition at every cutoff; the window closes inside the observed data. |
| Features computed "as of" the cutoff | Recency and spend windows are relative to the cutoff, not to the last transaction in the slice. |
| Purchase-level features per invoice | Line items on one invoice share a timestamp; per-line gaps would be mostly zero. |
| Log-ratio spend velocity | A plain ratio explodes when prior spend is 0; the log difference is defined everywhere. |
| Preprocessing inside one `Pipeline` | Transforms are re-fit per CV fold (no leakage) and the same object is served (no train/serve skew). |
| Logistic regression, random forest, gradient boosting | Interpretable baseline vs. non-linear models. Chosen by the simplest-within-one-standard-deviation rule on CV AUC; the test set is never used for selection. |
| Mocked model in API tests | CI stays fast and needs no data or trained artifact. Model quality is evaluated separately in the pipeline. |

## Results

Data: 4,372 identified customers. At the 2011-09-01 cutoff there are 3,317 customers, of whom 43.8% place no order in the next 90 days. Evaluation uses a stratified 80/20 customer split (664 test customers), 5-fold CV for tuning, and bootstrap confidence intervals.

| Model | CV AUC | Test AUC (95% CI) |
|---|---|---|
| Logistic regression (selected) | 0.748 | 0.747 (0.710 - 0.783) |
| Random forest | 0.752 | 0.742 (0.705 - 0.781) |
| Gradient boosting | 0.748 | 0.740 (0.702 - 0.777) |

| Baseline | Score |
|---|---|
| Predict "stays" for everyone (accuracy) | 0.562 |
| Recency alone (AUC) | 0.657 |
| Purchase frequency alone (AUC) | 0.723 |

Other checks:

- **Out-of-time:** trained on the 2011-06-01 cutoff, scored at 2011-09-01: AUC 0.746 (95% CI 0.729 - 0.762).
- **Calibration:** reasonable (customers scored near 0.7 churn at about 70%); Brier score 0.201 vs. 0.246 for always predicting the base rate.
- **Learning and validation curves:** train and validation AUC converge near 0.75 and the model is insensitive to regularisation strength, i.e. performance is limited by the features, not by tuning or data volume.
- **Lift:** the riskiest 40% of customers contain about 59% of churners, at roughly 65% precision (base rate 44%).

### What the results mean

- The three model families are statistically indistinguishable, so the simplest was shipped.
- **Purchase frequency carries most of the signal.** It alone reaches 0.723 AUC and dominates permutation importance. The full model's edge over it is within the confidence interval.
- Several engineered features (spend velocity, purchase-interval variability, hour of day) add no measurable signal.
- **The model is weak for low-history customers:** AUC 0.57 for one-order customers and 0.59 for two to three orders, versus 0.76 for customers with four or more orders. Part of the overall AUC comes from separating these groups, whose churn rates differ sharply (63%, 46%, 14%).
- Suitable for ranking customers for retention outreach; not suitable for automated decisions.

## Limitations and next steps

- One cutoff covers one season (September to November, the retailer's peak period); churn fell from 51% to 44% between the two periods tested, so probabilities would need recalibration over time.
- `unique_categories` uses a StockCode prefix as a crude proxy, not a real product taxonomy.
- Hyperparameters were selected on the September data, so the out-of-time check is slightly generous.
- No monitoring or scheduled retraining. CI tests the code and API contract, not model quality.
- Next steps: features that target the mechanism (time since last order relative to the customer's typical gap, customer tenure), a separate approach for one-time buyers, a paired test against the frequency-only baseline, and a probabilistic benchmark (BG/NBD).

## Testing and CI

`pytest tests/` covers the API (validation, status codes, risk flag) and regression tests for bugs found during review: customer-level sampling, reachable RFM segments, invoice-level purchase intervals, label-window logic, and a leakage guard that tampers with post-cutoff data and asserts the features do not change.

CI (`.github/workflows/ci.yml`) installs dependencies, runs the suite on Python 3.11 and 3.13, and checks that the API imports.
