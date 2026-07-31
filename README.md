# Telco Customer Churn — Production ML Pipeline

An end-to-end machine learning system that predicts telecom customer churn, built with production engineering practices: experiment tracking, automated testing, containerization, and CI/CD.

## Problem

Telco companies lose revenue when customers churn (cancel their subscription). This project builds a classifier that flags at-risk customers so a business can intervene (e.g. retention offers) before they leave.

**Business framing matters here more than accuracy**: missing a churner (false negative) costs the business a customer, while flagging a loyal customer (false positive) just costs an unnecessary offer. So this model is explicitly tuned to **maximize recall** on the churn class, accepting lower precision as a deliberate tradeoff.

## Dataset

[IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 20 features (demographics, account info, services subscribed), binary churn label. ~26.5% churn rate (class imbalance handled via threshold tuning rather than resampling).

## Architecture

Raw CSV → Validation → Preprocessing → Feature Engineering → Model Training
↓
MLflow tracking + model artifact
↓
FastAPI (/predict) + Gradio UI (/ui)
↓
Docker container → CI/CD → Docker Hub


## Key engineering decisions

- **Data validation before training**: a lightweight custom validator (schema + allowed-value + numeric-bounds checks) catches bad incoming data before it reaches the model, rather than failing silently or crashing downstream.
- **Multicollinearity investigated, not blindly fixed**: VIF analysis revealed severe multicollinearity from one-hot encoded "No internet service" flags — a structural artifact of the survey design, not noise. Since XGBoost splits on one feature at a time rather than inverting a correlation matrix, it's robust to this, so the features were kept rather than dropped.
- **Threshold chosen deliberately, not left at 0.5**: a sweep across thresholds (0.5 → 0.15) showed the recall/precision tradeoff explicitly. **Threshold = 0.2** was selected as a business decision — recall of ~86% at the cost of precision dropping to ~47% — rather than defaulting to whatever number a tutorial reported.
- **Hyperparameters tuned with Optuna**, optimizing directly for recall (not accuracy) across 30 trials.
- **Every experiment tracked in MLflow** (params, metrics, model artifacts) — reproducible, comparable across runs.
- **CI gates deployment on tests passing**: the GitHub Actions pipeline runs the full pytest suite before it's allowed to build/push the Docker image, so a broken pipeline can never ship.

## Model performance

| Threshold | Recall | Precision |
|---|---|---|
| 0.5 (default) | 52.4% | 67.1% |
| 0.3 | 78.1% | 53.4% |
| **0.2 (chosen)** | **86.1%** | **46.6%** |
| 0.15 | 91.7% | 43.1% |

Final model: XGBoost, tuned via Optuna (30 trials), threshold = 0.2.

## Tech stack

Python · pandas · scikit-learn · XGBoost · Optuna · MLflow · FastAPI · Gradio · Docker · GitHub Actions · pytest

## Project structure

telco-churn-mlops/
├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Encoded/cleaned dataset
├── notebooks/
│ └── eda.ipynb # Exploratory analysis, correlation, VIF
├── src/
│ ├── data/ # Loading, preprocessing
│ ├── features/ # Encoding logic
│ ├── validation/ # Data quality checks
│ ├── models/ # Training, tuning, MLflow logging
│ └── serving/ # FastAPI + Gradio
├── tests/ # pytest unit tests
├── models/ # Trained model artifact + feature schema
├── run_pipeline.py # End-to-end training entry point
├── Dockerfile
└── .github/workflows/ci.yml


## Running it locally

```bash
# Clone and set up
git clone https://github.com/aryankc70/telco-churn-mlops.git
cd telco-churn-mlops
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train the model (runs full pipeline: load → validate → preprocess → features → train)
python run_pipeline.py

# Run tests
pytest tests/ -v

# Serve the API + UI
uvicorn src.serving.main:app --reload --port 8000
```

Then visit:
- `http://127.0.0.1:8000/docs` — interactive API docs
- `http://127.0.0.1:8000/ui` — Gradio web interface

## Running with Docker

```bash
docker pull aryankc70/telco-churn-api:latest
docker run -d -p 8000:8000 aryankc70/telco-churn-api:latest
```

## CI/CD

Every push to `main` triggers:
1. **Test job**: installs dependencies, runs the full pytest suite
2. **Build & push job** (gated on tests passing): builds the Docker image and pushes it to Docker Hub

See `.github/workflows/ci.yml`.