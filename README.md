# Heart Disease Risk Predictor

A full Django web app that estimates a person's risk of heart disease from
11 clinical measurements, using a scikit-learn model trained on the
Heart Failure Prediction dataset (a harmonized merge of five real clinical
heart-disease cohorts).

**This is a portfolio / learning project. It is not a medical device and
must never be used for real diagnostic or treatment decisions.**

## What it does

- A form collects 11 clinical inputs (age, sex, chest pain type, resting
  blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart
  rate, exercise-induced angina, ST depression, and the slope of the peak
  exercise ST segment).
- On submit, Django loads a pre-trained scikit-learn `Pipeline`
  (`StandardScaler` + classifier) and returns a risk label with an
  estimated probability.
- An **About the model** page (`/about/`) documents the dataset, the
  train/test methodology, and each candidate model's measured accuracy —
  including its limitations. Nothing about the model's performance is
  hidden or rounded up.

## Honest note on the dataset and accuracy

The dataset used here (`heart.csv`, 918 rows) is the **Heart Failure
Prediction Dataset** — a harmonized merge of five real clinical
heart-disease cohorts (Cleveland, Hungarian, Switzerland, Long Beach VA,
and the Nashville/Stalog dataset). Every row is a distinct patient record
(verified with a plain duplicate-row count in `train_model.py`), so an
ordinary stratified 80/20 train/test split is honest here — no
duplicate-leakage correction is needed, unlike the smaller 303-patient
Cleveland-only dataset this project started with.

On one fixed stratified 80/20 split (`random_state=42`, the same split
used for every candidate model below):

| Model | 5-fold CV accuracy | Test accuracy | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.850 ± 0.041 | 0.886 | 0.930 |
| Random Forest | 0.858 ± 0.036 | 0.897 | 0.935 |
| Gradient Boosting | 0.854 ± 0.026 | 0.891 | 0.931 |
| **SVM, RBF kernel (deployed)** | 0.854 ± 0.045 | **0.897** | **0.949** |

SVM (RBF) was selected — it ties for the best test accuracy and has the
best ROC-AUC (best at ranking risk, not just classifying at one threshold).

**On hitting "90%":** this split lands at 89.7%, a real result, not an
inflated one. But re-running the same model across 30 different random
80/20 splits gives accuracy ranging from about 81% to 91%, averaging
~87% — only ~1 in 6 random splits clears 90%. Reporting a cherry-picked
lucky split as "90%+ accuracy" would be misleading, so this README and the
app's About page report the honest range rather than the single best
number. Still, switching from the smaller, duplicate-padded Cleveland
dataset (~79% honest test accuracy) to this larger, more diverse one is a
genuine, substantial improvement — not a re-measurement of the same data.

## CI/CD and MLOps

This project has three GitHub Actions workflows, each doing a distinct job rather than one big "deploy" script:

- **`ci.yml`** — on every push/PR to `main`: installs dependencies, runs `manage.py check`, runs the full test suite (`predictor/tests.py` — form validation, view behavior, and a sanity check on the committed model artifact itself), and verifies `collectstatic` succeeds under the production static-file config. This is the gate everything else depends on.
- **`retrain.yml`** — manual trigger, or automatic when `heart.csv` or `train_model.py` changes on `main`. Retrains the model, then runs `scripts/check_model_regression.py`, which compares the new test accuracy against the currently-committed baseline and **fails the job if accuracy drops by more than 3 percentage points** — a regression never gets the chance to be merged, automated or not. If the retrain passes the gate and the test suite, it opens a **pull request** with the updated model artifact rather than committing straight to `main`; a human still reviews and merges. Automating the retrain doesn't mean automating away the judgment call about whether to ship it.
- **`publish.yml`** — runs only after `ci.yml` succeeds on `main`, and builds + pushes the Docker image to GitHub Container Registry (`ghcr.io`) using the repo's own automatic `GITHUB_TOKEN` — no manually-managed registry credentials for anyone who forks this.

Why a regression gate and a PR instead of straight-to-`main` auto-deploy: the entire point of automating retraining is to catch data/behavior drift early, not to remove the human decision of "is this model actually better." An automated pipeline that can silently ship a worse model isn't safer than doing it by hand — it's just a faster way to ship a worse model. The gate and the PR step are what make this an actual MLOps workflow rather than a script that happens to run on a schedule.

## Project structure

```
heart-disease-prediction/
├── heart.csv                          # training data
├── train_model.py                     # trains & selects the model
├── scripts/
│   └── check_model_regression.py      # accuracy-regression gate used by retrain.yml
├── manage.py
├── Dockerfile
├── .github/workflows/
│   ├── ci.yml                         # test on every push/PR
│   ├── retrain.yml                    # gated retrain -> PR
│   └── publish.yml                    # build & push image to GHCR after CI passes
├── heart_disease_prediction/          # Django project settings/urls
└── predictor/                         # Django app
    ├── forms.py                       # HeartDiseasePredictionForm
    ├── views.py                       # predict() and about() views
    ├── tests.py                       # form/view/model-artifact tests
    ├── urls.py
    ├── ml_model/
    │   ├── heart_model.joblib         # trained sklearn Pipeline
    │   └── model_metrics.json         # measured accuracy per model
    └── templates/predictor/
        ├── base.html
        ├── predict.html
        └── about.html
static/css/style.css
```

## Running it locally

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python3 manage.py migrate
python3 manage.py runserver
```

Then open http://127.0.0.1:8000/ in a browser.

To run the test suite:

```bash
python3 manage.py test predictor -v 2
```

To retrain the model from scratch (e.g. after changing `heart.csv`):

```bash
python3 train_model.py
```

This regenerates `predictor/ml_model/heart_model.joblib` and
`model_metrics.json`.

## Running it with Docker

```bash
docker build -t heart-disease-prediction .
docker run -p 8000:8000 \
  -e DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1 \
  -e DJANGO_SECRET_KEY=some-real-secret-in-production \
  heart-disease-prediction
```

The container runs with `DEBUG=false` and Gunicorn (not the dev server), and serves static files via WhiteNoise with hashed, compressed filenames — the same production-shaped config `ci.yml` verifies on every push (`DJANGO_ALLOWED_HOSTS` must be set or Django will reject all requests with 400s; `DJANGO_SECRET_KEY` has a dev-only fallback that should never be used for a real deployment).

## Tech stack

Django 5, scikit-learn, pandas, numpy, joblib, Gunicorn, WhiteNoise, Docker, GitHub Actions (CI, gated retraining, GHCR image publishing). No JavaScript framework — plain HTML/CSS templates.
