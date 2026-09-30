# Heart Disease Risk Predictor

A full Django web app that estimates a person's risk of heart disease from
13 clinical measurements, using a scikit-learn model trained on the UCI
Cleveland Heart Disease dataset.

**This is a portfolio / learning project. It is not a medical device and
must never be used for real diagnostic or treatment decisions.**

## What it does

- A form collects 13 clinical inputs (age, sex, chest pain type, resting
  blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart
  rate, exercise-induced angina, ST depression, ST slope, number of major
  vessels, thalassemia).
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

## Project structure

```
heart-disease-prediction/
├── heart.csv                          # training data
├── train_model.py                     # trains & selects the model
├── manage.py
├── heart_disease_prediction/          # Django project settings/urls
└── predictor/                         # Django app
    ├── forms.py                       # HeartDiseasePredictionForm
    ├── views.py                       # predict() and about() views
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

To retrain the model from scratch (e.g. after changing `heart.csv`):

```bash
python3 train_model.py
```

This regenerates `predictor/ml_model/heart_model.joblib` and
`model_metrics.json`.

## Tech stack

Django 5, scikit-learn, pandas, numpy, joblib. No JavaScript framework —
plain HTML/CSS templates.
