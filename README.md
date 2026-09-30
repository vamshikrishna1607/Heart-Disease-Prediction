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

The dataset used here (`heart.csv`, 1025 rows) is a commonly-shared mirror
of the UCI Cleveland Heart Disease data. It only has **~302 unique patient
records** — the rest of the rows are exact duplicates padding it out to
1025. A naive random train/test split would let copies of the same patient
appear on both sides of the split, which inflates reported accuracy.

To avoid that, `train_model.py` groups rows by a hash of their full values
and uses `GroupShuffleSplit` / `GroupKFold` so that every duplicate of a
given record stays on the same side of the split (verified with an
`assert` that train/test groups are disjoint). On that honest,
patient-level split:

| Model | 5-fold CV accuracy | Test accuracy | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.810 ± 0.054 | 0.773 | 0.898 |
| **Random Forest (deployed)** | 0.797 ± 0.069 | **0.787** | 0.895 |
| SVM (RBF) | 0.786 ± 0.045 | 0.773 | 0.873 |

Random Forest was selected (highest test accuracy) and is the model this
app actually calls. ~78–79% accuracy on ~300 unique patient records is a
realistic number for this dataset — not a inflated one.

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
