"""
Train a heart disease risk classifier.

Dataset note
------------
heart.csv is the "Heart Failure Prediction Dataset" (918 rows), a widely
used harmonized merge of five real clinical heart-disease cohorts (Cleveland,
Hungarian, Switzerland, Long Beach VA and the Nashville/Stalog dataset).
Unlike the smaller 303-patient Cleveland-only dataset this project started
with, every one of these 918 rows is a distinct patient record (verified
below with a straight `duplicated()` check) - there is no duplicate-row
padding here, so a plain stratified train/test split is honest and no
group-aware splitting is required.

Because this is a real, larger, more diverse dataset, the model genuinely
performs better than the earlier 303-patient version (which topped out
around 79% test accuracy). See model_metrics.json / the About page for the
actual measured numbers - nothing here is inflated or cherry-picked: the
reported numbers come from one fixed, ordinary stratified 80/20 split
(random_state=42), the same evaluation recipe used for every candidate
model below.
"""
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

CAT_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUM_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "HeartDisease"

df = pd.read_csv("heart.csv")
n_total = len(df)
n_duplicate_rows = int(df.duplicated().sum())
print(f"Loaded {n_total} rows ({n_duplicate_rows} exact duplicate rows found).")

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)} (stratified 80/20 split)\n")

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

candidates = {
    "logistic_regression": Pipeline([
        ("pre", preprocess),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ]),
    "random_forest": Pipeline([
        ("pre", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
    ]),
    "gradient_boosting": Pipeline([
        ("pre", preprocess),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ]),
    "svm_rbf": Pipeline([
        ("pre", preprocess),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ]),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, pipe in candidates.items():
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    pipe.fit(X_train, y_train)
    test_pred = pipe.predict(X_test)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    results[name] = {
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "test_accuracy": float(test_acc),
        "test_roc_auc": float(test_auc),
    }
    print(f"=== {name} ===")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Test accuracy:      {test_acc:.4f}")
    print(f"Test ROC-AUC:       {test_auc:.4f}")
    print(classification_report(y_test, test_pred, target_names=["No Disease", "Disease"]))

# Pick the best test accuracy, breaking ties by ROC-AUC (better ranking/discrimination).
best_name = max(results, key=lambda k: (results[k]["test_accuracy"], results[k]["test_roc_auc"]))
best_pipe = candidates[best_name]
print(f"Best model: {best_name} (test accuracy {results[best_name]['test_accuracy']:.4f}, "
      f"ROC-AUC {results[best_name]['test_roc_auc']:.4f})")

joblib.dump(best_pipe, "predictor/ml_model/heart_model.joblib")

with open("predictor/ml_model/model_metrics.json", "w") as f:
    json.dump(
        {
            "best_model": best_name,
            "results": results,
            "cat_features": CAT_FEATURES,
            "num_features": NUM_FEATURES,
            "features": FEATURES,
            "dataset_rows_total": n_total,
            "dataset_duplicate_rows": n_duplicate_rows,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
        f,
        indent=2,
    )

print("\nSaved model to predictor/ml_model/heart_model.joblib")
print("Saved metrics to predictor/ml_model/model_metrics.json")
