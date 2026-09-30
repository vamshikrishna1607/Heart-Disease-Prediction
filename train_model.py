"""
Train a heart disease risk classifier.

Dataset note
------------
heart.csv is the popular 1025-row "Heart Disease Dataset" mirror of the UCI
Cleveland data. It is worth knowing that this file is the original ~303
patient Cleveland dataset with rows duplicated to pad it out to 1025 -
there are only ~302 distinct patient records in it. Training and evaluating
naively on a random row-level split of this file lets identical duplicate
rows land on both sides of the split, which leaks test answers into training
and inflates reported accuracy (this is a common, easy-to-miss mistake with
this particular dataset).

To use the full file honestly, every row is assigned a group id based on a
hash of its own values, and the train/test split is done at the GROUP level
(via GroupShuffleSplit) so duplicate copies of the same patient record
always stay together on one side of the split. This keeps the larger
dataset (and the extra training signal from class balance) while still
reporting a test accuracy that has no duplicate-leakage inflation.
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"

df = pd.read_csv("heart.csv")
n_total = len(df)
n_unique = df.drop_duplicates().shape[0]
print(f"Loaded {n_total} rows ({n_unique} distinct patient records, "
      f"{n_total - n_unique} duplicate copies).")

# Group id = hash of the full row, so duplicate rows share a group and can
# never be split across train and test.
df["_group"] = df.apply(lambda r: hash(tuple(r[FEATURES + [TARGET]])), axis=1)

X = df[FEATURES]
y = df[TARGET]
groups = df["_group"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Sanity check: no group should appear in both splits.
assert set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]) == set()
print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}  "
      f"(split at the patient-record level, no duplicate leakage)\n")

candidates = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "random_forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42
        )),
    ]),
    "svm_rbf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ]),
}

results = {}
train_groups = groups.iloc[train_idx]
cv = GroupKFold(n_splits=5)
for name, pipe in candidates.items():
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=cv, groups=train_groups, scoring="accuracy"
    )
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

best_name = max(results, key=lambda k: results[k]["test_accuracy"])
best_pipe = candidates[best_name]
print(f"Best model: {best_name} (test accuracy {results[best_name]['test_accuracy']:.4f})")

joblib.dump(best_pipe, "predictor/ml_model/heart_model.joblib")

with open("predictor/ml_model/model_metrics.json", "w") as f:
    json.dump(
        {
            "best_model": best_name,
            "results": results,
            "features": FEATURES,
            "dataset_rows_total": n_total,
            "dataset_rows_unique": n_unique,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
        f,
        indent=2,
    )

print("\nSaved model to predictor/ml_model/heart_model.joblib")
print("Saved metrics to predictor/ml_model/model_metrics.json")
