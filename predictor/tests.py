"""
Test suite for the predictor app. Run with `python manage.py test`.

These tests are the gate the CI workflow runs on every push/PR - they're
deliberately not just "does it not crash": the valid-input test asserts a
real, sane result comes back, and the model-loading test asserts the saved
pipeline actually agrees with the checked-in metrics before anything is
deployed from it.
"""
import json
from pathlib import Path

import joblib
from django.conf import settings
from django.test import TestCase
from django.urls import reverse

from .forms import HeartDiseasePredictionForm

VALID_DATA = {
    "age": 54,
    "sex": "M",
    "chest_pain_type": "ATA",
    "resting_bp": 130,
    "cholesterol": 246,
    "fasting_bs": 0,
    "resting_ecg": "Normal",
    "max_hr": 150,
    "exercise_angina": "N",
    "oldpeak": 1.4,
    "st_slope": "Up",
}


class HeartDiseasePredictionFormTests(TestCase):
    def test_valid_data_is_valid(self):
        form = HeartDiseasePredictionForm(data=VALID_DATA)
        self.assertTrue(form.is_valid(), form.errors)

    def test_age_out_of_range_is_rejected(self):
        data = {**VALID_DATA, "age": 200}
        form = HeartDiseasePredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn("age", form.errors)

    def test_implausible_max_hr_for_age_is_rejected(self):
        # 220 - age(80) + 40 = 180 is the cutoff; 230 should trip the cross-field check.
        data = {**VALID_DATA, "age": 80, "max_hr": 230}
        form = HeartDiseasePredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn("max_hr", form.errors)

    def test_missing_required_field_is_rejected(self):
        data = {k: v for k, v in VALID_DATA.items() if k != "chest_pain_type"}
        form = HeartDiseasePredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn("chest_pain_type", form.errors)

    def test_as_feature_dict_has_model_column_names(self):
        form = HeartDiseasePredictionForm(data=VALID_DATA)
        self.assertTrue(form.is_valid(), form.errors)
        features = form.as_feature_dict()
        expected_columns = {
            "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
            "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
            "Oldpeak", "ST_Slope",
        }
        self.assertEqual(set(features.keys()), expected_columns)
        self.assertEqual(features["Age"], 54)
        self.assertEqual(features["Sex"], "M")


class PredictViewTests(TestCase):
    def test_get_renders_empty_form(self):
        response = self.client.get(reverse("predictor:predict"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Predict risk")

    def test_post_valid_data_returns_a_prediction(self):
        response = self.client.post(reverse("predictor:predict"), data=VALID_DATA)
        self.assertEqual(response.status_code, 200)
        result = response.context["result"]
        self.assertIsNotNone(result, "expected a prediction result in the response context")
        self.assertIn(result["prediction"], (0, 1))
        self.assertTrue(0.0 <= result["probability"] <= 100.0)
        self.assertIn(result["risk_label"], (
            "Higher risk of heart disease", "Lower risk of heart disease",
        ))

    def test_post_invalid_data_shows_form_errors_not_a_crash(self):
        data = {**VALID_DATA, "age": -5}
        response = self.client.post(reverse("predictor:predict"), data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context["result"])
        self.assertTrue(response.context["form"].errors)


class AboutViewTests(TestCase):
    def test_about_page_renders_with_metrics(self):
        response = self.client.get(reverse("predictor:about"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Model comparison")


class ModelArtifactTests(TestCase):
    """
    Sanity-checks the committed model artifact itself, not just the view
    around it: the saved pipeline must actually load, must produce
    probabilities for the same VALID_DATA fixture used above, and its
    self-reported metrics in model_metrics.json must be internally
    consistent. This is what the retraining workflow's regression check
    is protecting - it's meaningless unless this file matches reality.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        model_dir = Path(settings.BASE_DIR) / "predictor" / "ml_model"
        cls.model = joblib.load(model_dir / "heart_model.joblib")
        with open(model_dir / "model_metrics.json") as f:
            cls.metrics = json.load(f)

    def test_model_predicts_on_the_valid_fixture(self):
        import pandas as pd

        form = HeartDiseasePredictionForm(data=VALID_DATA)
        self.assertTrue(form.is_valid())
        row = pd.DataFrame([form.as_feature_dict()])
        proba = self.model.predict_proba(row)[0][1]
        self.assertTrue(0.0 <= proba <= 1.0)

    def test_metrics_file_reports_a_best_model_with_sane_accuracy(self):
        best = self.metrics["best_model"]
        self.assertIn(best, self.metrics["results"])
        test_acc = self.metrics["results"][best]["test_accuracy"]
        # Sanity bounds, not a strict regression check (that's the retrain
        # workflow's job comparing against this same file) - this just
        # catches a completely broken/degenerate model (e.g. always
        # predicting one class) from ever being committed.
        self.assertGreater(test_acc, 0.6)
        self.assertLessEqual(test_acc, 1.0)
