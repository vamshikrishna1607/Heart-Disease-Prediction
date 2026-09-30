from django import forms


class HeartDiseasePredictionForm(forms.Form):
    """
    Captures the 13 clinical parameters used by the UCI Cleveland heart
    disease dataset. Field-level validation and sane min/max bounds are
    applied so obviously invalid clinical values are rejected before they
    ever reach the model.
    """

    age = forms.IntegerField(
        label="Age (years)",
        min_value=1,
        max_value=120,
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 54"}),
    )
    sex = forms.ChoiceField(
        label="Sex",
        choices=[(1, "Male"), (0, "Female")],
        widget=forms.Select,
    )
    cp = forms.ChoiceField(
        label="Chest pain type",
        choices=[
            (0, "Typical angina"),
            (1, "Atypical angina"),
            (2, "Non-anginal pain"),
            (3, "Asymptomatic"),
        ],
    )
    trestbps = forms.IntegerField(
        label="Resting blood pressure (mm Hg)",
        min_value=60,
        max_value=250,
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 130"}),
    )
    chol = forms.IntegerField(
        label="Serum cholesterol (mg/dl)",
        min_value=80,
        max_value=700,
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 246"}),
    )
    fbs = forms.ChoiceField(
        label="Fasting blood sugar > 120 mg/dl",
        choices=[(1, "Yes"), (0, "No")],
    )
    restecg = forms.ChoiceField(
        label="Resting ECG result",
        choices=[
            (0, "Normal"),
            (1, "ST-T wave abnormality"),
            (2, "Left ventricular hypertrophy"),
        ],
    )
    thalach = forms.IntegerField(
        label="Max heart rate achieved",
        min_value=50,
        max_value=250,
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 150"}),
    )
    exang = forms.ChoiceField(
        label="Exercise-induced angina",
        choices=[(1, "Yes"), (0, "No")],
    )
    oldpeak = forms.FloatField(
        label="ST depression induced by exercise",
        min_value=0.0,
        max_value=10.0,
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 1.4", "step": "0.1"}),
    )
    slope = forms.ChoiceField(
        label="Slope of peak exercise ST segment",
        choices=[(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
    )
    ca = forms.ChoiceField(
        label="Number of major vessels colored by fluoroscopy",
        choices=[(0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4")],
    )
    thal = forms.ChoiceField(
        label="Thalassemia",
        choices=[(0, "Unknown"), (1, "Normal"), (2, "Fixed defect"), (3, "Reversible defect")],
    )

    def clean(self):
        """
        Extra cross-field sanity checks beyond simple range validation -
        catches physiologically inconsistent combinations early rather than
        silently feeding them to the model.
        """
        cleaned_data = super().clean()
        thalach = cleaned_data.get("thalach")
        age = cleaned_data.get("age")
        if thalach is not None and age is not None:
            # A generous upper bound on achievable heart rate (220 - age is the
            # common estimate); flag values far outside anything plausible.
            if thalach > (220 - age) + 40:
                self.add_error(
                    "thalach",
                    "This max heart rate looks unusually high for the given age. "
                    "Please double-check the value.",
                )
        return cleaned_data

    def as_feature_row(self):
        """Convert validated form data into the ordered feature list the model expects."""
        data = self.cleaned_data
        return [
            int(data["age"]),
            int(data["sex"]),
            int(data["cp"]),
            int(data["trestbps"]),
            int(data["chol"]),
            int(data["fbs"]),
            int(data["restecg"]),
            int(data["thalach"]),
            int(data["exang"]),
            float(data["oldpeak"]),
            int(data["slope"]),
            int(data["ca"]),
            int(data["thal"]),
        ]
