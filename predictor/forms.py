from django import forms


class HeartDiseasePredictionForm(forms.Form):
    """
    Captures the 11 clinical parameters used by the Heart Failure
    Prediction dataset (a harmonized merge of the Cleveland, Hungarian,
    Switzerland, Long Beach VA and Stalog heart-disease cohorts).
    Field-level validation and sane min/max bounds are applied so obviously
    invalid clinical values are rejected before they ever reach the model.
    """

    age = forms.IntegerField(
        label="Age (years)",
        min_value=1,
        max_value=120,
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 54"}),
    )
    sex = forms.ChoiceField(
        label="Sex",
        choices=[("M", "Male"), ("F", "Female")],
        widget=forms.Select,
    )
    chest_pain_type = forms.ChoiceField(
        label="Chest pain type",
        choices=[
            ("TA", "Typical angina"),
            ("ATA", "Atypical angina"),
            ("NAP", "Non-anginal pain"),
            ("ASY", "Asymptomatic"),
        ],
    )
    resting_bp = forms.IntegerField(
        label="Resting blood pressure (mm Hg)",
        min_value=60,
        max_value=250,
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 130"}),
    )
    cholesterol = forms.IntegerField(
        label="Serum cholesterol (mg/dl)",
        min_value=0,
        max_value=700,
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 246"}),
    )
    fasting_bs = forms.ChoiceField(
        label="Fasting blood sugar > 120 mg/dl",
        choices=[(1, "Yes"), (0, "No")],
    )
    resting_ecg = forms.ChoiceField(
        label="Resting ECG result",
        choices=[
            ("Normal", "Normal"),
            ("ST", "ST-T wave abnormality"),
            ("LVH", "Left ventricular hypertrophy"),
        ],
    )
    max_hr = forms.IntegerField(
        label="Max heart rate achieved",
        min_value=50,
        max_value=250,
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 150"}),
    )
    exercise_angina = forms.ChoiceField(
        label="Exercise-induced angina",
        choices=[("Y", "Yes"), ("N", "No")],
    )
    oldpeak = forms.FloatField(
        label="ST depression induced by exercise",
        min_value=-3.0,
        max_value=10.0,
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 1.4", "step": "0.1"}),
    )
    st_slope = forms.ChoiceField(
        label="Slope of peak exercise ST segment",
        choices=[("Up", "Upsloping"), ("Flat", "Flat"), ("Down", "Downsloping")],
    )

    def clean(self):
        """
        Extra cross-field sanity checks beyond simple range validation -
        catches physiologically inconsistent combinations early rather than
        silently feeding them to the model.
        """
        cleaned_data = super().clean()
        max_hr = cleaned_data.get("max_hr")
        age = cleaned_data.get("age")
        if max_hr is not None and age is not None:
            # A generous upper bound on achievable heart rate (220 - age is the
            # common estimate); flag values far outside anything plausible.
            if max_hr > (220 - age) + 40:
                self.add_error(
                    "max_hr",
                    "This max heart rate looks unusually high for the given age. "
                    "Please double-check the value.",
                )
        return cleaned_data

    def as_feature_dict(self):
        """
        Convert validated form data into the column-name-keyed dict the
        model's ColumnTransformer expects (it selects columns by name, not
        position, so this must match the training column names exactly).
        """
        data = self.cleaned_data
        return {
            "Age": int(data["age"]),
            "Sex": data["sex"],
            "ChestPainType": data["chest_pain_type"],
            "RestingBP": int(data["resting_bp"]),
            "Cholesterol": int(data["cholesterol"]),
            "FastingBS": int(data["fasting_bs"]),
            "RestingECG": data["resting_ecg"],
            "MaxHR": int(data["max_hr"]),
            "ExerciseAngina": data["exercise_angina"],
            "Oldpeak": float(data["oldpeak"]),
            "ST_Slope": data["st_slope"],
        }
