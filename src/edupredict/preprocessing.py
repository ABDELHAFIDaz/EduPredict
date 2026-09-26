import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Exam_Score"

MISSING_COLS = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]

NUMERIC_COLS = [
    "Hours_Studied", "Attendance", "Sleep_Hours",
    "Previous_Scores", "Tutoring_Sessions", "Physical_Activity",
]

NOMINAL_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "School_Type": ["Public", "Private"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Internet_Access": ["No", "Yes"],
    "Learning_Disabilities": ["No", "Yes"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
}

ORDINAL_ORDERS = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
}


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, order in ORDINAL_ORDERS.items():
        df[col] = df[col].map({v: i for i, v in enumerate(order)})
        if df[col].isna().any():
            raise ValueError(f"Invalid value in '{col}', expected one of {order}")
    for col, cats in NOMINAL_CATEGORIES.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return pd.get_dummies(df, columns=list(NOMINAL_CATEGORIES), dtype=int)


preprocessor = ColumnTransformer(
    transformers=[("scale", StandardScaler(), NUMERIC_COLS)],
    remainder="passthrough",
)


def make_pipeline(model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])