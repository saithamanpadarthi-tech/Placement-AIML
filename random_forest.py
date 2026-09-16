import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# DATASET PATH
# =========================================================

DATA_PATH = r"placement_predict_50k Dataset (3) 1(in).csv"


# =========================================================
# RANDOM FOREST
# =========================================================

def run_random_forest():

    df = pd.read_csv(DATA_PATH)

    features = [
        "CGPA",
        "Attendance",
        "AptitudeTestScore",
        "CodingTestScore",
        "SoftSkillsRating",
        "MockInterviewScore"
    ]

    target = "PlacementStatus"

    features = [col for col in features if col in df.columns]

    data = df[features + [target]].copy()

    # Numeric conversion
    for col in features:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna()

    # Target conversion
    if data[target].dtype == "object":

        data[target] = (
            data[target]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        data[target] = data[target].map({
            "placed": 1,
            "yes": 1,
            "1": 1,
            "not placed": 0,
            "no": 0,
            "0": 0
        })

    data = data.dropna()

    X = data[features]
    y = data[target].astype(int)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    return {
        "model": "Random Forest Classifier",
        "features": features,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2)
    }