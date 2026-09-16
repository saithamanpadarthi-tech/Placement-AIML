import pandas as pd
import numpy as np

from load_data import load_data

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# MAIN PREPROCESSING FUNCTION
# =========================================================

def preprocess_data():

    print("\n========== PREPROCESSING STARTED ==========")

    # =====================================================
    # 1. LOAD DATA
    # =====================================================

    data = load_data()

    original_rows = int(data.shape[0])
    original_columns = int(data.shape[1])

    print("\nOriginal dataset shape:")
    print(data.shape)

    # =====================================================
    # 2. DUPLICATE HANDLING
    # =====================================================

    duplicate_count = int(
        data.duplicated().sum()
    )

    if duplicate_count > 0:

        data = data.drop_duplicates()

    duplicates_after = int(
        data.duplicated().sum()
    )

    # =====================================================
    # 3. IDENTIFY COLUMNS
    # =====================================================

    numeric_columns = data.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_columns = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # =====================================================
    # 4. EXCLUDED COLUMNS
    # =====================================================

    exclude_columns = [
        "StudentID",
        "PlacementStatus",
        "IsAnomaly",
        "Salary Package",
        "CGPA_Tier"
    ]

    numeric_columns = [
        col
        for col in numeric_columns
        if col not in exclude_columns
    ]

    categorical_columns = [
        col
        for col in categorical_columns
        if col not in exclude_columns
    ]

    # =====================================================
    # 5. HANDLE NUMERICAL MISSING VALUES
    # =====================================================

    numerical_missing = {}

    for col in numeric_columns:

        missing_count = int(
            data[col].isnull().sum()
        )

        numerical_missing[col] = missing_count

        if missing_count > 0:

            median_value = data[col].median()

            data[col] = data[col].fillna(
                median_value
            )

    # =====================================================
    # 6. HANDLE CATEGORICAL MISSING VALUES
    # =====================================================

    categorical_missing = {}

    for col in categorical_columns:

        missing_count = int(
            data[col].isnull().sum()
        )

        categorical_missing[col] = missing_count

        if missing_count > 0:

            mode_values = data[col].mode()

            if not mode_values.empty:

                mode_value = mode_values.iloc[0]

                data[col] = data[col].fillna(
                    mode_value
                )

            else:

                data[col] = data[col].fillna(
                    "Unknown"
                )

    # =====================================================
    # 7. TARGET SEPARATION
    # =====================================================

    target = "PlacementStatus"

    if target not in data.columns:

        raise ValueError(
            f"Target column '{target}' was not found."
        )

    columns_to_remove = [
        "PlacementStatus",
        "StudentID",
        "IsAnomaly",
        "Salary Package",
        "CGPA_Tier"
    ]

    columns_to_remove = [
        col
        for col in columns_to_remove
        if col in data.columns
    ]

    X = data.drop(
        columns=columns_to_remove
    )

    y = data[target]

    # =====================================================
    # 8. TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    # =====================================================
    # 9. PREPROCESSING PIPELINE
    # =====================================================

    transformers = []

    if numeric_columns:

        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_columns
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # =====================================================
    # 10. FIT TRAINING DATA
    # =====================================================

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    # =====================================================
    # 11. TRANSFORM TEST DATA
    # =====================================================

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # =====================================================
    # 12. RESULTS
    # =====================================================

    print("\n========== PREPROCESSING COMPLETED ==========")

    print(
        "Original shape:",
        data.shape
    )

    print(
        "X train:",
        X_train.shape
    )

    print(
        "X test:",
        X_test.shape
    )

    print(
        "Processed X train:",
        X_train_processed.shape
    )

    print(
        "Processed X test:",
        X_test_processed.shape
    )

    # =====================================================
    # 13. RETURN ORIGINAL RESULTS
    # =====================================================

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


# =========================================================
# FLASK-FRIENDLY PREPROCESSING
# =========================================================

def run_preprocessing():

    data = load_data()

    # =====================================================
    # ORIGINAL INFORMATION
    # =====================================================

    original_rows = int(
        data.shape[0]
    )

    original_columns = int(
        data.shape[1]
    )

    # =====================================================
    # DUPLICATES
    # =====================================================

    duplicates_before = int(
        data.duplicated().sum()
    )

    data_clean = data.drop_duplicates()

    duplicates_after = int(
        data_clean.duplicated().sum()
    )

    # =====================================================
    # COLUMNS
    # =====================================================

    numeric_columns = data_clean.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_columns = data_clean.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    excluded = [
        "StudentID",
        "PlacementStatus",
        "IsAnomaly",
        "Salary Package",
        "CGPA_Tier"
    ]

    numeric_features = [
        col
        for col in numeric_columns
        if col not in excluded
    ]

    categorical_features = [
        col
        for col in categorical_columns
        if col not in excluded
    ]

    # =====================================================
    # MISSING VALUES BEFORE
    # =====================================================

    missing_before = int(
        data_clean.isnull().sum().sum()
    )

    # =====================================================
    # HANDLE MISSING VALUES
    # =====================================================

    for col in numeric_features:

        if data_clean[col].isnull().sum() > 0:

            data_clean[col] = data_clean[col].fillna(
                data_clean[col].median()
            )

    for col in categorical_features:

        if data_clean[col].isnull().sum() > 0:

            mode_values = data_clean[col].mode()

            if not mode_values.empty:

                data_clean[col] = data_clean[col].fillna(
                    mode_values.iloc[0]
                )

            else:

                data_clean[col] = data_clean[col].fillna(
                    "Unknown"
                )

    # =====================================================
    # MISSING VALUES AFTER
    # =====================================================

    missing_after = int(
        data_clean.isnull().sum().sum()
    )

    # =====================================================
    # TARGET
    # =====================================================

    target = "PlacementStatus"

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X = data_clean.drop(
        columns=[
            col
            for col in excluded
            if col in data_clean.columns
        ]
    )

    y = data_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    # =====================================================
    # ENCODING + SCALING
    # =====================================================

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # =====================================================
    # RETURN WEB RESULTS
    # =====================================================

    return {

        "original_rows":
            original_rows,

        "original_columns":
            original_columns,

        "duplicates_before":
            duplicates_before,

        "duplicates_after":
            duplicates_after,

        "missing_before":
            missing_before,

        "missing_after":
            missing_after,

        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features,

        "target":
            target,

        "training_rows":
            int(len(X_train)),

        "testing_rows":
            int(len(X_test)),

        "processed_training_rows":
            int(X_train_processed.shape[0]),

        "processed_training_columns":
            int(X_train_processed.shape[1]),

        "processed_testing_rows":
            int(X_test_processed.shape[0]),

        "processed_testing_columns":
            int(X_test_processed.shape[1])
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    preprocess_data()