import os
import pandas as pd


# =========================================================
# DATASET PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    r"C:\Users\Thaman\PycharmProjects\PythonProject4\placement_predict_50k Dataset (3).csv"
)


# =========================================================
# LOAD DATASET
# =========================================================

def load_data():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    return df


# =========================================================
# DATASET SUMMARY
# =========================================================

def get_data_summary():

    df = load_data()

    # Convert NaN values to safe values for Jinja
    missing_counts = (
        df.isnull()
        .sum()
        .to_dict()
    )

    dtypes = {
        column: str(df[column].dtype)
        for column in df.columns
    }

    preview = (
        df.head(10)
        .fillna("")
        .to_dict(orient="records")
    )

    return {
        "n_rows": int(df.shape[0]),

        "n_cols": int(df.shape[1]),

        "columns": df.columns.tolist(),

        "dtypes": dtypes,

        "missing_counts": missing_counts,

        "preview": preview
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    df = load_data()

    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumns:")
    for column in df.columns:
        print("-", column)