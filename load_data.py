import os
import pandas as pd


import os
import pandas as pd


DATA_PATH = r"C:\Users\Thaman\PycharmProjects\PythonProject4\placement_predict_50k Dataset (3).csv"



def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("File does not exist.")

    df = pd.read_csv(path)
    return df



def get_data_summary(path:str=DATA_PATH) -> dict:
    df=load_data(path)
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col:int(df[col].isna().sum())for col in df.columns},
        "preview": df.head(10).to_dict(orient="records"),
    }
    return summary


if __name__ == "main":
    df = load_data()
    print(get_data_summary(df))


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("File does not exist.")

    df = pd.read_csv(path)
    return df



def get_data_summary(path:str=DATA_PATH) -> dict:
    df=load_data(path)
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col:int(df[col].isna().sum())for col in df.columns},
        "preview": df.head(10).to_dict(orient="records"),
    }
    return summary


if __name__ == "main":
    df = load_data()
    print(get_data_summary(df))