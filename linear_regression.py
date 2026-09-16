import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# DATASET PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FILE_PATH = os.path.join(
    BASE_DIR,
    "placement_predict_50k Dataset (3) 1(in).csv"
)


# ============================================================
# HELPER
# ============================================================

def to_numeric_clean(series):

    if pd.api.types.is_numeric_dtype(series):

        return pd.to_numeric(
            series,
            errors="coerce"
        )

    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize(x):

    x = np.asarray(
        x,
        dtype=float
    )

    mean = x.mean()

    std = x.std()

    if std == 0 or not np.isfinite(std):

        std = 1.0

    return (
        (x - mean) / std,
        mean,
        std
    )


# ============================================================
# BATCH GRADIENT DESCENT
# ============================================================

def batch_gradient_descent(
    x,
    y,
    alpha=0.05,
    epochs=6
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    m = len(x)

    theta0 = 0.0

    theta1 = 0.0

    mse_history = []

    for epoch in range(
        1,
        epochs + 1
    ):

        # Prediction

        y_hat = (
            theta0
            + theta1 * x
        )

        # Error

        error = (
            y_hat - y
        )

        # Gradients

        grad_theta0 = (
            (2.0 / m)
            * np.sum(error)
        )

        grad_theta1 = (
            (2.0 / m)
            * np.sum(
                error * x
            )
        )

        # Update

        theta0 = (
            theta0
            - alpha * grad_theta0
        )

        theta1 = (
            theta1
            - alpha * grad_theta1
        )

        # Updated prediction

        updated_y_hat = (
            theta0
            + theta1 * x
        )

        # MSE

        mse = np.mean(
            (
                updated_y_hat - y
            ) ** 2
        )

        mse_history.append({

            "epoch":
                epoch,

            "theta0":
                float(theta0),

            "theta1":
                float(theta1),

            "mse":
                float(mse)

        })

    return (
        theta0,
        theta1,
        mse_history
    )


# ============================================================
# CHART DIRECTORY
# ============================================================

def charts_directory():

    chart_dir = os.path.join(
        BASE_DIR,
        "static",
        "charts"
    )

    os.makedirs(
        chart_dir,
        exist_ok=True
    )

    return chart_dir


# ============================================================
# SAVE FIGURE
# ============================================================

def save_figure(
    fig,
    filename
):

    chart_dir = charts_directory()

    path = os.path.join(
        chart_dir,
        filename
    )

    fig.tight_layout()

    fig.savefig(
        path,
        format="png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return filename


# ============================================================
# CLEAR OLD CHARTS
# ============================================================

def clear_old_charts():

    chart_dir = charts_directory()

    chart_names = [

        "gradient_descent_mse.png",

        "gradient_descent_vs_sklearn.png",

        "scaling_comparison.png"

    ]

    for filename in chart_names:

        path = os.path.join(
            chart_dir,
            filename
        )

        if os.path.exists(path):

            try:

                os.remove(path)

            except OSError:

                pass


# ============================================================
# GENERATE CHARTS
# ============================================================

def generate_regression_charts(
    model_data
):

    clear_old_charts()

    charts = []

    if not model_data:

        return charts

    if model_data.get("error"):

        return charts

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    x_train = np.asarray(
        model_data["x_train"],
        dtype=float
    )

    y_train = np.asarray(
        model_data["y_train"],
        dtype=float
    )

    x_val = np.asarray(
        model_data["x_val"],
        dtype=float
    )

    y_val = np.asarray(
        model_data["y_val"],
        dtype=float
    )

    gd_predictions = np.asarray(
        model_data["gd_predictions"],
        dtype=float
    )

    sk_predictions = np.asarray(
        model_data["sk_predictions"],
        dtype=float
    )

    mse_history = model_data[
        "mse_history"
    ]

    # ========================================================
    # CHART 1
    # ========================================================

    epochs = [

        item["epoch"]

        for item in mse_history

    ]

    mse_values = [

        item["mse"]

        for item in mse_history

    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    ax.plot(
        epochs,
        mse_values,
        marker="o",
        linewidth=2
    )

    for epoch, mse in zip(
        epochs,
        mse_values
    ):

        ax.annotate(
            f"{mse:.2f}",
            (epoch, mse),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9
        )

    ax.set_title(
        "Gradient Descent MSE - 6 Iterations"
    )

    ax.set_xlabel(
        "Iteration"
    )

    ax.set_ylabel(
        "Mean Squared Error"
    )

    ax.set_xticks(
        epochs
    )

    ax.grid(
        True,
        alpha=0.25
    )

    filename = (
        "gradient_descent_mse.png"
    )

    save_figure(
        fig,
        filename
    )

    charts.append({

        "title":
            "Gradient Descent MSE",

        "filename":
            filename

    })

    # ========================================================
    # CHART 2
    # ========================================================

    sort_index = np.argsort(
        x_val
    )

    sorted_x = x_val[
        sort_index
    ]

    sorted_actual = y_val[
        sort_index
    ]

    sorted_gd = gd_predictions[
        sort_index
    ]

    sorted_sk = sk_predictions[
        sort_index
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    ax.scatter(
        sorted_x,
        sorted_actual,
        s=18,
        alpha=0.35,
        label="Actual Salary"
    )

    ax.plot(
        sorted_x,
        sorted_gd,
        linewidth=2,
        label="Gradient Descent"
    )

    ax.plot(
        sorted_x,
        sorted_sk,
        linewidth=2,
        linestyle="--",
        label="Scikit-learn"
    )

    ax.set_title(
        "Gradient Descent vs Scikit-learn"
    )

    ax.set_xlabel(
        "CGPA"
    )

    ax.set_ylabel(
        "Salary Package"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25
    )

    filename = (
        "gradient_descent_vs_sklearn.png"
    )

    save_figure(
        fig,
        filename
    )

    charts.append({

        "title":
            "Gradient Descent vs Scikit-learn",

        "filename":
            filename

    })

    # ========================================================
    # CHART 3
    # ========================================================

    x_mean = float(
        model_data["x_mean"]
    )

    x_std = float(
        model_data["x_std"]
    )

    gd_theta0 = float(
        model_data["gd_theta0"]
    )

    gd_theta1 = float(
        model_data["gd_theta1"]
    )

    x_original_line = np.linspace(

        float(x_train.min()),

        float(x_train.max()),

        200

    )

    x_scaled_line = (

        x_original_line
        - x_mean

    ) / x_std

    y_scaled_prediction = (

        gd_theta0
        + gd_theta1
        * x_scaled_line

    )

    original_slope = (

        gd_theta1
        / x_std

    )

    original_intercept = (

        gd_theta0
        -
        (
            gd_theta1
            * x_mean
            / x_std
        )

    )

    y_original_prediction = (

        original_intercept
        +
        original_slope
        * x_original_line

    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    ax.plot(
        x_original_line,
        y_original_prediction,
        linewidth=2,
        label="Original CGPA Scale"
    )

    ax.plot(
        x_original_line,
        y_scaled_prediction,
        linewidth=2,
        linestyle="--",
        label="Standardized CGPA Scale"
    )

    ax.scatter(
        x_train,
        y_train,
        s=15,
        alpha=0.25,
        label="Training Data"
    )

    ax.set_title(
        "Linear Regression - Scaling Comparison"
    )

    ax.set_xlabel(
        "CGPA"
    )

    ax.set_ylabel(
        "Salary Package"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25
    )

    filename = (
        "scaling_comparison.png"
    )

    save_figure(
        fig,
        filename
    )

    charts.append({

        "title":
            "Scaling Comparison",

        "filename":
            filename

    })

    return charts


# ============================================================
# MAIN LINEAR REGRESSION
# ============================================================

def run_linear_regression():

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not os.path.exists(
        FILE_PATH
    ):

        return {

            "error":
                "Dataset file was not found: "
                + FILE_PATH

        }

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        FILE_PATH
    )

    MODEL_FEATURE = (
        "CGPA"
    )

    MODEL_TARGET = (
        "Salary Package"
    )

    # --------------------------------------------------------
    # CHECK COLUMNS
    # --------------------------------------------------------

    if (

        MODEL_FEATURE
        not in df.columns

        or

        MODEL_TARGET
        not in df.columns

    ):

        return {

            "error":
                "CGPA or Salary Package column "
                "was not found."

        }

    # --------------------------------------------------------
    # SELECT DATA
    # --------------------------------------------------------

    model_df = df[

        [
            MODEL_FEATURE,
            MODEL_TARGET
        ]

    ].copy()

    # --------------------------------------------------------
    # CONVERT NUMERIC
    # --------------------------------------------------------

    model_df[
        MODEL_FEATURE
    ] = to_numeric_clean(

        model_df[
            MODEL_FEATURE
        ]

    )

    model_df[
        MODEL_TARGET
    ] = to_numeric_clean(

        model_df[
            MODEL_TARGET
        ]

    )

    model_df = model_df.dropna()

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if len(model_df) < 10:

        return {

            "error":
                "Not enough valid "
                "CGPA/Salary Package rows."

        }

    # --------------------------------------------------------
    # X AND Y
    # --------------------------------------------------------

    x = model_df[
        MODEL_FEATURE
    ].to_numpy(
        dtype=float
    )

    y = model_df[
        MODEL_TARGET
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    (
        x_train,
        x_val,
        y_train,
        y_val

    ) = train_test_split(

        x,
        y,

        test_size=0.20,

        random_state=42

    )

    # --------------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------------

    (
        x_train_std,
        x_mean,
        x_std

    ) = standardize(
        x_train
    )

    x_val_std = (

        x_val
        - x_mean

    ) / x_std

    # --------------------------------------------------------
    # GRADIENT DESCENT
    # --------------------------------------------------------

    gd_alpha = 0.05

    gd_epochs = 6

    (
        gd_theta0,
        gd_theta1,
        mse_history

    ) = batch_gradient_descent(

        x_train_std,
        y_train,

        alpha=gd_alpha,

        epochs=gd_epochs

    )

    # --------------------------------------------------------
    # GD PREDICTION
    # --------------------------------------------------------

    y_val_gd_pred = (

        gd_theta0
        +
        gd_theta1
        * x_val_std

    )

    # --------------------------------------------------------
    # GD METRICS
    # --------------------------------------------------------

    gd_mse = mean_squared_error(

        y_val,
        y_val_gd_pred

    )

    gd_rmse = np.sqrt(
        gd_mse
    )

    gd_mae = mean_absolute_error(

        y_val,
        y_val_gd_pred

    )

    gd_r2 = r2_score(

        y_val,
        y_val_gd_pred

    )

    # --------------------------------------------------------
    # ORIGINAL SCALE PARAMETERS
    # --------------------------------------------------------

    gd_original_slope = (

        gd_theta1
        / x_std

    )

    gd_original_intercept = (

        gd_theta0
        -
        (
            gd_theta1
            * x_mean
            / x_std
        )

    )

    # --------------------------------------------------------
    # SCIKIT-LEARN
    # --------------------------------------------------------

    sk_model = (
        LinearRegression()
    )

    sk_model.fit(

        x_train.reshape(
            -1,
            1
        ),

        y_train

    )

    y_val_sk_pred = (
        sk_model.predict(
            x_val.reshape(
                -1,
                1
            )
        )
    )

    # --------------------------------------------------------
    # SCIKIT METRICS
    # --------------------------------------------------------

    sk_mse = mean_squared_error(

        y_val,
        y_val_sk_pred

    )

    sk_rmse = np.sqrt(
        sk_mse
    )

    sk_mae = mean_absolute_error(

        y_val,
        y_val_sk_pred

    )

    sk_r2 = r2_score(

        y_val,
        y_val_sk_pred

    )

    # --------------------------------------------------------
    # CHART DATA
    # --------------------------------------------------------

    chart_data = {

        "x_train":
            x_train.tolist(),

        "y_train":
            y_train.tolist(),

        "x_val":
            x_val.tolist(),

        "y_val":
            y_val.tolist(),

        "gd_predictions":
            y_val_gd_pred.tolist(),

        "sk_predictions":
            y_val_sk_pred.tolist(),

        "x_mean":
            float(x_mean),

        "x_std":
            float(x_std),

        "gd_theta0":
            float(gd_theta0),

        "gd_theta1":
            float(gd_theta1),

        "mse_history":
            mse_history

    }

    # --------------------------------------------------------
    # GENERATE CHARTS
    # --------------------------------------------------------

    charts = (
        generate_regression_charts(
            chart_data
        )
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {

        "feature":
            MODEL_FEATURE,

        "target":
            MODEL_TARGET,

        "train_rows":
            int(len(x_train)),

        "validation_rows":
            int(len(x_val)),

        "alpha":
            gd_alpha,

        "epochs":
            gd_epochs,

        "standardization": {

            "mean":
                round(
                    float(x_mean),
                    6
                ),

            "std":
                round(
                    float(x_std),
                    6
                )

        },

        "gradient_descent": {

            "theta0":
                round(
                    float(gd_theta0),
                    6
                ),

            "theta1":
                round(
                    float(gd_theta1),
                    6
                ),

            "original_intercept":
                round(
                    float(
                        gd_original_intercept
                    ),
                    6
                ),

            "original_slope":
                round(
                    float(
                        gd_original_slope
                    ),
                    6
                ),

            "mse":
                round(
                    float(gd_mse),
                    6
                ),

            "rmse":
                round(
                    float(gd_rmse),
                    6
                ),

            "mae":
                round(
                    float(gd_mae),
                    6
                ),

            "r2":
                round(
                    float(gd_r2),
                    6
                )

        },

        "sklearn": {

            "intercept":
                round(
                    float(
                        sk_model.intercept_
                    ),
                    6
                ),

            "slope":
                round(
                    float(
                        sk_model.coef_[0]
                    ),
                    6
                ),

            "mse":
                round(
                    float(sk_mse),
                    6
                ),

            "rmse":
                round(
                    float(sk_rmse),
                    6
                ),

            "mae":
                round(
                    float(sk_mae),
                    6
                ),

            "r2":
                round(
                    float(sk_r2),
                    6
                )

        },

        "mse_history": [

            {

                "epoch":
                    item["epoch"],

                "theta0":
                    round(
                        item["theta0"],
                        6
                    ),

                "theta1":
                    round(
                        item["theta1"],
                        6
                    ),

                "mse":
                    round(
                        item["mse"],
                        6
                    )

            }

            for item in mse_history

        ],

        "charts":
            charts,

        "chart_version":
            str(
                int(
                    pd.Timestamp.now()
                    .timestamp()
                )
            )

    }

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = (
        run_linear_regression()
    )

    print("=" * 80)

    print(
        "LINEAR REGRESSION"
    )

    print("=" * 80)

    if "error" in result:

        print(
            "Error:",
            result["error"]
        )

    else:

        print(
            "Feature:",
            result["feature"]
        )

        print(
            "Target:",
            result["target"]
        )

        print(
            "Training rows:",
            result["train_rows"]
        )

        print(
            "Validation rows:",
            result["validation_rows"]
        )

        print(
            "\nGradient Descent:"
        )

        print(
            result["gradient_descent"]
        )

        print(
            "\nScikit-learn:"
        )

        print(
            result["sklearn"]
        )

        print(
            "\nCharts:"
        )

        for chart in result["charts"]:

            print(
                "-",
                chart["filename"]
            )

    print("=" * 80)