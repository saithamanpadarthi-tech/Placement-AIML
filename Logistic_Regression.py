import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "placement_predict_50k Dataset (3) 1(in).csv"
)

CHARTS_DIR = os.path.join(
    BASE_DIR,
    "static",
    "charts"
)

os.makedirs(
    CHARTS_DIR,
    exist_ok=True
)


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    return pd.read_csv(
        DATASET_PATH
    )


# =========================================================
# SAVE CHART
# =========================================================

def save_chart(
    fig,
    filename
):

    path = os.path.join(
        CHARTS_DIR,
        filename
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return filename


# =========================================================
# MAIN EDA
# =========================================================

def run_eda():

    df = load_dataset()

    charts = []


    # =====================================================
    # REMOVE OLD EDA CHARTS
    # =====================================================

    for file in os.listdir(CHARTS_DIR):

        if (
            file.startswith("eda_")
            and file.endswith(".png")
        ):

            try:

                os.remove(
                    os.path.join(
                        CHARTS_DIR,
                        file
                    )
                )

            except Exception:
                pass


    # =====================================================
    # BASIC DATASET INFORMATION
    # =====================================================

    rows = int(
        df.shape[0]
    )

    columns = int(
        df.shape[1]
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )


    # =====================================================
    # MISSING VALUES
    # =====================================================

    missing_series = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_series = missing_series[
        missing_series > 0
    ]


    missing_values = [

        {
            "column": str(column),

            "count": int(count)
        }

        for column, count
        in missing_series.items()

    ]


    # =====================================================
    # 1. MISSING VALUES
    # =====================================================

    if not missing_series.empty:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        missing_series.head(10).plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Missing Values by Column"
        )

        ax.set_xlabel(
            "Column"
        )

        ax.set_ylabel(
            "Missing Count"
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        filename = save_chart(
            fig,
            "eda_01_missing_values.png"
        )

        charts.append({

            "title":
                "Missing Values",

            "filename":
                filename

        })


    # =====================================================
    # PLACEMENT STATUS
    # =====================================================

    placement_counts = {}

    if "PlacementStatus" in df.columns:

        placement_counts = {

            str(index):
                int(value)

            for index, value
            in df[
                "PlacementStatus"
            ]
            .value_counts()
            .items()

        }


    # =====================================================
    # 2. PLACEMENT STATUS DISTRIBUTION
    # =====================================================

    if "PlacementStatus" in df.columns:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        counts = (
            df[
                "PlacementStatus"
            ]
            .value_counts()
            .sort_index()
        )

        counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Placement Status Distribution"
        )

        ax.set_xlabel(
            "Placement Status"
        )

        ax.set_ylabel(
            "Number of Students"
        )

        filename = save_chart(
            fig,
            "eda_02_placement_status.png"
        )

        charts.append({

            "title":
                "Placement Status Distribution",

            "filename":
                filename

        })


    # =====================================================
    # 3. SALARY PACKAGE DISTRIBUTION
    # =====================================================

    if "Salary Package" in df.columns:

        salary = pd.to_numeric(
            df[
                "Salary Package"
            ],
            errors="coerce"
        ).dropna()

        if not salary.empty:

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.hist(
                salary,
                bins=30
            )

            ax.set_title(
                "Salary Package Distribution"
            )

            ax.set_xlabel(
                "Salary Package"
            )

            ax.set_ylabel(
                "Frequency"
            )

            filename = save_chart(
                fig,
                "eda_03_salary_distribution.png"
            )

            charts.append({

                "title":
                    "Salary Package Distribution",

                "filename":
                    filename

            })


    # =====================================================
    # NUMERICAL DISTRIBUTIONS
    # =====================================================

    distribution_features = [

        (
            "CGPA",
            "CGPA Distribution"
        ),

        (
            "AptitudeTestScore",
            "Aptitude Test Score Distribution"
        ),

        (
            "CodingTestScore",
            "Coding Test Score Distribution"
        ),

        (
            "MockInterviewScore",
            "Mock Interview Score Distribution"
        ),

        (
            "SoftSkillsRating",
            "Soft Skills Rating Distribution"
        ),

        (
            "AttendancePercent",
            "Attendance Distribution"
        )

    ]


    # =====================================================
    # 4 - 9. FEATURE DISTRIBUTIONS
    # =====================================================

    chart_number = 4

    for column, title in distribution_features:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.hist(
            values,
            bins=25
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            column
        )

        ax.set_ylabel(
            "Frequency"
        )

        filename = save_chart(
            fig,
            f"eda_{chart_number:02d}_{column}.png"
        )

        charts.append({

            "title":
                title,

            "filename":
                filename

        })

        chart_number += 1


    # =====================================================
    # NUMERICAL PAIR HELPER
    # =====================================================

    def numeric_pair(
        x,
        y
    ):

        if x not in df.columns:
            return None, None

        if y not in df.columns:
            return None, None

        x_values = pd.to_numeric(
            df[x],
            errors="coerce"
        )

        y_values = pd.to_numeric(
            df[y],
            errors="coerce"
        )

        valid = (
            x_values.notna()
            &
            y_values.notna()
        )

        return (
            x_values[valid],
            y_values[valid]
        )


    # =====================================================
    # 10. CGPA VS SALARY
    # =====================================================

    x, y = numeric_pair(
        "CGPA",
        "Salary Package"
    )

    if x is not None and len(x) > 0:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            x,
            y,
            alpha=0.35,
            s=12
        )

        ax.set_title(
            "CGPA vs Salary Package"
        )

        ax.set_xlabel(
            "CGPA"
        )

        ax.set_ylabel(
            "Salary Package"
        )

        filename = save_chart(
            fig,
            "eda_10_cgpa_vs_salary.png"
        )

        charts.append({

            "title":
                "CGPA vs Salary Package",

            "filename":
                filename

        })


    # =====================================================
    # 11. APTITUDE VS SALARY
    # =====================================================

    x, y = numeric_pair(
        "AptitudeTestScore",
        "Salary Package"
    )

    if x is not None and len(x) > 0:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            x,
            y,
            alpha=0.35,
            s=12
        )

        ax.set_title(
            "Aptitude Score vs Salary Package"
        )

        ax.set_xlabel(
            "Aptitude Test Score"
        )

        ax.set_ylabel(
            "Salary Package"
        )

        filename = save_chart(
            fig,
            "eda_11_aptitude_vs_salary.png"
        )

        charts.append({

            "title":
                "Aptitude Score vs Salary Package",

            "filename":
                filename

        })


    # =====================================================
    # 12. CODING VS SALARY
    # =====================================================

    x, y = numeric_pair(
        "CodingTestScore",
        "Salary Package"
    )

    if x is not None and len(x) > 0:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            x,
            y,
            alpha=0.35,
            s=12
        )

        ax.set_title(
            "Coding Score vs Salary Package"
        )

        ax.set_xlabel(
            "Coding Test Score"
        )

        ax.set_ylabel(
            "Salary Package"
        )

        filename = save_chart(
            fig,
            "eda_12_coding_vs_salary.png"
        )

        charts.append({

            "title":
                "Coding Score vs Salary Package",

            "filename":
                filename

        })


    # =====================================================
    # 13. MOCK INTERVIEW VS SALARY
    # =====================================================

    x, y = numeric_pair(
        "MockInterviewScore",
        "Salary Package"
    )

    if x is not None and len(x) > 0:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            x,
            y,
            alpha=0.35,
            s=12
        )

        ax.set_title(
            "Mock Interview Score vs Salary Package"
        )

        ax.set_xlabel(
            "Mock Interview Score"
        )

        ax.set_ylabel(
            "Salary Package"
        )

        filename = save_chart(
            fig,
            "eda_13_mock_vs_salary.png"
        )

        charts.append({

            "title":
                "Mock Interview Score vs Salary Package",

            "filename":
                filename

        })


    # =====================================================
    # 14. CGPA VS PLACEMENT STATUS
    # =====================================================

    if (
        "CGPA" in df.columns
        and
        "PlacementStatus" in df.columns
    ):

        temp = df[
            [
                "CGPA",
                "PlacementStatus"
            ]
        ].copy()

        temp[
            "CGPA"
        ] = pd.to_numeric(
            temp["CGPA"],
            errors="coerce"
        )

        temp = temp.dropna()

        groups = []

        labels = []

        for status in sorted(
            temp[
                "PlacementStatus"
            ].unique()
        ):

            values = temp[
                temp[
                    "PlacementStatus"
                ] == status
            ][
                "CGPA"
            ].values

            if len(values) > 0:

                groups.append(
                    values
                )

                labels.append(
                    str(status)
                )

        if groups:

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.boxplot(
                groups
            )

            ax.set_xticks(
                range(
                    1,
                    len(labels) + 1
                )
            )

            ax.set_xticklabels(
                labels
            )

            ax.set_title(
                "CGPA vs Placement Status"
            )

            ax.set_xlabel(
                "Placement Status"
            )

            ax.set_ylabel(
                "CGPA"
            )

            filename = save_chart(
                fig,
                "eda_14_cgpa_vs_placement.png"
            )

            charts.append({

                "title":
                    "CGPA vs Placement Status",

                "filename":
                    filename

            })


    # =====================================================
    # 15. CODING VS PLACEMENT STATUS
    # =====================================================

    if (
        "CodingTestScore" in df.columns
        and
        "PlacementStatus" in df.columns
    ):

        temp = df[
            [
                "CodingTestScore",
                "PlacementStatus"
            ]
        ].copy()

        temp[
            "CodingTestScore"
        ] = pd.to_numeric(
            temp[
                "CodingTestScore"
            ],
            errors="coerce"
        )

        temp = temp.dropna()

        groups = []

        labels = []

        for status in sorted(
            temp[
                "PlacementStatus"
            ].unique()
        ):

            values = temp[
                temp[
                    "PlacementStatus"
                ] == status
            ][
                "CodingTestScore"
            ].values

            if len(values) > 0:

                groups.append(
                    values
                )

                labels.append(
                    str(status)
                )

        if groups:

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.boxplot(
                groups
            )

            ax.set_xticks(
                range(
                    1,
                    len(labels) + 1
                )
            )

            ax.set_xticklabels(
                labels
            )

            ax.set_title(
                "Coding Score vs Placement Status"
            )

            ax.set_xlabel(
                "Placement Status"
            )

            ax.set_ylabel(
                "Coding Test Score"
            )

            filename = save_chart(
                fig,
                "eda_15_coding_vs_placement.png"
            )

            charts.append({

                "title":
                    "Coding Score vs Placement Status",

                "filename":
                    filename

            })


    # =====================================================
    # 16. CORRELATION HEATMAP
    # =====================================================

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] > 1:

        correlation = (
            numeric_df.corr()
        )

        fig_width = max(
            12,
            correlation.shape[1] * 0.55
        )

        fig_height = max(
            9,
            correlation.shape[0] * 0.55
        )

        fig, ax = plt.subplots(
            figsize=(
                fig_width,
                fig_height
            )
        )

        image = ax.imshow(
            correlation,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            aspect="auto"
        )

        ax.set_title(
            "Correlation Heatmap",
            fontsize=14,
            pad=15
        )

        ax.set_xticks(
            range(
                len(
                    correlation.columns
                )
            )
        )

        ax.set_yticks(
            range(
                len(
                    correlation.columns
                )
            )
        )

        ax.set_xticklabels(
            correlation.columns,
            rotation=90,
            fontsize=7
        )

        ax.set_yticklabels(
            correlation.columns,
            fontsize=7
        )


        # =================================================
        # CORRELATION VALUES
        # =================================================

        for i in range(
            len(
                correlation.columns
            )
        ):

            for j in range(
                len(
                    correlation.columns
                )
            ):

                value = correlation.iloc[
                    i,
                    j
                ]

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6
                )


        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        filename = save_chart(
            fig,
            "eda_16_correlation_heatmap.png"
        )

        charts.append({

            "title":
                "Correlation Heatmap",

            "filename":
                filename

        })


    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "rows":
            rows,

        "columns":
            columns,

        "duplicate_rows":
            duplicate_rows,

        "missing_values":
            missing_values,

        "placement_counts":
            placement_counts,

        "charts":
            charts

    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = run_eda()

    print(
        "\nEDA completed successfully."
    )

    print(
        "Total charts generated:",
        len(result["charts"])
    )

    for chart in result["charts"]:

        print(
            "-",
            chart["title"]
        )