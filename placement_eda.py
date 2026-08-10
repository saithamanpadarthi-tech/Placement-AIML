import os
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from load_data import load_data

sns.set(style="whitegrid")

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), bbox_inches="tight", dpi=100)
    plt.close("all")


def run_eda() -> dict:
    data = load_data()
    charts = []
    # 1. LOAD DATA
    print("\n" + "=" * 80)
    print("1. LOAD DATA")
    print("=" * 80)

    print("Shape:", data.shape)
    print("\nFirst 5 Rows:")
    print(data.head())

    # 2. BASIC INFO / STRUCTURE
    print("\n" + "=" * 80)
    print("2. BASIC INFO / STRUCTURE")
    print("=" * 80)

    print("\nDataset Information:")
    data.info()

    print("\nColumn Data Types:")
    print(data.dtypes)

    print("\nDescriptive Statistics (Numeric Columns):")
    print(data.describe())

    print("\nDescriptive Statistics (Categorical Columns):")
    print(data.describe(include=["object"]))

    # 3. MISSING VALUES
    print("\n" + "=" * 80)
    print("3. MISSING VALUES")
    print("=" * 80)

    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ].sort_values(by="missing_count", ascending=False)

    print(missing_df)

    if not missing_df.empty:
        plt.figure(figsize=(10, 5), dpi=100)
        sns.barplot(x=missing_df.index, y=missing_df["missing_pct"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Missing %")
        plt.title("Missing values by column")
        _save("missing_values.png")
        charts.append("missing_values.png")

    # 4. DUPLICATES
    print("\n" + "=" * 80)
    print("4. DUPLICATE ROWS")
    print("=" * 80)
    print("Duplicate rows:", data.duplicated().sum())

    # 5. TARGET VARIABLE DISTRIBUTION
    print("\n" + "=" * 80)
    print("5. TARGET VARIABLE - Placementstatus")
    print("=" * 80)
    print(data["PlacementStatus"].value_counts())

    plt.figure(dpi=125)
    sns.countplot(x="PlacementStatus", data=data)
    plt.xlabel("PlacementStatus (0 = Not Placed, 1 = Placed)")
    plt.ylabel("Count")
    plt.title("Placement Status Distribution")
    _save("placement_status.png")
    charts.append("placement_status.png")

    # 6. NUMERIC DISTRIBUTION
    print("\n" + "=" * 80)
    print("6. NUMERIC DISTRIBUTION")
    print("=" * 80)

    hist_cols = [
        "CGPA",
        "Attendencepercent",
        "AptitudeTestScore",
        "softSkillsRating",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    hist_cols = [c for c in hist_cols if c in data.columns]

    fig = data[hist_cols].hist(figsize=(14, 10), bins=20)

    plt.tight_layout()
    plt.savefig(_chart_path("numeric_distribution.png"), dpi=100)
    plt.close("all")

    charts.append("numeric_distribution.png")

    plt.figure(dpi=125)
    sns.histplot(data["CGPA"], kde=True)
    plt.axvline(x=np.mean(data["CGPA"]), color="green", linestyle="--", label="Mean")
    plt.legend()
    plt.title("CGPA Distribution with Mean")
    _save("cgpa_distribution.png")
    charts.append("cgpa_distribution.png")

    # 7. OUTLIER DETECTION
    print("\n" + "=" * 80)
    print("7. OUTLIER DETECTION")
    print("=" * 80)

    box_cols = [
        "CGPA",
        "Attendencepercent",
        "AptitudeTestScore",
        "softSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "Salary Package"
    ]

    box_cols = [c for c in box_cols if c in data.columns]

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=data[box_cols], color="skyblue")

    plt.title("Boxplots of Numerical Features", fontsize=14)
    plt.xticks(rotation=45)
    _save("outlier_detection.png")
    charts.append("outlier_detection.png")

    # 8. CORRELATION ANALYSIS
    print("\n" + "=" * 80)
    print("8. CORRELATION ANALYSIS")
    print("=" * 80)

    corr = data.select_dtypes(include=[np.number]).corr()
    print(np.round(corr, 2))

    plt.figure(figsize=(16, 12), dpi=100)
    sns.heatmap(
        np.round(corr, 2),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap")
    _save("correlation_heatmap.png")
    charts.append("correlation_heatmap.png")

    # 9. RELATIONSHIP PLOTS
    print("\n" + "=" * 80)
    print("9. RELATIONSHIP PLOTS")
    print("=" * 80)

    plt.figure(figsize=(8, 5))
    sns.regplot(x="CGPA", y="Salary Package", data=data, scatter_kws={"alpha": 0.6})
    plt.title("CGPA vs Salary Package")
    _save("cgpa_salary.png")
    charts.append("cgpa_salary.png")

    plt.figure(figsize=(8, 5))
    sns.regplot(x="AptitudeTestScore", y="CodingTestScore", data=data, scatter_kws={"alpha": 0.6})
    plt.title("Aptitude Test Score vs Coding Test Score")
    _save("aptitude_coding.png")
    charts.append("aptitude_coding.png")

    # 10. CATEGORICAL FEATURE COUNTS
    print("\n" + "=" * 80)
    print("10. CATEGORICAL FEATURE COUNTS")
    print("=" * 80)

    cat_cols = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "CGPA_Tier"
    ]

    cat_cols = [c for c in cat_cols if c in data.columns]

    for col in cat_cols:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=col, data=data)
        plt.xticks(rotation=45)
        plt.title(f"{col} Count")

        filename = f"{col}_count.png"
        _save(filename)
        charts.append(filename)

        # 11. GENDER VS PLACEMENT STATUS
        print("\n" + "=" * 80)
        print("11. GENDER VS PLACEMENT STATUS")
        print("=" * 80)

        plt.figure(figsize=(8, 5))
        sns.countplot(x="Gender", hue="PlacementStatus", data=data)
        plt.title("Gender vs Placement Status")
        _save("gender_vs_placement.png")
        charts.append("gender_vs_placement.png")

        # 12. COLLEGE TIER / STREAM VS PLACEMENT STATUS
        print("\n" + "=" * 80)
        print("12. COLLEGE TIER / STREAM VS PLACEMENT STATUS")
        print("=" * 80)

        plt.figure(figsize=(8, 5))
        sns.countplot(x="CollegeTier", hue="PlacementStatus", data=data)
        plt.title("College Tier vs Placement Status")
        _save("collegetier_vs_placement.png")
        charts.append("collegetier_vs_placement.png")

        plt.figure(figsize=(10, 5))
        sns.countplot(x="Stream", hue="PlacementStatus", data=data)
        plt.xticks(rotation=45)
        plt.title("Stream vs Placement Status")
        _save("stream_vs_placement.png")
        charts.append("stream_vs_placement.png")

        # 13. SGPA TREND ACROSS SEMESTERS
        print("\n" + "=" * 80)
        print("13. SGPA TREND ACROSS SEMESTERS")
        print("=" * 80)

    return {
        "n_rows": data.shape[0],
        "n_cols": data.shape[1],
        "duplicate_count": int(data.duplicated().sum()),
        "missing": missing.to_dict(),
        "target_counts": data["PlacementStatus"].value_counts().to_dict(),
        "charts": charts
    }