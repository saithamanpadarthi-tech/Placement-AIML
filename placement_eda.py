import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def run_eda():

    data = load_data()

    charts = []

    sns.set_style("whitegrid")

    # Missing Values
    missing = data.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:

        plt.figure(figsize=(10,5))
        sns.barplot(x=missing.index, y=missing.values)

        plt.xticks(rotation=45)
        plt.title("Missing Values")

        plt.tight_layout()
        plt.savefig(_chart_path("missing_values.png"))
        plt.close()

        charts.append("missing_values.png")

        plt.figure(figsize=(12,6))
        sns.heatmap(data.isnull(), cbar=False)

        plt.title("Missing Value Heatmap")

        plt.tight_layout()
        plt.savefig(_chart_path("missing_heatmap.png"))
        plt.close()

        charts.append("missing_heatmap.png")

    duplicates = int(data.duplicated().sum())

    target_counts = data["PlacementStatus"].value_counts().to_dict()

    plt.figure(figsize=(6,5))

    sns.countplot(
        x="PlacementStatus",
        data=data
    )

    plt.title("Placement Status Distribution")

    plt.tight_layout()

    plt.savefig(_chart_path("placement_status.png"))
    plt.close()

    charts.append("placement_status.png")

    return {
        "rows": len(data),
        "columns": len(data.columns),
        "duplicates": duplicates,
        "missing": missing.to_dict(),
        "target_counts": target_counts,
        "charts": charts
    }