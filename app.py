from flask import Flask, render_template
from load_data import get_data_summary
from placement_eda import run_eda
from preprocessing import run_preprocessing
from linear_regression import run_linear_regression
from Logistic_Regression import run_logistic_regression
from decision_tree import run_decision_tree
from random_forest import run_random_forest
from bagging import run_bagging
from boosting import run_boosting


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        active="none",
        summary=None,
        error=None
    )
# =========================================================
# DATA LOADING
# =========================================================
@app.route("/data-loading")
def data_loading():
    try:
        summary = get_data_summary()
        return render_template(
            "index.html",
            active="data-loading",
            summary=summary,
            error=None
        )

    except Exception as e:

        return render_template(
            "index.html",
            active="data-loading",
            summary=None,
            error=str(e)
        )


# =========================================================
# EDA
# =========================================================

@app.route("/eda")
def eda():

    try:

        result = run_eda()

        return render_template(
            "eda.html",
            active="eda",
            eda=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "eda.html",
            active="eda",
            eda=None,
            error=str(e)
        )


# =========================================================
# PREPROCESSING
# =========================================================

@app.route("/preprocessing")
def preprocessing():

    try:

        result = run_preprocessing()

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            preprocessing=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            preprocessing=None,
            error=str(e)
        )


# =========================================================
# LINEAR REGRESSION
# =========================================================

@app.route("/linear-regression")
def linear_regression():

    try:

        result = run_linear_regression()

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            regression=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            regression=None,
            error=str(e)
        )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

@app.route("/logistic-regression")
def logistic_regression():

    try:

        result = run_logistic_regression()

        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            logistic=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            logistic=None,
            error=str(e)
        )


# =========================================================
# DECISION TREE
# =========================================================

@app.route("/decision-tree")
def decision_tree():

    try:

        result = run_decision_tree()

        return render_template(
            "decision_tree.html",
            active="decision-tree",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "decision_tree.html",
            active="decision-tree",
            result=None,
            error=str(e)
        )


# =========================================================
# RANDOM FOREST
# =========================================================

@app.route("/random-forest")
def random_forest():

    try:

        result = run_random_forest()

        return render_template(
            "random_forest.html",
            active="random-forest",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "random_forest.html",
            active="random-forest",
            result=None,
            error=str(e)
        )


# =========================================================
# BAGGING
# =========================================================

@app.route("/bagging")
def bagging():

    try:

        result = run_bagging()

        return render_template(
            "bagging.html",
            active="bagging",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "bagging.html",
            active="bagging",
            result=None,
            error=str(e)
        )


# =========================================================
# BOOSTING
# =========================================================

@app.route("/boosting")
def boosting():

    try:

        result = run_boosting()

        return render_template(
            "boosting.html",
            active="boosting",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "boosting.html",
            active="boosting",
            result=None,
            error=str(e)
        )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )