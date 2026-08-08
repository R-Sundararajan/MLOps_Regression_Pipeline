# src/model_evaluation.py

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_DIR = Path("data/processed")
CANDIDATE_DIR = Path("models/candidates")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


# ---------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------

def get_model_metadata(model_path):

    filename = model_path.stem

    # Example:
    # linear_regression_lr_v1
    # random_forest_rf_v1
    # xgboost_xgb_v1

    if filename.startswith("linear_regression_"):
        model_type = "linear_regression"
        model_name = filename.replace(
            "linear_regression_",
            "",
        )

    elif filename.startswith("random_forest_"):
        model_type = "random_forest"
        model_name = filename.replace(
            "random_forest_",
            "",
        )

    elif filename.startswith("xgboost_"):
        model_type = "xgboost"
        model_name = filename.replace(
            "xgboost_",
            "",
        )

    else:
        model_type = "unknown"
        model_name = filename

    return model_type, model_name


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(y_true, predictions):

    mse = mean_squared_error(
        y_true,
        predictions,
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                predictions,
            )
        ),
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "r2": float(
            r2_score(
                y_true,
                predictions,
            )
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    params = load_params()

    target = params["data"]["target"]
    experiment_name = params["experiment"]["name"]

    # ---------------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------------

    test = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    X_test = test.drop(
        columns=[target]
    )

    y_test = test[target]

    # ---------------------------------------------------------------
    # Configure MLflow
    # ---------------------------------------------------------------

    mlflow.set_experiment(
        experiment_name
    )

    # ---------------------------------------------------------------
    # Find candidate models
    # ---------------------------------------------------------------

    candidate_models = sorted(
        CANDIDATE_DIR.glob("*.joblib")
    )

    if not candidate_models:

        raise FileNotFoundError(
            "No candidate models found in "
            f"{CANDIDATE_DIR}"
        )

    print(
        f"\nFound {len(candidate_models)} "
        "candidate models."
    )

    # ---------------------------------------------------------------
    # Evaluate candidates
    # ---------------------------------------------------------------

    results = []

    champion_model = None
    champion_result = None
    best_r2 = float("-inf")

    for model_path in candidate_models:

        model_type, model_name = (
            get_model_metadata(model_path)
        )

        print(
            f"\nEvaluating: "
            f"{model_type} / {model_name}"
        )

        # -----------------------------------------------------------
        # Load model
        # -----------------------------------------------------------

        model = joblib.load(
            model_path
        )

        # -----------------------------------------------------------
        # Predict
        # -----------------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # -----------------------------------------------------------
        # Calculate metrics
        # -----------------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        print(
            f"R2={metrics['r2']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"MAE={metrics['mae']:.4f}"
        )

        # -----------------------------------------------------------
        # MLflow
        # -----------------------------------------------------------

        with mlflow.start_run(
            run_name=f"eval_{model_name}"
        ):

            mlflow.set_tags(
                {
                    "model_type": model_type,
                    "model_name": model_name,
                    "experiment_phase": "evaluation",
                }
            )

            mlflow.log_metrics(
                metrics
            )

            mlflow.log_artifact(
                str(model_path),
                artifact_path="candidate_model",
            )

        # -----------------------------------------------------------
        # Store result
        # -----------------------------------------------------------

        result = {
            "model_type": model_type,
            "model_name": model_name,
            "model_path": str(model_path),
            "metrics": metrics,
        }

        results.append(
            result
        )

        # -----------------------------------------------------------
        # Champion
        # -----------------------------------------------------------

        if metrics["r2"] > best_r2:

            best_r2 = metrics["r2"]

            champion_model = model

            champion_result = result

    # ---------------------------------------------------------------
    # Save champion
    # ---------------------------------------------------------------

    if champion_model is None:

        raise RuntimeError(
            "Unable to determine champion model."
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    champion_path = (
        MODEL_DIR
        / "champion_model.joblib"
    )

    joblib.dump(
        champion_model,
        champion_path,
    )

    # ---------------------------------------------------------------
    # Save experiment results
    # ---------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "champion": champion_result,
        "experiments": results,
    }

    report_path = (
        REPORT_DIR
        / "experiment_results.json"
    )

    with open(
        report_path,
        "w",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # ---------------------------------------------------------------
    # Print champion
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHAMPION MODEL")
    print("=" * 70)

    print(
        f"Model type : "
        f"{champion_result['model_type']}"
    )

    print(
        f"Model name : "
        f"{champion_result['model_name']}"
    )

    print(
        f"R2         : "
        f"{champion_result['metrics']['r2']:.4f}"
    )

    print(
        f"RMSE       : "
        f"{champion_result['metrics']['rmse']:.4f}"
    )

    print(
        f"MAE        : "
        f"{champion_result['metrics']['mae']:.4f}"
    )

    print(
        f"\nChampion saved to: "
        f"{champion_path}"
    )

    print(
        f"Experiment results saved to: "
        f"{report_path}"
    )


if __name__ == "__main__":
    main()
