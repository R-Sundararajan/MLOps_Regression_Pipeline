# src/model_building.py

from pathlib import Path
import hashlib
import json

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml

from mlflow import MlflowClient

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

CANDIDATE_MODEL_DIR = MODEL_DIR / "candidates"

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def load_params():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


# ---------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------

def build_model(model_type, config):

    if model_type == "linear_regression":

        return LinearRegression(
            fit_intercept=config["fit_intercept"],
            positive=config["positive"],
        )

    elif model_type == "random_forest":

        return RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_type == "xgboost":

        return XGBRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )


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
# Experiment identity
# ---------------------------------------------------------------------

def create_experiment_key(model_type, config):
    """
    Create a deterministic identifier for a model configuration.

    The same model + same hyperparameters always produces
    the same key.

    Changing a hyperparameter produces a different key.
    """

    experiment_definition = {
        "model_type": model_type,
        "config": config,
    }

    serialized = json.dumps(
        experiment_definition,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------
# Check MLflow for an existing successful run
# ---------------------------------------------------------------------

def successful_run_exists(
    client,
    experiment_id,
    experiment_key,
):
    """
    Return True if MLflow already contains a successful
    run for this exact experiment configuration.
    """

    filter_string = (
        "attributes.status = 'FINISHED' "
        f"AND tags.experiment_key = '{experiment_key}'"
    )

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=1,
    )

    return len(runs) > 0


# ---------------------------------------------------------------------
# Find successful run
# ---------------------------------------------------------------------

def get_successful_run(
    client,
    experiment_id,
    experiment_key,
):

    filter_string = (
        "attributes.status = 'FINISHED' "
        f"AND tags.experiment_key = '{experiment_key}'"
    )

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=1,
    )

    if not runs:
        return None

    return runs[0]


# ---------------------------------------------------------------------
# Train + evaluate + MLflow
# ---------------------------------------------------------------------

def run_experiment(
    client,
    experiment_id,
    model_type,
    config,
    X_train,
    y_train,
    X_test,
    y_test,
):

    model_name = config["name"]

    experiment_key = create_experiment_key(
        model_type,
        config,
    )

    # ---------------------------------------------------------------
    # Skip if successful run already exists
    # ---------------------------------------------------------------

    if successful_run_exists(
        client,
        experiment_id,
        experiment_key,
    ):

        existing_run = get_successful_run(
            client,
            experiment_id,
            experiment_key,
        )

        print(
            f"\nSkipping successful experiment: "
            f"{model_type} / {model_name}"
        )

        print(
            f"Existing run ID: "
            f"{existing_run.info.run_id}"
        )

        print(
            f"Existing R2: "
            f"{existing_run.data.metrics.get('r2', 'N/A')}"
        )

        return {
            "status": "skipped",
            "model_type": model_type,
            "model_name": model_name,
            "experiment_key": experiment_key,
            "run_id": existing_run.info.run_id,
            "metrics": dict(existing_run.data.metrics),
            "params": dict(existing_run.data.params),
            "model": None,
        }

    # ---------------------------------------------------------------
    # New experiment
    # ---------------------------------------------------------------

    print(
        f"\nRunning experiment: "
        f"{model_type} / {model_name}"
    )

    model = build_model(
        model_type,
        config,
    )

    with mlflow.start_run(
        run_name=model_name
    ):

        # -----------------------------------------------------------
        # Tags
        # -----------------------------------------------------------

        mlflow.set_tags(
            {
                "model_type": model_type,
                "model_name": model_name,
                "experiment_key": experiment_key,
                "experiment_phase": "model_experimentation",
            }
        )

        # -----------------------------------------------------------
        # Parameters
        # -----------------------------------------------------------

        params_to_log = {
            key: value
            for key, value in config.items()
            if key != "name"
        }

        mlflow.log_params(params_to_log)

        # -----------------------------------------------------------
        # Training
        # -----------------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # -----------------------------------------------------------
        # Prediction
        # -----------------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # -----------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        mlflow.log_metrics(metrics)

        # -----------------------------------------------------------
        # Log model using the correct MLflow flavor
        # -----------------------------------------------------------

        if model_type == "xgboost":

            # IMPORTANT:
            # XGBRegressor should be logged using MLflow's
            # native XGBoost flavor rather than mlflow.sklearn.
            mlflow.xgboost.log_model(
                xgb_model=model,
                name="model",
                model_format="json",
            )

        else:

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="cloudpickle",
            )

        # -----------------------------------------------------------
        # Save local candidate model
        # -----------------------------------------------------------

        CANDIDATE_MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidate_path = (
            CANDIDATE_MODEL_DIR
            / f"{model_type}_{model_name}.joblib"
        )

        joblib.dump(
            model,
            candidate_path,
        )

        # Log candidate model as an artifact too
        mlflow.log_artifact(
            str(candidate_path),
            artifact_path="joblib",
        )

        # -----------------------------------------------------------
        # Result
        # -----------------------------------------------------------

        run_id = mlflow.active_run().info.run_id

        print(
            f"R2={metrics['r2']:.4f} | "
            f"RMSE={metrics['rmse']:.4f}"
        )

        print(
            f"MLflow Run ID: {run_id}"
        )

        return {
            "status": "success",
            "model_type": model_type,
            "model_name": model_name,
            "experiment_key": experiment_key,
            "run_id": run_id,
            "metrics": metrics,
            "params": params_to_log,
            "model": model,
            "candidate_path": str(candidate_path),
        }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    params = load_params()

    target = params["data"]["target"]
    experiment_name = params["experiment"]["name"]

    # ---------------------------------------------------------------
    # Load processed data
    # ---------------------------------------------------------------

    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    test = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    X_train = train.drop(
        columns=[target]
    )

    y_train = train[target]

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

    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' "
            f"could not be found."
        )

    experiment_id = experiment.experiment_id

    print(
        f"MLflow experiment: "
        f"{experiment_name}"
    )

    print(
        f"Experiment ID: "
        f"{experiment_id}"
    )

    # ---------------------------------------------------------------
    # Run all experiments
    # ---------------------------------------------------------------

    results = []

    for model_type, configurations in (
        params["models"].items()
    ):

        for config in configurations:

            result = run_experiment(
                client=client,
                experiment_id=experiment_id,
                model_type=model_type,
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            results.append(result)

    # ---------------------------------------------------------------
    # Get ALL successful experiments from MLflow
    #
    # This is important:
    # skipped experiments must still participate in champion
    # selection.
    # ---------------------------------------------------------------

    successful_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED'"
        ),
        order_by=[
            "metrics.r2 DESC"
        ],
    )

    if not successful_runs:
        raise RuntimeError(
            "No successful MLflow experiments found."
        )

    # ---------------------------------------------------------------
    # Champion
    # ---------------------------------------------------------------

    champion_run = successful_runs[0]

    champion_run_id = champion_run.info.run_id

    champion_r2 = champion_run.data.metrics.get(
        "r2"
    )

    champion_model_type = (
        champion_run.data.tags.get(
            "model_type"
        )
    )

    champion_model_name = (
        champion_run.data.tags.get(
            "model_name"
        )
    )

    print("\n" + "=" * 70)
    print("CHAMPION MODEL")
    print("=" * 70)

    print(
        f"Model type : {champion_model_type}"
    )

    print(
        f"Model name : {champion_model_name}"
    )

    print(
        f"Run ID     : {champion_run_id}"
    )

    print(
        f"R2         : {champion_r2:.4f}"
    )

    # ---------------------------------------------------------------
    # Find the corresponding local candidate model
    # ---------------------------------------------------------------

    candidate_path = (
        CANDIDATE_MODEL_DIR
        / f"{champion_model_type}_"
          f"{champion_model_name}.joblib"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    champion_path = (
        MODEL_DIR
        / "champion_model.joblib"
    )

    # ---------------------------------------------------------------
    # If candidate exists locally, copy it.
    #
    # This handles both newly trained and previously successful
    # experiments.
    # ---------------------------------------------------------------

    if candidate_path.exists():

        champion_model = joblib.load(
            candidate_path
        )

        joblib.dump(
            champion_model,
            champion_path,
        )

    else:

        # -----------------------------------------------------------
        # This can happen if MLflow contains a successful run from
        # a previous execution but the local candidate file was
        # deleted.
        #
        # Download the MLflow artifact instead.
        # -----------------------------------------------------------

        print(
            "Local champion candidate not found."
        )

        print(
            "Downloading champion from MLflow..."
        )

        if champion_model_type == "xgboost":

            champion_model = (
                mlflow.xgboost.load_model(
                    f"runs:/{champion_run_id}/model"
                )
            )

        else:

            champion_model = (
                mlflow.sklearn.load_model(
                    f"runs:/{champion_run_id}/model"
                )
            )

        joblib.dump(
            champion_model,
            champion_path,
        )

    print(
        f"Champion saved to: "
        f"{champion_path}"
    )

    # ---------------------------------------------------------------
    # Save experiment summary
    # ---------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_summary = []

    for run in successful_runs:

        experiment_summary.append(
            {
                "run_id": run.info.run_id,
                "model_type": run.data.tags.get(
                    "model_type"
                ),
                "model_name": run.data.tags.get(
                    "model_name"
                ),
                "status": run.info.status,
                "r2": run.data.metrics.get(
                    "r2"
                ),
                "rmse": run.data.metrics.get(
                    "rmse"
                ),
                "mae": run.data.metrics.get(
                    "mae"
                ),
                "mse": run.data.metrics.get(
                    "mse"
                ),
                "params": dict(
                    run.data.params
                ),
            }
        )

    with open(
        REPORT_DIR
        / "experiment_results.json",
        "w",
    ) as file:

        json.dump(
            {
                "champion": {
                    "run_id": champion_run_id,
                    "model_type": champion_model_type,
                    "model_name": champion_model_name,
                    "r2": champion_r2,
                },
                "experiments": experiment_summary,
            },
            file,
            indent=4,
        )

    print(
        f"\nExperiment results saved to: "
        f"{REPORT_DIR / 'experiment_results.json'}"
    )


if __name__ == "__main__":
    main()
