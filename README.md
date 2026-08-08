# MLOps Regression Pipeline — Project Evidence

This folder contains the screenshots and supporting evidence for the MLOps Regression Pipeline capstone project.

The project follows the workflow specified in the project brief:

**Dataset → DVC Versioning → ML Pipeline → MLflow Tracking → Best Model → Model Registry → FastAPI Prediction API → Docker Container → GitHub Repository → GitHub Actions**

The project uses the **Wine Quality** dataset as a regression problem. The pipeline trains and compares Linear Regression, Random Forest, and XGBoost models, tracks experiments with MLflow, selects the best-performing model as the champion, serves it through FastAPI, containerizes the API with Docker, and validates the project through GitHub Actions.

> Reference: `Project_Briefing.pdf` specifies the required DVC, ML pipeline, MLflow, prediction API, Docker, and GitHub Actions phases.

## Project Evidence

All screenshots and supporting files are available in the output folder:

[View output folder](./output/)

### 1. DVC — Pipeline Reproduction

The DVC pipeline is reproduced using `dvc repro`. The screenshot shows the project stages executing and the DVC lock file being updated.


### 2. DVC — Pipeline DAG

The DVC DAG demonstrates the dependency flow between data ingestion, preprocessing, feature engineering, model building, and model evaluation.


### 3. MLflow — Experiment Tracking

MLflow is used to track the regression experiments. The experiment contains runs for:

- Linear Regression
- Random Forest
- XGBoost

Multiple hyperparameter configurations are evaluated and metrics including R², MAE, MSE, and RMSE are recorded. The MLflow UI is used to compare the candidate models and identify the strongest run.


### 4. FastAPI — API Running

The FastAPI application exposes a root endpoint confirming that the prediction service is running.

Endpoint:

`GET /`

Example response:

```json
{
  "message": "Wine Quality Prediction API is running"
}
```


### 5. FastAPI — Prediction Endpoint

The `/predict` endpoint accepts wine feature values as JSON and returns the model prediction.

Endpoint:

`POST /predict`

The screenshot demonstrates a successful HTTP 200 response from the prediction service.


### 6. Docker — Containerized Application

The FastAPI application and champion model are packaged into a Docker image. The container starts Uvicorn successfully and serves the API on port `8000`.

The Docker image contains:

- FastAPI application
- Python dependencies
- Champion model (`champion_model.joblib`)
- Uvicorn startup configuration


---

## Project Phases

### Phase 1 — Dataset Selection

The Wine Quality dataset was selected because it is suitable for a regression problem. The target variable is `quality`.

### Phase 2 — Data Versioning with DVC

DVC is used to version the dataset and define the reproducible ML pipeline.

Pipeline stages:

```text
data_ingestion
      ↓
data_preprocessing
      ↓
feature_engineering
      ↓
model_building
      ↓
model_evaluation
```

### Phase 3 — Machine Learning Pipeline

The training pipeline performs:

1. Data ingestion
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model evaluation

Three model families are experimented with:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Multiple hyperparameter configurations are evaluated for each model family.

### Phase 4 — MLflow Experiment Tracking

MLflow records the experiments and their parameters/metrics. The candidate models are compared using regression metrics such as:

- R²
- MAE
- MSE
- RMSE

The highest-performing model is selected as the champion model and stored locally as:

```text
models/champion_model.joblib
```

### Phase 5 — FastAPI Prediction API

The champion model is loaded by a FastAPI application.

Available endpoints:

```text
GET  /
GET  /health
POST /predict
```

The `/predict` endpoint accepts the Wine Quality feature values as JSON and returns the predicted quality score.

Interactive API documentation is available through FastAPI/Swagger at:

```text
/docs
```

### Phase 6 — Docker Containerization

The application is containerized with Docker.

The image packages the FastAPI service, required dependencies, and the champion model. Uvicorn is used as the application server.

The container exposes:

```text
8000
```

### Phase 7 — GitHub Actions

A GitHub Actions workflow is configured to execute on every push to the repository.

The CI workflow performs:

```text
Checkout repository
        ↓
Set up Python
        ↓
Install dependencies
        ↓
Run pytest
        ↓
Build Docker image
```

The workflow is defined in:

```text
.github/workflows/ci.yml
```

---

## Project Structure

```text
MLOps_Regression_Pipeline/
│
├── data/
├── models/
│   ├── candidates/
│   └── champion_model.joblib
│
├── reports/
│   └── experiment_results.json
│
├── server/
│   └── app.py
│
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── .dockerignore
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```

## Evidence Checklist

| Requirement | Evidence |
|---|---|
| Dataset selection | Wine Quality regression dataset |
| DVC initialization/versioning | DVC screenshots |
| Data ingestion | DVC pipeline |
| Preprocessing | DVC pipeline |
| Feature engineering | DVC pipeline |
| Model training | MLflow experiments |
| 3+ model types | Linear Regression, Random Forest, XGBoost |
| Multiple experiments | MLflow runs |
| Model evaluation | R², MAE, MSE, RMSE |
| Champion model | `models/champion_model.joblib` |
| FastAPI | Root and prediction screenshots |
| Docker | Docker screenshot |
| Automated tests | `tests/test_api.py` |
| GitHub Actions | `.github/workflows/ci.yml` |
| Successful GitHub Actions run | Add CI screenshot here if required for submission |
| MLflow Model Registry | Add registry screenshot here if required for submission |

## Additional Submission Evidence

The project brief also requests a screenshot of the **registered model** and a screenshot of a **successful GitHub Actions workflow**. Those two screenshots are not included in this output folder yet. Add them here when available, for example:

```text
07_mlflow_model_registry.png
08_github_actions_success.png
```

Then add links to them above.
