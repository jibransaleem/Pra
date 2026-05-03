from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import pickle
from utils import log, Files

# Logger
my_logger = log(Logger_name="Model", file_name="model").get_logger()

# MLflow (DagsHub)
mlflow.set_tracking_uri("https://dagshub.com/saleemjibran813/Pra.mlflow")
mlflow.set_experiment("PROJ")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(BASE_DIR, "data", "SPLIT")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")

# Save model locally (optional)
def save_model(model, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    file_path = os.path.join(MODELS_DIR, name)
    with open(file_path, "wb") as file:
        pickle.dump(model, file)
    my_logger.info("Model saved locally")

# Data paths
x_train = os.path.join(SPLIT_DIR, "x_train.csv")
y_train = os.path.join(SPLIT_DIR, "y_train.csv")
x_test  = os.path.join(SPLIT_DIR, "x_test.csv")
y_test  = os.path.join(SPLIT_DIR, "y_test.csv")

# Training
with mlflow.start_run(run_name="model-train") as run:

    # Params
    params = Files().load_params()
    n_neighbors = params['n_neighbors']

    # Model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Load data
    xtr = pd.read_csv(x_train)
    ytr = pd.read_csv(y_train).values.ravel()

    xtt = pd.read_csv(x_test)
    ytt = pd.read_csv(y_test).values.ravel()

    # Train
    model.fit(xtr, ytr)

    # Predict
    preds = model.predict(xtt)

    # Metrics
    acc = accuracy_score(ytt, preds)
    f1 = f1_score(ytt, preds)

    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log + Register Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="knn",
        registered_model_name="KNN_Model"
    )

    # Optional local save
    save_model(model, "knn.pkl")

    my_logger.info("Model trained and logged successfully")

# Register + Promote to Production
client = MlflowClient()

# Get latest version
latest_versions = client.get_latest_versions("KNN_Model")
latest_version = latest_versions[-1].version

# Transition stage
client.transition_model_version_stage(
    name="KNN_Model",
    version=latest_version,
    stage="Production"
)

my_logger.info(f"Model version {latest_version} moved to Production")