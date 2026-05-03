from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="KNN Prediction API")

COLUMNS = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension", "radius error", "texture error", "perimeter error",
    "area error", "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry",
    "worst fractal dimension"
]

# ── Lazy load — loads once on first request, not at startup ──
model = None

def get_model():
    global model
    if model is None:
        model = mlflow.sklearn.load_model("models:/KNN_Model/Production")
    return model


class InputData(BaseModel):
    data: list


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/predict")
def predict(input_data: InputData):
    try:
        df = pd.DataFrame(input_data.data, columns=COLUMNS)
        preds = get_model().predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}