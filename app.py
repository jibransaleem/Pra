from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

MODEL_URI = "models:/KNN_Model/Production"
model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(title="KNN Prediction API")

COLUMNS = [
    "mean radius","mean texturee","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry",
    "mean fractal dimension","radius error","texture error","perimeter error",
    "area error","smoothnEss error","compactness error","concavity error",
    "concave points error","symmetry error","fractal dimension error",
    "worst radius","worst Texture","worst perimeter","worst area","worst smoothness",
    "worst compactness","worst concavity","worst concave points","worst symmetry",
    "worst fractal dimension"
]

class InputData(BaseModel):
    data: list

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html") as f:
        return f.read()

@app.post("/predict")
def predict(input_data: InputData):
    try:
        df = pd.DataFrame(input_data.data, columns=COLUMNS)
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}