from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(title="KNN Prediction API")

COLUMNS = [
    "Unnamed: 0",
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension", "radius error", "texture error", "perimeter error",
    "area error", "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry",
    "worst fractal dimension"
]

# ── Lazy model load ──
model = None

def get_model():
    global model
    if model is None:
        model = mlflow.sklearn.load_model("models:/KNN_Model/Production")
    return model


class InputData(BaseModel):
    data: list


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Breast Cancer KNN Predictor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #f0f4f8;
    color: #1a202c;
    min-height: 100vh;
  }
  header {
    background: linear-gradient(135deg, #2b6cb0, #2c5282);
    color: white;
    padding: 2rem;
    text-align: center;
  }
  header h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.4rem; }
  header p  { font-size: 0.95rem; opacity: 0.85; }
  .container { max-width: 960px; margin: 2rem auto; padding: 0 1.5rem; }
  .presets { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem; }
  .preset-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 2px solid transparent;
    cursor: pointer;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }
  .preset-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
  .preset-card.benign   { border-color: #48bb78; }
  .preset-card.malignant { border-color: #fc8181; }
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  .benign   .badge { background: #c6f6d5; color: #276749; }
  .malignant .badge { background: #fed7d7; color: #9b2c2c; }
  .preset-card h3 { font-size: 1rem; margin-bottom: 0.25rem; }
  .preset-card p  { font-size: 0.82rem; color: #718096; }
  .load-btn {
    margin-top: 0.8rem;
    padding: 6px 14px;
    border: none;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
  }
  .benign   .load-btn { background: #48bb78; color: white; }
  .malignant .load-btn { background: #fc8181; color: white; }
  .form-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
  }
  .form-card h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; color: #2d3748; }
  .section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #718096;
    margin: 1.2rem 0 0.6rem;
    padding-bottom: 4px;
    border-bottom: 1px solid #e2e8f0;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 10px;
  }
  .field { display: flex; flex-direction: column; gap: 3px; }
  .field label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .field input {
    padding: 7px 9px;
    border: 1.5px solid #e2e8f0;
    border-radius: 7px;
    font-size: 0.85rem;
    color: #1a202c;
    background: #f7fafc;
    transition: border-color 0.15s;
  }
  .field input:focus { outline: none; border-color: #4299e1; background: white; }
  .actions { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 1.2rem; }
  .btn {
    padding: 10px 22px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
  }
  .btn:active { transform: scale(0.97); }
  .btn-primary { background: #3182ce; color: white; }
  .btn-primary:hover { background: #2b6cb0; }
  .btn-random  { background: #805ad5; color: white; }
  .btn-random:hover  { background: #6b46c1; }
  .btn-clear   { background: #e2e8f0; color: #4a5568; }
  .btn-clear:hover   { background: #cbd5e0; }
  #result-box {
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    display: none;
    margin-bottom: 1.5rem;
  }
  #result-box.benign    { background: #f0fff4; border: 2px solid #48bb78; }
  #result-box.malignant { background: #fff5f5; border: 2px solid #fc8181; }
  #result-box.error     { background: #fffbeb; border: 2px solid #f6ad55; }
  #result-label  { font-size: 0.85rem; color: #718096; margin-bottom: 0.3rem; }
  #result-value  { font-size: 2rem; font-weight: 700; }
  #result-box.benign    #result-value { color: #276749; }
  #result-box.malignant #result-value { color: #9b2c2c; }
  #result-box.error     #result-value { color: #c05621; font-size: 1rem; }
  #result-sub { font-size: 0.85rem; margin-top: 0.4rem; color: #4a5568; }
  .spinner {
    width: 28px; height: 28px;
    border: 3px solid #e2e8f0;
    border-top-color: #3182ce;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin: 0 auto 0.5rem;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  footer {
    text-align: center;
    padding: 1.5rem;
    font-size: 0.8rem;
    color: #a0aec0;
  }
</style>
</head>
<body>

<header>
  <h1>Breast Cancer KNN Predictor</h1>
  <p>Wisconsin Diagnostic Dataset — 30 tumor features → Benign or Malignant</p>
</header>

<div class="container">

  <div id="result-box">
    <div class="spinner" id="spinner"></div>
    <div id="result-label">Prediction</div>
    <div id="result-value">—</div>
    <div id="result-sub"></div>
  </div>

  <div class="presets">
    <div class="preset-card benign">
      <div class="badge">Sample A</div>
      <h3>Typical Malignant Case</h3>
      <p>High-risk values — larger radius, high concavity</p>
      <button class="load-btn" onclick="loadSample('malignant')">Load Sample A</button>
    </div>
    <div class="preset-card malignant">
      <div class="badge">Sample B</div>
      <h3>Typical Benign Case</h3>
      <p>Low-risk values — smaller radius, low concavity</p>
      <button class="load-btn" onclick="loadSample('benign')">Load Sample B</button>
    </div>
  </div>

  <div class="form-card">
    <h2>Enter Feature Values</h2>

    <div class="section-title">Mean values</div>
    <div class="grid" id="grid-mean"></div>

    <div class="section-title">Standard error values</div>
    <div class="grid" id="grid-error"></div>

    <div class="section-title">Worst values</div>
    <div class="grid" id="grid-worst"></div>

    <div class="actions">
      <button class="btn btn-primary" onclick="predict()">Predict</button>
      <button class="btn btn-random"  onclick="randomize()">Randomize</button>
      <button class="btn btn-clear"   onclick="clearAll()">Clear</button>
    </div>
  </div>

</div>

<footer>KNN Model — MLflow Registry — Breast Cancer Wisconsin Dataset</footer>

<script>
const FEATURES = [
  {key:"mean_radius",       label:"Mean radius",        min:6,     max:28,   dec:2, group:"mean"},
  {key:"mean_texture",      label:"Mean texture",       min:9,     max:40,   dec:2, group:"mean"},
  {key:"mean_perimeter",    label:"Mean perimeter",     min:40,    max:190,  dec:2, group:"mean"},
  {key:"mean_area",         label:"Mean area",          min:140,   max:2500, dec:1, group:"mean"},
  {key:"mean_smoothness",   label:"Mean smoothness",    min:0.05,  max:0.16, dec:4, group:"mean"},
  {key:"mean_compactness",  label:"Mean compactness",   min:0.02,  max:0.35, dec:4, group:"mean"},
  {key:"mean_concavity",    label:"Mean concavity",     min:0,     max:0.43, dec:4, group:"mean"},
  {key:"mean_concave_pts",  label:"Mean concave pts",   min:0,     max:0.20, dec:4, group:"mean"},
  {key:"mean_symmetry",     label:"Mean symmetry",      min:0.1,   max:0.3,  dec:4, group:"mean"},
  {key:"mean_fractal_dim",  label:"Mean fractal dim",   min:0.05,  max:0.1,  dec:5, group:"mean"},
  {key:"radius_error",      label:"Radius error",       min:0.1,   max:2.9,  dec:3, group:"error"},
  {key:"texture_error",     label:"Texture error",      min:0.3,   max:4.9,  dec:3, group:"error"},
  {key:"perimeter_error",   label:"Perimeter error",    min:0.7,   max:22,   dec:2, group:"error"},
  {key:"area_error",        label:"Area error",         min:6,     max:540,  dec:1, group:"error"},
  {key:"smoothness_error",  label:"Smoothness error",   min:0.001, max:0.031,dec:5, group:"error"},
  {key:"compactness_error", label:"Compactness error",  min:0.002, max:0.14, dec:5, group:"error"},
  {key:"concavity_error",   label:"Concavity error",    min:0,     max:0.4,  dec:5, group:"error"},
  {key:"concave_pts_error", label:"Concave pts error",  min:0,     max:0.053,dec:5, group:"error"},
  {key:"symmetry_error",    label:"Symmetry error",     min:0.007, max:0.08, dec:5, group:"error"},
  {key:"fractal_dim_error", label:"Fractal dim error",  min:0.0008,max:0.03, dec:6, group:"error"},
  {key:"worst_radius",      label:"Worst radius",       min:7,     max:36,   dec:2, group:"worst"},
  {key:"worst_texture",     label:"Worst texture",      min:12,    max:50,   dec:2, group:"worst"},
  {key:"worst_perimeter",   label:"Worst perimeter",    min:50,    max:250,  dec:2, group:"worst"},
  {key:"worst_area",        label:"Worst area",         min:185,   max:4250, dec:1, group:"worst"},
  {key:"worst_smoothness",  label:"Worst smoothness",   min:0.07,  max:0.22, dec:4, group:"worst"},
  {key:"worst_compactness", label:"Worst compactness",  min:0.027, max:1.06, dec:4, group:"worst"},
  {key:"worst_concavity",   label:"Worst concavity",    min:0,     max:1.25, dec:4, group:"worst"},
  {key:"worst_concave_pts", label:"Worst concave pts",  min:0,     max:0.29, dec:4, group:"worst"},
  {key:"worst_symmetry",    label:"Worst symmetry",     min:0.15,  max:0.66, dec:4, group:"worst"},
  {key:"worst_fractal_dim", label:"Worst fractal dim",  min:0.055, max:0.21, dec:5, group:"worst"},
];

const SAMPLES = {
  malignant: [17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189],
  benign:    [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
};

['mean','error','worst'].forEach(group => {
  const grid = document.getElementById('grid-' + group);
  FEATURES.filter(f => f.group === group).forEach(f => {
    const div = document.createElement('div');
    div.className = 'field';
    div.innerHTML = `<label for="f_${f.key}">${f.label}</label>
      <input id="f_${f.key}" type="number" step="any" placeholder="${((f.min+f.max)/2).toFixed(f.dec)}">`;
    grid.appendChild(div);
  });
});

function setValues(vals) {
  FEATURES.forEach((f, i) => {
    document.getElementById('f_' + f.key).value = vals[i];
  });
}

function loadSample(name) {
  setValues(SAMPLES[name]);
  document.getElementById('result-box').style.display = 'none';
}

function randomize() {
  setValues(FEATURES.map(f => parseFloat((f.min + Math.random() * (f.max - f.min)).toFixed(f.dec))));
  document.getElementById('result-box').style.display = 'none';
}

function clearAll() {
  FEATURES.forEach(f => { document.getElementById('f_' + f.key).value = ''; });
  document.getElementById('result-box').style.display = 'none';
}

async function predict() {
  const vals = FEATURES.map(f => parseFloat(document.getElementById('f_' + f.key).value));
  if (vals.some(v => isNaN(v))) {
    showResult('error', 'Missing values', 'Please fill all fields or use a sample / randomize', '');
    return;
  }

  const rb = document.getElementById('result-box');
  rb.className = '';
  rb.style.display = 'block';
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('result-label').textContent = 'Running prediction...';
  document.getElementById('result-value').textContent = '';
  document.getElementById('result-sub').textContent = '';

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [vals] })
    });
    const json = await res.json();
    document.getElementById('spinner').style.display = 'none';

    if (json.error) {
      showResult('error', 'Error', json.error, '');
    } else {
      const pred = json.predictions[0];
      const isMalignant = (pred === 1 || pred === '1' || pred === 'M');
      if (isMalignant) {
        showResult('malignant', 'Malignant', 'High-risk tumor detected', 'Further clinical evaluation strongly recommended');
      } else {
        showResult('benign', 'Benign', 'Low-risk — benign classification', 'Consistent with non-cancerous tissue');
      }
    }
  } catch(e) {
    document.getElementById('spinner').style.display = 'none';
    showResult('error', 'API Error', 'Could not reach prediction endpoint', e.message);
  }
}

function showResult(cls, value, label, sub) {
  const rb = document.getElementById('result-box');
  rb.className = cls;
  rb.style.display = 'block';
  document.getElementById('result-label').textContent = label;
  document.getElementById('result-value').textContent = value;
  document.getElementById('result-sub').textContent = sub;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    if os.path.exists("index.html"):
        with open("index.html") as f:
            return f.read()
    return HTMLResponse(content=HTML_PAGE)


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/predict")
def predict(input_data: InputData):
    try:
        rows = []
        for i, row in enumerate(input_data.data):
            rows.append([i] + list(row))  # prepend Unnamed: 0 index
        df = pd.DataFrame(rows, columns=COLUMNS)
        preds = get_model().predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}