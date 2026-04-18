from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score , f1_score
import mlflow
import os
import pickle
import dagshub
from utils import log , Files
my_logger = log(Logger_name="Model" , file_name="model").get_logger()
dagshub.init(
    repo_owner="saleemjibran813",
    repo_name="Pra",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/saleemjibran813/Pra.mlflow"
)
mlflow.set_experiment("PROJ")
def save_model(model , name):
    base_path  = r"C:\Users\ADIL TRADERS\OneDrive\Desktop\folder\Pra\artifacts\models"
    os.makedirs(base_path , exist_ok=True)
    file_path = os.path.join(base_path , name)
    with open(file_path , "wb") as file:
        pickle.dump(model , file)
    my_logger.info("model is saved")

x_train = r"C:\Users\ADIL TRADERS\OneDrive\Desktop\folder\Pra\data\SPLIT\x_train.csv"
y_train =  r"C:\Users\ADIL TRADERS\OneDrive\Desktop\folder\Pra\data\SPLIT\y_train.csv"
x_test  =  r"C:\Users\ADIL TRADERS\OneDrive\Desktop\folder\Pra\data\SPLIT\x_test.csv"
y_test = r"C:\Users\ADIL TRADERS\OneDrive\Desktop\folder\Pra\data\SPLIT\y_test.csv"  
with mlflow.start_run(run_name="model-train") as run :
    
    n_neighbors =  Files().load_params()['Model']['n_neighbors']
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    xtr = pd.read_csv(x_train)
    ytr =  pd.read_csv(y_train).values.ravel()
    model.fit(xtr , ytr)
    xtt = pd.read_csv(x_test)
    ytt =  pd.read_csv(y_test).values.ravel()
    preds =  model.predict(xtt) 
    acc =  accuracy_score(ytt , preds)
    f1= f1_score(ytt , preds)
    mlflow.log_metric("accuracy-score" , acc)
    mlflow.log_metric("f1-score", f1)
    mlflow.sklearn.log_model(sk_model=model , name="knn")
    my_logger.info("Model trained sucessfully")
