import os
import pandas as pd
from utils import log , Files
from sklearn.model_selection import train_test_split
import mlflow
import dagshub

dagshub.init(repo_owner='saleemjibran813', repo_name='Pra', mlflow=True)
mlflow.set_tracking_uri(
    "https://dagshub.com/saleemjibran813/Pra.mlflow"
)

mlflow.set_experiment("PROJ")

# your logger (assuming your class is already defined)
my_logger = log(
    Logger_name="data-ingestor",
    file_name="data-ingestion"
).get_logger()

class  Ingect_data:
    def __init__(self):
        self.file_path = r"C:\Users\ADIL TRADERS\OneDrive\Desktop\folder\Pra\data\data.csv"

    
    def load_data(self):
        try :
            if not os.path.isfile(self.file_path):
                raise FileNotFoundError(f"File not found at loc {self.file_path}")
            data = pd.read_csv(self.file_path)
            my_logger.info("Data is sucessfully injected from loc %s \n" , str(self.file_path))
            return data
        except FileNotFoundError as e:
            my_logger.error(str(e))
            raise
class Split_data(Ingect_data):
    def __init__(self):
        super().__init__()
        self.Base_path = r"C:\Users\ADIL TRADERS\OneDrive\Desktop\folder\Pra\data"
    def save_data(self ,data_object , name):
        FOLDER_NAME = "SPLIT"
        FOLDER   = os.path.join(self.Base_path , FOLDER_NAME)
        os.makedirs(FOLDER , exist_ok=True)
        file_path = os.path.join(FOLDER , name)
        data_object.to_csv(file_path ,index=False)
        my_logger.info("sucessfully saved %s", name)
    def split_data(self):
        params =  Files().load_params()
        test_size = params['split_ratio']
        data = self.load_data()
        x_train , x_test , y_train , y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1] , test_size=test_size , random_state=23)
        self.save_data(x_train , "x_train.csv")
        self.save_data(x_test , "x_test.csv")
        self.save_data(y_train ,"y_train.csv")
        self.save_data(y_test ,"y_test.csv")
        return test_size
with mlflow.start_run(run_name="Data-ingestion") as run:
    
    obj = Split_data()
    test_size = obj.split_data()
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("train_size" , 1-test_size)