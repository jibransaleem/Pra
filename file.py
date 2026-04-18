import yaml

pth  = r"C:\Users\ADIL TRADERS\OneDrive\Desktop\folder\Pra\params.yaml"
with open(pth , "r") as file:
    data=  yaml.safe_load(file)
    print(data)