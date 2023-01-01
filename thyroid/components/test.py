import pandas as pd

df= pd.read_csv("/config/workspace/artifact/01012023__110532/data_ingestion/feature_store/thyroid.csv")
print(df.isnull().sum())