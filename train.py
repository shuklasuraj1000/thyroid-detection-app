
from thyroid.pipeline.training_pipeline import start_training_pipeline


file_path="/config/workspace/artifact/01042023__234146/data_ingestion/dataset/test.csv"
print(__name__)
if __name__=="__main__":
    try:
        start_training_pipeline()
    except Exception as e:
        print(e)