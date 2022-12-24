from thyroid.logger import logging
from thyroid.exception import thyroidException
from thyroid.utils import get_collection_as_dataframe
import os, sys
from thyroid.entity import config_entity
from thyroid.components.data_ingestion import DataIngestion



if __name__=="__main__":
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        Data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(Data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config= Data_ingestion_config)
        print(data_ingestion.initiate_data_ingestion())
    

    except Exception as e:
        raise thyroidException(e, sys)