from thyroid.exception import thyroidException
from thyroid.logger import logging
from thyroid.predictor import ModelResolver
import pandas as pd
from thyroid.utils import load_object
import os,sys
from datetime import datetime
PREDICTION_DIR="prediction"

import numpy as np
def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"?":np.NAN},inplace=True)
        #validation
        
        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        input_encoder = load_object(file_path=model_resolver.get_latest_input_encoder_path())

        input_feature_name_num = list(transformer.feature_names_in_)
        input_feature_name_cat = list(input_encoder.feature_names_in_)
        input_arr_num =transformer.transform(df[input_feature_name_num])
        input_arr_cat =input_encoder.transform(df[input_feature_name_cat])
        logging.info(input_arr_cat)
        logging.info(input_arr_num)

        input_arr = np.c_[input_arr_cat,input_arr_num]
        logging.info(df[input_feature_name_cat + input_feature_name_num].columns)


        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)
        
        logging.info(f"Target encoder to convert predicted column into categorical")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())

        cat_prediction = target_encoder.inverse_transform(prediction.astype('int'))

        df["prediction"]=prediction
        df["cat_pred"]=cat_prediction


        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise thyroidException(e, sys)