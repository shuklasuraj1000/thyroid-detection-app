from thyroid.predictor import ModelResolver
from thyroid.entity import config_entity,artifact_entity
from thyroid.exception import thyroidException
from thyroid.logger import logging
from thyroid.utils import load_object
from sklearn.metrics import f1_score
import pandas  as pd
import numpy as np
import sys,os
from thyroid.config import TARGET_COLUMN

class ModelEvaluation:

    def __init__(self,
        model_eval_config:config_entity.ModelEvaluationConfig,
        data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
        data_transformation_artifact:artifact_entity.DataTransformationArtifact,
        model_trainer_artifact:artifact_entity.ModelTrainerArtifact      
        ):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise thyroidException(e,sys)



    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            #if saved model folder has model the we will compare 
            #which model is best trained or the model from saved model folder

            logging.info("if saved model folder has model the we will compare "
            "which model is best trained or the model from saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                improved_accuracy=None)
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact


            #Finding location of transformer model and target encoder
            logging.info("Finding location of transformer model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            input_encoder_path = self.model_resolver.get_latest_input_encoder_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()
            print(model_path)
            print(input_encoder_path)

            logging.info("Previous trained objects of transformer, model and target encoder")
            #Previous trained  objects
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            input_encoder = load_object(file_path=input_encoder_path)
            target_encoder = load_object(file_path=target_encoder_path)
            

            logging.info("Currently trained model objects")
            #Currently trained model objects
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model  = load_object(file_path=self.model_trainer_artifact.model_path)
            current_input_encoder = load_object(file_path=self.data_transformation_artifact.input_encoder_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)
            


            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            y_true =target_encoder.transform(target_df)
            # accuracy using previous trained model
            
            input_feature_name_num = list(transformer.feature_names_in_)
            input_feature_name_cat = list(input_encoder.feature_names_in_)
            input_arr_num =transformer.transform(test_df[input_feature_name_num])
            input_arr_cat =input_encoder.transform(test_df[input_feature_name_cat])
            logging.info(input_arr_cat)
            logging.info(input_arr_num)

            input_arr = np.c_[input_arr_cat,input_arr_num]
            logging.info(test_df[input_feature_name_cat + input_feature_name_num].columns)

            y_pred = model.predict(input_arr)
            print(f"Prediction using previous model: {target_encoder.inverse_transform(y_pred[:5].astype('int'))}")
            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")
           
            # accuracy using current trained model >>........
            #>input_feature_name = list(current_transformer.feature_names_in_)
            #>input_arr =current_transformer.transform(test_df[input_feature_name])

            input_feature_name_num = list(current_transformer.feature_names_in_)
            input_feature_name_cat = list(current_input_encoder.feature_names_in_)
            input_arr_num =current_transformer.transform(test_df[input_feature_name_num])
            input_arr_cat =current_input_encoder.transform(test_df[input_feature_name_cat])

            input_arr = np.c_[input_arr_cat,input_arr_num]


            y_pred = current_model.predict(input_arr)
            y_true =current_target_encoder.transform(target_df)
            #print([current_target_encoder.inverse_transform(i) for i in y_pred])
            print(f"Prediction using trained model: {current_target_encoder.inverse_transform(y_pred[:5].astype('int'))}")
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using current trained model: {current_model_score}")
            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise thyroidException(e,sys)