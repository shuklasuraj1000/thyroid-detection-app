from thyroid.entity import artifact_entity,config_entity
from thyroid.exception import thyroidException
from thyroid.logger import logging
from typing import Optional
import os,sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from thyroid import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
from thyroid.config import TARGET_COLUMN



class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise thyroidException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            scaler = StandardScaler()
            pipeline = Pipeline(steps=[
                    ('scaler',scaler)
                ])
            return pipeline
        except Exception as e:
            raise thyroidException(e, sys)


    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #Finding cotegorical and numerical columns in train and test dataset
            input_cat_col= [i for i in train_df.columns if train_df[i].dtype=='O']
            input_cat_col.remove(TARGET_COLUMN)
            logging.info(f"Categorical columns : {input_cat_col}")

            input_num_col= [i for i in train_df.columns if train_df[i].dtype!='O']
            logging.info(f"Numerical columns : {input_num_col}")
            
            
            #selecting input categorical & numerical feature datframe for train and test dataframe
            input_feature_train_df_cat=train_df[input_cat_col]
            input_feature_test_df_cat=test_df[input_cat_col]

            input_feature_train_df_num=train_df[input_num_col]
            input_feature_test_df_num=test_df[input_num_col]

            #Ordinal encoding for categorical input features >>>>>>>>>>>>>>>>>>>>>>
            enc = OrdinalEncoder()
            enc.fit(input_feature_train_df_cat)

            #transformation on input categorical features
            input_feature_train_arr_cat = enc.transform(input_feature_train_df_cat)
            input_feature_test_arr_cat = enc.transform(input_feature_test_df_cat)


            #selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            #transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)


            transformation_pipleine = DataTransformation.get_data_transformer_object()    #>>>>>>>>>>>>>>>
            transformation_pipleine.fit(input_feature_train_df_num)

            #transforming input numeral features
            input_feature_train_arr_num = transformation_pipleine.transform(input_feature_train_df_num)
            input_feature_test_arr_num = transformation_pipleine.transform(input_feature_test_df_num)

            # Joining input categorical and numerical feature after encoding & transformation
            input_feature_train_arr = np.c_[input_feature_train_arr_cat, input_feature_train_arr_num]
           
            input_feature_test_arr = np.c_[input_feature_test_arr_cat, input_feature_test_arr_num]
           

            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            
            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipleine)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=label_encoder)



            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path

            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise thyroidException(e, sys)


