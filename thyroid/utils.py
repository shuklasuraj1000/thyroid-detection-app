import pandas as pd
from thyroid.logger import logging
from thyroid.exception import thyroidException
from thyroid.config import mongo_client
from collections import Counter
from thyroid.config import numeric_features
import os,sys
import yaml
import numpy as np
import dill

numeric_feature= numeric_features

def missing_data_handler(df)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    data: df (data frame)
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info("missing data handling initiated")
        #replace na with Nan
        df.replace(to_replace='?',value=np.NAN,inplace=True)
        logging.info("Replaced ? with NaN")

        # Data typecasting of numerical features.
        for i in numeric_feature:
            df[i]=df[i].astype('float')
        logging.info("numeric feature converted to float")

        categorical_features = list((Counter(df.columns) - Counter(numeric_feature)).elements())
        logging.info(categorical_features)

        # Adding mode value for missing categorical data:
        for i in categorical_features:
            df[i].fillna(df[i].mode(), inplace=True)
        logging.info("categorical missing data filled")

        # Exception 'sex' column creatings issue, handling seperately. (################################)
        df['sex'].fillna(df['sex'].mode()[0], inplace=True)

        # Adding mean value for missing numerical data:

        for j in numeric_features:
            df[j].fillna(df[j].mean(), inplace=True)
        logging.info("numerical missing data filled")
        
        # Trimming string in target calumn.
        for i in range(len(df['status'])):
            df['status'][i]=str(df['status'][i])[slice(3)]
            if df['status'][i]=='neg':
                df['status'][i]='neg'
            else:
                df['status'][i]='pos'
        # Dropping unrelevant column which has one unique value.
        dump_col =[]
        for i in [j for j in df.columns]:
            if len(df[i].unique())>=2:
                pass
            else:
                dump_col.append(i)
        logging.info(f"column to drop which have one unique category : {dump_col}")
        df=df.drop(dump_col, axis=1)
        logging.info(f"columns after dropping : {df.columns}")
        logging.info(f"sex unique value : {df['sex'].unique()}")
        return df
    except Exception as e:
        raise thyroidException(e, sys)

def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id ")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and columns in df: {df.shape}")

        df_trans = missing_data_handler(df=df)
        
        return df_trans
    except Exception as e:
        raise thyroidException(e, sys)


def write_yaml_file(file_path,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise thyroidException(e, sys)
    
def convert_columns_float(df:pd.DataFrame,exclude_columns:list)->pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column]=df[column].astype('float')
        return df
    except Exception as e:
        raise e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise thyroidException(e, sys) from e


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise thyroidException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise thyroidException(e, sys) from e

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise thyroidException(e, sys) from e
