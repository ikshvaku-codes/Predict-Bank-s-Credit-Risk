import os, sys
import pandas as pd
import numpy as np
from predict.util import read_yaml_file, load_data, save_object, save_numpy_array_data
from predict.entity.config_entity import DataTransformationConfig
from predict.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from predict.exception import PredictException
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from predict.logger import logging

"""
        try:
            pass
        except Exception as e:
            raise PredictException(e,sys) from e
"""


class DataTransformation():
    def __init__(self,
                 data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact,
                 data_ingestion_aritfact:DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.data_ingestion_aritfact = data_ingestion_aritfact
        except Exception as e:
            raise PredictException(e,sys) from e
                
    def get_data_transformation_object(self)->ColumnTransformer:
        try:
            dataset = pd.read_csv(self.data_ingestion_aritfact.train_file_path)
            schema = read_yaml_file(self.data_validation_artifact.schema_file_path)
            target_column = schema["target_columns"]
            dataset = dataset.drop([target_column], axis=1)
            columns = dataset.columns
            descreet_columns = [col for col in columns if (len(dataset[col].unique()) <= 5 and (col is not target_column))]
            continous_columns = [col for col in columns if (len(dataset[col].unique()) > 5 and (col is not target_column))]
            des_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('scaler', StandardScaler(with_mean=False))
            ])

            con_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ])
            preprocessing = ColumnTransformer([
                ('des_pipeline', des_pipeline, descreet_columns),
                ('cat_pipeline', con_pipeline, continous_columns),
            ])
            logging.info(f"preprocessing object is ready to be processed: {preprocessing}")
            
            return preprocessing
        except Exception as e:
            raise PredictException(e,sys) from e
        
    def start_data_transformations(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_object()
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_aritfact.train_file_path
            test_file_path = self.data_ingestion_aritfact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)
            
            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema["target_columns"]
            
            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            print(input_feature_train_df.head())
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            


            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.processed_object_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)           
            is_transformed = True
            message = "Saving preprocessing object"
            data_transformation_artifact = DataTransformationArtifact(
                is_transformed,
                message,
                transformed_test_file_path,
                transformed_test_file_path,
                preprocessing_obj_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            raise PredictException(e,sys) from e
    
    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")