import sys, os
from predict.exception import PredictException
from predict.util import read_yaml_file
from predict.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, PushModelConfig
from predict.constant import *

"""
        try:
            pass
        except Exception as e:
            raise PredictException(e,sys) from e
"""


class Configuration:
    def __init__(self, 
                 config_file_path:str=CONFIG_FILE_PATH,
                 current_time_stamp:datetime = CURRENT_TIME_STAMP):
        try:
            self.config_file = read_yaml_file(config_file_path)
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.timestamp = current_time_stamp
        except Exception as e:
            raise PredictException(e,sys) from e
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            
            data_ingestion_config_info = self.config_file[DATA_INGESTION_CONFIG_KEY]
            
            data_ingestion_dir =  os.path.join(
                self.training_pipeline_config.artifact_dir,
                DATA_INGESTION_ARTIFACT_DIR,
                self.timestamp
            )
            
            download_url = data_ingestion_config_info[DATA_INGESTION_DOWNLOAD_URL_KEY]
            
            commpressed_data_dir = os.path.join(
                data_ingestion_dir,
                data_ingestion_config_info[DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY],
            )
            raw_data_dir = os.path.join(
                data_ingestion_dir,
                data_ingestion_config_info[DATA_INGESTION_RAW_DATA_DIR_KEY]
            )
            train_dataset_dir = os.path.join(
                data_ingestion_dir,
                data_ingestion_config_info[DATA_INGESTION_TRAIN_DIR_KEY]
            )
            test_dataset_dir = os.path.join(
                data_ingestion_dir,
                data_ingestion_config_info[DATA_INGESTION_TEST_DIR_KEY]
            )
            
            data_ingestion_config = DataIngestionConfig(
                download_url,
                commpressed_data_dir,
                raw_data_dir,
                train_dataset_dir,
                test_dataset_dir
            )
            return data_ingestion_config
        except Exception as e:
            raise PredictException(e,sys) from e
        
        
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            
            data_validation_config_info = self.config_file[DATA_VALIDATION_CONFIG_KEY]
            
            data_validation_folder = os.path.join(
                self.training_pipeline_config.artifact_dir,
                DATA_VALIDATION_ARTIFACT_DIR_NAME,
                self.timestamp
            )
            
            schema_file_path = os.path.join(
                ROOT_DIR,
                data_validation_config_info[DATA_VALIDATION_SCHEMA_DIR_KEY],
                data_validation_config_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]
            )
            report_file_path = os.path.join(
                data_validation_folder,
                data_validation_config_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY]
            )
            report_page_file_path = os.path.join(
                data_validation_folder,
                data_validation_config_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY]
            )
            data_validation_config = DataValidationConfig(
                schema_file_path,
                report_file_path,
                report_page_file_path
            )
            return data_validation_config
        except Exception as e:
            raise PredictException(e,sys) from e
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            
            trans_info = self.config_file[DATA_TRANSFORMATION_CONFIG_KEY]
                
            transformation_config_dir = os.path.join(
                artifact_dir,
                DATA_TRANSFORMATION_ARTIFACT_DIR,
                self.timestamp
            )
                       
            feature_1 = trans_info[DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY]
            transformed_train_dir = os.path.join(
                transformation_config_dir,
                trans_info[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY]
            )
            
            transformed_test_dir = os.path.join(
                transformation_config_dir,
                trans_info[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY]
            )
            
            processed_object_path = os.path.join(
                transformation_config_dir,
                trans_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                trans_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]
            )
            
            data_transformation_config = DataTransformationConfig(
                feature_1,
                transformed_train_dir, 
                transformed_test_dir, 
                processed_object_path
            )
            
            
            return data_transformation_config
        except Exception as e:
            raise PredictException(e,sys) from e
     
    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_file[TRAINING_PIPELINE_CONFIG_KEY]
            return TrainingPipelineConfig(
                artifact_dir = os.path.join(
                    ROOT_DIR,
                    training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                    training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
                )
            )
        except Exception as e:
            raise PredictException(e,sys) from e