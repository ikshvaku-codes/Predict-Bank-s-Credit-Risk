import os, sys, json
from predict.exception import PredictException
from predict.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from predict.entity.config_entity import DataValidationConfig
from predict.logger import logging
from predict.util import read_yaml_file
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

"""
        try:
            pass
        except Exception as e:
            raise PredictException(e,sys) from e
"""

class DataValidation:
    def __init__(self,
                 data_validation_config:DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise PredictException(e,sys) from e
        
        
    def get_train_test_df(self):
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            return train_df, test_df
        except Exception as e:
            raise PredictException(e,sys) from e

                
    def is_train_test_file_exists(self)->bool:
        try:
            is_train_file_exists = False
            is_test_file_exists = False
            
            training_path = self.data_ingestion_artifact.train_file_path
            test_path = self.data_ingestion_artifact.test_file_path
            
            is_train_file_exists = os.path.exists(training_path)
            is_test_file_exists = os.path.exists(test_path)
            
            logging.info(f'Is train and test files exists: {is_train_file_exists and is_test_file_exists}' )
            
            return is_train_file_exists and is_test_file_exists
        except Exception as e:
            raise PredictException(e,sys) from e

    def validate_dataset_schema(self)->bool:
        try:
            validation_status = True
            
            schema_file_path = self.data_validation_config.schema_file_path
            
            schema_info = read_yaml_file(schema_file_path)
            
            columns = schema_info["columns"]
            
            
            
            target_col = schema_info["target_columns"]
            
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            train_cols = train_df.columns
            
            test_cols = test_df.columns
            
            
            for key, value in columns.items():
                if key.strip() not in train_cols:
                    validation_status = False
                elif not train_df[key].dtypes == value.strip():
                    validation_status = False
                if validation_status == False:
                    break
            
            for key, value in columns.items():
                if key is target_col.strip():
                    continue
                if key.strip() not in test_cols:
                    validation_status = False
                elif not test_df[key].dtypes == value.strip():
                    validation_status = False
                if validation_status == False:
                    break
            
            return validation_status
        except Exception as e:
            raise PredictException(e,sys) from e

    def save_drift_report(self) :
        try:
            profile = Profile(sections = [DataDriftProfileSection()])
            
            train_df, test_df = self.get_train_test_df()
            
            profile.calculate(train_df, test_df)
            
            profile.json()
            
            report = json.loads(profile.json())
            
            report_file_dir = os.path.dirname(self.data_validation_config.report_file_path)
            
            os.makedirs(report_file_dir, exist_ok=True)
            
            with open(self.data_validation_config.report_file_path,"w") as report_file:
                json.dump(report,report_file, indent=6)
                
            return report
            
        except Exception as e:
            raise PredictException(e,sys) from e


    def save_drift_report_report_page(self) :
        try:
            dashboard = Dashboard(tabs = [DataDriftTab()])
            train_df, test_df = self.get_train_test_df()
            dashboard.calculate(train_df,test_df)
            dashboard.save(self.data_validation_config.report_page_file_path)
            
        except Exception as e:
            raise PredictException(e,sys) from e


    def is_data_drift_found(self) -> bool:
        try:
            report = self.save_drift_report()
            self.save_drift_report_report_page()
            data_drift = report["data_drift"]
            data = data_drift["data"]
            metric = data["metrics"]
            data_drift_metric = metric["dataset_drift"]
            
            
            return data_drift_metric
        except Exception as e:
            raise PredictException(e,sys) from e


    def start_data_validation(self)->DataValidationArtifact:
        try:
            
            message_ = None
            valid_success = True
            
            is_available = self.is_train_test_file_exists()
            if not is_available:
                message_ = f"Training file: {self.data_ingestion_artifact.train_file_path} or Test file: {self.data_ingestion_artifact.test_file_path} are not present"
                logging.error(message_)
                valid_success = False
                
            else:
                is_valid = self.validate_dataset_schema()  
                if not is_valid:
                    message_ = f"Training file: {self.data_ingestion_artifact.train_file_path} or Test file: {self.data_ingestion_artifact.test_file_path} are not matching with given Schema: {self.data_validation_config.schema_file_path}"
                    logging.error(message_)
                    valid_success = False
                    
                else:
                    drift_present = self.is_data_drift_found()
                    if drift_present:
                        message_ = f"Training file: {self.data_ingestion_artifact.train_file_path} or Test file: {self.data_ingestion_artifact.test_file_path} are having drift, drift reports are saved at: {self.data_validation_config.report_file_path} and {self.data_validation_config.report_page_file_path}"
                        logging.error(message_)
                        valid_success = False
                    else:
                        message_ = f"Training file: {self.data_ingestion_artifact.train_file_path} or Test file: {self.data_ingestion_artifact.test_file_path} are successfully validated"
                        
            
            data_val_atifact = DataValidationArtifact(
                schema_file_path = self.data_validation_config.schema_file_path,
                report_file_path = self.data_validation_config.report_file_path,
                report_page_file_path = self.data_validation_config.report_page_file_path,
                is_validated=valid_success,
                message=message_
            )
            logging.info(f"Data validation Artifact: {data_val_atifact}")  
            return data_val_atifact
        except Exception as e:
            raise PredictException(e,sys) from e
