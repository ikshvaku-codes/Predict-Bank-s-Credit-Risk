from predict.config.Configuration import Configuration
from predict.logger import logging
from predict.exception import PredictException
from predict.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from predict.component.DataIngestion import DataIngestion
from predict.component.DataValidation import DataValidation
from predict.component.DataTranformation import DataTransformation
import sys,  os

"""
        try:
            pass
        except Exception as e:
                    raise PredictException(e,sys) from e
"""


class Pipeline:
    def __init__(self,config: Configuration = Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise PredictException(e,sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(self.config.get_data_ingestion_config())
            data_ingestion_artifact = data_ingestion.start_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
                    raise PredictException(e,sys) from e
                
    def start_data_validation(self,
                               data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_val = DataValidation(self.config.get_data_validation_config(),
                                      data_ingestion_artifact)
            data_val_artifact =data_val.start_data_validation()
            return data_val_artifact
        except Exception as e:
                    raise PredictException(e,sys) from e
                
    def start_data_tranformation(self,
                                 data_ingestion_artifact:DataIngestionArtifact,
                                 data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        
        try:
            data_trans = DataTransformation(self.config.get_data_transformation_config(),
                                            data_validation_artifact,
                                            data_ingestion_artifact)
            data_trans_artifact = data_trans.start_data_transformations()
            return data_trans_artifact
        except Exception as e:
            raise PredictException(e,sys) from e
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_tranformation(data_ingestion_artifact,
                                                                          data_validation_artifact)
            print(data_transformation_artifact)
        except Exception as e:
                    raise PredictException(e,sys) from e
