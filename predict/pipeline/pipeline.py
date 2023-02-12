from predict.config.Configuration import Configuration
from predict.logger import logging
from predict.exception import PredictException
from predict.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact
from predict.component.DataIngestion import DataIngestion
from predict.component.DataValidation import DataValidation
from predict.component.DataTranformation import DataTransformation
import sys,  os
from predict.component.ModelTrainer import ModelTrainer
from predict.component.ModelEvaluation import ModelEvaluation
from predict.component.ModelPusher import ModelPusher
from threading import Thread
from predict.entity import model_factory
from collections import namedtuple
from predict.constant import *
import uuid
import pandas as pd


"""
        try:
            pass
        except Exception as e:
                    raise PredictException(e,sys) from e
"""


Experiment = namedtuple("Experiment",[
    "experiment_id",
    "initialization_timestamp", 
    "artifact_time_stamp",
    "running_status", 
    "start_time", 
    "stop_time", 
    "execution_time", 
    "message",
    "experiment_file_path", 
    "accuracy", 
    "is_model_accepted"
])

class Pipeline(Thread):
    
    experiment:Experiment = Experiment(*([None]*11))   
    experiment_file_path = None
    
    
    
    def __init__(self,config: Configuration = Configuration()) -> None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            Pipeline.experiment_file_path = os.path.join(config.training_pipeline_config.artifact_dir, EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            super().__init__(daemon=False,name="pipeline")
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
        
    def start_model_trainer(self,
                            data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        
        try:
            model_trainer = ModelTrainer(self.config.get_model_trainer_config(),
                                            data_transformation_artifact)
            model_trainer_artifact = model_trainer.start_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise PredictException(e,sys) from e
    
    def start_model_evolution(self,
                            data_val_artifact:DataValidationArtifact,
                            data_ingestion_artifact:DataIngestionArtifact,
                            model_train_artifact:ModelTrainerArtifact)->ModelEvaluationArtifact:
        try:
            model_eval = ModelEvaluation(self.config.get_model_eval_config(),
                                         data_ingestion_artifact,
                                         model_train_artifact,
                                         data_val_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        except Exception as e:
            raise PredictException(e,sys) from e
            
     
    def start_model_push(self,
                            model_evaluation_artifact:ModelEvaluationArtifact)->ModelPusherArtifact:
        try:
            model_push = ModelPusher(self.config.get_model_pusher_config(),
                                         model_evaluation_artifact)
            model_push_arti = model_push.initiate_model_pusher()
            return model_push_arti
        except Exception as e:
            raise PredictException(e,sys) from e
            
    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment
            
            logging.info("Pipeline starting...")
            
            experiment_id = str(uuid.uuid4())
            
            Pipeline.experiment = Experiment(
                experiment_id,
                self.config.timestamp,
                self.config.timestamp,
                True,
                datetime.now(),
                None,
                None,
                "Pipeline is Running",
                Pipeline.experiment_file_path,
                None,
                None
            )
            
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")

            self.save_experiment()

            #Training Begins Here
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_tranformation(data_ingestion_artifact,
                                                                          data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evolution(data_validation_artifact,
                                                                  data_ingestion_artifact,
                                                                  model_trainer_artifact)
            if model_evaluation_artifact.is_model_accepted:
                model_push_artifact = self.start_model_push(model_evaluation_artifact)
                logging.info(f"Model Pusher Artifact:{model_push_artifact}")
            else:
                logging.info(f"Trained Model is rejected")
            
            logging.info("Pipeline completed.")

            stop_time = datetime.now()  
            
            
            Pipeline.experiment = Experiment(
                Pipeline.experiment.experiment_id,
                Pipeline.experiment.initialization_timestamp,
                Pipeline.experiment.artifact_time_stamp,
                False,
                Pipeline.experiment.start_time,
                stop_time,
                stop_time - Pipeline.experiment.start_time,
                "Pipeline executed successfully",
                Pipeline.experiment_file_path,
                model_trainer_artifact.model_accuracy,
                model_evaluation_artifact.is_model_accepted
                
            )
            
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
            
        except Exception as e:
                    raise PredictException(e,sys) from e
                
    
    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                experiment_dict = experiment._asdict()
                experiment_dict: dict = {key: [value] for key, value in experiment_dict.items()}
                
                experiment_dict.update({
                    "created_time_stamp": [datetime.now()],
                    "experiment_file_path": [os.path.basename(Pipeline.experiment.experiment_file_path)]
                })
                
                experiment_report = pd.DataFrame(experiment_dict)
                
                os.makedirs(os.path.dirname(Pipeline.experiment_file_path), exist_ok=True)

                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_report.to_csv(Pipeline.experiment_file_path, index=False, header=False, mode="a")
                else:
                    experiment_report.to_csv(Pipeline.experiment_file_path, index=False, header=True, mode="w")

            else:
                print("First start experiment...")
        except Exception as e:
            raise PredictException(e,sys) from e
        
    
    
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise e
