import sys, os
from predict.entity.config_entity import ModelEvaluationConfig
from predict.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, DataValidationArtifact,ModelEvaluationArtifact
from predict.exception import PredictException
from predict.logger import logging
from predict.util import write_yaml_file, read_yaml_file, load_object, load_data
from predict.constant import *
from predict.entity.model_factory import evalute_classification_model
import numpy as np

class ModelEvaluation:
    
    def __init__(self, model_eval_config:ModelEvaluationConfig,
                 data_inj_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_val_artifact: DataValidationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")

            self.model_evaluation_config = model_eval_config
            self.data_ingestion_artifact = data_inj_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_validation_artifact = data_val_artifact
        except Exception as e:
            raise PredictException(e,sys) from e
    
    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path
            
            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(model_evaluation_file_path)
                return model
            
            model_eval_file_content = read_yaml_file(model_evaluation_file_path)
            
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content
            
            if BEST_MODEL_KEY  not in model_eval_file_content:
                return model
            
            model = load_object(model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise PredictException(e,sys) from e
        
    def update_evolution_report(self,
                                model_eval_artifact:ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            
            model_eval_content = read_yaml_file(eval_file_path)
            
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            previous_best_model = None
            
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]
                
            logging.info(f"Previous eval result: {model_eval_content}")
            
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_eval_artifact.evaluated_model_path,
                }
            }
            
            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_artifact:
                    history = {HISTORY_KEY:model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)
                    
            model_eval_content.update(eval_result)
            
            logging.info(f"Updated Eval Result: {model_eval_content}")
            
            write_yaml_file(eval_file_path, model_eval_content)
            
        except Exception as e:
            raise PredictException(e,sys) from e
        
        
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            
            trainned_model_object = load_object(trained_model_file_path)
            
            train_file_path = self.data_ingestion_artifact.train_file_path
            
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            schema_file_path = self.data_validation_artifact.schema_file_path
            
            train_dataframe = load_data(train_file_path,
                                        schema_file_path)
            
            test_dataframe = load_data(test_file_path,
                                       schema_file_path)
            
            schema_content = read_yaml_file(schema_file_path)
            
            target_column_name = schema_content[TARGET_COLUMN_KEY]
            
            # target_column
            logging.info(f"Converting target column into numpy array.")
            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")
            
            model = self.get_best_model()
            
            if model is None:
                """
                    If model is None, we will use the currently trained model as model.
                    and Update it in the file using update_evaluation_report function
                
                """
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evolution_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact
            
            """
            model:- Existing Best model
            trainned_model_object:- Best model of current pipeline
            """
            model_list = [model, trainned_model_object]
            
            metric_info_artifact = evalute_classification_model(
                model_list,
                train_dataframe,
                train_target_arr,
                test_dataframe,
                test_target_arr,
                self.model_trainer_artifact.model_accuracy
            )
            
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response
            
            if metric_info_artifact.index_number == 1:
                    model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                        is_model_accepted=True)
                    self.update_evaluation_report(model_evaluation_artifact)
                    logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact    
          
        except Exception as e:
            raise PredictException(e,sys) from e
    
    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")