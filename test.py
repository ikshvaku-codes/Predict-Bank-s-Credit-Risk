from predict.pipeline.pipeline import Pipeline
import sys, os
from predict.exception import PredictException
from predict.config.Configuration import Configuration
from predict.logger import logging
from predict.entity import model_factory
import numpy as np

def main():
    
    try:
        
        #comp = DataIngestion(conf.get_data_ingestion_config())
        #logging.info(comp.start_data_ingestion())
        #print_statement = Configuration().get_data_transformation_config()
        # model_trainer_config = Configuration().get_model_trainer_config()
        
        pipe = Pipeline()
        # trainData = np.load(r"G:\DeadLine Dec 2022\DSML\ineuron\MLEnd2EndDemo\Predict-Bank-s-Credit-Risk\predict\artifact\data_transformation\2023-01-15-19-09-55\train\SouthGermanCredit.npz")
        # X = trainData[:,:-1]
        # y = trainData[:,-1]
        # ini_mdl_lst = mf.get_initialized_model_list()
        # grid_list = []
        # for model in ini_mdl_lst:
        #     grid_list.append(mf.execute_grid_search_operation(model,X,y))
        # output = ""
        # for grid in zip(grid_list):
        #     for var in grid._fields:
        #         output+= f"{var} = {grid[var]}\n"
        print(pipe.start())
    except Exception as e:
        raise PredictException(e,sys) from e
        #logging.error(f"{e}")

if __name__ == "__main__":
    main()
