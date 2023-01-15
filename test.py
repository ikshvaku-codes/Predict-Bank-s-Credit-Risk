from predict.pipeline.pipeline import Pipeline
import sys
from predict.exception import PredictException
from predict.config.Configuration import Configuration
from predict.logger import logging


def main():
    
    try:
        pipe = Pipeline()
        #comp = DataIngestion(conf.get_data_ingestion_config())
        #logging.info(comp.start_data_ingestion())
        #print_statement = Configuration().get_data_transformation_config()
        print(pipe.run_pipeline())
    except Exception as e:
        raise PredictException(e,sys) from e
        #logging.error(f"{e}")

if __name__ == "__main__":
    main()
