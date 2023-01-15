import os, sys, zipfile
from predict.exception import PredictException
from predict.entity.config_entity import DataIngestionConfig
from predict.entity.artifact_entity import DataIngestionArtifact
from predict.logger import logging
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from six.moves import urllib


"""
        try:
            pass
        except Exception as e:
            raise PredictException(e,sys) from e
"""

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig) -> None:
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise PredictException(e,sys) from e
        
        
    def download_dataset(self) -> str:
        try:
            download_url = self.data_ingestion_config.download_url
            
            download_folder_location = self.data_ingestion_config.commpressed_data_dir
            
            os.makedirs(download_folder_location, exist_ok=True)
            
            file_name = os.path.basename(download_url)
            
            compressed_file_path = os.path.join(download_folder_location, file_name)
            
            logging.info(f"Downloading Compressed File begin from {download_url} to {compressed_file_path}...")
            
            urllib.request.urlretrieve(url = download_url, 
                                       filename = compressed_file_path)
            
            logging.info(f"Downloading Compressed File completed from {download_url} to {compressed_file_path} successfully.")

            return compressed_file_path
        except Exception as e:
            raise PredictException(e,sys) from e
        
    def extract_dataset(self, downloaded_dataset_path:str):
        try:
            raw_data_folder = self.data_ingestion_config.raw_data_dir
            
            if os.path.exists(raw_data_folder):
                os.remove(raw_data_folder)
            
            os.makedirs(raw_data_folder, exist_ok=True)
            
            logging.info(f"Extrating compressed dataset from {downloaded_dataset_path} to {raw_data_folder}.")
            
            with zipfile.ZipFile(downloaded_dataset_path, 'r') as zip_ref:
                zip_ref.extractall(raw_data_folder)
            
            logging.info(f"Extrated compressed dataset from {downloaded_dataset_path} to {raw_data_folder} successfully.")

            
        except Exception as e:
            raise PredictException(e,sys) from e
    
    def train_test_split(self)->DataIngestionArtifact:
        try:
            raw_data_folder = self.data_ingestion_config.raw_data_dir
            
            raw_file_names = os.listdir(raw_data_folder)
            
            raw_file_name = None
            
            if len(raw_file_names) > 1:
                for file_name in raw_file_names:
                    if file_name.endswith(".csv") or file_name.endswith(".asc"):
                        raw_file_name = file_name
                        break
            
            raw_file_path = os.path.join(raw_data_folder,raw_file_name)
            
            if os.path.isdir(raw_file_path):
                raw_data_folder = os.path.join(raw_data_folder,raw_file_name)

                
            
            logging.info(f"Raw data read from {raw_file_path}")
            raw_dataframe = pd.read_table(raw_file_path, sep=" ")
            strat_train_set = None
            strat_test_set = None
            
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            
            for train_index, test_index in split.split(raw_dataframe, raw_dataframe["kredit"]):
                strat_train_set = raw_dataframe.loc[train_index]
                strat_test_set = raw_dataframe.loc[test_index]           
            
            train_file_path = os.path.join(
                self.data_ingestion_config.train_dataset_dir,
                raw_file_name.replace(".asc",".csv")
            )
            
            test_file_path = os.path.join(
                self.data_ingestion_config.test_dataset_dir,
                raw_file_name.replace(".asc",".csv")
            )
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.train_dataset_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index=False)
                
            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.test_dataset_dir,exist_ok=True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index=False)
                
            data_ingested_artifact = DataIngestionArtifact(
                train_file_path,
                test_file_path,
                True,
                "Data Ingestion Completed Successfully"
            )
            logging.info("Data Ingestion Completed Success")
            return data_ingested_artifact
        except Exception as e:
            raise PredictException(e,sys) from e
    
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            downloaded_dataset_path = self.download_dataset()
            self.extract_dataset(downloaded_dataset_path)
            return self.train_test_split()
        except Exception as e:
            raise PredictException(e,sys) from e