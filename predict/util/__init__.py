import yaml,sys,os
from exception import PredictException

def read_yaml(filename:str)->dict:
    try:
        with open(filename, 'r') as stream:
            return yaml.safe_load(stream)
    except Exception as e:
        raise PredictException(e,sys) from e