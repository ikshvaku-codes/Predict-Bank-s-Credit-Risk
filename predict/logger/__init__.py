import logging, os
from datetime import datetime

LOG_DIR = "predict_logs"

CURRENT_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(LOG_DIR, exist_ok=True)

FILE_NAME = f'log_{CURRENT_TIMESTAMP}.log'

FILE_PATH = os.path.join(LOG_DIR, FILE_NAME)

logging.basicConfig(level=logging.INFO, 
                    filename=FILE_PATH,
                    filemode='w',
                    format="""[%(asctime)s] %(name)s 
                                            - %(levelname)s 
                                            - %(message)s"""
                    )