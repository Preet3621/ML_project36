import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import customException
from dataclasses import dataclass

@dataclass

class DataIngetionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str =os.path.join('artifacts','data.csv')

class DataIngetion:
    def __init__(self):
        self.ingetion_config = DataIngetionConfig()
        
    def initiate_data_ingetion(self):
        logging.info('Entering the dataingetion method or component')
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingetion_config.train_data_path),exist_ok = True)
            df.to_csv(self.ingetion_config.raw_data_path,index = False, header = True)

            logging.info('train test split initiated')
            train_set, test_set = train_test_split(df,test_size = 0.2,random_state =42)
            train_set.to_csv(self.ingetion_config.train_data_path,index = False, header = True)
            test_set.to_csv(self.ingetion_config.test_data_path, index = False, header = True)
            logging.info('ingetion of data has been completed')
            return (self.ingetion_config.train_data_path,self.ingetion_config.test_data_path)
        
        except Exception as e:
            raise customException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngetion()
    obj.initiate_data_ingetion()