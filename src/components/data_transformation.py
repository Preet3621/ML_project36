import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import customException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preproccesor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = ['gender', 'race_ethnicity', 
                            'parental_level_of_education', 'lunch',
                           'test_preparation_course']
            num_pipeline = Pipeline(
                steps = [('imputer',SimpleImputer(strategy = 'median')),
                         ('scaler',StandardScaler())]
            )
            cat_pipeline = Pipeline(
                steps = [('imputer',SimpleImputer(strategy='most_frequent')),
                           ('one_hot_encoder',(OneHotEncoder())),
                           ('scaler',(StandardScaler(with_mean = False)))]
            )
            logging.info('numerical coloumn standard scaling completed')
            logging.info('categorical columns encoding completed')

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_features),
                ('cat_pipeline',cat_pipeline,cat_features)
            ])
            return preprocessor
        except Exception as e:
            raise customException(e,sys)
    
    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Obtaining Preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()
            target_column = 'math_score'
            num_features = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns = [target_column],axis = 1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns = [target_column],axis = 1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing object on training and testing dataframes')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj)
                     
            return (
                train_arr,test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise customException(e,sys)


