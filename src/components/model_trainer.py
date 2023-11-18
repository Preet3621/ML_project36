# Basic Import
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
from dataclasses import dataclass
# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
from src.exception import customException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('splitting training and test input data') 
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours classifier": KNeighborsRegressor(),
                "Xgbclassifier": XGBRegressor(),
                "catboosting classifier": CatBoostRegressor(verbose = False),
                "Adaboost classifier": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            # to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dictionary
           # best_model_name = list(model_report.keys())[list(model_report.values().index(best_model_score))]
            best_model_name = [key for key, value in model_report.items() if value == best_model_score][0]


            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise customException("no best model found")
            logging.info(f'best found model on traing and testing dataset')

            save_object(file_path = self.model_trainer_config.trained_model_file_path,
                        obj = best_model)
            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
           raise customException(e,sys)


