import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

from util.config import Config
from typing import Callable

from util.time_counter import timeit


# Meachine Learning Util
class MLUtil(object):


    __model = None
    model_name = None
    f_predict: Callable[[Config], float] = None
    f_precict_all: Callable[[list[Config]], np.ndarray] = None
    f_train: Callable[[list[Config]], None] = None  # Train the model above
    f_acquisition: Callable[[Config], float] = None
    f_acquist_all: Callable[[list[Config]], np.ndarray] = None
    acquisition_function_name = None


    @staticmethod
    def using_xgboost() -> None:
        MLUtil.param = {
            'max_depth': 2,
            'eta': 1,
            'objective': 'reg:squarederror'
        }
        MLUtil.__model: xgb.Booster = None
        MLUtil.model_name = 'xgboost'

        @timeit
        def train(configs: list[Config]) -> None:
            X = MLUtil.__configs_to_nparray(configs)
            y = np.array([config.get_real_performance() for config in configs])
            dtrain = xgb.DMatrix(X, label=y)
            num_round = 10
            MLUtil.__model = xgb.train(MLUtil.param, dtrain, num_round)

        MLUtil.f_train = train
        MLUtil.f_predict = lambda config: MLUtil.__model.predict(xgb.DMatrix(np.array([config.config_options])))[0]
        MLUtil.f_precict_all = lambda configs: MLUtil.__model.predict(xgb.DMatrix(MLUtil.__configs_to_nparray(configs)))
        MLUtil.f_acquisition = MLUtil.f_predict
        MLUtil.f_acquist_all = MLUtil.f_precict_all
        MLUtil.acquisition_function_name = 'predicted_val'


    @staticmethod
    def using_cart() -> None:
        MLUtil.__model = DecisionTreeRegressor()
        MLUtil.model_name = 'CART'

        MLUtil.f_train = MLUtil.__train_sklearn_model
        MLUtil.f_predict = lambda config: MLUtil.__model.predict(np.array([config.config_options]))
        MLUtil.f_precict_all = lambda configs: MLUtil.__model.predict(MLUtil.__configs_to_nparray(configs))
        MLUtil.f_acquisition = MLUtil.f_predict
        MLUtil.f_acquist_all = MLUtil.f_precict_all
        MLUtil.acquisition_function_name = 'predicted_val'


    @staticmethod
    def using_random_forest() -> None:
        MLUtil.__model = RandomForestRegressor()
        MLUtil.model_name = 'RandomForest'

        MLUtil.f_train = MLUtil.__train_sklearn_model
        MLUtil.f_predict = lambda config: MLUtil.__model.predict(np.array([config.config_options]))
        MLUtil.f_precict_all = lambda configs: MLUtil.__model.predict(MLUtil.__configs_to_nparray(configs))
        MLUtil.f_acquisition = MLUtil.f_predict
        MLUtil.f_acquist_all = MLUtil.f_precict_all
        MLUtil.acquisition_function_name = 'mean_predicted_of_decision_trees'


    @staticmethod
    def using_random_forest_max_val() -> None:
        MLUtil.using_random_forest()

        def acquisition_function(config: Config) -> float:
            max_val = float('-inf')
            for dt in MLUtil.__model.estimators_:
                predicted_val = dt.predict(np.array(config.config_options))
                if predicted_val > max_val:
                    max_val = predicted_val
            return max_val

        # 速度要慢几十倍，效果也差不多
        def acquist_all(configs: list[Config]) -> np.ndarray:
            num_trees = len(MLUtil.__model.estimators_)
            predicted_mat = np.empty((num_trees, len(configs)), dtype=bool)     # TODO: 数据类型需要拓展到数值型
            for i in range(num_trees):
                predicted_mat[i] = MLUtil.__model.estimators_[i].predict(MLUtil.__configs_to_nparray(configs))
            return predicted_mat.max(axis=0)
    
        MLUtil.f_acquisition = acquisition_function
        MLUtil.f_acquist_all = acquist_all    
        MLUtil.acquisition_function_name = 'max_predicted_of_decision_trees'


    @timeit
    def __train_sklearn_model(configs: list[Config]) -> None:
        X = MLUtil.__configs_to_nparray(configs)
        y = np.array([config.get_real_performance() for config in configs])
        MLUtil.__model.fit(X, y)


    @staticmethod
    def __configs_to_nparray(configs: list[Config]) -> np.ndarray:
        return np.array([config.config_options for config in configs])
    