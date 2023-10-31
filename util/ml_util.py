import numpy as np
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

from util.config import Config
from typing import Callable


# Meachine Learning Util
class MLUtil(object):


    model = None
    f_predict: Callable[[Config], float] = None
    f_precict_all: Callable[[list[Config]], np.ndarray] = None
    f_train: Callable[[list[Config]], None] = None  # Train the model above
    f_acquisition: Callable[[Config], float] = None


    @staticmethod
    def using_xgboost() -> None:
        MLUtil.param = {
            'max_depth': 2,
            'eta': 1,
            'objective': 'reg:squarederror'
        }
        MLUtil.model: xgb.Booster = None

        def train(configs: list[Config]) -> None:
            X = np.array([config.config_options for config in configs])
            y = np.array([config.get_real_performance() for config in configs])
            dtrain = xgb.DMatrix(X, label=y)
            num_round = 10
            MLUtil.model = xgb.train(MLUtil.param, dtrain, num_round)

        MLUtil.f_train = train
        MLUtil.f_predict = lambda config: MLUtil.model.predict(xgb.DMatrix(np.array([config.config_options])))[0]
        MLUtil.f_precict_all = lambda configs: MLUtil.model.predict(xgb.DMatrix(np.array([config.config_options for config in configs])))
        MLUtil.f_acquisition = MLUtil.f_predict


    @staticmethod
    def using_cart() -> None:
        MLUtil.model = DecisionTreeRegressor()

        def train(configs: list[Config]) -> None:
            X = np.array([config.config_options for config in configs])
            y = np.array([config.get_real_performance() for config in configs])
            MLUtil.model.fit(X, y)

        MLUtil.f_train = train
        MLUtil.f_predict = lambda config: MLUtil.model.predict(np.array([config.config_options]))
        MLUtil.f_precict_all = lambda configs: MLUtil.model.predict(np.array([config.config_options for config in configs]))
        MLUtil.f_acquisition = MLUtil.f_predict
