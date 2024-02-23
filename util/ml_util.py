from multiprocessing import Pool
from statistics import LinearRegression
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

from util.config import Config
from typing import Callable, Dict
from util.multi_armed_bandit import EpsilonGreedy

from util.time_counter import timeit


# Machine Learning Util
class MLUtil(object):


    __model = None
    model_name = None
    f_predict: Callable[[Config], float] = None
    f_precict_all: Callable[[list[Config]], np.ndarray] = None
    f_train: Callable[[list[Config]], None] = None  # Train the model above
    f_acquisition: Callable[[Config], float] = None
    f_acquist_all: Callable[[list[Config]], np.ndarray] = None
    acquisition_function_name = None

    # multi model
    __n_carts = 4
    __alpha = 2.0


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
            X = MLUtil.configs_to_nparray(configs)
            y = np.array([config.get_real_performance() for config in configs])
            dtrain = xgb.DMatrix(X, label=y)
            num_round = 10
            MLUtil.__model = xgb.train(MLUtil.param, dtrain, num_round)

        MLUtil.f_train = train
        MLUtil.f_predict = lambda config: MLUtil.__model.predict(xgb.DMatrix(np.array([config.config_options])))[0]
        MLUtil.f_precict_all = lambda configs: MLUtil.__model.predict(xgb.DMatrix(MLUtil.configs_to_nparray(configs)))
        MLUtil.f_acquisition = MLUtil.f_predict
        MLUtil.f_acquist_all = MLUtil.f_precict_all
        MLUtil.acquisition_function_name = 'predicted_val'


    @staticmethod
    def using_cart() -> None:
        MLUtil.__model = DecisionTreeRegressor()
        MLUtil.model_name = 'CART'

        MLUtil.f_train = MLUtil.__train_sklearn_model
        MLUtil.f_predict = lambda config: MLUtil.__model.predict(np.array([config.config_options]))[0]
        MLUtil.f_precict_all = lambda configs: MLUtil.__model.predict(MLUtil.configs_to_nparray(configs))
        MLUtil.f_acquisition = MLUtil.f_predict
        MLUtil.f_acquist_all = MLUtil.f_precict_all
        MLUtil.acquisition_function_name = 'predicted_val'
        
        
    @staticmethod
    def using_sklearn_model(model) -> None:
        MLUtil.__model = model
        MLUtil.model_name = model.__class__.__name__
        
        MLUtil.f_train = MLUtil.__train_sklearn_model
        MLUtil.f_predict = lambda config: MLUtil.__model.predict(np.array([config.config_options]))[0]
        MLUtil.f_precict_all = lambda configs: MLUtil.__model.predict(MLUtil.configs_to_nparray(configs))
        MLUtil.f_acquisition = MLUtil.f_predict
        MLUtil.f_acquist_all = MLUtil.f_precict_all
        MLUtil.acquisition_function_name = 'predicted_val'


    @staticmethod
    def using_epsilon_greedy() -> None:
        MLUtil.__bandit = EpsilonGreedy()
        MLUtil.model_name = 'epsilon_greedy'
        MLUtil.f_train = MLUtil.__bandit.train
        MLUtil.f_acquist_all = MLUtil.__bandit.acquist_all
        MLUtil.acquisition_function_name = 'predicted_val'


    @staticmethod
    def using_random_forest() -> None:
        MLUtil.__model = RandomForestRegressor(n_estimators=MLUtil.__n_carts)
        MLUtil.model_name = 'RandomForest'

        MLUtil.f_train = MLUtil.__train_sklearn_model
        MLUtil.f_predict = lambda config: MLUtil.__model.predict(np.array([config.config_options]))[0]
        MLUtil.f_precict_all = lambda configs: MLUtil.__model.predict(MLUtil.configs_to_nparray(configs))
        MLUtil.f_acquisition = MLUtil.f_predict
        MLUtil.f_acquist_all = MLUtil.f_precict_all
        MLUtil.acquisition_function_name = 'mean_predicted_of_decision_trees'


    @staticmethod
    def using_random_forest_max_val() -> None:
        MLUtil.using_random_forest()

        def acquisition_function(config: Config) -> float:
            max_val = float('-inf')
            pool = Pool()
            res = []
            for dt in MLUtil.__model.estimators_:
                res.append(pool.apply_async(dt.predict, [np.array(config.config_options)]))
            pool.close()
            pool.join()
            for predicted_val in res:
                predicted_val = predicted_val.get()
                if predicted_val > max_val:
                    max_val = predicted_val
            return max_val

        # 速度要慢几十倍，效果也差不多
        def acquist_all(configs: list[Config]) -> np.ndarray:
            num_trees = len(MLUtil.__model.estimators_)
            predicted_mat = np.empty((num_trees, len(configs)))
            pool = Pool()
            res = []
            for i in range(num_trees):
                res.append(pool.apply_async(MLUtil.__model.estimators_[i].predict, [MLUtil.configs_to_nparray(configs)]))
            pool.close()
            pool.join()
            for i in range(num_trees):
                predicted_mat[i] = res[i].get()
            return predicted_mat.max(axis=0)
    
        MLUtil.f_acquisition = acquisition_function
        MLUtil.f_acquist_all = acquist_all    
        MLUtil.acquisition_function_name = 'max_predicted_of_decision_trees'


    @staticmethod
    def using_random_forest_ucb() -> None:
        MLUtil.using_random_forest()

        def acquist_all(configs: list[Config]) -> np.ndarray:
            predicted_mat = np.empty((MLUtil.__n_carts, len(configs)))
            for i in range(MLUtil.__n_carts):
                predicted_mat[i] = MLUtil.__model.estimators_[i].predict(MLUtil.configs_to_nparray(configs))
            return predicted_mat.mean(axis=0) + MLUtil.__alpha * predicted_mat.std(axis=0)
        
        MLUtil.f_acquist_all = acquist_all
        MLUtil.acquisition_function_name = 'random_forest_UCB'


    @staticmethod
    # FIXME: NotFittedError
    def using_n_carts_ucb() -> None:
        MLUtil.model: list[DecisionTreeRegressor] = [DecisionTreeRegressor() for _ in range(MLUtil.__n_carts)]
        MLUtil.model_name = 'n_carts_ucb'

        def train(configs: list[Config]) -> None:
            X = MLUtil.configs_to_nparray(configs)
            y = np.array([config.get_real_performance() for config in configs])
            # pool = Pool()
            # for i in range(n_carts):
            #     pool.apply_async(MLUtil.model[i].fit, [X, y])
            # pool.close()
            # pool.join()
            for dt in MLUtil.model:
                dt.fit(X, y)
            

        def acquist_all(configs: list[Config]) -> np.ndarray:
            predicted_mat = np.empty((MLUtil.__n_carts, len(configs)))
            # pool = Pool()
            # res = []
            # for i in range(n_carts):
            #     res.append(pool.apply_async(MLUtil.model[i].predict, [MLUtil.configs_to_nparray(configs)]))
            # pool.close()
            # pool.join()
            # for i in range(n_carts):
            #     predicted_mat[i] = res[i].get()
            for i in range(MLUtil.__n_carts):
                predicted_mat[i] = MLUtil.model[i].predict(MLUtil.configs_to_nparray(configs))
            # print(f'mean: {predicted_mat.mean(axis=0)}, std: {predicted_mat.std(axis=0)}')
            return predicted_mat.mean(axis=0) + MLUtil.__alpha * predicted_mat.std(axis=0)
        
        MLUtil.f_train = train
        MLUtil.f_acquist_all = acquist_all
        MLUtil.acquisition_function_name = 'n_carts_UCB'


    @timeit
    def __train_sklearn_model(configs: list[Config]) -> None:
        X = MLUtil.configs_to_nparray(configs)
        y = np.array([config.get_real_performance() for config in configs])
        MLUtil.__model.fit(X, y)


    kmeans_n_clusters = 8
    config_clazz: Dict[Config, int] = None
    @staticmethod
    def get_kmeans_clazz(configs: list[Config]) -> Dict[Config, int]:
        # if config_clazz is not None:
            # return config_clazz
        warnings.filterwarnings("ignore")
        X = MLUtil.configs_to_nparray(configs)
        kmeans = KMeans(n_clusters=MLUtil.kmeans_n_clusters).fit(X)
        config_clazz = {}
        for i in range(len(configs)):
            config_clazz[configs[i]] = kmeans.labels_[i]
        return config_clazz


    @staticmethod
    def configs_to_nparray(configs: list[Config]) -> np.ndarray:
        return np.array([config.config_options for config in configs])
    