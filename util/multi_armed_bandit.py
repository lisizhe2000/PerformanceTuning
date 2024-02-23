from enum import Enum
import math
import random
from typing import Callable
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from data_processing.common import Common
from scipy.optimize import curve_fit

from util.config import Config
from util.time_counter import timeit


class EpsilonGreedy(object):
    def __init__(self):
        self.count = 0
        self.models = [
            DecisionTreeRegressor(),
            KNeighborsRegressor(),
            SVR(),
            LinearRegression()
            ]
        self.best_scores = [float('inf') for _ in range(len(self.models))]
        self.best_model = None
        self.best_model_id = None
        
        # 每一轮的
        self.model_this_turn = None
        self.model_id = None
        
        self.get_epsilon = EpsilonGreedy.calc_epsilon_function()    # 随着迭代次数变小的epsilon
        self.epsilon = 0.2  # 固定的epsilon

    @timeit
    def train(self, configs: list[Config]) -> None:
        Operation = Enum('Operation', ['INIT', 'EXPLORE', 'EXPLOIT'])
        
        op = None
        if self.count < len(self.models):
            op = Operation.INIT
        elif random.random() < self.epsilon:    # explore
        # elif random.random() < self.get_epsilon(self.count):    # explore
            op = Operation.EXPLORE
        else:   # exploit
            op = Operation.EXPLOIT
        
        if op == Operation.INIT:
            self.best_model_id = self.count
            self.best_model = self.models[self.count]
            self.model_this_turn = self.best_model
            self.model_id = self.best_model_id
        elif op == Operation.EXPLORE:    # explore
            self.model_id = random.randint(0, len(self.models) - 1)
            while self.model_id == self.best_model_id:
                self.model_id = random.randint(0, len(self.models) - 1)
            self.model_this_turn = self.models[self.model_id]
        elif op == Operation.EXPLOIT:   # exploit
            self.model_this_turn = self.best_model
            self.model_id = self.best_model_id
        else:
            raise Exception('Unknown operation')
            
        X = self.configs_to_nparray(configs)
        y = np.array([config.get_real_performance() for config in configs])
        self.best_model.fit(X, y)
        self.count += 1

    @timeit
    def acquist_all(self, configs: list[Config]) -> list[float]:
        predicted =  self.model_this_turn.predict(self.configs_to_nparray(configs))
        best_performance = min(predicted)
        rank = self.get_rank(best_performance, to_minimize=True)
        if rank <= self.best_scores[self.model_id]:
            self.best_scores[self.model_id] = rank
            is_best = True
            for i in range(len(self.models)):
                if rank > self.best_scores[i]:
                    is_best = False
                    break
            if is_best:
                self.best_model = self.model_this_turn
                self.best_model_id = self.model_id
        # print(f'best model: {self.best_model_id}, model: {self.model_id}')
        return predicted

    @staticmethod
    def calc_epsilon_function() -> Callable[[int], float]:
        warnings.filterwarnings("ignore")
        def epsilon_function(x, a, b):
            return a * np.exp(-0.1 * x) + b
        total_iteration = Common().total_size - Common().init_size
        x_data = np.array([0, total_iteration])
        y_data = np.array([1, 0])
        params, covariance = curve_fit(epsilon_function, x_data, y_data, maxfev=5000)
        a, b = params
        return lambda x: epsilon_function(x, a, b)
    
    @staticmethod
    def get_rank(performance: float, to_minimize=True) -> int:
        sorted_performances = sorted(Common().all_performances, reverse = not to_minimize)
        for i in range(len(sorted_performances)):
            if math.isclose(performance, sorted_performances[i]):
                return i
        return -1

    @staticmethod
    def configs_to_nparray(configs: list[Config]) -> np.ndarray:
        return np.array([config.config_options for config in configs])
