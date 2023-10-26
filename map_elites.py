from functools import reduce
import numpy as np
import xgboost as xgb
from config import Config

from common import Common
from config_util import ConfigUtil


class MapElites(object):


    def __init__(self, model: xgb.Booster) -> None:
        self.num_options = Common().num_options

        # features
        num_selected = Feature(0, self.num_options + 1, self.num_options + 1, lambda configs: reduce(lambda x, y: x+y, configs.config_options))
        self.features: list[Feature] = [num_selected]

        self.dimension = len(self.features)
        self.shape = (self.num_options + 1, )
        self.archive = np.empty(self.shape, dtype=Config)
        self.model = model

        self.crossover_rate = 0.7
        self.mutate_rate = 1 - self.crossover_rate
        self.num_init_samples = 2 * self.num_options

        self.best: Config = None


    def search_configs(self, iteration: int) -> Config:
        print('\tMAP-Elites: Initializing......')
        # self.init()
        print('\tMAP-Elites: Starting evolve......')
        for i in range(iteration):
            if i % 20 == 0:
                print('\tMAP-Elits: iteration {}......'.format(i))
            self.evolve()
        return self.best
    

    def init(self) -> None:
        for i in range(self.num_init_samples):
            config = ConfigUtil.get_next_sat_config()
            self.update_archive(config)


    # 一次迭代
    def evolve(self) -> None:
        indices = self.select()
        old = self.get_from_archive(indices)
        new = self.mutate(old)
        self.update_archive(new)


    def select(self) -> list[int]:
        indices = []
        for length in self.shape:
            index = np.random.randint(length)
            indices.append(index)
        return indices

    
    def mutate(self, config: Config) -> Config:
        return ConfigUtil.get_next_sat_config()
    

    def get_from_archive(self, indices: list[int]) -> Config:
        tmp = self.archive
        for idx in indices:
            tmp = tmp[idx]
        return tmp


    # 子代Config无法保证有效性
    @DeprecationWarning
    def crossover(self, config_a: Config, config_b: Config) -> (Config, Config):
        ops_a = config_a.config_options
        ops_b = config_b.config_options
        # assert(len(ops_b) == len(ops_b))
        crossover_point = np.random.randint(1, self.num_options - 1)    # 交叉点在头尾相当于不变，没有意义
        a_copy = ops_a[:]
        b_copy = ops_b[:]
        for i in range(crossover_point):
            a_copy[i], b_copy[i] = b_copy[i], a_copy[i]
        child_a = Config(a_copy)
        child_b = Config(b_copy)
        return (child_a, child_b)
    

    def update_archive(self, config: Config) -> None:
        tmp = self.archive
        for i in range(self.dimension - 1):
            idx = self.features[i].get_index(config)
            tmp = self.archive[idx]
        last_idx = self.features[self.dimension - 1].get_index(config)
        old: Config = tmp[last_idx]
        val_acquisition = self.f_acquisition(config)
        if old == None or val_acquisition > self.f_acquisition(old):
            tmp[last_idx] = config  # update archive
            if self.best == None or val_acquisition > self.f_acquisition(self.best):
                self.best = config  # update best
            print('\t\tBetter config found. old_perf: {}, new_perf: {}'.format(None if old == None else 
                                                  self.f_acquisition(old), val_acquisition))
            
    
    def f_acquisition(self, config: Config) -> float:
        return config.get_or_eval_predicted_performance(self.model) # maximum mean




class Feature(object):
    def __init__(self, lb, ub, length, eval_func) -> None:
        self.lb = lb    # inclusive
        self.ub = ub    # exclusive
        self.length = length
        self.eval_func = eval_func
        self.step = (ub - lb) / length  # 必须能被整除
        
    def get_index(self, config: Config) -> int:
        feature_value = self.eval_func(config)
        return int((feature_value - self.lb) / self.step)
