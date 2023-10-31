from functools import reduce
import numpy as np
import xgboost as xgb
from config import Config

from common import Common
from ml_util import MLUtil


class MapElites(object):


    def __init__(self, model: xgb.Booster) -> None:
        self.num_options = Common().num_options

        # features
        num_selected = Feature(0, self.num_options + 1, self.num_options + 1, lambda configs: reduce(lambda x, y: x+y, configs.config_options))
        self.features: list[Feature] = [num_selected]

        self.dimension = len(self.features)
        self.shape = (self.num_options + 1, )
        self.archive = np.empty(self.shape, dtype=object)   # store a (config, val_acquisition) each cell
        self.model = model

        self.crossover_rate = 0.7
        self.mutate_rate = 1 - self.crossover_rate
        self.num_init_samples = 2 * self.num_options

        self.best_config: Config = None
        self.best_val_acq = float('-inf')


    def search_configs(self, iteration: int) -> Config:
        print('\tMAP-Elites: Initializing......')
        # self.init()
        print('\tMAP-Elites: Starting evolve......')
        for i in range(iteration):
            if i % 20 == 0:
                print('\tMAP-Elits: iteration {}......'.format(i))
            self.evolve()
        return self.best_config
    

    def init(self) -> None:
        for i in range(self.num_init_samples):
            config = Config.get_next_sat_config()
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
        return Config.get_next_sat_config()
    

    def get_from_archive(self, indices: list[int]) -> Config:
        tmp = self.archive
        for idx in indices:
            tmp = tmp[idx]
        return tmp[0]


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
        old_config: Config = tmp[last_idx][0] if tmp[last_idx] != None else None
        old_val_acq = tmp[last_idx][1] if tmp[last_idx] != None else None
        val_acquisition = MLUtil.f_acquisition(config)
        if old_config == None or val_acquisition > old_val_acq:
            tmp[last_idx] = (config, val_acquisition)  # update archive
            if self.best_config == None or val_acquisition > self.best_val_acq:
                self.best_config = config  # update best
                self.best_val_acq = MLUtil.f_acquisition(config)
            print('\t\tBetter config found. old_val_acq: {}, new_val_acq: {}'.format(old_val_acq, val_acquisition))




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
