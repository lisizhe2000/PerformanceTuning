from functools import reduce
from typing import Dict
import numpy as np
import xgboost as xgb
from util.config import Config

from data_processing.common import Common
from util.ml_util import MLUtil
from util.time_counter import timeit


class MapElites(object):

    @timeit
    def __init__(self, to_minimize=True) -> None:
        self.num_options = Common().num_options
        self.to_minimize = to_minimize

        # use num_selected as feature
        # num_selected = Feature(0, self.num_options + 1, self.num_options + 1, lambda configs: reduce(lambda x, y: x+y, configs.config_options))
        # self.features: list[Feature] = [num_selected]
        # self.shape = (self.num_options + 1, )
        
        # use k-means as feature
        if MLUtil.config_clazz is None:
            MLUtil.config_clazz = MLUtil.get_kmeans_clazz(Common().configs_pool)
        kmeans_clazz = Feature(0, MLUtil.kmeans_n_clusters, MLUtil.kmeans_n_clusters, lambda config: MLUtil.config_clazz[config])
        self.features: list[Feature] = [kmeans_clazz]
        self.shape = (MLUtil.kmeans_n_clusters, )

        self.dimension = len(self.features)
        self.archive: np.ndarray = np.empty(self.shape, dtype=object)   # store a (config, val_acquisition) each cell

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
    # generated way
    def evolve(self) -> None:
        new = Config.get_next_sat_config()
        self.update_archive(new)

    def select(self) -> list[int]:
        indices = []
        for length in self.shape:
            index = np.random.randint(length)
            indices.append(index)
        return indices

    def get_from_archive(self, indices: list[int]) -> Config:
        tmp = self.archive
        for idx in indices:
            tmp = tmp[idx]
        return tmp[0]
    
    @timeit
    def update_archive(self, config: Config, val_acquisition=None) -> None:
        tmp = self.archive
        for i in range(self.dimension - 1):
            idx = self.features[i].get_index(config)
            tmp = self.archive[idx]
        last_idx = self.features[self.dimension - 1].get_index(config)
        old_config: Config = tmp[last_idx][0] if tmp[last_idx] != None else None
        old_val_acq = tmp[last_idx][1] if tmp[last_idx] != None else None
        if val_acquisition is None:
            val_acquisition = MLUtil.f_acquisition(config)
        if old_config == None or (val_acquisition < old_val_acq if self.to_minimize else val_acquisition > old_val_acq):
            tmp[last_idx] = (config, val_acquisition)  # update archive
            if self.best_config == None or (val_acquisition < self.best_val_acq if self.to_minimize else val_acquisition > self.best_val_acq):
                self.best_config = config  # update best
                self.best_val_acq = val_acquisition
            # print('\t\tBetter config found. old_val_acq: {}, new_val_acq: {}'.format(old_val_acq, val_acquisition))

    @timeit
    def batch_update_archive(self, configs: list[Config]) -> None:
        val_acqs = MLUtil.f_acquist_all(configs)
        for i in range(len(configs)):
            config = configs[i]
            val_acquisition = val_acqs[i]
            self.update_archive(config, val_acquisition)

    def sample_from_archive(self, best_n=3) -> Config:
        elites = self.archive.flatten()
        # choose best_n elites
        if len(elites) < best_n:
            best_n = len(elites)
        elites = [elite for elite in elites if elite != None]
        elites = sorted(elites, key=lambda x: x[1], reverse=not self.to_minimize)[:best_n]  # sort by val_acquisition
        # choose one from best_n elites
        return elites[np.random.randint(best_n)][0]


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
