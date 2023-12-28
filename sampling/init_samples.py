from functools import reduce
import random
import numpy as np
from data_processing.common import Common
from util.config import Config
from util.distance_util import DistanceUtil
from util.ml_util import MLUtil
from util.time_counter import timeit


class InitSampling(object):


    @staticmethod
    def diversified_distance_based_sampling(size: int) -> list[Config]:
        # TODO
        pass


    @staticmethod
    def novelty_search(size: int) -> list[Config]:
        # TODO
        pass


    @staticmethod
    @timeit
    # fixed-size-candidate-set
    def fscs(size: int) -> list[Config]:
        samples = []
        pool = Common().configs_pool
        num_all_valid = len(pool)
        samples.append(pool[np.random.randint(num_all_valid)])
        while len(samples) < size:
            furthest_config = None
            furthest_distance = None
            for config in pool:
                distance = 0
                for sample in samples:
                    distance += DistanceUtil.f_get_distance(config, sample)
                if furthest_config is None or furthest_distance < distance:
                    furthest_config = config
                    furthest_distance = distance
            samples.append(furthest_config)
        return samples

    
    @staticmethod
    def random(size: int) -> list[Config]:
        pool = Common().configs_pool
        clone = [config for config in pool]
        random.shuffle(clone)
        return clone[:size]
    

    @staticmethod
    def random_each_num_selected(size: int) -> list[Config]:
        pool = Common().configs_pool
        configs_per_num_selected = {}
        for config in pool:
            num_selected = reduce(lambda x, y: x+y, config.config_options)
            if num_selected not in configs_per_num_selected:
                configs_per_num_selected[num_selected] = []
            configs_per_num_selected[num_selected].append(config)

        samples = []
        num_samples_per_num_selected = size // len(configs_per_num_selected)
        num_samples = [num_samples_per_num_selected] * len(configs_per_num_selected)
        tmp = [i for i in range(len(configs_per_num_selected))]
        random.shuffle(tmp)
        tmp = tmp[:size % len(configs_per_num_selected)]
        for i in tmp:
            num_samples[i] += 1
        # TODO: 有可能会出现某个num_selected的config不够的情况

    
    @staticmethod
    def random_each_kmeans_clazz(size: int) -> list[Config]:
        pool = Common().configs_pool
        config_clazz = MLUtil.get_kmeans_clazz(pool)
        # TODO

        