import random
import numpy as np
from data_processing.common import Common
from util.config import Config
from util.distance_util import DistanceUtil
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
    def random(size: int) -> None:
        pool = Common().configs_pool
        clone = [config for config in pool]
        random.shuffle(clone)
        return clone[:size]
