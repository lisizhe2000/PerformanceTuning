from typing import Callable

import numpy as np
from common import Common
from config import Config
from distance_util import DistanceUtil


class InitSampling(object):

    f_init_samples: Callable[[], list[Config]] = None


    @staticmethod
    def diversified_distance_based_sampling(size: int) -> list[Config]:
        # TODO
        pass


    @staticmethod
    def novelty_search(size: int) -> list[Config]:
        # TODO
        pass


    @staticmethod
    def fixed_size_candidate_set(size: int) -> list[Config]:
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
                    distance += DistanceUtil.hamming_distance(config, sample)
                if furthest_config is None or furthest_distance < distance:
                    furthest_config = config
                    furthest_distance = distance
            samples.append(furthest_config)
        return samples
            
