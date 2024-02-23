from typing import Callable
from data_processing.common import Common
from util.config import Config


class DistanceUtil(object):

    f_get_distance: Callable[[Config, Config], int | float] = None

    @staticmethod
    def hamming_distance(config_a: Config, config_b: Config) -> int:
        ops_a = config_a.config_options
        ops_b = config_b.config_options
        distance = 0
        for i in range(Common().num_options):
            distance += (ops_a[i] != ops_b[i])
        return distance
    
    @staticmethod
    def squared_sum(config_a: Config, config_b: Config) -> float:
        ops_a = config_a.config_options
        ops_b = config_b.config_options
        distance = 0
        for i in range(Common().num_options):
            distance += abs(ops_a[i] - ops_b[i]) ** 2
        return distance
