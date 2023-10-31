from common import Common
from config import Config


class DistanceUtil(object):

    @staticmethod
    def hamming_distance(config_a: Config, config_b: Config) -> int:
        ops_a = config_a.config_options
        ops_b = config_b.config_options
        distance = 0
        for i in range(Common().num_options):
            distance += ops_a[i] ^ ops_b[i]
        return distance