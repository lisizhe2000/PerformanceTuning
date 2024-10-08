import math
from data_processing.common import Common
from util.config import Config


class ExprUtil(object):

    @staticmethod
    def get_rank(config: Config) -> int:
        sorted_performances = sorted(Common().all_performances, reverse=True)
        for i in range(len(sorted_performances)):
            if math.isclose(config.get_real_performance(), sorted_performances[i]):
                return i
        return -1
