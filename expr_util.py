import math
from common import Common
from config import Config


class ExprUtil(object):

    @staticmethod
    def get_rank(config: Config) -> int:
        sorted_performances = sorted(Common().all_performances, reverse=True)
        for i in range(len(sorted_performances)):
            if math.isclose(config.get_or_eval_real_performance(), sorted_performances[i]):
                return i
        return -1
