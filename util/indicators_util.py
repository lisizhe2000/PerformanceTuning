import math

import numpy as np

from data_processing.common import Common
from util.config import Config
from util.ml_util import MLUtil


class IndicatorsUtil(object):

    @staticmethod
    def get_rank(config: Config, to_minimize=True) -> int:
        sorted_performances = sorted(Common().all_performances, reverse=not to_minimize)
        for i in range(len(sorted_performances)):
            if math.isclose(config.get_real_performance(), sorted_performances[i]):
                return i
        return -1

    @staticmethod
    def get_rank_no_duplicates(config: Config, to_minimize=True) -> int:
        sorted_performances = sorted(set(Common().all_performances), reverse=not to_minimize)
        for i in range(len(sorted_performances)):
            if math.isclose(config.get_real_performance(), sorted_performances[i]):
                return i
        return -1

    @staticmethod
    def get_performance_rank(performance: float, to_minimize=True) -> int:
        sorted_performances = sorted(Common().all_performances, reverse = not to_minimize)
        for i in range(len(sorted_performances)):
            if math.isclose(performance, sorted_performances[i]):
                return i
        return -1

    @staticmethod
    def print_indicators(**kwargs) -> None:
        rank = kwargs['rank']
        predicted_performances = MLUtil.f_precict_all(Common().configs_pool)
        error_rate = IndicatorsUtil.eval_error_rate(predicted_performances)
        n3_percent_error_rate = IndicatorsUtil.eval_n_percent_error_rate(predicted_performances)
        n0_5_percent_error_rate = IndicatorsUtil.eval_n_percent_error_rate(predicted_performances, 0.005)
        best_error_rate = IndicatorsUtil.eval_best_error_rate()
        n_pred_better_than_best = IndicatorsUtil.eval_n_pred_better_than_best(predicted_performances)
        print(
            f'rank: {rank}, '
            f'error_rate: {error_rate:.3f}, '
            f'3%_e: {n3_percent_error_rate:.3f}, '
            f'0.5%_e: {n0_5_percent_error_rate:.3f}, '
            f'best_e: {best_error_rate:.3f}, '
            f'n_pred_better_than_best: {n_pred_better_than_best}, '
        )

    @staticmethod
    def eval_error_rate(predicted_performances: list[float] | np.ndarray) -> float:
        configs = Common().configs_pool
        error_c = []
        for i in range(len(configs)):
            e = abs(configs[i].get_real_performance() - predicted_performances[i]) / configs[i].get_real_performance()
            error_c.append(e)
        error_rate = sum(error_c) / len(error_c)
        return error_rate

    @staticmethod
    def eval_n_percent_error_rate(predicted_performances, percent=0.03, to_minimize=True) -> float:
        configs = Common().configs_pool
        zip_configs_perfs = list(zip(configs, predicted_performances))
        zip_configs_perfs.sort(key=lambda x: x[1], reverse=not to_minimize)
        num_configs = len(zip_configs_perfs)
        num_to_eval = int(num_configs * percent)
        error_c = []
        for i in range(num_to_eval):
            config, predicted_perf = zip_configs_perfs[i]
            e = abs(config.get_real_performance() - predicted_perf) / config.get_real_performance()
            error_c.append(e)
        error_rate = sum(error_c) / len(error_c)
        return error_rate

    @staticmethod
    def eval_n_pred_better_than_best(predicted_performances, to_minimize=True) -> int:
        configs = Common().configs_pool
        best = min(configs, key=lambda config: config.get_real_performance()) if to_minimize else max(configs,
                                                                                                      key=lambda
                                                                                                          config: config.get_real_performance())
        best_pred_perf = MLUtil.f_predict(best)

        num_better = 0
        for predicted_perf in predicted_performances:
            if predicted_perf < best_pred_perf if to_minimize else predicted_perf > best_pred_perf:
                num_better += 1

        return num_better

    @staticmethod
    def eval_best_error_rate(to_minimize=True) -> float:
        configs = Common().configs_pool
        best = min(configs, key=lambda config: config.get_real_performance()) if to_minimize else max(configs,
                                                                                                      key=lambda
                                                                                                          config: config.get_real_performance())
        best_pred_perf = MLUtil.f_predict(best)
        best_error_rate = abs(best.get_real_performance() - best_pred_perf) / best.get_real_performance()
        return best_error_rate
