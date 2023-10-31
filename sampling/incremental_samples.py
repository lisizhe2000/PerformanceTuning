from typing import Callable
from data_processing.common import Common
from util.config import Config
from util.ml_util import MLUtil


class IncrementalSampling(object):

    f_single_sampling: Callable[[], Config] = None
    f_batch_sampling: Callable[[], Config] = None


    @staticmethod
    def single_sample(already_sampled: list[Config]) -> Config:
        pool = Common().configs_pool
        best_config = None
        best_val_acq = None
        for config in pool:
            print(config.to_bin_str())
            if config in already_sampled:
                continue
            val_acq = MLUtil.f_acquisition(config)
            if best_config is None or best_val_acq < val_acq:
                best_config = config
                best_val_acq = val_acq
        return best_config


    @staticmethod
    def batch_sample_nslc(already_sampled: list[Config]) -> list[Config]:
        # TODO
        pass


    @staticmethod
    def maximum_mean_in_once_prediction(already_sampled: list[Config]) -> Config:
        pool = Common().configs_pool
        configs = [config for config in pool if config not in already_sampled]
        predictions = MLUtil.f_precict_all(configs)
        max_mean = float('-inf')
        best = None
        for i in range(len(predictions)):
            if predictions[i] >  max_mean:
                max_mean = predictions[i]
                best = configs[i]
        return best
