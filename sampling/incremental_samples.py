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
