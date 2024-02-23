from __future__ import annotations
from functools import reduce
import random
import numpy as np
from data_processing.common import Common


class Config(object):

    def __init__(self, config_options: list[ int | float ], performance: float = None) -> None:
        assert(len(config_options) == Common().num_options)

        self.config_options = config_options

        # Generated config, performance unknown
        if performance == None:
            self.__real_perf_evaluated = False
            self.__real_performance = None
        # Config from pool, performance already known
        else:
            self.__real_perf_evaluated = True
            self.__real_performance = performance

    # 如果从xml中读取，可能不存在，会抛异常，需要进行异常处理
    # 缓存未命中时复杂度为O(n)，n为num_options
    def get_real_performance(self) -> float:
        if not self.__real_perf_evaluated:
            self.__real_performance = Common().measurement_tree.get_performance(self.config_options)
            self.__real_perf_evaluated = True
        return self.__real_performance

    # for bool type config
    def to_bin_str(self) -> str:
        bin_str = ''
        for option in self.config_options:
            bin_str += '1' if option else '0'
        return bin_str

    # for bool type config
    def get_selected_options_names(self) -> str:
        config_names = ''
        for i in range(Common().num_options):
            if self.config_options[i]:
                config_names += Common().id_to_option[i + 1]
                config_names += ','
        config_names = config_names[:-1]
        return config_names

    # 该方法需要尝试几十万上百万次才能找到一个有效配置，效率极低
    # 或许加载mandatory和dead后能提高n倍效率（n = 2 ** (num_mandatory + num_dead)）
    @DeprecationWarning
    @classmethod
    def gen_random_config(cls) -> Config:
        print('generating random config......')
        performance = None
        while performance == None:
            try:
                num_selected = np.random.randint(Common().num_options + 1)
                config_options = [True] * num_selected + [False] * (Common().num_options - num_selected)
                random.shuffle(config_options)
                config = cls(config_options)
                performance = config.get_or_eval_real_performance()
            except:
                pass
        print('generation done. config = {}'.format(config))
        return config

    @classmethod
    def get_next_sat_config(cls) -> Config:
        solution = next(Common().sat_config_iter, None)
        config_options = [True if lit > 0 else False for lit in solution]
        return cls(config_options)
