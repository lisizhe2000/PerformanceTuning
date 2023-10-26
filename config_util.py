import random

import numpy as np
from common import Common
from config import Config


class ConfigUtil(object):


    # 该方法需要尝试几十万上百万次才能找到一个有效配置，效率极低
    # 或许加载mandatory和dead后能提高n倍效率（n = 2 ** (num_mandatory + num_dead)）
    @DeprecationWarning
    @staticmethod
    def gen_random_config() -> Config:
        print('generating random config......')
        performance = None
        while performance == None:
            try:
                num_selected = np.random.randint(Common().num_options + 1)
                config_options = [True] * num_selected + [False] * (Common().num_options - num_selected)
                random.shuffle(config_options)
                config = Config(config_options)
                performance = config.get_or_eval_real_performance()
            except:
                pass
        print('generation done. config = {}'.format(config))
        return config
    
    
    @staticmethod
    def get_next_sat_config() -> Config:
        solution = next(Common().sat_config_iter, None)
        config_options = [True if lit > 0 else False for lit in solution]
        return Config(config_options)
    

    @staticmethod
    def mutate(config: Config) -> Config:
        flip_rate = 0.1
        pass
                