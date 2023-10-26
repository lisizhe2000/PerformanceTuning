from functools import reduce
import numpy as np
import xgboost as xgb
from common import Common


class Config(object):


    def __init__(self, config_options: list[bool]) -> None:
        assert(len(config_options) == Common().num_options)

        self.config_options = config_options
        self.__real_perf_evaluated = False
        self.__pred_perf_evalueted = False
        self.__real_performance = None
        self.__predicted_performance = None
    

    # 可能不存在，会抛异常，需要进行异常处理
    def get_or_eval_real_performance(self) -> float:
        if not self.__real_perf_evaluated:
            # self.eval_real_performance()
            self.__real_performance = Common().measurement_tree.get_performance(self.config_options)
            self.__real_perf_evaluated = True
        return self.__real_performance
    

    def get_or_eval_predicted_performance(self, model: xgb.Booster) -> float:
        if not self.__pred_perf_evalueted:
            self.__predicted_performance = model.predict(xgb.DMatrix(np.array([self.config_options])))[0]
            self.__pred_perf_evalueted = True
        return self.__predicted_performance
    

    def to_bin_str(self) -> str:
        bin_str = ''
        for option in self.config_options:
            bin_str += '1' if option else '0'
        return bin_str
    

    def get_selected_options_names(self) -> str:
        config_names = ''
        for i in range(Common().num_options):
            if self.config_options[i]:
                config_names += Common().id_to_option[i + 1]
                config_names += ','
        config_names = config_names[:-1]
        return config_names
