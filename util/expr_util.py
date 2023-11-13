import math 
import os
import re
import subprocess
from typing import Callable, Iterable

from matplotlib import pyplot as plt
from data_processing.common import Common
from util.config import Config
from util.distance_util import DistanceUtil
from util.ml_util import MLUtil
from util.time_counter import timeit


class ExprUtil(object):

    @staticmethod
    def get_rank(config: Config, toMinimize=True) -> int:
        sorted_performances = sorted(Common().all_performances, reverse = not toMinimize)
        for i in range(len(sorted_performances)):
            if math.isclose(config.get_real_performance(), sorted_performances[i]):
                return i
        return -1


    @staticmethod
    @timeit
    def run_flash(filename) -> (int, int):
        cwd = os.getcwd()
        flash_dir = '/home/sizhe/code/Flash-SingleConfig'

        os.chdir(flash_dir)
        ret = subprocess.run(['python', 'progressive_active_learning.py', '-f', filename], capture_output=True, text=True)
        os.chdir(cwd)
        
        rank = re.match(r'.*rank:(\d+)', ret.stdout).group(1)
        evals = re.match(r'.*evals:(\d+)', ret.stdout).group(1)
        best_performance = re.search(r'\[([\d.]+)\]', ret.stdout).group(1)
        print(f'flash: best performance: {best_performance}')
        return (int(rank), int(evals))
    
    
    @staticmethod
    @timeit
    def run_sail(
        filename: str, 
        f_ml_init: Callable[[], None], 
        f_distance: Callable[[Config, Config], int | float],
        init_size: int,
        total_size: int,
        f_init_sampling: Callable[[int], list[Config]],
        f_incremental_sampling: Callable[[list[Config]], Config]
        ) -> (int, int):

        Common().load_csv(filename)
        f_ml_init()
        DistanceUtil.f_get_distance = f_distance
        
        samples = f_init_sampling(init_size)
        while len(samples) < total_size:
            MLUtil.f_train(samples)
            new = f_incremental_sampling(samples)
            if isinstance(new, Iterable):
                samples.extend(new)
            else:
                samples.append(new)

        best = min(samples, key = lambda config: config.get_real_performance())
        rank = ExprUtil.get_rank(best, toMinimize=True)
        print(f'sail: best performance: {best.get_real_performance()}')
        
        return (rank, total_size)


    @staticmethod
    def comparative_boxplot(rank_dict: dict) -> None:
        plt.figure(figsize=(7,7))
        plt.boxplot(rank_dict.values(), labels=rank_dict.keys())
        plt.show()
        
