from datetime import datetime
import math 
import os
from pathlib import Path
import re
import subprocess
from typing import Callable, Iterable

from matplotlib import pyplot as plt
from data_processing.common import Common
from sampling.incremental_samples import IncrementalSampling
from sampling.init_samples import InitSampling
from util.config import Config
from util.distance_util import DistanceUtil
from util.ml_util import MLUtil
from util.time_counter import timeit


class ExprUtil(object):

    @staticmethod
    def get_rank(config: Config, to_minimize=True) -> int:
        sorted_performances = sorted(Common().all_performances, reverse = not to_minimize)
        for i in range(len(sorted_performances)):
            if math.isclose(config.get_real_performance(), sorted_performances[i]):
                return i
        return -1
    

    @staticmethod
    def get_rank_no_duplicates(config: Config, to_minimize=True) -> int:
        sorted_performances = sorted(set(Common().all_performances), reverse = not to_minimize)
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
    @timeit
    def run_flash(filename) -> (int, int):
        cwd = os.getcwd()
        flash_dir = '/home/sizhe/code/Flash-SingleConfig'

        os.chdir(flash_dir)
        ret = subprocess.run(['python', 'progressive_active_learning.py', '-f', filename], capture_output=True, text=True)
        os.chdir(cwd)
        
        rank = re.match(r'.*rank:(\d+)', ret.stdout).group(1)
        evals = re.match(r'.*evals:(\d+)', ret.stdout).group(1)
        best_performance = re.search(r'performance=([\d.]+)', ret.stdout).group(1)
        rank = ExprUtil.get_performance_rank(float(best_performance))
        # rank = str(rank)
        print(f'flash: best performance: {best_performance}')
        # return (int(rank), int(evals))
        return (rank, int(evals))
    

    @staticmethod
    @timeit
    def reproduce_flash(filename) -> (int, int):
        rank, evals = ExprUtil.run(
            filename,
            MLUtil.using_cart,
            None,
            20,
            50,
            InitSampling.random,
            IncrementalSampling.min_acquisition_in_once
            )
        return (rank, evals)
    
    
    @staticmethod
    @timeit
    def run(
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
        rank = ExprUtil.get_rank_no_duplicates(best, to_minimize=True)
        print(f'sail: best performance: {best.get_real_performance()}')
        
        return (rank, total_size)


    @staticmethod
    def comparative_boxplot(rank_dict: dict) -> None:
        plt.figure(figsize=(7,7))
        plt.boxplot(rank_dict.values(), labels=rank_dict.keys(), showmeans=True)
        # plt.show()
        path = f'./Data/plot/{datetime.today().strftime("%Y%m%d")}'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{path}/{Common().sys_name}.png')


    @staticmethod
    def run_batch_comparative(sys_name: str, repeats: int) -> None:
    
        ranks_flash = []
        evals_flash = []
        ranks_sail = []
        evals_sail = []
        
        for _ in range(repeats):

            # init_size = evals // 2
            init_size = 20
            rank, evals = ExprUtil.run(
                sys_name, 
                MLUtil.using_cart, 
                DistanceUtil.squared_sum,
                init_size,
                50,
                InitSampling.random,
                IncrementalSampling.map_elites_num_selected
            )
            ranks_sail.append(rank)
            evals_sail.append(evals)
            print(f'sail:  rank={rank}, evals={evals}')

            # rank, evals = ExprUtil.run_flash(sys_name)
            rank, evals = ExprUtil.reproduce_flash(sys_name)
            ranks_flash.append(rank)
            evals_flash.append(evals)
            print(f'flash: rank={rank}, evals={evals}')
            
        rank_dict = {}
        rank_dict['sail'] = ranks_sail
        rank_dict['flash'] = ranks_flash
        ExprUtil.comparative_boxplot(rank_dict)
        
