from datetime import datetime
import math 
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Iterable

from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from data_processing.common import Common
from sampling.incremental_samples import IncrementalSampling
from sampling.init_samples import InitSampling
from util.config import Config
from util.distance_util import DistanceUtil
from util.indicators_util import IndicatorsUtil
from util.ml_util import MLUtil
from util.time_counter import timeit


class ExprRunningUtil(object):

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
        rank = ExprRunningUtil.get_performance_rank(float(best_performance))
        # rank = str(rank)
        print(f'flash: best performance: {best_performance}')
        # return (int(rank), int(evals))
        return (rank, int(evals))
    

    @staticmethod
    @timeit
    def reproduce_flash(init_size, total_size) -> (int, int):
        rank, evals = ExprRunningUtil.run(
            MLUtil.using_cart,
            None,
            init_size,
            total_size,
            InitSampling.random,
            IncrementalSampling.min_acquisition_in_once
            )
        return (rank, evals)
    
    
    @staticmethod
    @timeit
    def run(
        f_ml_init: Callable[[], None], 
        f_distance: Callable[[Config, Config], int | float],
        init_size: int,
        total_size: int,
        f_init_sampling: Callable[[int], list[Config]],
        f_incremental_sampling: Callable[[list[Config]], Config],
        ml_model=None,
        ) -> (int, int):

        # print('!... new run ...!')

        if ml_model is not None:
            f_ml_init(ml_model)
        else:
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

            # rank = ExprUtil.get_rank(new)
            # ExprUtil.print_indicators(rank=rank)

        best = min(samples, key = lambda config: config.get_real_performance())
        rank = IndicatorsUtil.get_rank(best, to_minimize=True)
        # print(f'best performance: {best.get_real_performance()}')
        
        return (rank, total_size)


    @staticmethod
    def comparative_boxplot(rank_dict: dict, fig_size=(7,7)) -> None:
        plt.figure(figsize=fig_size)
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
        
        Common().load_csv(sys_name)
        for _ in range(repeats):

            # init_size = evals // 2
            rank, evals = ExprRunningUtil.run(
                MLUtil.using_epsilon_greedy,
                DistanceUtil.squared_sum,
                Common().init_size,
                Common().total_size,
                InitSampling.random,
                IncrementalSampling.min_acquisition_in_once
            )
            ranks_sail.append(rank)
            evals_sail.append(evals)
            print(f'sail:  rank={rank}, evals={evals}')

            # rank, evals = ExprUtil.run_flash(sys_name)
            rank, evals = ExprRunningUtil.reproduce_flash(Common().init_size, Common().total_size)
            ranks_flash.append(rank)
            evals_flash.append(evals)
            print(f'flash: rank={rank}, evals={evals}')
            
        rank_dict = {}
        rank_dict['sail'] = ranks_sail
        rank_dict['flash'] = ranks_flash
        ExprRunningUtil.comparative_boxplot(rank_dict)
        
        
    @staticmethod
    def run_different_models(sys_name: str, repeats: int) -> None:
        models = [
            LinearRegression(), 
            SVR(),
            KNeighborsRegressor(),
            # LogisticRegression(),     # 逻辑回归是分类模型
            DecisionTreeRegressor(),
            # RandomForestRegressor(),
            # GaussianNB(),   # 可以用partial_fit做增量学习
            # AdaBoostRegressor(),
            ]
        Common().load_csv(sys_name)
        init_size = 20
        total_size = 50
        rank_dict = {}
        
        for model in models:
            start_time = time.perf_counter()
            ranks = []
            for _ in range(repeats):
                rank, evals = ExprRunningUtil.run(
                    MLUtil.using_sklearn_model,
                    DistanceUtil.squared_sum,
                    init_size,
                    total_size,
                    InitSampling.random,
                    IncrementalSampling.min_acquisition_in_once,
                    ml_model=model
                )
                ranks.append(rank)
            elapse = time.perf_counter() - start_time
            rank_dict[MLUtil.model_name] = ranks
            print(f'{MLUtil.model_name}: mean_rank={sum(ranks) / len(ranks)}, evals={evals}, elapse={elapse:.3f}s')
        
        ExprRunningUtil.comparative_boxplot(rank_dict, fig_size=(12, 7))

