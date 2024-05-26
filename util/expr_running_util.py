from datetime import datetime
import math 
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Dict, Iterable

from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
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
    def reproduce_flash() -> int:
        rank = ExprRunningUtil.run(
            MLUtil.using_cart,
            None,
            InitSampling.random,
            IncrementalSampling.min_acquisition_in_once
            )
        return rank
    
    @staticmethod
    @timeit
    def reproduce_gil() -> int:
        rank = ExprRunningUtil.run(
            MLUtil.using_ridge,
            None,
            InitSampling.random,
            IncrementalSampling.min_acquisition_in_once
            )
        return rank

    @staticmethod
    @timeit
    def run(
        f_ml_init: Callable[[], None], 
        f_distance: Callable[[Config, Config], int | float],
        f_init_sampling: Callable[[int], list[Config]],
        f_incremental_sampling: Callable[[list[Config]], Config],
        ml_model=None,
        ) -> int:

        # print('!... new run ...!')

        if ml_model is not None:
            f_ml_init(ml_model)
        else:
            f_ml_init()
        DistanceUtil.f_get_distance = f_distance
        
        samples = f_init_sampling(Common().init_size)
        while len(samples) < Common().total_size:
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
        
        ExprRunningUtil.cleanUp()
        
        return rank
    
    @staticmethod
    def cleanUp():
        MLUtil.config_clazz = None

    @staticmethod
    def comparative_boxplot(rank_dict: dict, fig_size=(7,7)) -> None:
        plt.figure(figsize=fig_size)
        plt.boxplot(rank_dict.values(), labels=rank_dict.keys(), showmeans=True)
        # plt.show()
        path = f'./Data/plot/{datetime.today().strftime("%Y%m%d")}'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{path}/{Common().sys_name}.png')

    @staticmethod
    def save_result(rank_dict: Dict[str, list[int]]):
        path = f'./Data/result/{datetime.today().strftime("%Y%m%d")}'
        for key, ranks in rank_dict.items():
            Path(f'{path}/{key}').mkdir(parents=True, exist_ok=True)
            with open(f'{path}/{key}/{Common().sys_name}.txt', 'w') as f:
                f.writelines([str(rank) + '\n' for rank in ranks])

    @staticmethod
    def run_batch_comparative(sys_name: str, repeats: int) -> None:
    
        ranks_flash = []
        ranks_sail = []
        ranks_gil = []

        Common().load_csv(sys_name)
        for _ in range(repeats):

            rank = ExprRunningUtil.run(
                MLUtil.using_epsilon_greedy,
                DistanceUtil.squared_sum,
                InitSampling.random,
                IncrementalSampling.map_elites
            )
            ranks_sail.append(rank)
            print(f'sail:  rank={rank}')

            # rank = ExprUtil.run_flash(sys_name)
            rank = ExprRunningUtil.reproduce_flash()
            ranks_flash.append(rank)
            print(f'flash: rank={rank}')
            
            rank = ExprRunningUtil.reproduce_gil()
            ranks_gil.append(rank)
            print(f'gil: rank={rank}')
            
        rank_dict = {}
        rank_dict['sail'] = ranks_sail
        rank_dict['flash'] = ranks_flash
        rank_dict['gil'] = ranks_gil
        ExprRunningUtil.comparative_boxplot(rank_dict)
        ExprRunningUtil.save_result(rank_dict)

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
        rank_dict = {}
        
        for model in models:
            start_time = time.perf_counter()
            ranks = []
            for _ in range(repeats):
                rank = ExprRunningUtil.run(
                    MLUtil.using_sklearn_model,
                    DistanceUtil.squared_sum,
                    InitSampling.random,
                    IncrementalSampling.min_acquisition_in_once,
                    ml_model=model
                )
                ranks.append(rank)
            elapse = time.perf_counter() - start_time
            rank_dict[MLUtil.model_name] = ranks
            print(f'{MLUtil.model_name}: mean_rank={sum(ranks) / len(ranks)}, elapse={elapse:.3f}s')
        
        ExprRunningUtil.comparative_boxplot(rank_dict, fig_size=(12, 7))

