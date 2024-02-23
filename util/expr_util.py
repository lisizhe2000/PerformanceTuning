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
    def reproduce_flash(init_size, total_size) -> (int, int):
        rank, evals = ExprUtil.run(
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
        rank = ExprUtil.get_rank(best, to_minimize=True)
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
            rank, evals = ExprUtil.run(
                MLUtil.using_cart,
                DistanceUtil.squared_sum,
                Common().init_size,
                Common().total_size,
                InitSampling.random,
                IncrementalSampling.map_elites
            )
            ranks_sail.append(rank)
            evals_sail.append(evals)
            print(f'sail:  rank={rank}, evals={evals}')

            # rank, evals = ExprUtil.run_flash(sys_name)
            rank, evals = ExprUtil.reproduce_flash(Common().init_size, Common().total_size)
            ranks_flash.append(rank)
            evals_flash.append(evals)
            print(f'flash: rank={rank}, evals={evals}')
            
        rank_dict = {}
        rank_dict['sail'] = ranks_sail
        rank_dict['flash'] = ranks_flash
        ExprUtil.comparative_boxplot(rank_dict)
        
        
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
                rank, evals = ExprUtil.run(
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
        
        ExprUtil.comparative_boxplot(rank_dict, fig_size=(12,7))


    @staticmethod
    def print_indicators(**kwargs) -> None:
        rank = kwargs['rank']
        predicted_performances = MLUtil.f_precict_all(Common().configs_pool)
        error_rate = ExprUtil.eval_errror_rate(predicted_performances)
        n3_percent_error_rate = ExprUtil.eval_n_percent_error_rate(predicted_performances)
        n0_5_percent_error_rate = ExprUtil.eval_n_percent_error_rate(predicted_performances, 0.005)
        best_error_rate = ExprUtil.eval_best_error_rate()
        n_pred_better_than_best = ExprUtil.eval_n_pred_better_than_best(predicted_performances)
        print(
            f'rank: {rank}, '
            f'error_rate: {error_rate:.3f}, '
            f'3%_e: {n3_percent_error_rate:.3f}, '
            f'0.5%_e: {n0_5_percent_error_rate:.3f}, '
            f'best_e: {best_error_rate:.3f}, '
            f'n_pred_better_than_best: {n_pred_better_than_best}, '
            )

    
    @staticmethod
    def eval_errror_rate(predicted_performances: list[float]) -> float:
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
        best = min(configs, key=lambda config: config.get_real_performance()) if to_minimize else max(configs, key=lambda config: config.get_real_performance())
        best_pred_perf = MLUtil.f_predict(best)
        
        num_better = 0
        for predicted_perf in predicted_performances:
            if predicted_perf < best_pred_perf if to_minimize else predicted_perf > best_pred_perf:
                num_better += 1

        return num_better
    

    @staticmethod
    def eval_best_error_rate(to_minimize=True) -> float:
        configs = Common().configs_pool
        best = min(configs, key=lambda config: config.get_real_performance()) if to_minimize else max(configs, key=lambda config: config.get_real_performance())
        best_pred_perf = MLUtil.f_predict(best)
        best_error_rate = abs(best.get_real_performance() - best_pred_perf) / best.get_real_performance()
        return best_error_rate
