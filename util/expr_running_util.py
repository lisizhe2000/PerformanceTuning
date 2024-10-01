from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Callable, Dict, Iterable, List

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from settings import Settings

matplotlib.use('Agg')
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
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

systems = [
    'Apache',
    'BDBC',
    'Dune',
    'HSMGP',
    'lrzip',
    'SQL',
    'WGet',
    'X264',

    'SS-A1',
    'SS-A2',
    'SS-B1',
    'SS-B2',
    'SS-C1',
    'SS-C2',
    'SS-D1',
    'SS-D2',
    'SS-E1',
    'SS-E2',
    'SS-F1',
    'SS-F2',
    'SS-G1',
    'SS-G2',
    'SS-H1',
    'SS-H2',
    'SS-I1',
    'SS-I2',
    'SS-J1',
    'SS-J2',
    'SS-K1',
    'SS-K2',
    'SS-L1',
    'SS-L2',

    # 'SS-M1',
    # 'SS-M2',
    # 'SS-N1',
    # 'SS-N2',
    # 'SS-O1',
    # 'SS-O2',

    # 'JavaGC_num',
    # 'sac_compile-cpu',
    # 'sac_compile-maxmem',
    # 'sac_run-cpu',
    # 'sac_run-maxmem',
]


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
        
        samples = f_init_sampling(Settings.init_size)
        while len(samples) < Settings.total_size:
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
        
        ExprRunningUtil.clean_up()
        
        return rank
    
    @staticmethod
    def clean_up():
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
    def save_result_rank(rank_dict: Dict[str, list[int]]):
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
                IncrementalSampling.min_acq_with_shapley_mutated,
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
        ExprRunningUtil.save_result_rank(rank_dict)

    @staticmethod
    def get_evaluation_count(f_ml_init: Callable[[], None],
                             f_init_sampling: Callable[[int], list[Config]],
                             f_incremental_sampling: Callable[[list[Config]], Config],
                             ) -> List[int]:
        f_ml_init()
        samples = f_init_sampling(Settings.init_size)
        evals = Settings.init_size
        pool_size = len(Common().configs_pool)
        n1, n2, n3 = pool_size * Settings.r1, pool_size * Settings.r2, pool_size * Settings.r3
        r1, r2, r3 = 0, 0, 0
        while True:
            MLUtil.f_train(samples)
            new = f_incremental_sampling(samples)
            if isinstance(new, Iterable):
                samples.extend(new)
            else:
                samples.append(new)
            evals += 1
            best = min(samples, key = lambda config: config.get_real_performance())
            rank = IndicatorsUtil.get_rank(best, to_minimize=True)
            if r1 == 0 and rank <= n1:
                r1 = evals
            if r2 == 0 and rank <= n2:
                r2 = evals
            if r3 == 0 and rank <= n3:
                r3 = evals
                break
        return [r1, r2, r3]

    @staticmethod
    def run_evaluation_count(sys_name: str, repeats: int) -> None:
        Common().load_csv(sys_name)
        r1s, r2s, r3s = [], [], []
        for _ in range(repeats):
            r1, r2, r3 = ExprRunningUtil.get_evaluation_count(
                MLUtil.using_epsilon_greedy,
                InitSampling.random,
                IncrementalSampling.min_acq_with_shapley_mutated,
            )
            r1s.append(r1)
            r2s.append(r2)
            r3s.append(r3)
        print(f'r1: {r1s}, r2: {r2s}, r3: {r3s}')
        print(f'mean r1: {sum(r1s) / len(r1s)}, mean r2: {sum(r2s) / len(r2s)}, mean r3: {sum(r3s) / len(r3s)}')
        ExprRunningUtil.save_result_evaluation_count(r1s, r2s, r3s, 'sail')
        r1s, r2s, r3s = [], [], []
        for _ in range(repeats):
            r1, r2, r3 = ExprRunningUtil.get_evaluation_count(
                MLUtil.using_cart,
                InitSampling.random,
                IncrementalSampling.min_acquisition_in_once,
            )
            r1s.append(r1)
            r2s.append(r2)
            r3s.append(r3)
        print(f'r1: {r1s}, r2: {r2s}, r3: {r3s}')
        print(f'mean r1: {sum(r1s) / len(r1s)}, mean r2: {sum(r2s) / len(r2s)}, mean r3: {sum(r3s) / len(r3s)}')
        ExprRunningUtil.save_result_evaluation_count(r1s, r2s, r3s, 'flash')
        r1s, r2s, r3s = [], [], []
        for _ in range(repeats):
            r1, r2, r3 = ExprRunningUtil.get_evaluation_count(
                MLUtil.using_ridge,
                InitSampling.random,
                IncrementalSampling.min_acquisition_in_once,
            )
            r1s.append(r1)
            r2s.append(r2)
            r3s.append(r3)
        print(f'r1: {r1s}, r2: {r2s}, r3: {r3s}')
        print(f'mean r1: {sum(r1s) / len(r1s)}, mean r2: {sum(r2s) / len(r2s)}, mean r3: {sum(r3s) / len(r3s)}')
        ExprRunningUtil.save_result_evaluation_count(r1s, r2s, r3s, 'gil')

    @staticmethod
    def save_result_evaluation_count(r1s: List[int], r2s: List[int], r3s: List[int], alg: str):
        path = f'./Data/result/{datetime.today().strftime("%Y%m%d")}/{alg}'
        Path(path).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({f'到达前{Settings.r1}评估次数': r1s, f'到达前{Settings.r2}评估次数': r2s, f'到达前{Settings.r3}评估次数': r3s})
        df.to_csv(f'{path}/{Common().sys_name}_eval_count.csv', index=False)

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

    @staticmethod
    def get_mse() -> None:
        models = [
            DecisionTreeRegressor(),
            KNeighborsRegressor(),
            SVR(),
            SVR(kernel='poly', C=1.0, epsilon=0.1),
            LinearRegression(),
            Ridge(),
            lgb.LGBMRegressor(verbosity=-1),
            RandomForestRegressor(),
            ]
        data = []
        for sys in systems:
            Common().load_csv(sys)
            configs = Common().configs_pool
            np.random.shuffle(configs)
            X = MLUtil.configs_to_nparray(configs)
            y = np.array([config.get_real_performance() for config in configs])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            X_train, y_train = X_train[:50], y_train[:50]
            for model in models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                print(f'{sys:10s} {model.__class__.__name__:30s} rmse: {rmse:.3f}')
                data.append([sys, model.__class__.__name__, rmse])
            data.append(['', '', ''])
        df = pd.DataFrame(data, columns=['sys', 'model', 'MSE'])
        df.to_csv('./Data/rmse.csv', index=False)

    @staticmethod
    def get_training_time() -> None:
        models = [
            DecisionTreeRegressor(),
            KNeighborsRegressor(),
            SVR(),
            SVR(kernel='poly', C=1.0, epsilon=0.1),
            LinearRegression(),
            Ridge(),
            lgb.LGBMRegressor(verbosity=-1),
            GaussianProcessRegressor(kernel=RBF()),
            ]
        data = []
        for sys in systems:
            Common().load_csv(sys)
            configs = Common().configs_pool
            np.random.shuffle(configs)
            X = MLUtil.configs_to_nparray(configs)
            y = np.array([config.get_real_performance() for config in configs])
            for model in models:
                start_time = time.perf_counter()
                for _ in range(50):
                    model.fit(X[:50], y[:50])
                train_elapse = time.perf_counter() - start_time
                start_time = time.perf_counter()
                for _ in range(50):
                    model.predict(X)
                predict_elapse = time.perf_counter() - start_time
                print(f'{sys:10s} {model.__class__.__name__:30s} train: {train_elapse:.3f}, predict: {predict_elapse:.3f}')
                data.append([sys, model.__class__.__name__, train_elapse, predict_elapse])
