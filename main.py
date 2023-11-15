import csv
import time
from data_processing.common import Common
from util.distance_util import DistanceUtil
from util.expr_util import ExprUtil
from sampling.incremental_samples import IncrementalSampling
from sampling.init_samples import InitSampling
from util.ml_util import MLUtil
from util.time_counter import TimeCounter, timeit


@timeit
def main():
    start_time = time.perf_counter()

    Common().load_csv('lrzip')
    MLUtil.using_cart()
    DistanceUtil.f_get_distance = DistanceUtil.squared_sum

    init_size = 2
    total_size = 36
    f_init_sampling = InitSampling.fscs
    f_incremental_sampling = IncrementalSampling.max_acquisition_in_once
    
    samples = f_init_sampling(init_size)
    while len(samples) < total_size:
        MLUtil.f_train(samples)
        new = f_incremental_sampling(samples)
        if type(new) == list or type(new) == tuple:   # batch sampling
            samples.extend(new)
        else:                   # single sampling
            samples.append(new)
        # print(f'sample size: {len(samples)}')
    best = max(samples, key = lambda config: config.get_real_performance())
    rank = ExprUtil.get_rank(best)
    print(f'rank: {rank}')

    execution_time = time.perf_counter() - start_time
    
    results = [
        Common().sys_name,
        f_init_sampling.__name__,
        f_incremental_sampling.__name__,
        MLUtil.acquisition_function_name,
        init_size,
        total_size,
        MLUtil.model_name,
        execution_time,
        rank
    ]

    with open('./results.csv' ,'a') as f:
        writer = csv.writer(f)
        writer.writerow(results)
    

if __name__ == '__main__':
    # main()

    systems = [
        # 'Apache',
        # 'BDBC',
        # 'Dune',
        # 'HSMGP',
        # 'lrzip',
        'SQL',
        # 'WGet',
        # 'X264'
    ]

    for sys in systems:
        print(f'------ {sys} ------')
        ExprUtil.run_batch_comparative(sys, 20)

    print(TimeCounter.execution_time)
