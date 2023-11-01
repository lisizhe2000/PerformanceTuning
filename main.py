import csv
import time
from data_processing.common import Common
from util.expr_util import ExprUtil
from sampling.incremental_samples import IncrementalSampling
from sampling.init_samples import InitSampling
from util.ml_util import MLUtil
from util.time_counter import TimeCounter, timeit


@timeit
def main():
    start_time = time.perf_counter()

    Common().load_csv('SQL')
    MLUtil.using_cart()

    init_size = 15
    total_size = 40
    f_init_sampling = InitSampling.fscs
    f_incremental_sampling = IncrementalSampling.maximum_mean_in_once_prediction
    
    samples = f_init_sampling(init_size)
    while len(samples) < total_size:
        MLUtil.f_train(samples)
        new = f_incremental_sampling(samples)
        if type(new) == list:   # batch sampling
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
    main()
    print(TimeCounter.execution_time)
