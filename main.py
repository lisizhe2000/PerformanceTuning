import csv
import time
from data_processing.common import Common
from util.distance_util import DistanceUtil
from util.expr_running_util import ExprRunningUtil
from sampling.incremental_samples import IncrementalSampling
from sampling.init_samples import InitSampling
from util.indicators_util import IndicatorsUtil
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
    rank = IndicatorsUtil.get_rank(best)
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
        # 'SQL',
        # 'WGet',
        # 'X264',

        # 'SS-A1',
        # 'SS-A2',
        # 'SS-B1',
        # 'SS-B2',
        # 'SS-C1',
        # 'SS-C2',
        # 'SS-D1',
        # 'SS-D2',
        # 'SS-E1',
        # 'SS-E2',
        # 'SS-F1',
        # 'SS-F2',
        # 'SS-G1',
        # 'SS-G2',
        # 'SS-H1',
        # 'SS-H2',
        # 'SS-I1',
        # 'SS-I2',
        # 'SS-J1',
        # 'SS-J2',
        # 'SS-K1',
        # 'SS-K2',
        # 'SS-L1',
        # 'SS-L2',
        
        'SS-M1',
        'SS-M2',
        'SS-N1',
        'SS-N2',
        'SS-O1',
        'SS-O2',
        #        # 'JavaGC',
        'JavaGC_num',
        'sac_compile-cpu',
        'sac_compile-maxmem',
        'sac_run-cpu',
        'sac_run-maxmem',
    ]

    for sys in systems:
        print(f'------ {sys} ------')
        ExprRunningUtil.run_batch_comparative(sys, 20)
        # ExprRunningUtil.run_different_models(sys, 10)

    print(TimeCounter.execution_time)
