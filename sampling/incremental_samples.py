import random
from data_processing.common import Common
from sampling.map_elites import MapElites
from util.config import Config
from util.distance_util import DistanceUtil
from util.ml_util import MLUtil
from util.time_counter import timeit


class IncrementalSampling(object):


    @staticmethod
    @timeit
    def single_sample(already_sampled: list[Config]) -> Config:
        pool = Common().configs_pool
        best_config = None
        best_val_acq = None
        for config in pool:
            print(config.to_bin_str())
            if config in already_sampled:
                continue
            val_acq = MLUtil.f_acquisition(config)
            if best_config is None or best_val_acq < val_acq:
                best_config = config
                best_val_acq = val_acq
        return best_config


    @staticmethod
    def batch_sample_knn_substitution(already_sampled: list[Config]) -> tuple[Config]:
        pool = Common().configs_pool
        configs = [config for config in pool if config not in already_sampled]
        predicted_perfs = MLUtil.f_precict_all(configs)
        configs = list(zip(configs, predicted_perfs))
        random.shuffle(configs)

        archive = []
        n = 5
        k = 3
        
        for config, predicted_perf in configs:
            
            if len(archive) < n:
                archive.append((config, predicted_perf))
                continue
            
            distances = []
            for arc_conf, _ in archive:
                d = DistanceUtil.f_get_distance(arc_conf, config)
                distances.append(d)
            
            # find config's knn in archive
            knn_id = []
            for _ in range(k):
                furthest_d = float('-inf')
                furthest_id = None
                for i in range(n):
                    d = distances[i]
                    if i not in knn_id and d > furthest_d:
                        furthest_d = d
                        furthest_id = i
                knn_id.append(furthest_id)
            
            # find the config that it's predicted performance is worst
            worst_perf_in_knn = float('inf')
            worst_perf_id_in_knn = None
            for i in range(n):
                arc_pred_perf = archive[i][1]
                if arc_pred_perf > worst_perf_in_knn:
                    worst_perf_in_knn = arc_pred_perf
                    worst_perf_id_in_knn = i
            # substitute if config is better than the worst
            if predicted_perf > worst_perf_in_knn:
                archive[worst_perf_id_in_knn] = (config, predicted_perf)
        
        new_samples, _ = zip(*archive)
        return new_samples
            

    @staticmethod
    @timeit
    # 一次预测完所有的，速度要快几十上百倍
    def max_acquisition_in_once(already_sampled: list[Config]) -> Config:
        pool = Common().configs_pool
        configs = [config for config in pool if config not in already_sampled]
        acq_vals = MLUtil.f_acquist_all(configs)
        max_acq_val = float('-inf')
        best = None
        for i in range(len(acq_vals)):
            if acq_vals[i] > max_acq_val:
                max_acq_val = acq_vals[i]
                best = configs[i]
        return best
    

    @staticmethod
    @timeit
    def min_acquisition_in_once(already_sampled: list[Config]) -> Config:
        pool = Common().configs_pool
        configs = [config for config in pool if config not in already_sampled]
        acq_vals = MLUtil.f_acquist_all(configs)
        min_acq_val = float('inf')
        best = None
        for i in range(len(acq_vals)):
            if acq_vals[i] < min_acq_val:
                min_acq_val = acq_vals[i]
                best = configs[i]
        return best
    

    @staticmethod
    @timeit
    def map_elites(already_sampled: list[Config]) -> Config:
        pool = Common().configs_pool
        configs = [config for config in pool if config not in already_sampled]
        map_elites = MapElites()
        map_elites.batch_update_archive(configs)
        return map_elites.sample_from_archive(best_n=3)

