from data_processing.common import Common
from util.expr_util import ExprUtil
from sampling.incremental_samples import IncrementalSampling
from sampling.init_samples import InitSampling
from util.ml_util import MLUtil


if __name__ == '__main__':
    Common().load_csv('BDBC')
    MLUtil.using_cart()
    size = 40
    samples = InitSampling.fixed_size_candidate_set(5)
    while len(samples) < size:
        MLUtil.f_train(samples)
        new = IncrementalSampling.maximum_mean_in_once_prediction(samples)
        samples.append(new)
        print(f'sample size: {len(samples)}')
    best = max(samples, key = lambda config: config.get_real_performance())
    print(f'rank: {ExprUtil.get_rank(best)}')
