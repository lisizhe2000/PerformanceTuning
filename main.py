from data_processing.common import Common
from util.config import Config
from util.expr_util import ExprUtil
from sampling.incremental_samples import IncrementalSampling
from sampling.init_samples import InitSampling
from util.ml_util import MLUtil


if __name__ == '__main__':
    Common().load_csv('Apache')
    MLUtil.set_xgboost()
    size = 6
    samples = InitSampling.fixed_size_candidate_set(5)
    while len(samples) < size:
        MLUtil.f_train(samples)
        new = IncrementalSampling.single_sample(samples)
        samples.append(new)
    best = max(samples, key = lambda config: config.get_real_performance())
    print(f'rank: {ExprUtil.get_rank(best)}')
