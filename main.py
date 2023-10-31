from common import Common
from config import Config
from expr_util import ExprUtil
from incremental_samples import IncrementalSampling
from init_samples import InitSampling
from ml_util import MLUtil


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
