from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# inclusive
def get_ranks(df: pd.DataFrame, data_range_in_csv: tuple[int]) -> list[int]:
    ranks = []
    for i in range(data_range_in_csv[0] - 2, data_range_in_csv[1] - 1):
        rank = int(df.loc[i]['rank'])
        ranks.append(rank)
    return ranks


if __name__ == '__main__':
    df = pd.read_csv('./results.csv')

    # inclusive
    start_line = 68
    end_line = 77

    # data
    rank_dict = {}
    i29_t30_maxMean = (184, 203)
    i15_t30_knnSub = (100, 119)
    i10_t30_knnSub = (121, 140)
    i15_t30_maxMean = (142, 161)
    i10_t30_maxMean = (163, 182)
    i15_t30_meanPredicted_randomForest = (226, 245)
    i10_t30_meanPredicted_randomForest = (205, 224)
    i15_t30_maxPredicted_randomForest = (247, 266)
    i10_t30_maxPredicted_randomForest = (268, 287)


    rank_dict['init_size=29\ntotal=30\nmax_mean\nCART'] = get_ranks(df, i29_t30_maxMean)
    # rank_dict['init_size=15\ntotal=30\nknn-sub'] = get_ranks(df, i15_t30_knnSub)
    rank_dict['init_size=10\ntotal=30\nknn-sub\nCART'] = get_ranks(df, i10_t30_knnSub)
    # rank_dict['init_size=15\ntotal=30\nmax_mean'] = get_ranks(df, i15_t30_maxMean)
    rank_dict['init_size=10\ntotal=30\nmax_mean\nCART'] = get_ranks(df, i10_t30_maxMean)

    # 随机森林，弟中之弟！
    # rank_dict['init_size=15\ntotal=30\nmean_predicted\nrandom_forest'] = get_ranks(df, i15_t30_meanPredicted_randomForest)
    rank_dict['init_size=10\ntotal=30\nmean_predicted\nrandom_forest'] = get_ranks(df, i10_t30_meanPredicted_randomForest)
    # rank_dict['init_size=15\ntotal=30\nmax_predicted\nrandom_forest'] = get_ranks(df, i15_t30_maxPredicted_randomForest)
    rank_dict['init_size=10\ntotal=30\nmax_predicted\nrandom_forest'] = get_ranks(df, i10_t30_maxPredicted_randomForest)

    
    ranks = [264,86,129,123,88,162,341,337,184,455,133,64,86,41,262,13,57,25,3,152]     # FLASH-SQL
    rank_dict['FLASH'] = ranks

    data = np.array(ranks)

    plt.figure(figsize=(10, 10))
    plt.boxplot(rank_dict.values(), labels=rank_dict.keys())
    plt.show()
    