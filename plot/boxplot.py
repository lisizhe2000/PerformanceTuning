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

    # data
    rank_dict = {}
    
    i29_t30_maxMean_SQL = (184, 203)
    i15_t30_knnSub_SQL = (100, 119)
    i10_t30_knnSub_SQL = (121, 140)
    i15_t30_maxMean_SQL = (142, 161)
    i10_t30_maxMean_SQL = (163, 182)
    i15_t30_meanPredicted_randomForest_SQL = (226, 245)
    i10_t30_meanPredicted_randomForest_SQL = (205, 224)
    i15_t30_maxPredicted_randomForest_SQL = (247, 266)
    i10_t30_maxPredicted_randomForest_SQL = (268, 287)

    # rank_dict['init_size=29\ntotal=30\nmax_mean\nCART'] = get_ranks(df, i29_t30_maxMean)
    # rank_dict['init_size=15\ntotal=30\nknn-sub'] = get_ranks(df, i15_t30_knnSub)
    # rank_dict['init_size=10\ntotal=30\nknn-sub\nCART'] = get_ranks(df, i10_t30_knnSub)
    # rank_dict['init_size=15\ntotal=30\nmax_mean'] = get_ranks(df, i15_t30_maxMean)
    # rank_dict['init_size=10\ntotal=30\nmax_mean\nCART'] = get_ranks(df, i10_t30_maxMean)

    # 随机森林，弟中之弟！
    # rank_dict['init_size=15\ntotal=30\nmean_predicted\nrandom_forest'] = get_ranks(df, i15_t30_meanPredicted_randomForest)
    # rank_dict['init_size=10\ntotal=30\nmean_predicted\nrandom_forest'] = get_ranks(df, i10_t30_meanPredicted_randomForest)
    # rank_dict['init_size=15\ntotal=30\nmax_predicted\nrandom_forest'] = get_ranks(df, i15_t30_maxPredicted_randomForest)
    # rank_dict['init_size=10\ntotal=30\nmax_predicted\nrandom_forest'] = get_ranks(df, i10_t30_maxPredicted_randomForest)

    # 2023.11.09
    i15_t32_maxPredicted_CART_Apache = (288, 307)
    i15_t44_maxPredicted_CART_BDBC = (308, 327)
    i15_t36_maxPredicted_CART_lrzip = (328, 347)
    i15_t33_maxPredicted_CART_SQL = (348, 367)
    i15_t34_maxPredicted_CART_WGet = (368, 387)
    i15_t41_maxPredicted_CART_X264 = (388, 407)
    i15_t39_maxPredicted_CART_Dune = (408, 427)
    i15_t43_maxPredicted_CART_HSMGP = (428, 447)
    # discrete
    flash_ranks_Apache = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]                               # mean_size = 32.35
    flash_ranks_BDBC = [0, 3, 8, 128, 128, 128, 3, 128, 1, 3, 0, 128, 1, 0, 136, 0, 0, 1, 0, 136]                   # mean_size = 43.9
    flash_ranks_lrzip = [5, 0, 0, 0, 0, 0, 0, 0, 2, 5, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]                                # mean_size = 36.15
    flash_ranks_SQL = [12, 19, 79, 19, 214, 167, 81, 32, 35, 28, 65, 124, 342, 113, 13, 546, 349, 98, 155, 112]     # mean_size = 33.15
    flash_ranks_WGet = [0, 1, 0, 0, 1, 1, 0, 3, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 8, 1]                                 # mean_size = 34.35
    flash_ranks_X264 = [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0]                                 # mean_size = 40.8
    # numeric
    flash_ranks_Dune = [3, 3, 5, 8, 82, 3, 3, 2, 3, 3, 20, 7, 3, 12, 5, 5, 1, 7, 21, 10]                            # mean_size = 39.65
    flash_ranks_HSMGP = [5, 58, 1, 0, 5, 0, 0, 0, 0, 5, 8, 0, 6, 0, 5, 0, 8, 0, 1, 5]                               # mean_size = 42.9

    # 2023.11.9
    i10_t36_maxPredicted_CART_lrzip = (448, 467)
    i5_t36_maxPredicted_CART_lrzip = (468, 487)
    i3_t36_maxPredicted_CART_lrzip = (488, 507)
    i2_t36_maxPredicted_CART_lrzip = (508, 527)


    # plot dict
    # rank_dict['our'] = get_ranks(df, i15_t43_maxPredicted_CART_HSMGP)
    rank_dict['i2'] = get_ranks(df, i2_t36_maxPredicted_CART_lrzip)
    rank_dict['i3'] = get_ranks(df, i3_t36_maxPredicted_CART_lrzip)
    rank_dict['i5'] = get_ranks(df, i5_t36_maxPredicted_CART_lrzip)
    rank_dict['i10'] = get_ranks(df, i10_t36_maxPredicted_CART_lrzip)
    rank_dict['i15'] = get_ranks(df, i15_t36_maxPredicted_CART_lrzip)
    rank_dict['FLASH'] = flash_ranks_HSMGP


    plt.figure(figsize=(7, 7))
    plt.boxplot(rank_dict.values(), labels=rank_dict.keys())
    plt.show()
    