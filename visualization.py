from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from data_processing.common import Common
from util.ml_util import MLUtil


class Visualization(object):

    @staticmethod
    def visualize_tsne() -> None:
        configs = Common().configs_pool
        X = MLUtil.configs_to_nparray(configs)
        y = np.array([config.get_real_performance() for config in configs])
        tsne_res = TSNE().fit_transform(X, y)
        plt.scatter(*zip(*tsne_res))
        plt.show()

    @staticmethod
    def visualize_kmeans() -> None:
        configs = Common().configs_pool
        X = MLUtil.configs_to_nparray(configs)
        kmeans = KMeans().fit(X)
        for i in range(len(configs)):
            configs[i].clazz = kmeans.labels_[i]

        # put different clazz configs into different lists
        configs_by_clazz = [[] for _ in range(kmeans.n_clusters)]
        for config in configs:
            configs_by_clazz[config.clazz].append(config)

        perfs_by_clazz = []
        for configs in configs_by_clazz:
            perfs = [config.get_real_performance() for config in configs]
            perfs_by_clazz.append(perfs)
        plt.boxplot(perfs_by_clazz, showmeans=True)
        plt.savefig(f'./Data/plot/kmeans/{Common().sys_name}.png')
        plt.clf()

    @staticmethod
    def visualize_clazz_by_num_selected() -> None:
        configs = Common().configs_pool
        # init a list of lists
        perfs_by_num_selected = [[] for _ in range(Common().num_options + 1)]
        for config in configs:
            num_selected = int(sum(config.config_options))
            perfs_by_num_selected[num_selected].append(config.get_real_performance())
        plt.boxplot(perfs_by_num_selected, showmeans=True)
        plt.savefig(f'./Data/plot/num_selected/{Common().sys_name}.png')
        plt.clf()


if __name__ == '__main__':
    # Common().load_csv('SQL')
    systems = [
        'Apache',
        'BDBC',
        'Dune',
        'HSMGP',
        'lrzip',
        'SQL',
        'WGet',
        'X264'
    ]
    for system in systems:
        print(f'------ {system} ------')
        Common().load_csv(system)
        # Visualization.visualize_kmeans()
        Visualization.visualize_clazz_by_num_selected()