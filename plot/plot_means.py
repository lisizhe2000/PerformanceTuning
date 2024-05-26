import matplotlib.pyplot as plt
import statistics
import numpy as np

# 示例数据
sys_name2primitive_name = {
    'Apache' : 'Apache',
    'BDBC' : 'BDBC',
    'Dune' : 'Dune',
    'HSMGP' : 'HSMGP',
    'lrzip' : 'lrzip',
    'SQL' : 'SQL',
    'WGet' : 'WGet',
    'X264' : 'X264',
    'sort' : 'SS-B1',
    # 'wc-C' : 'SS-C1',
    # 'wc-D' : 'SS-D2',
    'noc' : 'SS-H1',
    # 'rs' : 'SS-J1',
    'wc' :'SS-K1',
    'LLVM' : 'SS-L2',
    'Trimesh' : 'SS-M1',
    'SaC' : 'SS-O2',
    # 'sac-cc' : 'sac_compile-cpu',
    # 'sac-cm' : 'sac_compile-maxmem',
    # 'sac-rc' : 'sac_run-cpu',
    # 'sac-rm' : 'sac_run-maxmem',
}

if __name__ == '__main__':
    
    rank_diffs = []
    path = f'/home/sizhe/code/SailHCS/Data/result/20240519/epsilon_greedy+map_elites_kmeans_best_n=3'
    # sys_names = python 
    
    sys_names = list(sys_name2primitive_name.keys())
    primitive_names = list(sys_name2primitive_name.values())
    ranks_sail = {}
    ranks_flash = {}
    ranks_sail_means = []
    ranks_flash_means = []
    for sys_name in sys_names:
        primitive_name = sys_name2primitive_name[sys_name]
        with open(f'{path}/sail/{primitive_name}.txt') as f:
            ranks_sail[primitive_name] = []
            for line in f.readlines():
                ranks_sail[primitive_name].append(int(line))
        ranks_sail_means.append(statistics.mean(ranks_sail[primitive_name]))
        
        with open(f'{path}/flash/{primitive_name}.txt') as f:
            ranks_flash[primitive_name] = []
            for line in f.readlines():
                ranks_flash[primitive_name].append(int(line))
        ranks_flash_means.append(statistics.mean(ranks_flash[primitive_name]))
        
    rank_diffs = [y - x for x, y in zip(ranks_sail_means, ranks_flash_means)]
    print(rank_diffs)
    
    # 设置图表尺寸
    plt.figure(figsize=(10, 6))

    plt.bar(sys_names, rank_diffs, width=0.5)

    # 添加标题和标签
    # plt.title('Performance Metrics of Algorithms on Different Datasets')
    # plt.xlabel('Datasets')
    plt.ylabel('Rank Difference')
    # plt.legend()
    plt.axhline(y=0, color='black', linewidth=0.5)

    # 显示图表
    # plt.show()
    plt.savefig('./tmp.png')
