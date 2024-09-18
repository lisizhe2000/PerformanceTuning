import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def draw(data, system):
    df = pd.DataFrame(data)

    # 创建柱状图
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # 使用seaborn画柱状图
    sns.barplot(x='Model', y='MSE', data=df, palette='muted')

    plt.grid(False)
    # 设置标签和标题
    plt.ylabel('RMSE')
    plt.xlabel(system)
    plt.savefig(f'./plot/mse_50sample/{system}.png', dpi=600)

draw(data = {'Model': ['CART', 'LightGBM', 'Ridge', 'KNN', 'SVR(RBF)', 'SVR(Poly)'],
             'MSE': [1.4174547001363027,0.8579955429904813,4.734352906198423,3.71707916574694,2.4063163982691287,1.8769412746013878]},
     system='LLVM')
draw(data = {'Model': ['CART', 'LightGBM', 'Ridge', 'KNN', 'SVR(RBF)', 'SVR(Poly)'],
             'MSE': [1.0507862419995873,0.8035890165250278,0.950391171695566,0.8087233702519275,0.8151184829009822,0.9061009110862349]},
     system='SQL')
draw(data = {'Model': ['CART', 'LightGBM', 'Ridge', 'KNN', 'SVR(RBF)', 'SVR(Poly)'],
             'MSE': [37.16840778289549,51.78665246776725,110.4134791708776,162.76835620676982,60.50381478092308,40.25656298810347]},
     system='X264')