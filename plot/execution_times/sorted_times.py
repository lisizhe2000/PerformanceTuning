import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_sorted_times(system, perf):
    # Load the data
    data = pd.read_csv(f'./Data/CsvMeasurements/{system}.csv')
    times = data.loc[:, perf].to_numpy()
    times.sort()
    sns.lineplot(x=range(len(times)), y=times)
    plt.fill_between(range(len(times)), times, alpha=0.2)
    plt.ylabel('Execution Time (s)')
    plt.xlabel('Configuration Rank')
    plt.savefig(f'./plot/execution_times/{system}.png', dpi=300)
    plt.close()

plot_sorted_times('X264', '$<Performance')
# plot_sorted_times('lrzip', '$<CompressionTime')
