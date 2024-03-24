import seaborn as sns
from matplotlib import pyplot as plt


def plot_time_series(df, x, y, title='', xlabel='Date', ylabel='Value'):
    plt.figure(figsize=(12, 4))
    sns.lineplot(df, x=x, y=y)
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    return
