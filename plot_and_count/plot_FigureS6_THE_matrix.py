from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def autocorr_ranks():
    dir_in = r'../dataframes/times_higher_ed'
    dfs = []
    for year in range(2011, 2025):
        fn = f'{year}_rankings.csv'
        fp = f'{dir_in}/{fn}'
        df = pd.read_csv(fp)
        print(f'{year}: {len(df)=}')
        df = df[['name', 'scores_research']]
        df.rename(columns={'scores_research': f'score_{year}'}, inplace=True, )
        df.set_index('name', inplace=True)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)

    df.sort_values('score_2024', ascending=False, inplace=True)

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 10002)
    pd.set_option('display.width', 300)

    plt.figure(figsize=(6, 5))
    matplotlib.rc('font', **{'size': 10})
    corr = df.corr(method='pearson')

    plt.pcolormesh(corr, cmap='inferno', )
    plt.xticks(np.arange(14) + .5, list(range(2011, 2025)), rotation=45)
    plt.yticks(np.arange(14) + .5, list(range(2011, 2025)))

    plt.title('Times Higher Education\nWorld University Rankings',
              fontsize=11)
    plt.ylabel('Year', fontsize=12)
    plt.gca().set_xticks(np.linspace(0., 13, 14, endpoint=True), minor=True)
    plt.gca().set_yticks(np.linspace(0., 13, 14, endpoint=True), minor=True)
    plt.gca().tick_params(which='minor', bottom=False, left=False)
    plt.grid(which='minor', color='k', linestyle='-', linewidth=0.5,
             alpha=.15)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    cbar.ax.set_title('Spearman\ncorrelation', fontsize=10)
    Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)
    plt.savefig('../figs_and_tables/Figure_S6_corr_THE_ranks.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    autocorr_ranks()
