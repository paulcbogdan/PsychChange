from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import read_csv_fast

pd.options.mode.chained_assignment = None


def do_paper_corr(key='p_fragile'):
    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    df.dropna(subset=[key], inplace=True)
    if key == 'p_fragile_implied':
        df = df[df['sig_implied'] > 1]

    plt.figure(figsize=(7, 2.5))

    matplotlib.rc('font', **{'size': 14})
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.gca().set_facecolor('whitesmoke')
    plt.gca().set_axisbelow(True)
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5,
             axis='y')
    df_ = df[(df['year'] >= 2020) & (df['year'] < 2024)]

    plot_bar_continuous(df_, 'log_cites_year', key)

    plt.tight_layout()
    Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)
    if key == 'p_fragile':
        plt.savefig('../figs_and_tables/Figure_5A_cites.png', dpi=600)
    elif key == 'p_fragile_implied':
        plt.savefig('../figs_and_tables/Figure_S11A_cites_implied.png', dpi=600)
    plt.show()


def get_bootstrap_se(df, y, x, tile=.025, n_pulls=1000):
    df_grpd = df.groupby(['year', f'{x}_int'])[y]
    grp2vals_all = defaultdict(list)
    for _ in range(n_pulls):
        x2vals = defaultdict(list)
        for i, (grp, df_grp) in enumerate(df_grpd):
            df_grp = df_grp.sample(frac=1, replace=True)
            x2vals[grp[1]].append(df_grp.mean())
        for key, vals in x2vals.items():
            grp2vals_all[key].append(np.mean(vals))

    lows = []
    highs = []
    for key in range(len(grp2vals_all)):
        vals = grp2vals_all[key]
        vals = np.sort(vals)
        lows.append(vals[int(tile * len(vals))])
        highs.append(vals[int((1 - tile) * len(vals))])
    ar = np.array([lows, highs])
    ar = np.exp(ar) - 1
    return ar


def dull_color(color):
    new_colors = list(color)
    for i in range(3):
        new_colors[i] = (new_colors[i] + 0.3) / 1.3
    return tuple(new_colors)


def plot_bar_continuous(df, y, key, ):
    df = df.dropna(subset=[y, key])
    df[f'{key}_int'] = (df[key] * 10).astype(int)
    df[f'{key}_int'] = df[f'{key}_int'].apply(lambda x: min(x, 6))

    # average by year, then across years, the undo the log-transformation
    df_grp = df.groupby(['year', f'{key}_int'])[y].mean()
    df_x = df_grp.groupby(f'{key}_int').mean()
    if 'log' in y: df_x = np.exp(df_x) - 1

    df_x = df_x.loc[list(range(7))]  # Over 1000 count

    cmap = plt.get_cmap('coolwarm')
    colors = [cmap(0.95), cmap(0.95), cmap(0.95),
              cmap(0.85), cmap(0.65), cmap(0.45),
              cmap(0.125), cmap(0.05), cmap(0.00)]
    colors = colors[::-1]
    colors = [dull_color(c) for c in colors]

    xt = ['<\u200910%', '<\u200920%', '<\u200930%', '<\u200940%',
          '<\u200950%', '<\u200960%', '≥\u200960%']

    plt.xticks(list(range(7)), xt, fontsize=12.5)
    plt.gca().tick_params(axis='x', length=0, pad=5)

    yerr = get_bootstrap_se(df, y, key)
    yerr -= df_x.values
    yerr[0] = -yerr[0]
    plt.bar(list(range(len(df_x))), df_x.values, color=colors,
            yerr=yerr, error_kw={'capsize': 4, 'elinewidth': 1},
            edgecolor='k', linewidth=0.5)

    last = df_x.values[-1]
    first = df_x.values[0]
    last_up = last + yerr[1][-1] * 2.5
    plt.plot([6.02, 6.02], [first, last_up], color='k', linewidth=1)
    plt.plot([5.92, 6.02], [last_up, last_up], color='k', linewidth=1)
    plt.plot([5.92, 6.02], [first, first], color='k', linewidth=1)
    M = (first + last_up) / 2
    gain = first / last - 1

    if key == 'p_fragile':
        plt.title('Fragile p-values (.01 ≤ p < .05) x Citations',
                  fontsize=14, pad=12)
        plt.text(5.9, M, f'+{gain:.0%}', va='center', ha='right', fontsize=14, )
        plt.yticks([0, 0.5, 1.0, 1.5, 2.])
    elif key == 'p_fragile_implied':
        plt.title('Fragile implied p-values x Citations',
                  fontsize=14, pad=12)
        plt.text(5.9, M * 1.01,
                 f'+{gain:.0%}', va='center', ha='right', fontsize=14, )
        plt.yticks([0, 0.5, 1.0, 1.5])
    plt.ylim(0, None)

    if y == 'log_cites_year':
        plt.ylabel('Citations / year',
                   fontsize=14, labelpad=10)
    else:
        plt.ylabel('Cites / year')


if __name__ == '__main__':
    do_paper_corr(key='p_fragile')
    do_paper_corr(key='p_fragile_implied')
