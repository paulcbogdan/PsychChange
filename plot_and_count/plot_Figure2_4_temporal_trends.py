from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from utils import read_csv_fast

pd.options.mode.chained_assignment = None


def pool_neighoring_years(df_s, grp_range=1):
    years = df_s['year'].unique()
    for i in range(-grp_range, grp_range + 1):
        if i == 0: continue
        df_s_cop = df_s.copy()
        df_s_cop['year'] = df_s_cop['year'] + i
        df_s = pd.concat([df_s, df_s_cop])
    df_s = df_s[df_s.year.isin(years)]
    return df_s


def plot_key(df, key='marginal', group_keys=None, ax=None,
             marker=None, shaded=True, linewidth=1):
    if ax is not None:
        plt.sca(ax)
    plt.gca().spines[['right', 'top']].set_visible(False)

    if not isinstance(key, tuple):
        df = df.dropna(subset=[key])

    df = pool_neighoring_years(df)

    group2full = {'General': 'General_Psychology',
                  'Exp. & Cog.': 'Experimental_and_Cognitive_Psychology',
                  'Dev. & Edu.': 'Developmental_and_Educational_Psychology',
                  'Social': 'Social_Psychology',
                  'Clinical': 'Clinical_Psychology',
                  'Applied': 'Applied_Psychology'}

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3)),
                  (0, (3, 1, 1, 1, 1, 1))]

    for subject in group_keys:
        df_s = df[df[group2full[subject]] == True]
        yrs = []
        Ms = []
        SE_low = []
        SE_high = []
        for year, df_sy in df_s.groupby('year'):
            if isinstance(key, tuple):
                M = df_sy[key[0]].mean() - df_sy[key[1]].mean()
                if 'p_fragile_only_w_implied' in key:
                    SE = (df_sy[key[0]] - df_sy[key[1]]).sem()
                else:
                    SE = np.sqrt(df_sy[key[0]].sem() ** 2 + df_sy[key[1]].sem() ** 2)
            elif key in ['d', 't_N', 'd_sig', 't_N_sig']:
                vals = df_sy[key].values
                low = np.nanquantile(vals, .25)
                high = np.nanquantile(vals, .75)
                M = np.nanmedian(vals)
                SD = (high - low) / 1.349
                SE = SD / (np.sum(~np.isnan(vals)) ** 0.5)
            else:
                M = df_sy[key].mean()
                SE = df_sy[key].std() / (df_sy[key].count() ** 0.5)
            SE_low.append(M - SE)  # *1.96)
            SE_high.append(M + SE)  # *1.96)
            Ms.append(M)
            yrs.append(year)
        plt.plot(yrs, Ms, label=subject,
                 linewidth=linewidth, linestyle=linestyles.pop(0),
                 marker=marker, markersize=4)
        if shaded:
            plt.fill_between(yrs, SE_low, SE_high, alpha=0.2)
    plt.xlim(min(df.year), max(df.year))

    plt.gca().set_facecolor('whitesmoke')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.yticks(fontsize=11)

    if key == 'p_fragile':  # 0.2998, 0.2998
        plt.title('Mean p-fragile (%)\n'
                  '(.01 ≤ p < .05)',
                  fontsize=12.5, pad=9)
        plt.plot([min(yrs), max(yrs)], [.26, .26], '--', color='k',
                 linewidth=1, zorder=-1)
        plt.text(2012, 0.253, '(Expected if 80% power)',
                 fontsize=11, ha='center', va='top')
        plt.ylim(.21, .343)
    elif key in ['p_fragile_implied', 'p_fragile_implied_rel_sig', ]:  # 0.2998, 0.2998
        plt.title('Mean implied p-fragile (%)\n(.01 < p < .05)',
                  fontsize=12.5, pad=9)
        plt.plot([min(yrs), max(yrs)], [.26, .26], '--', color='k',
                 linewidth=1, zorder=-1)
        plt.text(2011.5, 0.253, '(Expected if\n  80% power)',
                 fontsize=11, ha='center', va='top')
        plt.ylim(.21, .343)
    elif key == 'p_bad':
        plt.title('Flagrantly unlikely to replicate\n(p-fragile ≥ 50%)',
                  fontsize=12.5, pad=9)
        plt.plot([min(yrs), max(yrs)], [.11, .11], '--', color='k',
                 linewidth=1, zorder=-1)
        plt.text(2014, 0.123, '(Expected if 80% power)',
                 fontsize=11, ha='center')
        plt.yticks([.1, .15, .2, .25, .3, .35])
    elif key == 'p_good':
        plt.title('Optimistically replicable\n(p-fragile < 32%)',
                  fontsize=12.5, pad=9)
        plt.plot([min(yrs), max(yrs)], [.66, .66], '--', color='k',
                 linewidth=1, zorder=-1)
        plt.text(2014, 0.64, '(Expected if 80% power)',
                 fontsize=11, ha='center')
    elif key in ['t_N', 't_N_sig']:
        plt.title('Median sample sizes',
                  fontsize=12.5, pad=9)
        plt.ylim(0, 260)
    elif key in ['d', 'd_sig']:
        plt.title('Median Cohen’s d',
                  fontsize=12.5, pad=9)
    elif isinstance(key, tuple):
        if 'implied' in key[0]:
            plt.title('Observed minus implied p-fragile\n(only papers with test statistics)',
                      fontsize=12.5, pad=9)
        else:
            plt.title('Observed minus implied p-fragile',
                      fontsize=12.5, pad=9)
        plt.ylim(-.005, .062)

    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024], fontsize=10.75)

    if isinstance(key, tuple):
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('+{x:.0%}'))
    elif 'p_fragile' in key:
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    if ax is None:
        plt.show()


@cache
def get_df_for_temporal_trends(bias_adjustment=.023):
    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)

    # Apply the adjustment after calculating the p-values, as otherwise
    #   the adjustment may cause 2-p-value or 3-p-value papers to flip
    df['p_bad'] = (df['p_fragile'] > .5 - 1e-6).astype(int)
    df['p_good'] = (df['p_fragile'] < .319).astype(int)
    df['p_fragile'] -= bias_adjustment
    df['gap'] = df['p_fragile'] - df['p_fragile_implied']
    M_gap = df['gap'].mean()
    print(f'Mean gap: {M_gap=:.3f}')

    df_ = df.dropna(subset=['p_fragile_implied'])
    df['p_fragile_only_w_implied'] = df_['p_fragile']

    gap = df_['gap'].mean()
    print(f'Mean gap only test statistics: {gap=:.3f}')
    return df


def plot_subject_over_time(keys, bias_adjustment=.023):
    df = get_df_for_temporal_trends(bias_adjustment=bias_adjustment)

    group_keys = ['General', 'Exp. & Cog.', 'Dev. & Edu.', 'Social',
                  'Clinical', 'Applied']

    pd.set_option('display.max_rows', 2000)
    # for key in keys:
    #     if key in ['gap', 'gap_drop']: continue
    #     df_m = df.groupby('year')[key].mean()

    if len(keys) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.7))
    elif len(keys) == 3:
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.7))
    elif len(keys) == 6:
        fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.3))
    else:
        fig, axs = plt.subplots(2, 2, figsize=(6.5, 6.3))

    for i, key in enumerate(keys):
        if len(keys) in [2, 3]:
            plt.sca(axs[i])
        else:
            if len(keys) == 6:
                row = i // 3
                col = i % 3
            else:
                row = i // 2
                col = i % 2
            plt.sca(axs[row, col])
        # df_ = df.dropna(subset=[key])
        if key == 'p_fragile_implied_rel_sig':
            df_ = df[df['p_fragile_implied_rel_sig'] <= 1]
        else:
            df_ = df

        if key == 'gap':
            key = ('p_fragile', 'p_fragile_implied')
        elif key == 'gap_drop':
            key = ('p_fragile_only_w_implied', 'p_fragile_implied')
        plot_key(df_, key=key, ax=plt.gca(), group_keys=group_keys)

    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    leg = fig.legend(handles, labels, loc='lower center', ncol=6,
                     fontsize=12.5 if len(keys) == 6 else 11,
                     frameon=False, columnspacing=0.8, handletextpad=0.3,
                     markerscale=2, handlelength=1.5)
    for line in leg.get_lines():
        line.set_linewidth(1.5)

    if len(keys) in [2, 3]:
        plt.subplots_adjust(left=0.075, bottom=0.20, right=0.96,
                            top=0.85, wspace=0.35 if 'gap' in keys else 0.24,
                            )
    else:
        plt.subplots_adjust(left=0.075,
                            bottom=0.125 if len(keys) == 6 else 0.11,
                            right=0.96, top=0.915, wspace=0.24,
                            hspace=0.45 if len(keys) == 6 else (.4 if 'gap' in keys else 0.325)
                            )
    Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)
    if keys == ['p_fragile', 'p_fragile_implied', ]:
        plt.savefig('../figs_and_tables/Figure_2_temporal_p_frag.png', dpi=600)
    elif keys == ['t_N_sig', 'd_sig']:
        plt.savefig('../figs_and_tables/Figure_4_temporal_sample_effect.png', dpi=600)
    elif keys == ['gap', 'gap_drop']:
        plt.savefig('../figs_and_tables/Figure_S10_temporal_gap.png', dpi=600)
    else:
        print(f'No file saved: {keys=}')
    plt.show()


if __name__ == '__main__':
    KEYS = ['p_fragile', 'p_fragile_implied']  # Figure 2
    plot_subject_over_time(KEYS)
    KEYS = ['t_N_sig', 'd_sig']  # Figure 4
    plot_subject_over_time(KEYS)
    KEYS = ['gap', 'gap_drop']  # Figure S10
    plot_subject_over_time(KEYS)
