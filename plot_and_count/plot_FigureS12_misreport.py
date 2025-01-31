import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

from make_dataset.make_p_z_df import stat2p_z_single
from plot_and_count.plot_Figure2_4_temporal_trends import plot_key
from utils import read_csv_fast, PSYC_SUBJECTS_


def get_num_digits(val):
    s_val = str(val)
    if 'e' in s_val:
        return 0
    num_digits = len(s_val.split('.')[1])
    return num_digits


def eval_similarity_wiggle(p_val, p_implied, stat_type, stat, df1, df2, ):
    if pd.isna(stat):
        return pd.NA, pd.NA, pd.NA

    stat = np.abs(stat)
    stat = max(stat, .0001)  # there is at least one case of this
    stat_digits = get_num_digits(stat)  # will error if scientific notation
    stat_low = round(stat - 0.5 * (10 ** -stat_digits), stat_digits + 1)
    stat_high = round(stat + 0.5 * (10 ** -stat_digits), stat_digits + 1)
    p_implied_high = stat2p_z_single(stat_type, stat_low, df1, df2, p_val)
    p_implied_low = stat2p_z_single(stat_type, stat_high, df1, df2, p_val)

    p_val_ = max(p_val, .001)
    p_val_ = min(p_val_, .999)
    num_digits_p = get_num_digits(p_val_)
    p_implied_low = round(p_implied_low, num_digits_p)
    p_implied_high = round(p_implied_high, num_digits_p)

    p_implied_low_rnd = max(p_implied_low, .001)
    p_implied_low_rnd = min(p_implied_low_rnd, .999)
    p_implied_high_rnd = max(p_implied_high, .001)
    p_implied_high_rnd = min(p_implied_high_rnd, .999)

    consistent = p_implied_low_rnd <= p_val_ <= p_implied_high_rnd
    one_tailed = False
    if not consistent:
        p_val_half = p_val_ * 2
        p_implied_low_rnd = max(p_implied_low, .0005)
        p_implied_high_rnd = max(p_implied_high, .0005)
        good_half = p_implied_low_rnd <= p_val_half <= p_implied_high_rnd
        if good_half:
            one_tailed = True

    if consistent:
        z_dif = 0
    else:
        z_val = stats.norm.isf(p_val_)
        p_implied_ = max(p_implied, .001)
        p_implied_ = min(p_implied_, .999)
        z_implied = stats.norm.isf(p_implied_)
        z_dif = z_val - z_implied  # positive = p_val is stronger than p_implied

    return consistent, one_tailed, z_dif


def plot_misreporting_over_time(df):
    years = sorted(df['year'].unique())
    df['abs_z_dif'] = df['z_dif'].abs()
    df['overreport'] = (df['z_dif'] > 0).astype(int)

    df_my_tail_drop = df[df['one_tailed'] == False]
    df_nj_tail_drop = df[df['has_tail'] == False]

    pd.set_option('display.max_rows', None)

    df_grp = df.groupby('year')['consistent'].mean()
    v_og = [1 - df_grp[year] for year in years]
    df_my = df_my_tail_drop.groupby('year')['consistent'].mean()
    v_my = [1 - df_my[year] for year in years]
    df_nj = df_nj_tail_drop.groupby('year')['consistent'].mean()
    v_nj = [1 - df_nj[year] for year in years]

    matplotlib.rc('font', **{'size': 12, })
    fig, axs = plt.subplots(3, figsize=(6.5, 8))

    plt.sca(axs[0])
    plt.title('Overall misreporting')
    plt.plot(years, v_og, label='No drop', color='k', marker='.')
    plt.plot(years, v_my, label='Statcheck drop', linestyle='--', color='red', marker='.')
    plt.plot(years, v_nj, label='Word drop', linestyle='-.', color='dodgerblue', marker='.')
    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024], fontsize=11)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.75)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xlim(2003.5, 2024.5)
    plt.yticks([.1, .11, .12, .13, .14], fontsize=11)
    plt.ylabel('Misreporting rate', labelpad=8, fontsize=13)

    plt.sca(axs[1])
    df_inc = df[df['consistent'] == False]
    df_my_tail_drop = df_inc[df_inc['one_tailed'] == False]
    df_nj_tail_drop = df_inc[df_inc['has_tail'] == False]
    df_grp = df_inc.groupby('year')['abs_z_dif'].median()
    v_og = [df_grp[year] for year in years]
    df_my = df_my_tail_drop.groupby('year')['abs_z_dif'].median()
    v_my = [df_my[year] for year in years]
    df_nj = df_nj_tail_drop.groupby('year')['abs_z_dif'].median()
    v_nj = [df_nj[year] for year in years]
    plt.title('Misreport magnitude')
    plt.plot(years, v_og, label='No drop', color='k', marker='.')
    plt.plot(years, v_my, label='Statcheck drop', linestyle='--', color='red', marker='.')
    plt.plot(years, v_nj, label='Word drop', linestyle='-.', color='dodgerblue', marker='.')
    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024], fontsize=11)
    plt.ylabel('|z($p_{reported}$) - z($p_{implied}$)|', labelpad=8, fontsize=13)
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.75)
    plt.yticks([.25, .3, .35])
    plt.ylim(None, .355)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xlim(2003.5, 2024.5)

    df_grp = df_inc.groupby('year')['overreport'].mean()
    v_og = [df_grp[year] for year in years]
    df_my = df_my_tail_drop.groupby('year')['overreport'].mean()
    v_my = [df_my[year] for year in years]
    df_nj = df_nj_tail_drop.groupby('year')['overreport'].mean()
    v_nj = [df_nj[year] for year in years]

    plt.sca(axs[2])
    plt.title('Over vs. underreporting')
    plt.plot(years, v_og, label='No one-tail drop', color='k', marker='.')
    plt.plot(years, v_my, label='Statcheck drop', linestyle='--', color='r', marker='.')
    plt.plot(years, v_nj, label='Word drop', linestyle='-.', color='dodgerblue', marker='.')
    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024], fontsize=11)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.75)
    plt.gca().spines[['right', 'top']].set_visible(False)
    # plt.legend(frameon=False)
    plt.yticks([.35, .4, .45, .5, .55, .6], fontsize=11)
    plt.ylim(.345, .615)
    plt.xlim(2003.5, 2024.5)
    plt.ylabel('Overreported rate', labelpad=8, fontsize=13)
    plt.xlabel('Year', fontsize=13, labelpad=6)

    plt.subplots_adjust(left=0.17, bottom=0.115,
                        right=0.95, top=0.94,
                        wspace=0.24, hspace=.4)

    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12,
               frameon=False, columnspacing=1,  # handletextpad=0.3,
               markerscale=1, handlelength=1.5, bbox_to_anchor=(.54, 0))
    fp_out = r'../figs_and_tables/Figure_S2ABC_misreport_temporal.png'
    plt.savefig(fp_out, dpi=600)

    plt.show()


def plot_misreport_hist(df):
    pd.set_option('display.max_rows', None)
    matplotlib.rc('font', **{'size': 12, })
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 3))
    df = df[df['consistent'] == False]
    plt.sca(axs[0])
    plt.hist(df['z_dif'].to_list(), bins=50, range=(-1, 1), stacked=True, color='k')
    plt.ylabel('Frequency', fontsize=12, labelpad=8)
    plt.xticks([-1, -.5, 0, .5, 1], ['-1', '-.5', '0', '.5', '1'], fontsize=12)
    plt.yticks(fontsize=12)
    axs[0].spines[['right', 'top']].set_visible(False)

    plt.sca(axs[1])
    df_one_tail = df[df['one_tailed'] == True]
    df_not_one_tail = df[df['one_tailed'] == False]
    plt.hist([df_not_one_tail['z_dif'].to_list(),
              df_one_tail['z_dif'].to_list(), ],
             bins=50, range=(-1, 1), stacked=True, color=['k', 'red'])
    plt.xlabel('z($p_{reported}$) - z($p_{implied}$)', fontsize=12)
    plt.xticks([-1, -.5, 0, .5, 1], ['-1', '-.5', '0', '.5', '1'], fontsize=12)
    axs[1].spines[['left', 'right', 'top']].set_visible(False)
    plt.yticks([])

    plt.sca(axs[2])
    df_one_tail = df[df['has_tail'] == True]
    df_not_one_tail = df[df['has_tail'] == False]
    plt.hist([df_not_one_tail['z_dif'].to_list(),
              df_one_tail['z_dif'].to_list(), ],
             bins=50, range=(-1, 1), stacked=True, color=['k', 'dodgerblue'])
    plt.xticks([-1, -.5, 0, .5, 1], ['-1', '-.5', '0', '.5', '1'], fontsize=12)
    axs[2].spines[['left', 'right', 'top']].set_visible(False)
    plt.yticks([])
    plt.suptitle('Misreported results summary')
    plt.tight_layout()

    fp_out = r'../figs_and_tables/Figure_S1ABC_misreport_magnitude.png'
    plt.savefig(fp_out, dpi=600)

    plt.show()


def plot_1tail_rate(df):
    years = sorted(df['year'].unique())
    df['abs_z_dif'] = df['z_dif'].abs()
    df['overreport'] = (df['z_dif'] > 0).astype(int)

    pd.set_option('display.max_rows', None)

    df['one_tailed_inc'] = df['one_tailed'] & ~df['consistent']
    df['has_tail_inc'] = df['has_tail'] & ~df['consistent']

    df_my = df.groupby('year')['one_tailed_inc'].mean()
    v_my = [df_my[year] for year in years]
    df_nj = df.groupby('year')['has_tail_inc'].mean()
    v_nj = [df_nj[year] for year in years]

    matplotlib.rc('font', **{'size': 12, })
    fig = plt.figure(figsize=(6.5, 3))
    plt.title('One-tailed analysis rate')
    plt.plot(years, v_my, label='Statcheck one-tail', linestyle='--', color='red', marker='.')
    plt.plot(years, v_nj, label='Word one-tail', linestyle='-.', color='dodgerblue', marker='.')
    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024], fontsize=11)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.1%}'))
    plt.yticks([.01, .015, .02, .025], fontsize=11)
    plt.ylim(.0095, .0265)
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.75)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xlabel('Year', labelpad=-36, fontsize=13)
    plt.xlim(2003.5, 2024.5)

    plt.subplots_adjust(left=0.13, bottom=0.22,
                        right=0.95, top=0.9,
                        wspace=0.24, hspace=.4)

    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12,
               frameon=False, columnspacing=1,
               markerscale=1, handlelength=1.5, bbox_to_anchor=(.54, 0))

    plt.ylabel('One-tail rate', labelpad=8, fontsize=13)
    fp_out = r'../figs_and_tables/Figure_S1D_one-tail_temporal.png'
    plt.savefig(fp_out, dpi=600)
    plt.show()


def plot_key_field(df):
    df['one_tailed_inc'] = df['one_tailed'] & ~df['consistent']
    df['has_tail_inc'] = df['has_tail'] & ~df['consistent']
    df['inconsistent'] = ~df['consistent']

    df_ = df.groupby('doi')[['one_tailed_inc', 'has_tail_inc', 'inconsistent']].mean()
    for subject in PSYC_SUBJECTS_:
        df_[subject] = df.groupby('doi')[subject].first()
    df_['year'] = df.groupby('doi')['year'].first()
    df = df_

    matplotlib.rc('font', **{'size': 12, })
    fig, axs = plt.subplots(2, figsize=(6.5, 6))
    group_keys = ['General', 'Exp. & Cog.', 'Dev. & Edu.', 'Social',
                  'Clinical', 'Applied']

    plt.sca(axs[0])
    plot_key(df, key='inconsistent', ax=axs[0], group_keys=group_keys,
             marker='.', shaded=True, linewidth=1)
    plt.title('Overall misreporting')
    plt.ylabel('Misreport rate')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024], fontsize=12)
    plt.yticks(fontsize=12)

    plt.sca(axs[1])
    plot_key(df, key='one_tailed_inc', ax=axs[1], group_keys=group_keys,
             marker='.', shaded=True, linewidth=1)
    plt.title('One-tailed analysis rate')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.xlabel('Year', fontsize=12, labelpad=0)
    plt.ylabel('One-tail rate', fontsize=12, labelpad=6)
    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024], fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 0.045)

    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    leg = fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12,
                     frameon=False, columnspacing=0.8, handletextpad=0.3,
                     markerscale=2, handlelength=1.5, bbox_to_anchor=(.54, 0))
    for line in leg.get_lines():
        line.set_linewidth(1.5)

    plt.subplots_adjust(left=0.13, bottom=0.18, right=0.95, top=0.9,
                        wspace=0.24, hspace=.4)
    fp_out = r'../figs_and_tables/Figure_S2DE_misreport_field.png'
    plt.savefig(fp_out, dpi=600)
    plt.show()


def calc_over_under():
    fp = fr'../dataframes/df_by_pval_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)

    df = df[df['sign'] == '=']
    df.dropna(subset=['p_implied'], inplace=True)

    df = df[df['p_val'] < .051]

    df['stat_digits'] = df['stat'].apply(get_num_digits)
    df['p_digits'] = df['p_val'].apply(get_num_digits)
    df_yr = df.groupby('year')[['stat_digits', 'p_digits']].mean()
    print(df_yr)

    tqdm.pandas()  # adds pd.Dataframe.progress_apply with a progress bar
    df[['consistent', 'one_tailed', 'z_dif']] = df.progress_apply(
        lambda x: eval_similarity_wiggle(x['p_val'], x['p_implied'], x['stat_type'],
                                         x['stat'], x['df1'], x['df2'], ),
        axis=1, result_type='expand')

    pd.set_option('display.max_rows', 100)
    plot_misreport_hist(df)
    plot_1tail_rate(df)
    plot_misreporting_over_time(df)
    plot_key_field(df)


if __name__ == '__main__':
    calc_over_under()
