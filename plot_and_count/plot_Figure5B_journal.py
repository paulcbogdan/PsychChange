from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from matplotlib import ticker as mtick
from matplotlib.markers import MarkerStyle
from statsmodels.formula import api as smf

from plot_and_count.plot_Figure5C_university import control_vars
from utils import read_csv_fast, PSYC_SUBJECTS_

pd.options.mode.chained_assignment = None
import numpy as np


def make_journal_scatter(key, bias_adjustment=.023):
    np.random.seed(0)
    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    df = df[(df['year'] <= 2024) & (df['year'] >= 2020)]
    df.dropna(subset=['SNIP'], inplace=True)
    df['SNIP'] = df['SNIP'].astype(float)

    df = control_vars(df, ['year', ], target=key)
    if key == 'p_fragile':
        df['p_fragile'] -= bias_adjustment

    j2subjs = {}
    subj2j = defaultdict(list)
    for journal in df['journal'].unique():
        df_j = df[df['journal'] == journal]
        num_subj = 0
        subjs = []
        for subject in PSYC_SUBJECTS_:
            if subject == 'Psychology_Miscellaneous': continue
            num_subj += df_j[subject].any()
            if df_j[subject].any():
                subjs.append(subject)
        j2subjs[journal] = subjs
        subj2j[tuple(subjs)].append(journal)

    df_j = df.groupby('journal')[[key, 'SNIP']].mean()
    df_j['cnt'] = df.groupby('journal')[key].count()
    df_j['SE'] = df.groupby('journal')[key].sem()

    df_j = df_j[df_j['SE'] < .04]
    df_j = df_j[df_j['cnt'] >= 10]

    subj2color = {'General_Psychology': 'tab:blue',
                  'Experimental_and_Cognitive_Psychology': 'tab:orange',
                  'Developmental_and_Educational_Psychology': 'tab:green',
                  'Social_Psychology': 'tab:red',
                  'Clinical_Psychology': 'tab:purple',
                  'Applied_Psychology': 'tab:brown',
                  }

    subj2label = {'General_Psychology': 'General',
                  'Experimental_and_Cognitive_Psychology': 'Exp. & Cog.',
                  'Developmental_and_Educational_Psychology': 'Dev. & Edu.',
                  'Social_Psychology': 'Social',
                  'Clinical_Psychology': 'Clinical',
                  'Applied_Psychology': 'Applied', }

    more_than2_subj = 0
    fig = plt.figure(figsize=(6, 5))
    for subj in subj2j:
        journals = subj2j[subj]
        journals = list(set(journals) & set(df_j.index))
        df_subj = df_j.loc[journals]
        if len(subj) > 2:
            subj = list(subj)
            np.random.shuffle(subj)
            subj = subj[:2]
            more_than2_subj += 1

        if len(subj) == 1:
            plt.scatter(df_subj['SNIP'], df_subj[key],
                        color=subj2color[subj[0]], label=subj2label[subj[0]],
                        alpha=.7)
        elif len(subj) == 2:
            plt.scatter(df_subj['SNIP'], df_subj[key],
                        color=subj2color[subj[0]],
                        marker=MarkerStyle("o", fillstyle="left"),
                        alpha=.7)
            plt.scatter(df_subj['SNIP'], df_subj[key],
                        color=subj2color[subj[1]],
                        marker=MarkerStyle("o", fillstyle="right"),
                        alpha=.7)
    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    plt.xlim(0.5, 3.7)
    print(f'Number of journals with more than two Scopus subjects: '
          f'{more_than2_subj}')

    r, p = stats.spearmanr(df_j[key], df_j['SNIP'])
    if key == 'p_fragile':
        plt.text(2.18, .360, '$r_{' + str(len(df_j) - 2) + '}$ = ' +
                 f'{r:.2f}', fontsize=14,
                 va='top', ha='left')
    elif key == 'p_fragile_implied':
        plt.text(2.18, .333, '$r_{' + str(len(df_j) - 2) + '}$ = ' +
                 f'{r:.2f}', fontsize=14,
                 va='top', ha='left')

    res = smf.ols(formula=f'{key} ~ SNIP', data=df_j).fit()
    df_pred = pd.DataFrame({'SNIP': [0, 4]})
    pred = res.predict(df_pred)
    plt.plot(df_pred['SNIP'], pred, color='black', linestyle='--')
    plt.gca().spines[['right', 'top', ]].set_visible(False)
    plt.xticks([.5, 1, 1.5, 2, 2.5, 3, 3.5], fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Journal SNIP (2020-2024)', fontsize=14)
    plt.gca().set_facecolor('whitesmoke')
    if key == 'p_fragile':
        plt.ylabel('Fragile p-value (%)', fontsize=14, labelpad=8)
    elif key == 'p_fragile_implied':
        plt.ylabel('Fragile implied p-value (%)', fontsize=14, labelpad=8)
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))

    tuples_lohand_lolbl = [plt.gca().get_legend_handles_labels()]
    tolohs = zip(*tuples_lohand_lolbl)
    handles, labels = (sum(list_of_lists, []) for list_of_lists in tolohs)
    leg = fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=12,
                     frameon=False, columnspacing=0.35, handletextpad=-0.2,
                     markerscale=1, handlelength=1.5,
                     scatteryoffsets=[0.55], scatterpoints=1)

    plt.subplots_adjust(left=0.15,
                        bottom=0.15,
                        right=0.985,
                        top=0.925,
                        wspace=0.24,
                        hspace=.36
                        )

    Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)
    if key == 'p_fragile':
        plt.savefig('../figs_and_tables/Figure_5B_SNIP.png', dpi=600)
    elif key == 'p_fragile_implied':
        plt.savefig('../figs_and_tables/Figure_S11B_SNIP_implied.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    make_journal_scatter(key='p_fragile')
    make_journal_scatter(key='p_fragile_implied')
