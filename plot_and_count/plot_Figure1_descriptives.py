from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from make_dataset.make_affiliation_df import get_affiliation_mapper
from utils import read_csv_fast, PSYC_SUBJECTS_, PSYC_SUBJECTS_space

# filter SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def make_p_stackplot():
    fp = r'..\dataframes\df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)

    df = df[df['cond'] != 'all_less0.05']
    # drop papers that always do "p < .05"
    # drop papers that don't report less than below "p < .01"
    point_o1_cond = df['cond'].str.contains('0.01')
    df = df[~point_o1_cond]
    end = ''

    df['n05'] = df[f'n05{end}']
    df['n005_l'] = df[f'n005_l{end}']
    df['n005_h'] = df[f'n005_h{end}']
    df['n001'] = df[f'n001{end}']
    df['num_ps'] = df[f'num_ps{end}']

    df['05'] = df['n05'] / df['num_ps']
    df['005_l'] = df['n005_l'] / df['num_ps']
    df['005_h'] = df['n005_h'] / df['num_ps']
    df['005'] = df['005_l'] + df['005_h']
    df['001'] = df['n001'] / df['num_ps']
    df['insignif'] = 1 - df['05'] - df['005_l'] - df['005_h'] - df['001']
    l05 = []
    l005 = []
    l001 = []
    linsignif = []
    years = []

    for year, df_sy in df.groupby('year'):
        years.append(year)
        l05.append(df_sy['05'].mean())
        l005.append(df_sy['005'].mean())
        l001.append(df_sy['001'].mean())
        linsignif.append(df_sy['insignif'].mean())
    l_props = np.array([l001, l005, l05, linsignif])

    color = '#404040'
    cmap = plt.get_cmap('RdYlGn')
    colors = [cmap(.95), cmap(.5), cmap(.05), (0, 0, 0)]
    for i in range(3):
        colors[i] = list(colors[i])

    colors[0][1] += .1
    colors[2][0] += .1

    plt.stackplot(years, *l_props,
                  labels=['$p < .001$', '$.001 < p < .01$',
                          '$.01 < p < .05$', '$p ≥ .05$'],
                  alpha=0.6, colors=colors,
                  edgecolor=color, linewidth=0.5)

    running = 0
    for l in l_props[:-1]:
        running += l
        plt.scatter(years, running, c=color, s=5, marker='.')
    color = '#202020'

    fs_text = 13
    plt.text(2014, 0.875, 'p > .05', color='w',
             fontsize=fs_text, ha='center')
    plt.text(2014, 0.625, '.01 ≤ p ≤ .05', color=color,
             fontsize=fs_text, ha='center')
    plt.text(2014, 0.403, '.001 ≤ p < .01', color='#202020',
             fontsize=fs_text, ha='center')
    plt.text(2014, 0.14, 'p < .001', color=color, fontsize=fs_text,
             ha='center')

    plt.xticks([2006, 2010, 2014, 2018, 2022])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlim(min(years), max(years))
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.ylabel('P-values (%)')
    plt.gca().set_facecolor('whitesmoke')
    plt.grid(color='lightgray', linestyle='-',
             linewidth=0.5, alpha=0.5)
    plt.title('Distribution of p-values', pad=9)


def plot_year_x_n_papers():
    fp = r'..\dataframes\df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)

    years = []
    mapper_subj = {'Developmental and Educational Psychology':
                       'Dev. & Edu. Psychology',
                   'Experimental and Cognitive Psychology':
                       'Exp. & Cog. Psychology',
                   'Cognitive Neuroscience':
                       'Cog. Neuro.',
                   'Psychology (Miscellaneous)': 'Miscellaneous'}

    group_keys = ['Applied', 'Clinical', 'Dev. & Edu.', 'Exp. & Cog.',
                  'General', 'Social', ]
    d2scores = {k: [] for k in group_keys}

    df['num_areas'] = df[PSYC_SUBJECTS_].sum(axis=1)

    for year, df_sy in df.groupby('year'):
        if year == 2024: continue
        for psych_subject, psych_subject_ in zip(PSYC_SUBJECTS_space,
                                                 PSYC_SUBJECTS_):
            df_g = df_sy[df_sy[psych_subject_] == True]
            df_g['weight'] = 1 / df_g['num_areas']
            score = df_g['weight'].sum()

            if 'miscellaneous' in psych_subject.lower():
                continue
            group = mapper_subj[psych_subject] if psych_subject in mapper_subj \
                else psych_subject
            group = group.replace(' Psychology', '')
            d2scores[group].append(score)
        years.append(year)

    color = '#202020'

    l_scores = [d2scores[k] for k in group_keys]
    plt.stackplot(years, *l_scores, labels=group_keys,
                  alpha=0.6, edgecolor=color, linewidth=0.5, )

    l_scores = np.array(l_scores)
    running = 0
    for l in l_scores:
        running += l
        plt.scatter(years, running, c=color, s=5, marker='.')

    handles, labels = plt.gca().get_legend_handles_labels()

    legend = plt.gca().legend(handles[:4][::-1], labels[:4][::-1],
                              loc='upper left', bbox_to_anchor=[0., 1.023],
                              fontsize=12., ncol=1, borderpad=0.4,
                              borderaxespad=0.3, frameon=False)
    plt.gca().add_artist(legend)

    plt.gca().legend(handles[4:][::-1], labels[4:][::-1], loc='upper left',
                     bbox_to_anchor=[0.355, 1.023], fontsize=12., ncol=1,
                     borderpad=0.4, borderaxespad=0.3, frameon=False)

    plt.xticks([2006, 2010, 2014, 2018, 2022])

    plt.ylabel('Papers (#)', )

    plt.gca().set_facecolor('whitesmoke')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.xlim(2004, 2023)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.title('Papers collected by year', pad=9)


def hist_num_ps():
    fp = r'..\dataframes\df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    plt.hist(df['num_ps'], range=(1, 100), bins=100,
             color='dodgerblue', alpha=0.6)
    plt.xticks([1, 20, 40, 60, 80, 100])
    plt.gca().set_facecolor('whitesmoke')
    plt.ylabel('Papers (#)')
    plt.xlabel('P-values (#)')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.25)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xlim(1, 100)
    plt.title('Numbers of p-values', pad=9)


def citations_histogram():
    fp = r'..\dataframes\df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    df = df[df['year'] < 2024]  # NaN cites_year value
    df['year_dif'] = 2024 - df['year']
    df['cites_year'] = df['cites'] / df['year_dif']

    plt.hist(df['cites_year'], range=(0, 25), bins=25, color='dodgerblue',
             alpha=0.6, log=False, label='Cites', density=False)

    plt.xticks([0, 5, 10, 15, 20, 25])

    plt.gca().set_facecolor('whitesmoke')
    plt.ylabel('Papers (#)')
    plt.xlabel('Citations / year')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.25)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xlim(0, 25)
    plt.title('Numbers of citations', pad=9)


def plot_university_x_research_score():
    fp = r'..\dataframes\df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    affs_in = set(df['school'].dropna().unique())
    print(f'Number of affs: {len(affs_in)=}')

    aff2scores, re_str, score_columns, affs_no_parenthesis_map = (
        get_affiliation_mapper())

    df_aff = pd.DataFrame(aff2scores).T
    df_aff.sort_values('rank_2024', inplace=True)
    df_aff['rank_2024'] = np.arange(1, 1001)

    d_out = {'rank': [], 'research': []}
    d_in = {'rank': [], 'research': []}
    for idx, row in df_aff.iterrows():
        if idx in affs_in:
            # print('in')
            d_in['rank'].append(-row['rank_2024'])
            d_in['research'].append(row['score_2024'])
        else:
            d_out['rank'].append(-row['rank_2024'])
            d_out['research'].append(row['score_2024'])

    df_grp_in = pd.DataFrame(d_in)
    df_grp_in['color'] = 'dodgerblue'
    df_grp_out = pd.DataFrame(d_out)
    df_grp_out['color'] = 'red'
    assert len(df_grp_in) + len(df_grp_out) == 1000

    df_grp_both = pd.concat([df_grp_in, df_grp_out])

    plt.scatter(df_grp_both['rank'].values, df_grp_both['research'].values,
                s=0.03, alpha=0.3, color=df_grp_both['color'])
    df_grp_both.sort_values('rank', inplace=True)
    for _, row in df_grp_both.iterrows():
        plt.plot([row['rank'], row['rank']], [0, row['research']],
                 color=row['color'], linewidth=0.1,
                 alpha=0.9 if row['color'] != 'red' else 0.75)

    plt.xticks([-1000, -750, -500, -250, -1],
               labels=['#1000', '#750',
                       '#500', '#250', '#1'],
               fontsize=13)
    plt.xlim(-1000, 1)
    plt.ylim(0, 100)
    plt.xlabel('Times Higher Ed. Ranking')
    plt.ylabel('Research score')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='dodgerblue',
                              label=f'Found ({len(affs_in)})', linewidth=0,
                              alpha=.75),
                       Line2D([0], [0], marker='o', color='red',
                              label=f'Missing ({1000 - len(affs_in)})',
                              linewidth=0, alpha=.75)]
    plt.legend(handles=legend_elements, loc='upper left', frameon=False,
               handletextpad=0, fontsize=12)

    plt.gca().set_facecolor('whitesmoke')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.25)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.title('University ranking')
    plt.tight_layout()


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)

    matplotlib.rc('font', **{'size': 12, })
    fig, axs = plt.subplots(3, 6, figsize=(8.5, 9))
    gs = axs[0, 0].get_gridspec()
    gs = axs[0, 0].get_gridspec()

    for ax in axs[0, :]:
        ax.remove()
    for ax in axs[1, :]:
        ax.remove()
    for ax in axs[2, :]:
        ax.remove()
    #
    axbig = fig.add_subplot(gs[0, 1:5])
    plot_year_x_n_papers()
    axbig = fig.add_subplot(gs[1, :3])
    hist_num_ps()
    axbig = fig.add_subplot(gs[1, 3:])
    make_p_stackplot()
    axbig = fig.add_subplot(gs[2, :3])
    citations_histogram()

    axbig = fig.add_subplot(gs[2, 3:])
    plot_university_x_research_score()

    Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)

    plt.savefig('../figs_and_tables/Figure_1_dataset_descriptives.png', dpi=600)
    plt.show()
