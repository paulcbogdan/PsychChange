from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

from utils import read_csv_fast

pd.options.mode.chained_assignment = None


def get_word_scores(cols, num_bars, do_pad=True, do_logistic=False,
                    include_SNIP=False, num_words=2000, single_regr=True,
                    paper_any=False, stat_type='t', single_frag=False,
                    p_implied=False, regress_specific=None,
                    control_subject=True, whole_paper_stats=True,
                    method='holm-sidak', alpha=.05):
    signif_str = '_signif' if True else ''
    nwords_str = f'_nw{num_words}'
    drop_dup_str = '_drop_dup' if True else ''
    stat_type_str = f'_{stat_type}' if stat_type != 'all' else ''
    p_implied_str = f'_p_implied' if p_implied else ''

    w_logit = '_logit' if do_logistic else ''
    jif_str = '_SNIP' if include_SNIP else ''
    single_str = '_single' if single_regr else ''
    paper_any_str = '_paper_any' if paper_any else ''
    is_frag_str = '_single_frag' if single_frag else ''
    ctrl_subject_str = '_ctrl_subj' if control_subject else ''
    regress_specific_str = f'_R{regress_specific}' if regress_specific else ''
    whole_paper_str = '_whole' if whole_paper_stats else ''

    fp_in = (fr'../dataframes/word_regr/df_word_regr_'
             fr'{w_logit}{nwords_str}{signif_str}{jif_str}{single_str}'
             fr'{paper_any_str}{stat_type_str}{is_frag_str}{drop_dup_str}'
             fr'{ctrl_subject_str}{regress_specific_str}{p_implied_str}'
             fr'{whole_paper_str}_Jan21.csv')

    if p_implied:
        cols_ = []
        for col in cols:
            if col == 'p_fragile_if_stat': col = 'p_fragile_implied'
            cols_.append(col)
        cols = cols_

    df = read_csv_fast(fp_in, check_dup=False)
    df = deepcopy(df)

    ts = [f't_{col}' for col in cols]
    coefs = [f'coef_{col}' for col in cols]

    df.dropna(subset=coefs, inplace=True)

    pd.set_option('display.max_rows', 50000)
    pd.set_option('display.width', 1000)

    non_words = {'fi', 'tl', 'tli', 'cf', 'rmse', 'tl', 'srm', 'cfi',
                 'rmsea', 'tli', 'srmr', 'df', 'cfa',
                 'prep'}  # prep is generally "p" with a "rep" subscript?

    df = df[df['word'].apply(lambda x: x not in non_words)]
    df['word'] = df['word'].str.replace('_', '')
    map_ha = {'ha': 'has', 'thi': 'this'}
    df['word'] = df['word'].apply(lambda x: map_ha.get(x, x))

    # drop words associated with nationality
    country_words = ['dutch', 'chinese', 'english', 'japanese']
    df = df[~df['word'].isin(country_words)]

    df = df.sort_index()

    words_l = []
    coefs_l = []
    for i, (coef, col, t) in enumerate(zip(coefs, cols, ts)):
        df[f'{col}_p'] = stats.norm.sf(df[t].abs()) * 2
        df.dropna(subset=[t], inplace=True)
        _, df[f'{col}_corr_p'], _, _ = multipletests(df[f'{col}_p'],
                                                     method=method)

        pre_len = len(df)

        df_col = df[df[f'{col}_corr_p'] < alpha]

        print(f'{col}: Number of ts that are signif: {len(df_col)} '
              f'({len(df_col) / pre_len:.1%})')

        if 'p_key' in coef:
            df_col[coef] *= -1
        if not do_logistic:
            df_col[coef] /= df_col['prop']
            df_col[coef] *= 100
        else:
            df_col[coef] = np.exp(df_col[coef]) - 1
            df_col[coef] *= 100

        if do_pad:
            df_col = pad_blanks(df_col, coef, num_bars)
        df_col.sort_values(coef, inplace=True, ascending=False)
        head_words = df_col['word'].head(num_bars).values
        tail_words = df_col['word'].tail(num_bars).values
        head_coef = df_col[coef].head(num_bars).values
        tail_coef = df_col[coef].tail(num_bars).values
        both_coefs = list(head_coef) + list(tail_coef)
        both_words = list(head_words) + list(tail_words)

        words_l.append(both_words)
        coefs_l.append(both_coefs)

    return words_l, coefs_l


def make_word_ar(df, n_col=7):
    words, col_coef, p_coef = (df['word'].to_list(), df['coef_col'].to_list(),
                               df['coef_p'].to_list())
    con_ar = []
    con_row = []
    while len(words):
        if len(con_row) == n_col:
            con_ar.append(con_row)
            con_row = []
        word = words.pop(0)
        if len(word) > 11:
            word = word[:10] + '.'
        c = col_coef.pop(0)
        p = p_coef.pop(0)
        s = f'{word}'
        con_row.append(s)
    if len(con_row):
        con_ar.append(con_row)
    return con_ar


def pad_blanks(df, col_name, num_bars):
    df = pd.concat([df,
                    pd.DataFrame({'word': [''] * num_bars * 2,
                                  'prop': [0] * num_bars * 2,
                                  col_name: [0] * num_bars * 2, })])
    return df


def get_base_kwargs(analysis_name='paper_any_logit', include_SNIP=False,
                    single_regr=True, p_implied=False, regress_specific=False,
                    whole_paper_stats=False, num_words=5000,
                    control_subject=False, ):
    kwargs = {'include_SNIP': include_SNIP,
              'num_words': num_words,
              'single_regr': single_regr,
              'p_implied': p_implied,
              'regress_specific': regress_specific,
              'whole_paper_stats': whole_paper_stats,
              'control_subject': control_subject,
              }

    if analysis_name == 'single_p_logit':
        kwargs['do_logistic'] = True
        kwargs['single_frag'] = True
        kwargs['paper_any'] = False
    elif analysis_name == 'paper_any_logit':
        kwargs['do_logistic'] = True
        kwargs['single_frag'] = False
        kwargs['paper_any'] = True
    elif analysis_name == 'paper_sum_reg':
        kwargs['do_logistic'] = False
        kwargs['single_frag'] = False
        kwargs['paper_any'] = False
    else:
        raise ValueError

    return kwargs


def plot_4types(single_regr=True, p_implied=False, final=True,
                alpha=.05):
    if final:
        kwargs = get_base_kwargs(analysis_name='paper_sum_reg',
                                 include_SNIP=True, single_regr=single_regr,
                                 p_implied=p_implied, whole_paper_stats=False,
                                 num_words=2500,
                                 )
    else:
        raise ValueError

    words_l, coefs_l = [], []
    cols = ['p_key']
    num_bars = 50
    stat_types = ['t', 'F', 'chi', 'rB']
    for stat_type in stat_types:
        kwargs['stat_type'] = stat_type
        words_, coefs_ = get_word_scores(cols, num_bars, method='holm-sidak',
                                         alpha=alpha, **kwargs, )
        words_l.extend(words_)
        coefs_l.extend(coefs_)

    cols = stat_types
    fp = f'../figs_and_tables/Figure_6_language_fragile_p-values.png'
    plot_regr_data(words_l, coefs_l, cols, do_logistic=False,
                   green=1., red=.0, fp=fp)


def plot_regr_data(words_l, coefs_l, cols, do_logistic, top=.96,
                   green=.99, red=.00, fp=None):
    col2title = {'p_fragile_if_stat': 'Strong p-values',
                 'year': 'Year',
                 'target_score': 'Prestige',
                 'log_cites_year_z': 'Citations',
                 'jif': 'Journal impact factor',
                 'SNIP': 'SNIP',
                 'p_key': 'Strong p-values',
                 't': 't-values', 'F': 'F-values',
                 'chi': 'Chi-squared values',
                 'rB': 'Correlations &\nregressions'}

    fig, axs = plt.subplots(1, len(cols), figsize=(8, 11.5))
    half_boost = 0.5
    word_shift = -.1
    ylim = 80
    plt.rcParams["font.family"] = "Arial"
    cmap = plt.get_cmap('coolwarm_r')
    green = cmap(green)
    # green = 'blue'
    red = cmap(red)

    black_words = ['repeatedmeasures', 'asd', 'multivariate', 'hierarchical',
                   'validity', 'priming', 'genotype', 'completer', 'infant',
                   'intervention', 'pupil', 'cortisol', 'amplitude', 'gyrus',
                   'betweengroup', 'moderated', 'ancova', 'left', 'right',
                   'hemisphere', 'sex']

    for i, (both_words, both_coefs, col) in enumerate(zip(words_l, coefs_l,
                                                          cols)):
        both_words = both_words[::-1]
        both_coefs = both_coefs[::-1]

        plt.sca(axs[i])
        num_bars = len(both_words) // 2

        ypos = list(range(len(both_words)))
        for j in range(num_bars):
            ypos[j + num_bars] += half_boost

        head_words = both_words[:len(both_words) // 2]
        tail_words = both_words[len(both_words) // 2:]
        fs = 9.5 if num_bars == 50 else 8.5
        fs_over = 8 if num_bars == 50 else 7

        for y, word in enumerate(head_words):
            plt.text(ylim / 20, y, word,
                     color='k' if word in black_words else red,
                     va='center', ha='left',
                     )

            if both_coefs[y] < -ylim:
                plt.text(-ylim * 1.02, y + word_shift,
                         f'{both_coefs[y]:.0f}',
                         color='k', va='center', ha='right', fontsize=fs_over)

        for y, word in enumerate(tail_words):

            plt.text(-ylim / 20, y + num_bars + half_boost + word_shift, word,
                     color='k' if word in black_words else green,
                     va='center', ha='right', fontsize=fs)

            if both_coefs[y + num_bars] > ylim:
                plt.text(ylim * 1.02, y + num_bars + half_boost + word_shift,
                         f'{both_coefs[y + num_bars]:.0f}',
                         color='k', va='center', ha='left', fontsize=fs_over)

        plt.plot([-1, 1], [num_bars + .5 - 1 + half_boost / 2,
                           num_bars + .5 - 1 + half_boost / 2], color='black',
                 lw=1)

        # Just picking some numbers to divide and add so it looks nice
        color_red = cmap(np.array(both_coefs[:len(both_coefs) // 2]) / 180
                         + 0.39)
        color_green = cmap(np.array(both_coefs[len(both_coefs) // 2:]) / 130
                           + 0.6)

        plt.barh(ypos, both_coefs, align='center',
                 color=list(color_red) + list(color_green),
                 linewidth=0.5, edgecolor='k', alpha=.9, height=0.75)

        plt.xlim(-ylim, ylim)
        plt.ylim(-1, 2 * num_bars)
        plt.gca().spines[['right', 'top', 'left']].set_visible(False)
        plt.plot([0, 0], [-1, 2 * num_bars], color='black', lw=1)
        plt.yticks([])
        if ylim == 50:
            plt.xticks([-40, -20, 0, 20, 40],
                       ['-0.4', '-0.2', '0', '0.2', '0.4'])
        if do_logistic:
            plt.xlabel('Odds ratio')
        else:
            plt.xlabel('Difference in usage (%)')
        plt.title(col2title[col])

    plt.subplots_adjust(left=0.025,
                        bottom=0.04,
                        right=0.975,
                        top=top,
                        wspace=0.1,
                        hspace=1.)
    if fp is not None:
        plt.savefig(fp, dpi=600)
    plt.show()


if __name__ == '__main__':
    plot_4types()
