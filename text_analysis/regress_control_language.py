import sys

import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

from text_analysis.build_df_word import get_word_lists, clean_words, prune_to_stat_type
from text_analysis.mass_regressions import clean_df_words
from utils import get_formula_cols
from utils import pickle_wrap, read_csv_fast

if __name__ == '__main__':
    # Honestly, this regression with 2500+ predictors has crashed my computer
    #   before. Maybe it's a skill issue on my part.
    # Either way, fingers crossed you have good RAM

    num_words = 2500
    stat_type = 'all'
    p_implied = False
    nwords_str = f'_nw{num_words}'
    stat_type_str = f'_{stat_type}' if stat_type != 'all' else ''
    p_implied_str = f'_p_implied' if p_implied else ''
    fp_top2500 = (f'../cache/get_word_lists'
                  f'{stat_type_str}{p_implied_str}{nwords_str}.pkl')
    word_counter, words_above, hyphenwords = (
        pickle_wrap(get_word_lists, filepath=fp_top2500,
                    kwargs={'num_words': num_words, 'stat_type': stat_type,
                            'p_implied': p_implied, },
                    easy_override=False))
    words_above, word_counter = clean_words(word_counter, words_above)

    signif_str = '_signif' if True else ''
    nwords_str = f'_nw{num_words}'
    drop_dup_str = '_drop_dup' if True else ''
    stat_type_str = f'_{stat_type}' if stat_type != 'all' else ''
    p_implied_str = f'_p_implied' if p_implied else ''

    fp_words = (fr'../dataframes/df_words/df_'
                fr'{nwords_str}{stat_type_str}{p_implied_str}'
                fr'_Jan21.csv')

    df_result_words = read_csv_fast(fp_words, easy_override=False,
                                    check_dup=False)
    df_result_words = prune_to_stat_type(df_result_words, stat_type)
    df_result_words = clean_df_words(df_result_words)

    assert np.sum(df_result_words[words_above].isna().sum()) == 0

    do_logistic = False
    paper_any = False
    whole_paper_stats = True

    fp_combined = fr'../dataframes/df_combined_pruned_Jan21.csv'
    df = read_csv_fast(fp_combined, easy_override=False)
    df = df[df['doi_str'].isin(set(df_result_words['doi_str'].to_list()))]
    df = df[df['cond'] != 'all_less0.05']
    df_result_words = df_result_words[df_result_words['doi_str'].isin(
        set(df['doi_str'].to_list()))]
    analysis_cols = ['year', 'target_score', 'log_cites_year_z', 'SNIP']

    if do_logistic and not paper_any:
        assert not whole_paper_stats
        df_result_words['num_words'] = df_result_words[words_above].sum(axis=1)
        df = df_result_words.merge(df[analysis_cols + ['doi_str', 'num_ps']],
                                   on='doi_str')
        df['p_key'] = df['p_is_fragile_implied'] if p_implied else (
            df)['p_is_fragile']

        print(f'Merged... : {len(df)=}')

        df['i_num_ps'] = 1 / df['num_ps']
        assert np.sum(df['i_num_ps'].isna().sum()) == 0
    else:
        if paper_any:
            assert do_logistic
            df_paper_words = df_result_words.groupby('doi_str').any()
            df_paper_words['num_words'] = df_paper_words[words_above].sum(axis=1)
        else:
            df_paper_words = df_result_words.groupby('doi_str').sum()
            df_paper_words['num_words'] = df_paper_words[words_above].sum(axis=1)
            normed = (df_paper_words[words_above].values /
                      df_paper_words['num_words'].values[:, None])
            df_paper_words[words_above] = normed
        imp_str = '_implied' if p_implied else ''
        if not whole_paper_stats:
            df_result_words['p_key'] = df_result_words[f'p_is_fragile{imp_str}']
            df_p = df_result_words.groupby('doi_str')[['p_key']].mean()
            df_p.reset_index(inplace=True)
            df = df.merge(df_p, on='doi_str')

        df = df_paper_words.merge(df[analysis_cols +
                                     ['doi_str', 'p_fragile',
                                      'p_fragile_implied', 'doi'] +
                                     (['p_key'] if 'p_key' in df.columns else
                                      [])], on='doi_str')
        if whole_paper_stats:
            df['p_key'] = df[f'p_fragile{imp_str}']
        pre_len = len(df)
        df.dropna(subset=['p_key'], inplace=True)  # for p_implied
        if not p_implied: assert len(df) == pre_len
        df['p_key'] = df['p_key'].astype(float)

    print(f'{len(df)=}')

    formula = (f'p_key ~ target_score + '
               + ' + '.join(words_above))

    sys.setrecursionlimit(100_000)  # needed for this big word regression

    cols = get_formula_cols(df, formula)
    for col in cols:
        n_nan = df[col].isna().sum()
        df[col] = df[col].astype(float)
        df[col] = stats.zscore(df[col], nan_policy='omit')

        print(f'{col} {n_nan=}')
        if df[col].isna().sum() > 10_000:
            formula = formula.replace(' + ' + col, '')
            continue

    df.dropna(subset=cols, inplace=True)

    res = smf.ols(formula=formula, data=df).fit()
    print(res.summary())

    print(res.params['target_score'])

    formula = (f'p_key ~ target_score')
    res = smf.ols(formula=formula, data=df).fit()
    print(res.summary())
    print(res.params['target_score'])
