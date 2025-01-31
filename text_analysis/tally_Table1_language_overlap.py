from pathlib import Path

import numpy as np
import pandas as pd

from text_analysis.plot_Figure6_language import make_word_ar, get_base_kwargs, get_word_scores, pad_blanks


def save_overlap_table(df, col, n_col=7, report_only=False):
    pd.set_option('display.max_rows', 50000)
    df_con = df[df['consistent'] == True]
    print(f'Number of {col} words consistent with p-val effect: {len(df_con)=}')
    df_con_p = (df_con[df_con['coef_col'] > 0].
                sort_values('coef_col', ascending=False))
    df_con_n = (df_con[df_con['coef_col'] < 0].
                sort_values('coef_col', ascending=True))
    p_title = (['Positively linked with ..., '
                'consistently linked with fragile p-values'] +
               [''] * (n_col - 1))
    con_p_ar = make_word_ar(df_con_p, n_col)

    n_title = (['Negatively linked with ..., '
                'consistently linked with fragile p-values'] +
               [''] * (n_col - 1))
    con_n_ar = make_word_ar(df_con_n, n_col)
    blank = [''] * n_col
    df_con = pd.DataFrame([p_title] + con_p_ar + [blank] + [n_title] + con_n_ar,
                          columns=list(range(n_col)))

    if col == 'target_score':
        Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)
        df_con.to_csv(fr'../figs_and_tables'
                      r'/Table_1_and_S2_prestige_consistent.csv', index=False,
                      header=False)
    else:
        fp_out = fr'../figs_and_tables/SuppMat_language_tables' \
                 fr'/Table_SuppMat_{col}_consistent.csv'
        Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
        df_con.to_csv(fp_out, index=False, header=False)

    df_inc = df[(df['consistent'] == False) & (~pd.isna(df['coef_p']))]
    print(f'Number of {col} words inconsistent with p-val effect: '
          f'{len(df_con)=}')
    df_inc_p = (df_inc[df_inc['coef_col'] > 0].
                sort_values('coef_col', ascending=False))
    ip_title = (['Positively linked with ..., '
                 'inconsistently linked with fragile p-values'] +
                [''] * (n_col - 1))
    inc_p_ar = make_word_ar(df_inc_p, n_col)
    df_inc_n = (df_inc[df_inc['coef_col'] < 0].
                sort_values('coef_col', ascending=True))
    in_title = (['Negatively linked with ..., '
                 'inconsistently linked with fragile p-values'] +
                [''] * (n_col - 1))
    inc_n_ar = make_word_ar(df_inc_n, n_col)
    blank = [''] * n_col
    df_inc = pd.DataFrame([ip_title] + inc_p_ar + [blank] + [in_title] +
                          inc_n_ar, columns=list(range(n_col)))
    if col == 'target_score':
        df_inc.to_csv(fr'../figs_and_tables'
                      r'/Table_S3_prestige_inconsistent.csv', index=False,
                      header=False)
    else:
        fp_out = fr'../figs_and_tables/SuppMat_language_tables' \
                 fr'/Table_SuppMat_{col}_inconsistent.csv'
        df_inc.to_csv(fp_out, index=False, header=False)

    # Not reported because there are plenty of tables already...
    if report_only:
        df_only_col = df[pd.isna(df['coef_p'])]
        print(f'Number of {col} words significant with no p-val effect: '
              f'{len(df_con)=}')
        df_col_p = (df_only_col[df_only_col['coef_col'] > 0].
                    sort_values('coef_col', ascending=False))
        col_p_title = (['Positively linked with ..., no fragile p-values link'] +
                       [''] * (n_col - 1))
        col_p_ar = make_word_ar(df_col_p, n_col)
        df_col_n = (df_only_col[df_only_col['coef_col'] < 0].
                    sort_values('coef_col', ascending=True))
        col_n_title = (['Negatively linked with ..., no fragile p-values link'] +
                       [''] * (n_col - 1))
        col_n_ar = make_word_ar(df_col_n, n_col)
        df_col_only = pd.DataFrame([col_p_title] + col_p_ar + [blank] +
                                   [col_n_title] + col_n_ar,
                                   columns=list(range(n_col)))
        if col == 'target_score':
            fp_out = fr'../figs_and_tables/SuppMat_language_tables' \
                     fr'/Table_SuppMat_prestige_only.csv'
            df_inc.to_csv(fp_out, index=False, header=False)
        else:
            fp_out = fr'../figs_and_tables/SuppMat_language_tables' \
                     fr'/Table_SuppMat_{col}_only.csv'
            df_inc.to_csv(fp_out, index=False, header=False)


def make_overlaps_table(stat_type='all', p_implied=False, single_regr=True,
                        num_bars=60, nonconsistent=False, final=True, ):
    if final:
        kwargs = get_base_kwargs(analysis_name='paper_sum_reg',
                                 include_SNIP=True, single_regr=single_regr,
                                 p_implied=p_implied, whole_paper_stats=True,
                                 num_words=2500, regress_specific='p_key'
                                 )

        stat_type = 'all'
        # kwargs['method'] = 'holm-sidak'
        # kwargs['alpha'] = .05

        kwargs['method'] = 'fdr_bh'
        kwargs['alpha'] = .05
    elif p_implied:
        kwargs = get_base_kwargs(analysis_name='paper_sum_reg',
                                 include_SNIP=True, single_regr=single_regr,
                                 p_implied=True, whole_paper_stats=True,
                                 num_words=2500,
                                 regress_specific='p_key'
                                 )
    else:
        raise ValueError

    kwargs['stat_type'] = stat_type
    cols = ['year', 'SNIP', 'log_cites_year_z', 'target_score', ]
    col2expected_sign = {'year': 1, 'log_cites_year_z': 1,
                         'target_score': -1, 'SNIP': 1}

    words_plot = []
    coefs_plot = []
    for col in cols:
        print(f'---- {col} ----')
        words_col, coefs_col = get_word_scores([col], 2500, do_pad=False,
                                               **kwargs)

        df_col = pd.DataFrame({'word': words_col[0], 'coef_col': coefs_col[0]})
        df_col.drop_duplicates('word', inplace=True)  # head and tail duped
        words_p, coefs_p = get_word_scores([f'{col}_rp_key'], 2500,
                                           do_pad=False, **kwargs)
        df_p = pd.DataFrame({'word': words_p[0], 'coef_p': coefs_p[0]})
        df_p.drop_duplicates('word', inplace=True)
        words_overlap = set(df_col['word']).intersection(set(df_p['word']))
        words_overlap -= {''}

        df_col['overlap'] = df_col['word'].isin(words_overlap)
        df_p = df_p[df_p['word'].isin(words_overlap)]
        df = df_col.merge(df_p, on='word', how='left')

        pd.set_option('display.max_rows', 50000)

        df['consistent'] = (np.sign(df['coef_p'] * df['coef_col']) ==
                            col2expected_sign[col])
        assert df['consistent'].isna().sum() == 0

        df['better_p'] = df['coef_p'] > 0
        df['worse_p'] = df['coef_p'] < 0
        if nonconsistent:
            df['consistent'] = ~df['consistent']
            df['worse_p'] = ~df['worse_p']
            df['better_p'] = ~df['better_p']

        if col == 'target_score':
            df['better_p_consistent'] = df['consistent'] & df['worse_p']
            df['worse_p_consistent'] = df['consistent'] & df['better_p']
        else:
            df['worse_p_consistent'] = df['consistent'] & df['worse_p']
            df['better_p_consistent'] = df['consistent'] & df['better_p']

        save_overlap_table(df, col)
        df = df[df['overlap']]

        df_worse_p = df[df['worse_p_consistent']]
        df_worse_p = pad_blanks(df_worse_p, 'coef_col', num_bars)
        df_worse_p.sort_values('coef_col', inplace=True, ascending=False)
        tail_words = df_worse_p['word'].tail(num_bars)
        tail_coefs = df_worse_p['coef_col'].tail(num_bars)
        df_better_p = df[df['better_p_consistent']]
        df_better_p = pad_blanks(df_better_p, 'coef_col', num_bars)
        df_better_p.sort_values('coef_col', inplace=True, ascending=False)
        head_words = df_better_p['word'].head(num_bars)
        head_coefs = df_better_p['coef_col'].head(num_bars)
        head_words = list(head_words)
        tail_words = list(tail_words)
        head_coefs = list(head_coefs)
        tail_coefs = list(tail_coefs)
        words_plot.append(head_words + tail_words)
        coefs_plot.append(head_coefs + tail_coefs)

        p_consistent = df['consistent'].mean()

        print(f'{col} {len(words_overlap)=} ({p_consistent:.1%})')
        print([x for x in df_better_p['word'].head(1000) if x != ''])
        print([x for x in df_worse_p['word'].tail(1000) if x != ''])


if __name__ == '__main__':
    make_overlaps_table()
