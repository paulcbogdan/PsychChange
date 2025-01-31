from pathlib import Path

import pandas as pd

from utils import read_csv_fast


def get_general_cond(cond):
    if 'all_less_mixed' in cond:
        return 'all_less_mixed'
    elif 'all_less' in cond:
        return 'all_less'
    elif 'equal_less' in cond:
        return 'equal_less'
    elif 'all_equal' in cond:
        return 'all_equal'
    elif 'eclectic' in cond:
        return 'eclectic'
    elif 'no_sig' in cond:
        return 'no_sig'
    return pd.NA


def get_general_cutoff(cutoff):
    if pd.isna(cutoff):
        return pd.NA
    elif cutoff > .01:
        return .05
    elif cutoff > .001:
        return .01
    elif cutoff > .0001:
        return .001
    else:
        return .0001


def count_pval_reporting_style():
    pd.set_option('display.max_rows', None)

    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    df['p_fragile'] = df['p_fragile_orig']  # before overriding with .55
    df['cond_gen'] = df['cond'].apply(get_general_cond)
    df['cutoff_gen'] = df['p_cutoff'].apply(get_general_cutoff)

    cond_names = ['all_less', 'all_less_mixed', 'equal_less', 'all_equal']
    cutoffs = [.05, .01, .001, .0001]
    tups = []
    for cond_name in cond_names:
        for cutoff in cutoffs:
            tups.append((cond_name, cutoff))
    tups.append(('eclectic', pd.NA))

    cnt = df[['cond_gen', 'cutoff_gen']].value_counts(
        normalize=False, dropna=False)
    p_implied = (df.groupby(['cond_gen', 'cutoff_gen'], dropna=False)
                 ['p_fragile_implied'].mean())
    p_fragile = (df.groupby(['cond_gen', 'cutoff_gen'], dropna=False)
                 ['p_fragile'].mean())
    df_out = pd.concat([cnt, p_implied, p_fragile, ], axis=1)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    df_out['p_implied_count'] = (
        df.groupby(['cond_gen', 'cutoff_gen'],
                   dropna=False)['p_fragile_implied'].count())

    # check that the total adds up to 1
    #   (account for float precision; maybe only relevant to older Python)

    def n2str(count, n):
        prop = count / n
        return f'{prop:.1%}'

    APA_cols = [('all_less', .001), ('all_less', .0001),
                ('equal_less', .001), ('equal_less', .0001),
                ('all_equal', .05), ('all_equal', .01), ('all_equal', .001),
                ('all_equal', .0001)]
    df_APA = df_out.loc[APA_cols]
    APA_cnt = df_APA['count'].sum()
    APA_p = df_APA['count'].sum() / df_out['count'].sum()
    print(f'Total number of papers: {df_out["count"].sum()}')
    print(f'\tAPA+ style count = {APA_cnt} ({APA_p:.1%})')

    df_out['percentage'] = df_out['count'].apply(lambda x: n2str(x, len(df)))
    assert df_out['count'].sum() == len(df) == 240_355

    df_out = df_out.reindex(tups).reset_index()
    df_out['p_fragile'] = df_out['p_fragile'].apply(lambda x: f'{x:.1%}')
    df_out['p_fragile_implied'] = df_out['p_fragile_implied'].apply(
        lambda x: f'{x:.1%}')
    df_out['has_implied_percent'] = df_out['p_implied_count'] / df_out['count']
    df_out['has_implied_percent'] = df_out['has_implied_percent'].apply(
        lambda x: f'{x:.1%}')
    print(df_out)

    cols = ['cond_gen', 'cutoff_gen', 'count', 'percentage',
            'p_fragile', 'p_fragile_implied', 'p_implied_count',
            'has_implied_percent']
    df_out = df_out[cols]

    mapper = {'cond_gen': 'Style', 'cutoff_gen': 'Cutoff',
              'percentage': '(%)', 'count': 'Count',
              'p_fragile': 'P fragile\npercentage',
              'p_fragile_implied': 'P-implied fragile\npercentage',
              'p_implied_count': 'Implied-p\ncount',
              'has_implied_percent': 'Has implied\npercentage'}
    df_out.rename(columns=mapper, inplace=True)

    style_mapper = {'all_less': 'All less', 'all_less_mixed': 'All less mixed',
                    'equal_less': 'Equal less', 'all_equal': 'All equal',
                    'eclectic': 'Eclectic', 'no_sig': 'No sig'}
    df_out['Style'] = df_out['Style'].map(style_mapper)

    Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)
    fp_out = fr'../figs_and_tables/Table_S1_pval_survey.csv'
    df_out.to_csv(fp_out, index=False)


if __name__ == '__main__':
    count_pval_reporting_style()
