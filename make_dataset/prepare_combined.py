from collections import defaultdict

import numpy as np
import pandas as pd

from utils import read_csv_fast, save_csv_pkl

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None


def interpolate_journal(df, key='p_fragile'):
    df_j = df.groupby(['journal_clean', 'year'])[key].mean()
    p_na = pd.isna(df[key])
    p_na = p_na & df['journal_clean'].notna() & df['year'].notna()
    df[f'{key}_j'] = df[key]
    df.loc[p_na, f'{key}_j'] = df[p_na].apply(
        lambda row: df_j[row['journal_clean'], row['year']], axis=1)
    return df


def add_jrl_cnt(df, limit=5):
    df_j = df[df['has_results'] == True].groupby(
        ['journal_clean', 'year'])['doi_str'].count()
    df['jrl_cnt'] = df.apply(
        lambda row: df_j[row['journal_clean'], row['year']] if
        (row['journal_clean'], row['year']) in df_j else np.nan, axis=1)
    df['has_results_j'] = df['has_results']
    df.loc[df['jrl_cnt'] < limit, 'has_results_j'] = False


def get_non_ps_df(all_aff=False):
    fp_aff = fr'../dataframes/df_affiliation_Aug24.csv'
    df_aff = read_csv_fast(fp_aff)

    # made many affiliation columns, lets just pick the ones we care about

    if all_aff:
        aff_types = ['Random', 'TargetMax', 'TargetMin', 'TargetMed',
                     'YearMax', 'YearMin', 'YearMed',
                     'Mean', 'Mode']
        col_specific = ['school', 'country',
                        'target_rank', 'target_score',
                        'year_rank', 'year_score', ]
        cols = []
        for aff_type in aff_types:
            for col in col_specific:
                if aff_type == 'Mean' and col in ['school', 'country']:
                    continue
                cols.append(f'{aff_type}_{col}')
        cols += ['doi_str', 'num_affil']

        df_aff = df_aff[cols]
    else:
        cols = ['Mode_school', 'Mode_country',
                'Mode_year_rank', 'Mode_target_rank',
                'Mode_year_score', 'Mode_target_score',
                'doi_str', 'num_affil']
        df_aff = df_aff[cols]
        mapper = {col: col.replace('Mode_', '') for col in cols}
        df_aff.rename(columns=mapper, inplace=True)

    df_aut = read_csv_fast(f'../dataframes/df_author_ages_Aug24.csv')

    cols = list(df_aut.columns.difference(df_aff.columns)) + ['doi_str']
    df = df_aff.merge(df_aut[cols], on='doi_str', how='outer')

    df_subjects = read_csv_fast(f'../dataframes/df_subjects_Aug24.csv')
    cols = list(df_subjects.columns.difference(df.columns)) + ['doi_str']
    df = df.merge(df_subjects[cols], on='doi_str', how='outer')

    df_SNIP = read_csv_fast(f'../dataframes/df_SNIP_Aug24.csv')
    cols = list(df_SNIP.columns.difference(df.columns)) + ['doi_str']
    df = df.merge(df_SNIP[cols], on='doi_str', how='outer')

    fp_text = fr'../dataframes/df_text_Aug24.csv'
    df_text = read_csv_fast(fp_text, easy_override=False)
    df_text = df_text[['doi_str', 'baye_paper', 'freq_paper', 'ML_paper']]
    df = df.merge(df_text, on='doi_str', how='outer')

    if not all_aff:
        mthd2school2M = {'baye': {}, 'freq': {}, 'ML': {}}
        for method in ['baye', 'freq', 'ML']:
            for school in df['school'].unique():
                df_school = df[df['school'] == school]
                mthd2school2M[method][school] = (
                    df_school[f'{method}_paper'].mean())
            df[f'school_M_{method}'] = df['school'].map(mthd2school2M[method])

    fp_has_results = f'../dataframes/df_has_results_Aug24.csv'
    df_has_results = read_csv_fast(fp_has_results)
    df = df.merge(df_has_results[['doi_str', 'has_results']],
                  on='doi_str', how='outer')

    fp_cites = r'../dataframes/df_cites_Aug24.csv'
    df_cites = read_csv_fast(fp_cites)
    df = df.merge(df_cites, on='doi_str', how='outer')

    fp_journal_clean = r'../dataframes/df_lens_Aug24.csv'
    df_journal_clean = read_csv_fast(fp_journal_clean)
    df = df.merge(df_journal_clean[['doi_str', 'journal_clean']],
                  on='doi_str', how='outer')
    print('Made most of the non-p component of the final dataframe')
    print('-*' * 10 + '-')
    return df


def make_combined_df(all_aff=False):
    fp_p = r'../dataframes/df_p_process_Jan21.csv'
    df_p = read_csv_fast(fp_p)
    df_non_p = get_non_ps_df(all_aff=all_aff)

    print(f'p/z-value df: {len(df_p)=}')

    cols = list(df_p.columns.difference(df_non_p.columns)) + ['doi_str']
    df = df_non_p.merge(df_p[cols], on='doi_str', how='outer')
    df['num_ps'] = df['num_ps'].fillna(0)

    df_lens = read_csv_fast(r'..\dataframes\df_lens_Aug24.csv', verbose=0)
    if 'journal_clean' in df.columns:
        df.drop(columns=['journal_clean'], inplace=True)
    df = df.merge(df_lens[['doi_str', 'journal_clean']], on='doi_str',
                  how='left')

    interpolate_journal(df, key='p_fragile')
    add_jrl_cnt(df)

    if 'journal_clean_x' in df.columns:
        df.drop(columns=['journal_clean_x', 'journal_clean_y'], inplace=True)

    # these are redundant or not meaningful
    df.drop(columns=['index', 'lens_cites', 'cnt',
                     'insig_exact_implied', 'insig_less_implied',
                     'n001_exact_implied', 'n001_less_implied',
                     'n005_h_exact_implied', 'n005_h_less_implied',
                     'n005_l_exact_implied', 'n005_l_less_implied',
                     'n05_exact_implied', 'n05_less_implied',
                     'n_exact05_implied', 'num_ps_exact_implied',
                     'num_ps_less_implied', 'sig_exact_implied',
                     'sig_less_implied',
                     ],
            inplace=True)

    if not all_aff:
        df_school = df[df['num_ps'] > 0].groupby('school')['country'].first()
        country2school_cnt = defaultdict(int)
        for school, country in df_school.items():
            country2school_cnt[country] += 1
        df['country_school_count'] = df['country'].map(country2school_cnt)

        # needed for the df_by_pval, which can't calculate jrnl cnt or school count
        save_csv_pkl(df[['doi_str', 'jrl_cnt', 'country_school_count']],
                     r'../dataframes/df_journal_school_count_Jan21.csv')

    aff_all_str = '_all_aff' if all_aff else ''
    fp_out = f'../dataframes/df_combined{aff_all_str}_Jan21.csv'
    print(f'{fp_out=}')

    save_csv_pkl(df, fp_out)
    print('Made combined dataframe')


def make_combined_by_pval(all_aff=False):
    df_non_p = get_non_ps_df(all_aff=all_aff)
    fp_p = r'../dataframes/df_p_processed_by_p_Jan21.csv'
    df_by_p = read_csv_fast(fp_p)
    df_p_by_p_cols = set(df_by_p.columns)
    df_cols = set(df_non_p.columns)
    overlap_cols = df_p_by_p_cols.intersection(df_cols)
    cols = list(df_p_by_p_cols - overlap_cols) + ['doi_str']
    df_by_p = df_by_p[cols].merge(df_non_p, on='doi_str')
    df_other = read_csv_fast(r'../dataframes/df_journal_school_count_Jan21.csv')
    df_by_p = df_by_p.merge(df_other, on='doi_str')

    aff_all_str = '_all_aff' if all_aff else ''
    fp_out = fr'../dataframes/df_by_pval_combined{aff_all_str}_Jan21.csv'
    if 'journal_clean_x' in df_by_p.columns:
        df_by_p.drop(columns=['journal_clean_x', 'journal_clean_y'])
    save_csv_pkl(df_by_p, fp_out, check_dup=False)
    print('Made combined-by-p-value dataframe')
    print('-*' * 10 + '-')


if __name__ == '__main__':
    make_combined_df()
    make_combined_by_pval()
    make_combined_df(all_aff=True)
