import pandas as pd

from make_dataset.prune_to_final import NEURO_W_PSYCH_JOURNALS, NEURO_JOURNALS
from utils import read_csv_fast

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    df = read_csv_fast(r'..\dataframes\df_lens_Aug24.csv', verbose=0)
    lowest_year = df['year'].min()
    highest_year = df['year'].max()
    print('-*- PRODUCE NUMBERS IN SUPPLEMENTAL 1 and METHODS 2.2 -*- ')
    print(f'Number of Elsevier, Springer-Nature, Wiley, SAGE, Frontiers '
          f'records: {len(df):,} ({lowest_year} - {highest_year}) (SM)')
    df = df[df['journal'] != 'journal_of_fluorescence']  # lens.org error entry?
    pd.set_option('display.max_rows', None)
    df = df[~df['journal'].isin(NEURO_JOURNALS)]
    print(f'Number of records after dropping neuro: {len(df):,} (SM)')
    df = df[df['year'] >= 2004]

    df_ta = read_csv_fast(r'..\dataframes\df_lens_ta_Aug24.csv', verbose=0)
    lowest_year = df_ta['year'].min()
    highest_year = df_ta['year'].max()
    print(f'\tNumber of APA and Taylor & Francis papers: {len(df_ta):,} '
          f'({lowest_year} - {highest_year})')
    df_ta = df_ta[df_ta['year'] >= 2004]
    print(f'\tNumber of APA/T & Fpapers in year range (2004-2024): '
          f'{len(df_ta):,} (SM)')
    print()
    print(f'Number of non-neuro records from 2004-2024: {len(df):,}')

    df = read_csv_fast(r'..\dataframes\df_combined_Jan21.csv', verbose=0)
    df = df[df['year'] >= 2004]
    df = df[df['journal'] != 'journal_of_fluorescence']  # lens.org error entry?
    df = df[(df['is_neuro'] != True) |
            (df['journal'].isin(NEURO_W_PSYCH_JOURNALS))]
    pubs = ['Elsevier_BV', 'Springer_Science_and_Business_Media_LLC',
            'SAGE_Publications', 'Wiley', 'Frontiers_Media_SA']
    df = df[df['publisher'].isin(pubs)]
    print(f'\tNumber of non-neuro papers that could be retrieved: {len(df):,} '
          f'(SM)')
    len_2004 = len(df)
    df_results = df[df['has_results'] == True]

    len_pre = len(df_results)
    print(f'Number of papers with Results: '
          f'{len(df_results):,}')
    df_results = df_results[df_results['jrl_cnt'] >= 5]
    num_dropped = len_pre - len(df_results)
    print(f'Number of papers with Results in valid journals: '
          f'{len(df_results):,}')

    n_w_ps = (df_results['num_ps_any'] > 0).sum()

    df_results['has_ps'] = df_results['num_ps_any'] > 0
    pre_len = len(df_results)
    df_results = df_results[df_results['has_ps'] == True]
    print(f'Number of papers with p-values: {len(df_results):,} '
          f'({len(df_results) / pre_len:.1%})')

    all_insig = ((df_results['num_ps_any'] > 0) & (df_results['sig'] == 0)).sum()
    print(f'\tNumber of papers with only insignificant results: {all_insig:} '
          f'({all_insig / n_w_ps:.1%})')
    all_insig = ((df_results['num_ps_any'] > 1) & (df_results['sig'] == 0)).sum()
    print(f'\tNumber of papers with only insignificant results (2+ p-values): {all_insig:} '
          f'({all_insig / n_w_ps:.1%})')

    df_results['has_signif_ps'] = df_results['sig'] > 1
    df_results = df_results[df_results['has_signif_ps'] == True]
    print(f'Number of papers with 2+ signif. p-values: {len(df_results):,} '
          f'(ie, final dataset before dropping papers without SNIP or school)')

    total_num_ps = df_results['num_ps_any'].sum().astype(int)
    total_num_ps_implied = df_results['num_ps_any_implied'].sum().astype(int)
    percent_implied = total_num_ps_implied / total_num_ps
    M_num_ps = df_results['num_ps_any'].mean()
    Med_num_ps = df_results['num_ps_any'].median()
    SD_num_ps = df_results['num_ps_any'].std()
    print(f'Number of p-values: M = {M_num_ps:.1f} [SD = {SD_num_ps:.1f}], '
          f'median = {Med_num_ps:.1f} (total num ps = {total_num_ps:,})')
    print(f'\tTotal num ps implied = {total_num_ps_implied:,} '
          f'({percent_implied:.1%})')

    n_nan_snip = df_results['SNIP'].isna().sum()
    n_school_semifinal = df_results['school'].nunique()
    print(f'Number of papers with NaN SNIP: {n_nan_snip:,}')
    n_nan_affil = df_results['school'].isna().sum()
    print(f'Number of papers with NaN affiliation: {n_nan_affil:,}')
    df_results = df_results[df_results['SNIP'].notna() &
                            df_results['school'].notna()]
    print(f'Final dataset size w/ SNIP and school: {len(df_results):,}')
    # 'num_ps_any' includes '>' and '>='
    total_num_ps = df_results['num_ps_any'].sum().astype(int)
    total_num_ps_implied = df_results['num_ps_any_implied'].sum().astype(int)
    M_num_ps = df_results['num_ps_any'].mean()
    Med_num_ps = df_results['num_ps_any'].median()
    SD_num_ps = df_results['num_ps_any'].std()
    print(f'\tNumber of p-values: M = {M_num_ps:.1f} [SD = {SD_num_ps:.1f}], '
          f'median = {Med_num_ps:.1f} (total num ps = {total_num_ps:,})')
    df_results['journal'] = df_results['journal'].str.lower()
    num_journals = len(df_results['journal'].unique())
    print(f'\tNumber of journals: {num_journals:}')
    num_schools = df_results['school'].nunique()
    print()
    print('-*- COUNT P-IMPLIED -*-')
    percent_implied = total_num_ps_implied / total_num_ps
    print(f'\tTotal num ps implied = {total_num_ps_implied:,} '
          f'({percent_implied:.1%})')
    print()

    n_schools_final = df_results['school'].nunique()

    fp = r'..\dataframes\df_affiliation_all_lens_Aug24.csv'
    df_aff = read_csv_fast(fp, easy_override=False, verbose=0)
    n_school_any = df_aff['Mode_school'].dropna().nunique()
    print('-*- PRODUCE COUNTS ON THE NUMBER OF UNIQUE UNIVERSITIES FOUND -*-')
    print('Number of schools in final dataset regardless of SNIP: '
          f'{n_school_semifinal}')
    # print(f'\tFinal pruned dataset with SNIP: {n_schools_final}')
    print(f'\tAll downloaded papers even ones without Results: {n_school_any}')

    M_num_affil = df_results['num_affil'].mean()
    SD_num_affil = df_results['num_affil'].std()
    med_num_affil = df_results['num_affil'].median()
    print(f'\tMean number of affiliations per paper: '
          f'{M_num_affil:.2f} [SD  {SD_num_affil:.2f}], '
          f'median = {med_num_affil}')

    for col in ['SNIP', 'year', 'log_cites_year_z', 'target_score']:
        assert pd.notna(df_results[col]).all(), f'{col} has NaNs'

    # Sanity checkcount...
    df_by_p = read_csv_fast(r'..\dataframes'
                            r'\df_by_pval_combined_pruned_Jan21.csv',
                            verbose=0)
    assert total_num_ps == len(df_by_p), f'{total_num_ps=:,}, {len(df_by_p)=:,}'

    df_by_p_imp = df_by_p.dropna(subset=['p_implied'])
    assert len(df_by_p_imp) == total_num_ps_implied

    df_empirical = read_csv_fast(
        r'..\dataframes\df_combined_all_empirical_Jan21.csv', verbose=0)
    print(f'Number of paper with a Results section along with SNIP/school:'
          f' {len(df_empirical):,} (SM)')
