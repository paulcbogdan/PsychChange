from collections import defaultdict

import pandas as pd

from utils import read_csv_fast, save_csv_pkl


def make_author_age_df():
    df = read_csv_fast(fr'../dataframes/df_lens_Aug24.csv')
    df_ta = read_csv_fast(fr'../dataframes/df_lens_ta_Aug24.csv')
    df = pd.concat([df, df_ta]).reset_index()


    pre_len = len(df)
    df.dropna(subset=['authors_str'], inplace=True)
    df = df[df['authors_str'] != 'null No authorship indicated']
    print(f'Dropped {pre_len - len(df)} rows with missing authors')

    first_aut2first_year = defaultdict(list)
    last_aut2first_year = defaultdict(list)
    auts_first = []
    auts_last = []

    for tup in df.itertuples():
        authors = tup.authors_str.split('; ')
        assert len(authors)
        auts_first.append(authors[0])
        auts_last.append(authors[-1])
        first_aut2first_year[authors[0]].append(tup.year)
        last_aut2first_year[authors[-1]].append(tup.year)
    df['author_first'] = auts_first
    df['author_last'] = auts_last

    pd.set_option('display.max_rows', None)

    cnt = df['author_last'].value_counts()
    cnt = cnt[cnt > 10]
    print(len(cnt))
    last_names = cnt.index.str.split(' ').str[-1]
    print(len(set(last_names)))

    first_names = cnt.index.str.split(' ').str[0]
    print(len(set(first_names)))
    names_cnt = cnt.index.str.split(' ').str.len()
    print(names_cnt.value_counts(normalize=True))

    for aut, l in first_aut2first_year.items():
        first_aut2first_year[aut] = min(l)
    for aut, l in last_aut2first_year.items():
        last_aut2first_year[aut] = min(l)

    first_aut2first_year = defaultdict(lambda: pd.NA, first_aut2first_year)
    last_aut2first_year = defaultdict(lambda: pd.NA, last_aut2first_year)

    df['age_first'] = df['author_first'].map(first_aut2first_year)
    df['age_last'] = df['author_last'].map(last_aut2first_year)
    print('Made author df')
    cnt = (df[['author_last', 'age_last']].groupby('author_last').first().
           value_counts(dropna=False))
    print('Counted...')
    total = 0
    for year in range(1904, 2025):  # 1904 is the first year in lens.org
        total += int(cnt[year])
        print(f'{year}: {cnt[year]:>6,} | total: {total:>9,}')

    pubs = ['Elsevier_BV', 'Springer_Science_and_Business_Media_LLC',
            'SAGE_Publications', 'Wiley', 'Frontiers_Media_SA']
    df = df[df['publisher'].isin(pubs)]

    fp_out = f'../dataframes/df_author_ages_Aug24.csv'
    save_csv_pkl(df, fp_out)


if __name__ == '__main__':
    make_author_age_df()
