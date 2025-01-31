import os
from functools import cache

import pandas as pd
from tqdm import tqdm

from utils import save_csv_pkl

# These entries create duplicates in doi_str unless handled as exceptional cases
#   They may be typos in the lens.org database
PREVENT_DOI2STR_DUPES = {'10.17105/spr-2015-0118.v46-2',
                         '10.1037/0003-066x.59.1.52a',
                         '10.1037/0003.066x.59.1.52c',
                         '10.1037/0003-066x.59.1.52b',
                         '10.1037/0003-066x.59.1.54',
                         '10.1037/0003.066x.59.1.53',
                         '10.1037/0003.066x.59.1.52b',
                         '10.1037/0003-066x.59.1.48',
                         '10.1037/0003.066x.59.1.47',
                         '10.1037/0003.066x.59.1.48',
                         '10.1037/0003.066x.59.1.49',
                         '10.1037/0003-066x.59.1.49',
                         '10.1037/0003.066x.59.1.54',
                         '10.1037/0003-066x.59.1.47',
                         '10.1037/0003-066x.59.1.53',
                         '10.1037/0003.066x.59.1.51',
                         '10.1037/0003.066x.59.1.52a',
                         '10.1037/0003-066x.59.1.51',
                         '10.17105/spr-2015-0118.v46.2',
                         '10.1037/0003-066x.59.1.52c'}


def doi2doi_str(doi):
    if doi in PREVENT_DOI2STR_DUPES:
        return doi.replace('-', '--').replace('/', '_').replace('.', '-')
    else:
        return doi.replace('/', '_').replace('.', '-')


@cache
def get_singular_ISSN(ISSN_str):
    if pd.isna(ISSN_str):
        print('bad ISSN')
        return None
    ISSNs = ISSN_str.split('; ')
    ISSN = sorted(ISSNs)[0]
    return ISSN


def make_ISSN_journal_consistent(df):
    map_high2low_ISSN = {}
    map_low2high_ISSN = {}
    map_lowISSN2journal = {}
    num_nan = 0
    for ISSNs in tqdm(df['ISSNs'].values, desc='processing ISSNs'):
        if pd.isna(ISSNs):
            num_nan += 1
            continue
        ISSNs = ISSNs.split('; ')
        if len(ISSNs) == 2:
            low_ISSN, high_ISSN = sorted(ISSNs)
            map_high2low_ISSN[high_ISSN] = low_ISSN
            map_high2low_ISSN[low_ISSN] = low_ISSN
            map_low2high_ISSN[low_ISSN] = high_ISSN
    print(f'Number with bad journal names: {num_nan=} '
          f'({num_nan / len(df):.1%})')

    ISSN_clean = []
    journal_clean = []
    backup_ISSNs = []
    for ISSNs, journal in tqdm(zip(df['ISSNs'].values, df['journal'].values),
                               desc='Fixing ISSN and journal'):
        if pd.isna(ISSNs):
            ISSN_clean.append(None)
            journal_clean.append(journal)
            backup_ISSNs.append(None)
            print(f'Bad missing ISSN: {journal}')
            continue

        ISSNs = ISSNs.split('; ')
        if len(ISSNs) == 2:
            ISSN = sorted(ISSNs)[0]
            ISSN = map_high2low_ISSN[ISSN]
        else:
            ISSN = ISSNs[0]
        if ISSN in map_lowISSN2journal:
            journal = map_lowISSN2journal[ISSN]
        else:
            map_lowISSN2journal[ISSN] = journal
        if ISSN in map_low2high_ISSN:
            backup_ISSN = map_low2high_ISSN[ISSN]
        else:
            backup_ISSN = None
        ISSN_clean.append(ISSN)
        journal_clean.append(journal)
        backup_ISSNs.append(backup_ISSN)
    df['ISSN'] = ISSN_clean
    df['backup_ISSN'] = backup_ISSNs
    df['journal_clean'] = journal_clean
    prop_fixed = (df['journal'] != df['journal_clean']).mean()
    print(f'Percentage of journal names tweaked for consistency: '
          f'{prop_fixed:.1%}')


def make_lens_df(key='all'):
    dir_in = r'..\dataframes\lens_org'
    fps = os.listdir(dir_in)
    if key == 'all':
        fps = [fp for fp in fps if key in fp]
    elif key == 'ta':
        fps = [fp for fp in fps if 'tanf' in fp or 'apa' in fp]
    else:
        raise ValueError

    dfs_lens = []
    keep_cols = ['Date Published', 'Publisher', 'MeSH Terms', 'DOI',
                 'Publication Year', 'Source URLs', 'External URL',
                 'Is Open Access', 'ISSNs',
                 # 'Open Access License', 'Open Access Colour',
                 'Author/s', 'Citing Works Count', 'Title', 'Source Title']
    fps = sorted(list(fps))
    for fp in tqdm(fps):
        df = pd.read_csv(fr'{dir_in}\{fp}', low_memory=False)
        dfs_lens.append(df[keep_cols])
    df = pd.concat(dfs_lens)
    df.drop_duplicates(subset=['DOI'], inplace=True)
    df.dropna(subset=['DOI'], inplace=True)

    renamer = {'Date Published': 'date', 'Publisher': 'publisher',
               'MeSH Terms': 'mesh_terms', 'DOI': 'doi',
               'Publication Year': 'year', 'Source URLs': 'source_url',
               'External URL': 'external_url',
               'Is Open Access': 'is_open_access', 'Author/s': 'authors_str',
               'Citing Works Count': 'lens_cites', 'Title': 'title',
               'Source Title': 'journal'}
    df.rename(columns=renamer, inplace=True)
    df['journal'] = df['journal'].str.replace(' ', '_').str.replace(':', '')
    df['journal'] = df['journal'].str.lower()

    df['publisher'] = df['publisher'].str.replace(' ', '_')
    df['doi_str'] = df['doi'].apply(doi2doi_str)

    df['doi_str'] = df['doi_str'].str.replace('(', '_').str.replace(')', '_')

    df.dropna(subset=['journal'], inplace=True)
    df['journal'] = df['journal'].str.replace('&', 'amp').str.replace('?', '')

    make_ISSN_journal_consistent(df)
    df['journal_clean'] = df['journal_clean'].str.replace('amp', 'and')
    df['journal_clean'] = (df['journal_clean'].str.replace('.', '').str.
                           replace(',', ''))

    if key == 'all':
        save_csv_pkl(df, fr'../dataframes/df_lens_Aug24.csv', no_csv=False)
    elif key == 'ta':
        save_csv_pkl(df, fr'../dataframes/df_lens_ta_Aug24.csv', no_csv=False)
    else:
        raise ValueError

    total = 0
    cnt = df[['year']].value_counts()
    for year in range(1904, 2025):  # 1904 is the first year in lens.org
        total += int(cnt[year])
        print(f'{year}: {cnt[year]:>6,} | total: {total:>9,}')


def verify_no_duplicates():
    df = pd.read_csv(r'..\dataframes\df_lens_Aug24.csv')
    df_ta = pd.read_csv(r'..\dataframes\df_lens_ta_Aug24.csv')
    df = pd.concat([df, df_ta])
    assert not df['doi_str'].duplicated().any()
    assert pd.isna(df['doi_str']).sum() == 0


if __name__ == '__main__':
    make_lens_df('all')
    make_lens_df('ta')
    verify_no_duplicates()
