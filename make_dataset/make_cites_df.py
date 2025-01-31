import numpy as np

from utils import read_csv_fast, save_csv_pkl


def make_cites_df():
    df = read_csv_fast(fr'../dataframes/df_lens_Aug24.csv')
    df['cites'] = df['lens_cites']
    df['cites_year'] = df['cites'] / (2024 - df['year'])
    df.loc[df['year'] == 2024, 'cites_year'] = np.nan
    assert not df['cites_year'].max() > 1e6, \
        f'cites_year is inf: {df["cites_year"].max()}'
    df['log_cites_year'] = np.log(df['cites_year'] + 1)
    for year in df['year'].unique(): # must do after dropping ps < 4
        df_year = df[df['year'] == year]
        df_year_M = df_year['log_cites_year'].mean()
        df_year_sd = df_year['log_cites_year'].std()
        df.loc[df['year'] == year, 'log_cites_year_z'] = \
            (df_year['log_cites_year'] - df_year_M) / df_year_sd

    fp_out = r'../dataframes/df_cites_Aug24.csv'
    df = df[['doi_str', 'cites', 'cites_year', 'log_cites_year',
             'log_cites_year_z']]
    save_csv_pkl(df, fp_out)

if __name__ == '__main__':
    make_cites_df()
