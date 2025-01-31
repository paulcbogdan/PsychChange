import pandas as pd
from scipy import stats

from utils import read_csv_fast

pd.options.mode.chained_assignment = None


def validate_p_x_p_implied():
    fp_p_by_p = fr'../dataframes/df_by_pval_combined_semi_pruned_Jan21.csv'
    df_p_by_p = read_csv_fast(fp_p_by_p, easy_override=False)

    assert df_p_by_p['p_val'].isna().sum() == 0
    pre_len = len(df_p_by_p)

    equal_len = len(df_p_by_p)
    implied_len = len(df_p_by_p.dropna(subset=['p_implied']))
    p = implied_len / equal_len
    print(f'Number of ps with p_implied: {implied_len:,} ({p:.1%})')

    df_p_by_p = df_p_by_p[df_p_by_p['sign'] == '=']
    equal_len = len(df_p_by_p)
    print(f'\tPercentage reported with equal sign: {equal_len / pre_len:.1%}')
    df_p_by_p.dropna(subset=['p_implied'], inplace=True)
    implied_len = len(df_p_by_p)
    print(f'\tNumber of exact with p_implied: {implied_len:,}')
    print(f'\tFinal len: {implied_len=}')
    r, p = stats.spearmanr(df_p_by_p['p_val'], df_p_by_p['p_implied'])
    print(f'p_val x p_implied Spearmanr: {r=:.4f}')

    df_p_by_p['p_val'] = df_p_by_p['p_val'].apply(
        lambda x: min(max(x, .001), .999))

    df_p_by_p['z_val'] = stats.norm.isf(df_p_by_p['p_val'] / 2)
    min_z = df_p_by_p['z_val'].min()
    max_z = df_p_by_p['z_val'].max()
    df_p_by_p['z_implied'] = df_p_by_p['z_implied'].apply(
        lambda x: min(max(x, min_z), max_z))
    r, p = stats.pearsonr(df_p_by_p['z_val'], df_p_by_p['z_implied'])
    print(f'\tz_val x z_implied Pearsonr: {r=:.4f}')


if __name__ == '__main__':
    validate_p_x_p_implied()
