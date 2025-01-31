import pandas as pd

from plot_and_count.tally_TableS1_pval_reporting import (
    count_pval_reporting_style)
from text_analysis.build_df_word import get_is_fragile
from utils import read_csv_fast


def under_overreporting_p_fragile():
    fp = fr'../dataframes/df_by_pval_combined_semi_pruned_Aug24.csv'
    df = read_csv_fast(fp, easy_override=False)

    df.dropna(subset=['p_val', 'p_implied'], inplace=True)

    df = df[df['sig'] == True]

    df['p_is_fragile'] = df.apply(lambda x:
                                  get_is_fragile(x['p_val'], x['sign']), axis=1)
    df['p_is_fragile_implied'] = df['p_implied'].apply(get_fragile_implied_str)
    df = df[df['cond'] != 'all_less0.05']
    df['p_is_fragile_implied'] = df['p_implied'].apply(get_fragile_implied_str)

    cnt = df[['p_is_fragile', 'p_is_fragile_implied']].value_counts(
        normalize=True, dropna=False)

    prop_p_fragile = (cnt[(True, 'FRAGILE')] + cnt[(True, 'STRONG')] +
                      cnt[(True, 'INSIG')])
    prop_p_fragile_strong = cnt[(True, 'STRONG')]
    prop_not_really_fragile = prop_p_fragile_strong / prop_p_fragile
    print(f'\t{prop_not_really_fragile:.1%} of p-values reported as fragile '
          f'are actually strong per p_implied')
    prop_p_strong = (cnt[(False, 'FRAGILE')] + cnt[(False, 'STRONG')] +
                     cnt[(False, 'INSIG')])
    prop_actually_fragile = cnt[(False, 'FRAGILE')] / prop_p_strong
    print(f'\t{prop_actually_fragile:.1%} of p-values reported as strong are '
          f'actually fragile per p_implied')
    print(cnt)
    base_rate = .35
    p_fragile_change = (prop_not_really_fragile * base_rate -
                        prop_actually_fragile * (1 - base_rate))
    print(f'Bias: {p_fragile_change:.1%}')
    print(' ^^ quantity used for adustment ^^')
    print()

    print('Ignoring p = .01...')
    df = df[(df['p_val'] > .011) | (df['p_val'] < .009)]
    cnt = df[['p_is_fragile', 'p_is_fragile_implied']].value_counts(
        normalize=True, dropna=False)

    prop_p_fragile = (cnt[(True, 'FRAGILE')] + cnt[(True, 'STRONG')] +
                      cnt[(True, 'INSIG')])
    prop_p_fragile_strong = cnt[(True, 'STRONG')]
    prop_not_really_fragile = prop_p_fragile_strong / prop_p_fragile
    print(f'\t{prop_not_really_fragile:.1%} of p-values reported as fragile '
          f'are actually strong per p_implied (ignoring p = .01)')
    prop_actually_fragile = cnt[(False, 'FRAGILE')] / prop_p_strong
    print(f'\t{prop_actually_fragile:.1%} of p-values reported as strong are '
          f'actually fragile per p_implied (ignoring p = .01)')
    p_fragile_change = (prop_not_really_fragile * base_rate -
                        prop_actually_fragile * (1 - base_rate))
    print(f'\tBias: {p_fragile_change:.1%} (ignoring p = .01)')
    print(cnt)


def get_fragile_implied_str(p_implied):
    if pd.isna(p_implied):
        return pd.NA
    elif p_implied > 0.05:
        return 'INSIG'
    elif p_implied < .01:
        return 'STRONG'
    else:
        return 'FRAGILE'


if __name__ == '__main__':
    count_pval_reporting_style()
    under_overreporting_p_fragile()
