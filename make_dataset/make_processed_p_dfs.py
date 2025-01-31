from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import read_csv_fast, pickle_wrap, save_csv_pkl

smol = 1e-6

pd.options.mode.chained_assignment = None


def categorize_paper_pvals(df_by_p, thresh=.999999, p_key=''):
    doi2cond = {}
    doi2lower_cutoff = {}
    if p_key is None:
        p_key = 'p_val'
    else:
        p_key = f'p_val{p_key}'

    df_p_by_p_ = df_by_p[['doi_str', 'sign', p_key, ]]
    for doi_str, df_doi in tqdm(df_p_by_p_.groupby('doi_str'),
                                desc='categorizing papers'):
        doi2cond[doi_str] = 'ERROR'

        is_sig = ((df_doi[p_key] < .05 + smol) &
                  (df_doi['sign'].isin(['<', '='])))
        df_sig = df_doi[is_sig]
        if not len(df_sig):
            doi2cond[doi_str] = 'no_sig'
            doi2lower_cutoff[doi_str] = pd.NA
            continue

        # Ignore "p = .05"
        equal05 = ((df_sig[p_key] < .05 + smol) &
                   (df_sig[p_key] > .05 - smol) &
                   (df_sig['sign'] == '='))

        df_sig = df_sig[~equal05]

        if not len(df_sig):
            doi2cond[doi_str] = 'all_equal0.05'
            doi2lower_cutoff[doi_str] = .05
            continue

        prop_less_than = (df_sig['sign'] == '<').mean()
        if prop_less_than > thresh:
            for cutoff_ in [.000, .0001, .0005, .001, .005, .01, .05,
                            '_custom']:
                # .000 is strange, but some papers report "p < .000"
                if cutoff_ == '_custom':
                    # e.g., if an author, always reports p < .0125 because
                    #   they like having the same less-than p-value everywhere
                    #   and use a p < .05 threshold bonferroni corrected 4x
                    cutoff = df_sig[p_key].mode().iloc[0]
                else:
                    # e.g., always reports p < .05 or always p < .01
                    cutoff = cutoff_

                is_less_cut = ((df_sig[p_key] < cutoff + smol) &
                               (df_sig[p_key] > cutoff - smol) &
                               (df_sig['sign'] == '<'))
                prop_less_cut = is_less_cut.mean()
                if prop_less_cut > thresh:
                    doi2cond[doi_str] = f'all_less{cutoff_}'
                    doi2lower_cutoff[doi_str] = cutoff
                    break
            else:
                lowest_p = df_sig[p_key].min()
                for cutoff_ in [.000, .0001, .0005, .001, .005, .01]:
                    # e.g., if an author, always reports "<" but in different
                    #   ways, like p < .05 and p < .001.
                    # cutoff_ = .001 means that the lowest is p < .001 or lower
                    if cutoff_ - smol < lowest_p < cutoff_ + smol:
                        doi2cond[doi_str] = f'all_less_mixed{cutoff_}'
                        doi2lower_cutoff[doi_str] = cutoff_
                        break
                else:
                    doi2cond[doi_str] = f'all_less_mixed_custom'
                    doi2lower_cutoff[doi_str] = lowest_p
                continue
            continue

        is_equal = df_sig['sign'] == '='
        prop_equal = is_equal.mean()
        if prop_equal > thresh:
            doi2cond[doi_str] = 'all_equal'
            # Not actually a cutoff, but it's still useful to note lowest p = x
            doi2lower_cutoff[doi_str] = df_sig[p_key].min()
            continue

        for cutoff_ in [.000, .0001, .0005, .001, .005, .01, '_custom']:
            if cutoff_ == '_custom':
                cutoff = (
                    df_sig[df_sig['sign'] == '<'][p_key].mode().iloc[0])
            else:
                cutoff = cutoff_
            is_less_cut = ((df_sig[p_key] < cutoff + smol) &
                           (df_sig[p_key] > cutoff - smol) &
                           (df_sig['sign'] == '<'))
            is_equal_cut = ((df_sig['sign'] == '=') &
                            (df_sig[p_key] > cutoff - smol))
            is_equal_less_cut = is_equal_cut | is_less_cut
            prop_equal_less_cut = is_equal_less_cut.mean()
            if prop_equal_less_cut > thresh:
                doi2cond[doi_str] = f'equal_less{cutoff_}'
                doi2lower_cutoff[doi_str] = cutoff
                break
        else:
            doi2cond[doi_str] = 'eclectic'
            doi2lower_cutoff[doi_str] = pd.NA

    cond2cnt = defaultdict(lambda: 0)
    for cond in doi2cond.values():
        cond2cnt[cond] += 1

    return df_by_p, doi2cond, doi2lower_cutoff


def semi_prep(key=None):
    fp_p_by_p = fr'../dataframes/df_by_pval_Jan21.csv'
    df_by_p = read_csv_fast(fp_p_by_p, check_dup=False)
    sign_mapper = {'=': '=', '<': '<', '>': '>',
                   '≤': '<', '≥': '>'}
    df_by_p['sign'] = df_by_p['sign'].map(sign_mapper)

    df_by_p, doi2cond, doi2lower_cutoff = (
        categorize_paper_pvals(df_by_p, p_key=key))
    return df_by_p, doi2cond, doi2lower_cutoff


def parse_p(stat, sign):
    out = {'sig_exact': 0, 'n05_exact': 0, 'n005_h_exact': 0, 'n005_l_exact': 0,
           'n001_exact': 0, 'num_ps_exact': 0,
           'sig_less': 0, 'n05_less': 0, 'n005_h_less': 0, 'n005_l_less': 0,
           'n001_less': 0, 'num_ps_less': 0,

           'insig_exact': 0, 'insig_less': 0, 'insig_over': 0,
           'num_ps_any': 0, 'n_exact05': 0
           }

    if not pd.isna(stat):
        if sign == '=':
            if 0.05 - smol < stat < .05 + smol:  # Note: p = .05 is excluded
                out['n_exact05'] = 1
            elif stat < .05 + smol:
                out['num_ps_exact'] = 1
                out['sig_exact'] = 1
                if stat > .01 - smol:
                    out['n05_exact'] = 1
                elif stat > .005 - smol:
                    out['n005_h_exact'] = 1
                elif stat > .001 - smol:
                    out['n005_l_exact'] = 1
                else:
                    out['n001_exact'] = 1
            else:
                out['num_ps_exact'] = 1
                out['insig_exact'] = 1
        elif sign == '<':
            out['num_ps_less'] = 1
            if stat < .05 + smol:
                out['sig_less'] = 1
                if stat > .01 + smol:
                    out['n05_less'] = 1
                elif stat > .005 + smol:
                    out['n005_h_less'] = 1
                elif stat > .001 + smol:
                    out['n005_l_less'] = 1
                else:
                    # Note: p < .001 is considered both n001_less and n001_exact
                    out['n001_less'] = 1
                    out['n001_exact'] = 1
            else:
                out['insig_less'] = 1
        elif sign == '>':
            out['insig_over'] = 1
        else:
            raise ValueError

    out['sig'] = out['sig_exact'] or out['sig_less']
    out['n05'] = out['n05_exact'] or out['n05_less']
    out['n005_h'] = out['n005_h_exact'] or out['n005_h_less']
    out['n005_l'] = out['n005_l_exact'] or out['n005_l_less']
    out['n01'] = out['n005_h'] or out['n005_l']
    out['n001'] = out['n001_exact'] or out['n001_less']
    out['n01_001'] = out['n01'] or out['n001']
    out['num_ps'] = out['num_ps_exact'] or out['num_ps_less']
    out['num_ps_any'] = out['num_ps'] or out['insig_over'] or out['n_exact05']
    out['insig'] = out['insig_exact'] or out['insig_less'] or out['insig_over']

    # Papers that report exactly still usually have p < .001
    out['sig_exact'] = out['sig_exact'] or out['n001_less']
    out['num_ps_exact'] = out['num_ps_exact'] or out['n001_less']

    return out


def make_p_processed_dfs():
    # fp_semi_prep = r'../cache/semi_prep_p.pkl'
    fp_semi_prep = r'../cache/semi_prep_p_Jan21.pkl'
    df_by_p, doi2cond, doi2lower_cutoff = pickle_wrap(
        semi_prep, fp_semi_prep, RAM_cache=False, easy_override=False,
        dt_max=datetime(2024, 8, 26, 10, 20, 0, 0))

    p_cats = df_by_p[['p_val', 'sign']].apply(
        lambda x: parse_p(x['p_val'], x['sign']), axis=1)
    p_cats = pd.DataFrame(p_cats.tolist())

    p_implied_cats = df_by_p['p_implied'].apply(lambda x: parse_p(x, '='))
    p_implied_cats = pd.DataFrame(p_implied_cats.tolist())
    p_implied_cats['has'] = (~pd.isna(df_by_p['p_implied'])).astype(int)
    p_implied_cats = p_implied_cats.add_suffix('_implied')

    df_by_p = pd.concat([df_by_p, p_cats, p_implied_cats, ], axis=1)

    df_by_p['cnt'] = 1
    df_by_p['cond'] = df_by_p['doi_str'].map(doi2cond)
    # df_test = df_by_p[df_by_p['cond'] == 'all_less0.05']
    # print(df_test['p_fragile_implied'])
    # quit()

    df_by_p['sig_has_implied'] = df_by_p[['sig', 'has_implied']].all(axis=1).astype(int)

    cols = (list(p_cats.columns) + list(p_implied_cats.columns) +
            ['cnt', 'sig_has_implied'])

    df = df_by_p.groupby('doi_str')[cols].sum()
    for end in ['_val', '_implied', ]:
        df[f'lowest_p{end}'] = (
            df_by_p.groupby('doi_str')[f'p{end}'].min())

    df_t = df_by_p[df_by_p['stat_type'] == 't']
    df_t['t_N'] = df_t['df1'] + 1
    df_t['d'] = df_t['stat'].abs() / np.sqrt(df_t['t_N'])
    df[['t_N', 'd']] = df_t.groupby('doi_str')[['t_N', 'd']].median()

    df_t_sig = df_by_p[df_by_p['sig'] == 1]
    df_t_sig = df_t_sig[df_t_sig['stat_type'] == 't']
    df_t_sig['t_N_sig'] = df_t_sig['df1'] + 1
    df_t_sig['d_sig'] = (df_t_sig['stat'].abs() /
                            np.sqrt(df_t_sig['t_N_sig']))

    df[['t_N_sig', 'd_sig']] = df_t_sig.groupby('doi_str')[[
        't_N_sig', 'd_sig']].median()

    df.reset_index(inplace=True)
    df['cond'] = df['doi_str'].map(doi2cond)

    # not used for manuscript
    df['p_cutoff'] = df['doi_str'].map(doi2lower_cutoff)

    df['prop_implied'] = df['num_ps_implied'] / df['num_ps']

    for end in ['', '_implied', ]:
        df[f'p_fragile{end}'] = df[f'n05{end}'] / df[f'sig{end}']

    # counts reported/interpreted one-tailed significance in the denominator
    df[f'p_fragile_implied_rel_sig'] = (df[f'n05_implied'] /
                                        (df['sig_has_implied']))

    df[f'p_fragile_w_exact05'] = ((df[f'n05'] + df['n_exact05']) /
                                  (df[f'sig'] + df['n_exact05']))
    df['p_fragile_orig'] = df['p_fragile']

    # cnt_na = df['p_fragile'].isna().sum()
    # print(F'{cnt_na=}')
    # print(f'{len(df)=}')
    # df_ = df[df['cond'] == 'all_less0.05']
    # print(df_[['p_fragile', 'p_fragile_implied']])

    df.loc[df['cond'] == 'all_less0.05', 'p_fragile'] = (
        df)[df['cond'] == 'all_less0.05']['p_fragile_implied']
    # cnt_na = df['p_fragile'].isna().sum()
    # print(F'{cnt_na=}')
    # df_ = df[df['cond'] == 'all_less0.05']
    # print(df_[['p_fragile', 'p_fragile_implied']])
    # M = np.nanmean(df_['p_fragile'])
    # print(f'{M=}')
    # M = np.nanmean(df_['p_fragile_implied'])
    # print(f'{M=}')
    # quit()

    df.loc[df['cond'] == 'all_less0.05', 'p_fragile'] = (
        df.loc[df['cond'] == 'all_less0.05', 'p_fragile'].fillna(0.509))

    # cnt_na = df['p_fragile'].isna().sum()
    # print(F'{cnt_na=}')
    # quit()


    fp_out_by_p = fr'../dataframes/df_p_processed_by_p_Jan21.csv'
    save_csv_pkl(df_by_p, fp_out_by_p, check_dup=False)

    fp_out = r'../dataframes/df_p_process_Jan21.csv'
    save_csv_pkl(df, fp_out, check_dup=False)
    return df


if __name__ == '__main__':
    make_p_processed_dfs()
