import os.path
import re
from datetime import datetime
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from make_dataset.make_p_z_df import add_row_to_df, remove_diacritics_fast
from utils import pickle_wrap, read_csv_fast, save_csv_pkl

np.random.seed(0)


def map_UCL(name):
    if name == 'ucl':
        return 'university college london'
    return name


def sort_re_str(re_str):
    print('Cleaning regex string for affiliations...')
    # If s0 is a subset of s1, we want s1 to be earlier in the regex pattern
    re_spl = re_str[1:-1].split('|')
    not_in_any_other = []
    in_another = []
    for s0 in re_spl:
        for s1 in re_spl:
            if s0 == s1: continue
            if s0 in s1:
                in_another.append(s0)
                break
        else:
            not_in_any_other.append(s0)

    re_spl_new = '|'.join(not_in_any_other + in_another)
    re_spl_new = f'({re_spl_new})'
    if re_spl_new != re_str:  # in case there are multiple to fix
        return sort_re_str(re_spl_new)
    return re_spl_new


@cache
def get_affiliation_mapper(rank_cutoff=1000):
    dir_in = r'../dataframes/times_higher_ed'
    dfs = []
    for year in range(2011, 2025):
        fn = f'{year}_rankings.csv'
        fp = f'{dir_in}/{fn}'
        df = pd.read_csv(fp)
        cols = ['name', 'scores_research']
        df['name'] = df['name'].apply(remove_diacritics_fast)
        df['name'] = df['name'].str.lower()
        df['name'] = df['name'].apply(map_UCL)

        if year == 2024: cols.append('location')
        df = df[cols]
        df.rename(columns={'scores_research': f'score_{year}',
                           'name': 'school',
                           'location': 'country'}, inplace=True, )
        # add backslash before parentheses for later regex
        df['school'] = df['school'].str.replace('(', r'\(').str.replace(')',
                                                                        r'\)')
        df.set_index('school', inplace=True, drop=year != 2024)
        df[f'rank_{year}'] = df[f'score_{year}'].rank(ascending=False)

        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    assert 'ucl' not in df.index
    assert 'university college london' in df.index

    df.sort_values('score_2024', ascending=False, inplace=True)
    df = df.iloc[:rank_cutoff]
    pd.set_option('display.max_rows', 1000)
    l_of_d = df.to_dict('records')
    aff2scores = dict(zip(df.index, l_of_d))
    re_str = '|'.join(aff2scores.keys())
    to_add = {}
    remove_affs = []
    affs_no_parenthesis = []
    affs_no_parenthesis_map = {}  # for catching and mapping, e.g., 'Ohio state
    #    university' to the formal name 'Ohio state
    #    university (main campus)'
    for aff, scores in aff2scores.items():
        aff_reverted = aff.replace(r'\(', '(').replace(r'\)', ')')
        to_add[aff_reverted] = scores
        if aff != aff_reverted:
            remove_affs.append(aff)
            aff_no_paren = aff.split(r'\(')[0].strip()
            affs_no_parenthesis_map[aff_no_paren] = aff_reverted
            affs_no_parenthesis.append(aff_no_paren)
    for aff in remove_affs:
        del aff2scores[aff]
    aff2scores.update(to_add)
    for aff, d in aff2scores.items():
        del d['school']
    re_str += '|' + '|'.join(affs_no_parenthesis)
    re_str = f'({re_str})'
    re_str = sort_re_str(re_str)
    score_columns = [f'score_{year}' for year in range(2011, 2025)]
    return aff2scores, re_str, score_columns, affs_no_parenthesis_map

def get_paper_affil(fp_text, row, target_year=2024,  # omit_repeats=False,
                    rank_cutoff=1000):
    year = row.year
    if year < 2011:  # Lowest possible THE ranking
        year = 2011
    with open(fp_text, 'r', encoding='utf-8') as file:
        plain_text = file.read()
    if len(plain_text) == 0: return {}  # rare

    # Drop back half to ignore references (e.g., "Princeton university press")
    plain_text = plain_text[:len(plain_text) // 2]
    plain_text = remove_diacritics_fast(plain_text).lower()
    aff2scores, re_str, score_columns, affs_no_paren_map = (
        get_affiliation_mapper(rank_cutoff=rank_cutoff))

    matches = re.findall(re_str, plain_text)

    l_ds = []
    for match in matches:
        if match in affs_no_paren_map:
            match = affs_no_paren_map[match]
        aff_scores = aff2scores[match]
        target_score = aff_scores[f'score_{target_year}']
        year_score = aff_scores[f'score_{year}']
        year_rank = aff_scores[f'rank_{year}']
        if np.isnan(year_score):
            for year_ in range(year + 1, 2025):
                year_score = aff_scores[f'score_{year_}']
                if not np.isnan(year_score):
                    year_rank = aff_scores[f'rank_{year_}']
                    break
            else:
                year_ = np.nan  # rare
        else:
            year_ = year

        target_rank = aff_scores[f'rank_{target_year}']
        country = aff_scores['country']
        l_ds.append({'school': match, 'country': country,
                     'year_score': year_score, 'year_rank': year_rank,
                     'target_score': target_score, 'target_rank': target_rank,
                     'rank_year': year_})
    df_matches = pd.DataFrame(l_ds)
    if not len(df_matches):
        return {}

    # Below code gets max/min/etc. of ranks with respect to the potentially
    #   multiple schools detected in a wide variety of ways

    idx_year_random = np.random.choice(df_matches.index)
    d_random = df_matches.loc[idx_year_random].add_prefix('Random_')
    df_l = [d_random]

    df_matches_ = df_matches.dropna(subset=['target_score'])
    if len(df_matches_):
        idx_target_max = df_matches_['target_score'].idxmax()
        d_target_max = df_matches.loc[idx_target_max].add_prefix('TargetMax_')
        df_l += [d_target_max]
        idx_target_min = df_matches_['target_score'].idxmin()
        d_target_max = df_matches.loc[idx_target_min].add_prefix('TargetMin_')
        df_l += [d_target_max]

    if (~df_matches['year_score'].isna()).any():  # check if at least 1 isn't nan
        idx_year_max = df_matches['year_score'].idxmax(skipna=True)
        d_year_max = df_matches.loc[idx_year_max].add_prefix('YearMax_')
        df_l += [d_year_max]

        idx_year_min = df_matches['year_score'].idxmin(skipna=True)
        d_year_min = df_matches.loc[idx_year_min].add_prefix('YearMin_')
        df_l += [d_year_min]

    d_mean = df_matches[['year_score', 'target_score',
                         'year_rank', 'target_rank']].mean().add_prefix('Mean_')
    df_l += [d_mean]

    # calculate median separately for year and target
    #   need to do this elaborate method below rather than dataframe.median
    #   to get the index and in turn the location
    df_matches_yr = df_matches.dropna(subset='year_score')
    if len(df_matches_yr):
        idx_year_median = df_matches_yr['year_score'].sort_values(
            ascending=False).index[len(df_matches_yr) // 2]
        d_year_med = df_matches.loc[idx_year_median].add_prefix('YearMed_')
        df_l += [d_year_med]

    df_matches_tar = df_matches.dropna(subset='target_score')
    if len(df_matches_tar):
        idx_target_median = df_matches_tar['target_score'].sort_values(
            ascending=False).index[len(df_matches_tar) // 2]
        d_target_med = df_matches.loc[idx_target_median].add_prefix(
            'TargetMed_')
        df_l += [d_target_med]

    # if not omit_repeats:
    # get random if multiple schools have the same number of entries
    mode_school = df_matches['school'].mode().sample(1)
    df_mode = df_matches[df_matches['school'] == mode_school.iloc[0]]
    d_mode = df_mode.iloc[0].add_prefix('Mode_')
    num_matches = len(df_mode)
    d_mode['Mode_matches'] = num_matches
    df_l += [d_mode]

    d_all = {}
    for df in df_l:
        d_all.update(df.to_dict())
    d_all = add_row_to_df(d_all, row)
    d_all['num_affil'] = len(df_matches)
    return d_all


def make_affiliation_df(rank_cutoff=1000, only_w_results=False):
    if only_w_results:
        df = read_csv_fast(fr'../dataframes/df_has_results_Aug24.csv')
        df = df[df['has_results'] == True]
    else:
        df = read_csv_fast(fr'../dataframes/df_lens_Aug24.csv')

    list_of_dicts = []
    dt_max = datetime(2024, 8, 24, 8, 0, 0, 0)

    skip_count = 0
    for row in tqdm(df.itertuples(), desc='adding universities',
                    total=df.shape[0]):
        # TODO: Change to Aug24
        fp_text = (fr'..\data\plaintexts_Aug15\{row.publisher}\{row.journal}' +
                   fr'\{row.year}\{row.doi_str}.txt')
        if not os.path.exists(fp_text):  # papers with no web version
            skip_count += 1
            if skip_count % 10_000 == 0:
                print(f'Skip count: {skip_count}')
            continue
        fp_pkl = f'../cache/paper_affil/{row.doi_str}_rank{rank_cutoff}.pkl'
        Path(fp_pkl).parent.mkdir(parents=True, exist_ok=True)
        d_affil = pickle_wrap(get_paper_affil, fp_pkl,
                              kwargs={'fp_text': fp_text, 'row': row,
                                      'rank_cutoff': rank_cutoff,
                                      'target_year': 2024},
                              verbose=-1, easy_override=False,
                              dt_max=dt_max)
        if len(d_affil) == 0:
            continue
        list_of_dicts.append(d_affil)

    df_new = pd.DataFrame(list_of_dicts)
    if only_w_results:
        fp_out = fr'../dataframes/df_affiliation_Aug24.csv'
    else:
        fp_out = fr'../dataframes/df_affiliation_all_lens_Aug24.csv'
    save_csv_pkl(df_new, fp_out)


if __name__ == '__main__':
    make_affiliation_df()
