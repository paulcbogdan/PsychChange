import pickle

import numpy as np
import pandas as pd

from utils import read_csv_fast, save_csv_pkl

pd.options.mode.chained_assignment = None


def make_subject_df():
    def make_subject_tuple(l):
        if l is None:  # Scopus API generates nothing for ~1.2% of cases
            return tuple([pd.NA] * 9)
        out = []
        for area in targets:
            out.append(area in l)
        is_psych = any(out)

        for l_area in l:
            if 'neuro' in l_area.lower():
                is_neuro = True
                break
        else:
            is_neuro = False
        return (is_psych, is_neuro, *out)

    targets = ['Developmental and Educational Psychology',
               'Psychology (all)', 'Social Psychology',
               'Applied Psychology', 'Clinical Psychology',
               'Experimental and Cognitive Psychology',
               'Psychology (miscellaneous)', ]

    with open(rf'../cache/journal_to_SCOPUS_subject_Jan21.pkl', 'rb') as f:
        journal_to_ASJC_ = pickle.load(f)

    journal_to_ASJC = {}
    for journal, l in journal_to_ASJC_.items():
        journal_to_ASJC[journal] = make_subject_tuple(l)

    df = read_csv_fast(rf'../dataframes/df_has_results_Aug24.csv')

    # Rename Psychology ('all') to 'General Psychology' for consistency with
    #    my initial submission and the older CrossRef naming
    new_cols = ['is_psych', 'is_neuro',
                'Developmental_and_Educational_Psychology',
                'General_Psychology', 'Social_Psychology',
                'Applied_Psychology', 'Clinical_Psychology',
                'Experimental_and_Cognitive_Psychology',
                'Psychology_Miscellaneous']

    # SettingWithCopyWarning is annoying. Not even sure what triggers it here
    pd.options.mode.chained_assignment = None
    df[new_cols] = df['journal'].map(journal_to_ASJC).apply(pd.Series)
    df = df[['doi_str'] + new_cols]

    fp_out = rf'../dataframes/df_subjects_Jan21.csv'
    save_csv_pkl(df, fp_out)
    print('Saved df_subjects')


def make_SNIP_df():
    def get_SNIP(journal, year):
        try:
            return journal_to_SNIP[journal][year]
        except KeyError:
            try:
                d_journal = journal_to_SNIP[journal]
                while year < 2024:
                    year += 1
                    if year in d_journal:
                        return d_journal[year]
                return np.nan
            except KeyError:
                return np.nan
        except TypeError:
            return np.nan

    def check_if_exact_SNIP(journal, year):
        if journal in journal_to_SNIP:
            if journal_to_SNIP[journal] is None:
                return np.nan
            if year in journal_to_SNIP[journal]:
                return 1
            else:
                return 0
        else:
            return np.nan

    with open(rf'../cache/journal_to_SNIP_Jan21.pkl', 'rb') as f:
        journal_to_SNIP = pickle.load(f)
    for journal, d in journal_to_SNIP.items():
        if d is None:
            journal_to_SNIP[journal] = None
            continue

        journal_to_SNIP[journal] = {int(year): v for year, v in d.items()}

    df = read_csv_fast(rf'../dataframes/df_has_results_Aug24.csv')
    df = df[df['has_results'] == True]
    covered_journals = set(journal_to_SNIP.keys())
    df = df[df['journal'].isin(covered_journals)]

    df['year'] = df['year'].astype(int)
    df['exact_SNIP'] = df[['journal', 'year']].apply(
        lambda row: check_if_exact_SNIP(row['journal'], row['year']), axis=1)
    print(df['exact_SNIP'].value_counts(dropna=True, normalize=True))

    df['SNIP'] = df[['journal', 'year']].apply(
        lambda row: get_SNIP(row['journal'], row['year']), axis=1)
    df['SNIP'] = df['SNIP'].astype(float)

    num_nan = df['SNIP'].isna().sum()
    print(f'Number with missing SNIP: {num_nan}')
    fp_out = rf'../dataframes/df_SNIP_Jan21.csv'
    save_csv_pkl(df[['doi_str', 'SNIP']], fp_out)
    print('Saved df_jif')
    # Sorry but the _Aug24.csv cannot be produced again. I foolishly deleted my
    #   "journal_to_SCOPUS_subject.pkl" from when I originally made it in august
    #   now, when make_journal_d_scopus.py is run, the API will return slightly
    #   different things (particularly for 2024 papers). The difference is very small
    #   and I believe stems from more paper being added to Scopus
    # Just use df_SNIP_Aug24.csv if you want to reproduce the paper

if __name__ == '__main__':
    make_subject_df()
    make_SNIP_df()
