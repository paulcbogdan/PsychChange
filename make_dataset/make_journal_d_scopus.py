import pickle
from pathlib import Path

import pandas as pd
import pybliometrics
from crossref.restful import Works
from pybliometrics.scopus import SerialTitle
from pybliometrics.scopus.exception import Scopus404Error, Scopus401Error
from tqdm import tqdm

from utils import read_csv_fast

pd.options.mode.chained_assignment = None

if __name__ == '__main__':
    pybliometrics.scopus.init()

    works = Works()
    df = read_csv_fast(rf'../dataframes/df_has_results_Aug24.csv')

    df = df[df['has_results'] == True]

    df.sort_values('year', inplace=True, ascending=False)
    df_per_journal = df.groupby('journal').first()  # also includes some books
    df_per_journal.reset_index(inplace=True)

    d_subject = {}
    d_yr2SNIP = {}
    bad_journals = []
    for row in tqdm(df_per_journal.itertuples(),
                    total=df_per_journal.shape[0],
                    desc='Making journal to ASJC'):
        ISSN = row.ISSNs
        if pd.isna(ISSN):
            print(f'Missing ISSN key: {row.journal}')
            bad_journals.append(row.journal)
            d_subject[row.journal] = None
            continue
        backup_ISSN = row.backup_ISSN
        try:
            res = SerialTitle(ISSN, years='2004-2024', view='ENHANCED')
        except Scopus404Error as e:
            if pd.isna(backup_ISSN):
                print(f'Bad ISSN: {row.journal}: {e}')
                bad_journals.append(row.journal)
                d_subject[row.journal] = None
                continue
            else:
                print('Trying backup ISSN after 404 error')
                try:
                    res = SerialTitle(backup_ISSN, years='2004-2024',
                                      view='ENHANCED')
                    print('\tSuccesfully used backup!')
                except Scopus404Error as e:
                    print(f'Bad ISSNs: {row.journal}: {e}')
                    bad_journals.append(row.journal)
                    d_subject[row.journal] = None
                    continue
        except Scopus401Error as e:
            # 10,000 or so lines in I cba to turn below into a function...
            if pd.isna(backup_ISSN):
                bad_journals.append(row.journal)
                d_subject[row.journal] = None
                continue
            else:
                print('Trying backup ISSN after 401 error')
                try:
                    res = SerialTitle(backup_ISSN, years='2004-2024',
                                      view='ENHANCED')
                    print('\tSuccesfully used backup!')
                except Scopus404Error as e:
                    print(f'???: {row.journal}: {e}')
                    bad_journals.append(row.journal)
                    d_subject[row.journal] = None
                    continue

        d_yr2SNIP[row.journal] = {}
        try:
            SNIP_info = vars(res)['_entry']['SNIPList']['SNIP']
            for entry in SNIP_info:
                year = entry['@year']
                SNIP = entry['$']
                d_yr2SNIP[row.journal][int(year)] = SNIP
        except KeyError as e:
            print(f'No SNIP: {row.journal}')
            d_yr2SNIP[row.journal] = None
            bad_journals.append(row.journal)

        try:
            subject_areas = res.subject_area
        except KeyError as e:
            print(f'No subject areas: {row.journal}')
            bad_journals.append(row.journal)
            d_subject[row.journal] = None
            continue
        subject_areas = [sa[0] for sa in subject_areas]
        d_subject[row.journal] = subject_areas
    print(f'{bad_journals=}')

    df_ = df[df['journal'].isin(bad_journals)]
    n_all = len(df)
    n_bad = len(df_)
    print(f'Bad journals: {n_bad}/{n_all} ({n_bad / n_all:.1%})')  # Only ~1.2%
    fp_out = rf'../cache/journal_to_SCOPUS_subject_Jan21.pkl'

    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    with open(fp_out, 'wb') as f:
        pickle.dump(d_subject, f)

    fp_out = rf'../cache/journal_to_SNIP_Jan21.pkl'

    with open(fp_out, 'wb') as f:
        pickle.dump(d_yr2SNIP, f)
