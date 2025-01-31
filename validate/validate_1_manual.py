import pandas as pd

from utils import read_csv_fast


def verify_validation(fp_in):
    df_p = read_csv_fast(fp_in, check_dup=False)
    df_manual = pd.read_csv('manual_validation_sheet.csv',
                            encoding='unicode_escape')
    df_manual['doi_str'] = (df_manual['doi'].str.replace('/', '_').str.
                            replace('.', '-'))
    n_entries = len(df_manual)

    has_miss = 0
    for tup in df_manual.itertuples():
        num_ps = tup.num_ps
        doi_str = tup.doi_str
        df_extracted = df_p[df_p['doi_str'] == doi_str]

        num_ps_extracted = len(df_extracted)
        if num_ps_extracted != num_ps:
            has_miss += 1

    if has_miss == 0:
        print(f'Successful validation ({n_entries} hits)')


if __name__ == '__main__':
    verify_validation(fr'../dataframes/df_by_pval_Jan21.csv')
    verify_validation(fr'../dataframes/df_p_processed_by_p_Jan21.csv')

    # There was also a 41th hit... I meant to get 40 but counting is hard...
    # 6 p-values in 10.1177/1046496414532954
