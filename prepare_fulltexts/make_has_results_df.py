import os

from tqdm import tqdm

from utils import read_csv_fast, save_csv_pkl


def make_df_has_results():
    def check_is_res_exists(row):
        fp_text = (fr'..\data\plaintexts_res_Aug24\{row.publisher}' +
                   fr'\{row.journal}\{row.year}\{row.doi_str}.txt')
        return os.path.exists(fp_text)

    def get_res_len(row):
        fp_text = (fr'..\data\plaintexts_res_Aug24\{row.publisher}' +
                   fr'\{row.journal}\{row.year}\{row.doi_str}.txt')
        with open(fp_text, 'r', encoding='utf-8') as file:
            text = file.read()
        text = text.replace('FOUND_RESULTS_SECTION', '').replace('--', '')
        return len(text)

    tqdm.pandas()

    df = read_csv_fast(fr'../dataframes/df_lens_Aug24.csv')
    df.dropna(subset=['doi_str'], inplace=True)
    df.drop_duplicates(subset=['doi_str'], inplace=True)

    df['has_results'] = df.progress_apply(check_is_res_exists, axis=1)
    print(df['has_results'].value_counts())

    df.loc[df['has_results'], 'results_len'] = (
        df[df['has_results']].progress_apply(get_res_len, axis=1))

    # Tiny scraps of text. Sometimes triggered by a Review paper that has an
    #   Results section in the Abstract if the journal requirse it. Note
    #   that 100 represents just 100 characters, not words. Sometimes also
    #   a results section will be saved as just "Results", which is an error.
    #   Omitting papers with results section under 100 characters fixes this.
    df_results = df[df['has_results']]
    df_results['super_short_results'] = df_results['results_len'] < 100
    print(df_results['super_short_results'].value_counts())

    # super_short_results
    # False    393308
    # True       2894
    # Name: count, dtype: int64

    df.loc[df['results_len'] < 100, 'has_results'] = False

    fp_out = fr'../dataframes/df_has_results_Aug24.csv'
    save_csv_pkl(df, fp_out)


if __name__ == '__main__':
    make_df_has_results()
