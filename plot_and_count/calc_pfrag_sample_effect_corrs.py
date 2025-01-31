import scipy.stats as stats

from utils import read_csv_fast

if __name__ == '__main__':
    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)

    df.dropna(subset=['d_sig', 't_N_sig', 'p_fragile'], inplace=True)

    eff = df['d_sig']
    N = df['t_N_sig']
    p_frag = df['p_fragile']
    print(f'{len(df)=:,}')
    r_N_p, _ = stats.spearmanr(N, p_frag)
    r_eff_N, _ = stats.spearmanr(eff, N)
    r_eff_p, _ = stats.spearmanr(eff, p_frag)
    print(f'Sample size (N) x p_fragile: {r_N_p=:.3f}')
    print(f'Sample size (N) x effect size (d): {r_eff_N=:.3f}')
    print(f'effect size (d) x p_fragile: {r_eff_p=:.3f}')
