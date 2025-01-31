from copy import deepcopy
from functools import cache

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker as mtick
from scipy import stats
from tqdm import tqdm

from utils import read_csv_fast, save_csv_pkl


def gaussian_func(X, M, sigma=1):
    # Just a unit (sigma = 1) gaussian function centered at M.
    #   To be clear, no truncation is implemented here.
    return np.exp(-(X - M) ** 2 / (2 * sigma ** 2))


def evaluate_fit(Z, M):
    # Calculate the likelihood of a test-stat (z) from a gaussian centered at M.
    #   Then, calculate the percentage of the guassian that is to the right of
    #   1.96 (the z-value for p = .05). Then, divide the likelihood of the
    #   gaussian by this percentage. Then, we take the produce of every
    #   test-stat's likelihod divided by the percentage.
    # This procedure above is all done in terms of log-likelihoods to avoid
    #   numerical underflow (i.e., taking products like .01 ** 100).
    # Note that np.sum(np.log(...) is equal to np.log(np.prod(...)).

    p = gaussian_func(Z, M)
    LL_total = np.sum(np.log(p))

    X_range = np.linspace(-6, 6, 1000)
    X_low_range = X_range[X_range < 1.96]
    p_low_range = gaussian_func(X_low_range, M)
    X_high_range = X_range[X_range > 1.96]
    p_high_range = gaussian_func(X_high_range, M)
    p_range_rel = (np.sum(p_high_range) /
                   (np.sum(p_high_range) + np.sum(p_low_range)))
    LL_range_total = np.log(p_range_rel) * len(Z)

    LL_total_rel = LL_total - LL_range_total
    return LL_total_rel


def zcurve(Zs, resolution=3, verbose=0):
    # The goal is to identify the mean for which a folded normal maximizes
    #   the likelihood of the observed z-scores. In turn, we calculate the
    #   implied plot_and_count power and return it.
    # The resolution, represents how precise we want to calculate power
    #   (e.g., 2 means 1% increments, 3 means 0.1% increments, etc.).
    # To accomplish this, we use a binary search. We start by considering the
    #   M associated with 50% power, test the M's fit, then test the Ms
    #   associated with 25% power or 75% power. If 25% or 75% are better fits
    #   we move to them. Suppose 25% is better. We would then test 12.5% vs.
    #   37.5% power, and so on with the increment halving each time. Given the
    #   monotonic relationship between M and power and given that we are only
    #   interested in optimizing one M (i.e., as opposed to a mixture model with
    #   multiple Ms/gaussians, like Z-Curve 2), then this binary search will
    #   always arrive at the optimal solution.
    # For information on what exactly is being optimized (the fit of a truncated
    #   gaussian, centered at M and folded at 1.96), see evaluate_fit

    power_l = get_power_l(resolution)
    num_iter = np.log2(len(power_l))
    num_iter = np.ceil(num_iter).astype(int) - 2

    ten_pow = 10 ** resolution
    power = ten_pow // 2
    cur_fit = evaluate_fit(Zs, power_l[power])
    incrament = ten_pow // 4

    idx05 = ten_pow // 20  # minimum possible power (power cant be below 5%)
    for _ in range(num_iter):
        idx_low = max(power - incrament, idx05)
        idx_high = min(power + incrament, ten_pow - 1)
        lower_fit = evaluate_fit(Zs, power_l[idx_low])
        upper_fit = evaluate_fit(Zs, power_l[idx_high])
        if verbose:
            print(f'{power} | {incrament} | {cur_fit=:.1f}, {lower_fit=:.1f}, '
                  f'{upper_fit=:.1f}')
        if lower_fit > cur_fit:
            power -= incrament
            cur_fit = lower_fit
            incrament = incrament // 2
        elif upper_fit > cur_fit:
            power += incrament
            cur_fit = upper_fit
            incrament = incrament // 2
        else:
            incrament = incrament // 2

    power = power / ten_pow
    if verbose: print(f'Final: {power:.1%}')
    return power


def M2power(M):
    power = 1 - stats.norm.cdf(1.96 - M) + stats.norm.cdf(-1.96 - M)
    return power


@cache
def get_power_l(resolution=2):
    ten_pow = 10 ** resolution
    power2M = np.full(ten_pow, np.nan)  # 999 corresponds to 99.9% power
    for M in np.linspace(0, 6, 6 * ten_pow + 1):
        power = M2power(M)
        power_round = int(power * ten_pow + .5)
        if power_round >= ten_pow:
            power_round = ten_pow - 1
            if np.isnan(power2M[power_round]):
                power2M[power_round] = M
            break
        if np.isnan(power2M[power_round]):
            power2M[power_round] = M
        else:
            continue
    cnt_nan = np.sum(np.isnan(power2M[int(ten_pow * .05):]))
    assert cnt_nan == 0
    return power2M


def make_zcurve_df(req_num_p=1, only_interesting=True,
                   classic=True):
    fp_p_by_p = fr'../dataframes/df_by_pval_Jan21.csv'
    df_p_by_p = read_csv_fast(fp_p_by_p, check_dup=False)
    df_p_by_p = deepcopy(df_p_by_p)

    df_p_by_p.dropna(subset=['p_implied'], inplace=True)

    # sometimes extremely small implied p-values became 0, which causes z = inf
    df_p_by_p.loc[df_p_by_p['p_implied'] < 1e-10, 'p_implied'] = 1e-10
    df_p_by_p['z'] = stats.norm.isf(df_p_by_p['p_implied'] / 2)

    df_p_by_p.dropna(subset=['z'], inplace=True)

    if only_interesting:
        df_p_by_p.loc[df_p_by_p['z'] > 6, 'z'] = np.nan
        df_p_by_p.dropna(subset=['z'], inplace=True)

    df_p_by_p_ = df_p_by_p[['doi_str', 'sign', 'p_implied', 'p_val', 'z']]

    dois = []
    powers = []
    p_frgaile_implieds = []
    for doi_str, df_doi in tqdm(df_p_by_p_.groupby('doi_str'),
                                desc='z curving papers'):
        df_sig = df_doi[df_doi['p_implied'] < .05]
        if len(df_sig) < req_num_p:  # Even for a single p-value, the fit
            # seems reasonable.
            # See sanity_test_zcurve
            continue
        p_fragile_implied = (df_sig['p_implied'] > .01).astype(int).mean()
        if classic:
            power = zcurve(df_sig['z'].values)
        else:
            power = zcurve_single(df_sig['z'].values)

        dois.append(doi_str)
        powers.append(power)
        p_frgaile_implieds.append(p_fragile_implied)
    df_out = pd.DataFrame({'doi_str': dois, 'implied_power': powers,
                           'p_fragile_implied': p_frgaile_implieds})
    r, p = stats.spearmanr(df_out['p_fragile_implied'], df_out['implied_power'],
                           nan_policy='omit')
    plt.scatter(df_out['p_fragile_implied'], df_out['implied_power'], alpha=.01)
    plt.ylabel('implied_power')
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.xlabel('p_fragile_implied')
    plt.xlim(0, 1)
    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.show()

    print(f'{classic=}, {only_interesting=}')
    print(f'p_fragile_implied x power: {r=:.5f}')

    # p_fragile is primarily calculated and saved elsewhere.
    #   The above test is just a quick result
    df_out = df_out[['doi_str', 'implied_power']]
    if classic:
        if only_interesting:
            df_out.rename(columns={'implied_power': 'implied_power_int'},
                          inplace=True)
            save_csv_pkl(df_out, fr'../dataframes/df_z_curve_int_Aug15.csv')
        else:
            save_csv_pkl(df_out, f'../dataframes/df_z_curve_Aug15.csv')
    else:
        if only_interesting:
            df_out.rename(columns={'implied_power': 'implied_power_int_single'},
                          inplace=True)
            save_csv_pkl(df_out,
                         fr'../dataframes/df_z_curve_int_single_Aug15.csv')
        else:
            df_out.rename(columns={'implied_power': 'implied_power_single'},
                          inplace=True)
            save_csv_pkl(df_out, fr'../dataframes/df_z_curve_single_Aug15.csv')


@cache
def get_single_z_to_power_l():
    d = {}
    for z in np.linspace(1.95, 6, 406):
        z = np.round(z, 2)
        power = zcurve([z])
        d[z] = power
    return d


def get_power_of_z(z):
    z2power = get_single_z_to_power_l()
    z_round = np.round(z, 2)
    return z2power[z_round]


def zcurve_single(vals):
    vals[vals > 6] = 6
    powers = []
    for val in vals:
        power = get_power_of_z(val)
        powers.append(power)
    return np.median(powers)

def power_to_cohens_dz(power, n, alpha=0.05):
    """
    Converts power and sample size to Cohen's dz for a one-sample t-test.

    Parameters:
        power (float): Desired statistical power (e.g., 0.8 for 80% power).
        n (int): Sample size.
        alpha (float): Significance level (default is 0.05).

    Returns:
        float: Cohen's dz.
    """
    # Degrees of freedom for one-sample t-test
    df = n - 1
    from statsmodels.stats.power import TTestPower

    # Create a TTestPower instance
    ttest_power = TTestPower()
    print(f'{n=}, {alpha}, {power=}')
    # Solve for the effect size (delta) that corresponds to the given power
    effect_size = ttest_power.solve_power(effect_size=None, nobs=n, alpha=alpha, power=power, alternative='two-sided')

    # For a one-sample t-test, Cohen's dz is equivalent to the effect size
    return effect_size

def test_power_d():
    fp_power = r'../dataframes/df_z_curve_int_single_Aug15.csv'
    df_power = read_csv_fast(fp_power)

    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df_main = read_csv_fast(fp)
    dois_power = set(df_power['doi_str'])
    dois_main = set(df_main['doi_str'])
    overlap = dois_power.intersection(dois_main)
    df_power = df_power[df_power['doi_str'].isin(overlap)]
    df_main = df_main[df_main['doi_str'].isin(overlap)]
    df = df_power.merge(df_main, on='doi_str')
    df['implied_power'] = df['implied_power_int_single']

    df_y = df.groupby('year')[['implied_power']].median()

    df = df[df['t_N'] > 1]
    df = df[df['implied_power'] > .05]
    df = df[df['implied_power'] < .99]

    # df = df.iloc[::10]

    tqdm.pandas()

    df['d_z'] = (df[['implied_power', 't_N']].
                 progress_apply(lambda x: power_to_cohens_dz(x[0], x[1]),
                                axis=1))
    pd.set_option('display.max_rows', 100)
    df_y = df.groupby('year')[['d_z', 'implied_power']].median()
    print(df_y)



if __name__ == '__main__':
    test_power_d()
    quit()

    make_zcurve_df(only_interesting=True, classic=False)
    make_zcurve_df(only_interesting=False, classic=False)
    make_zcurve_df(only_interesting=True, classic=True)
    make_zcurve_df(only_interesting=False, classic=True)
