import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

from utils import read_csv_fast


def sample_from_num_ps():
    # read_csv_fast caches, so its not loaded every time this func is called
    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    return df['num_ps'].sample(20, replace=True)


def sim_80_power_implied_low(beta=2.8, nsim=10_000):
    np.random.seed(0)
    p_fragiles = []

    p_sig_cutoff = .05
    z_sig_cutoff = stats.norm.isf(p_sig_cutoff / 2)
    p_fragile_cutoff = .01
    z_fragile_cutoff = stats.norm.isf(p_fragile_cutoff / 2)

    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    num_ps = df['num_ps'].astype(int).sample(nsim, replace=True)
    prop_sigs = []
    for num_p in num_ps:
        sig_zs = []
        while len(sig_zs) < 1:  # no sig
            zs = np.random.normal(beta, size=num_p)
            sig_zs = zs[zs > z_sig_cutoff]
            if len(sig_zs) == 0:
                prop_sigs.append(0)
            # prop_sigs will be over nsim but that's fine
        prop_sig = len(sig_zs) / len(zs)
        prop_sigs.append(prop_sig)
        p_fragile = np.mean(sig_zs < z_fragile_cutoff)
        p_fragiles.append(p_fragile)

    power = np.mean(prop_sigs)
    print(f'Calculated power: {power:.2%}')
    if beta == 2.8:
        assert 0.798 < power < 0.802
    p_fragiles = np.array(p_fragiles)
    M_fragile = np.mean(p_fragiles)
    print(f'Mean p-fragile expected: {M_fragile:.0%}')

    prop_fragile_over_50 = np.mean(p_fragiles > .5 - 1e-6)
    prop_fragile_under_32 = np.mean(p_fragiles < .319)

    print(f'Percentage of papers with over 50% expected: '
          f'{prop_fragile_over_50:.1%}')
    print(f'Percentage of papers with under 31.9% expected: '
          f'{prop_fragile_under_32:.1%}')


def power2p05_prop(beta, nsim=1000):
    np.random.seed(0)

    p_sig_cutoff = .05
    z_sig_cutoff = stats.norm.isf(p_sig_cutoff / 2)
    p_fragile_cutoff = .01
    z_fragile_cutoff = stats.norm.isf(p_fragile_cutoff / 2)

    zs = np.random.normal(beta, size=nsim)
    zs_sig = zs[zs > z_sig_cutoff]
    n_sig = len(zs_sig)
    power = n_sig / nsim
    zs_fragile = zs_sig[zs_sig < z_fragile_cutoff]
    n_fragile = len(zs_fragile)
    p_fragile = n_fragile / n_sig
    return power, p_fragile


def plot_power2p05_prop():
    fontsize = 14

    plt.rcParams.update({'font.size': fontsize,
                         'font.sans-serif': 'Arial',
                         'figure.figsize': (7, 4), })

    betas = np.linspace(0, 10.0, 1001, endpoint=True)
    powers = []
    p_fragiles = []
    for beta in tqdm(betas, desc='Running betas'):
        power, p_fragile = power2p05_prop(beta, nsim=100_000)
        powers.append(power)
        p_fragiles.append(p_fragile)

    plt.plot(powers, p_fragiles, linewidth=2, color='r')
    plt.xlabel('Power')
    plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.ylabel('Percentage of p-values\n(.01 < p < .05)')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.yticks(fontsize=12)
    plt.yticks(np.linspace(0, 1, 11), fontsize=12)
    plt.xlim(0, 1.002)
    plt.xticks(np.linspace(0, 1, 11), fontsize=12)
    plt.ylim(0, 0.81)
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.75)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_power2p05_prop()
    sim_80_power_implied_low()

    # Show that 44% power causes 50% of significant p-values to be fragile
    sim_80_power_implied_low(beta=1.8)
