import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import read_csv_fast

# disable .loc warning in pd
pd.options.mode.chained_assignment = None


def plot_percent_insig():
    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)

    df['prop_sig'] = 1 - df['sig'] / df['num_ps']

    df_y = df.groupby('year')[['prop_sig', ]].mean()
    df_y_se = df.groupby('year')[['prop_sig', ]].sem()

    v_prop_sig = df_y['prop_sig'].values
    fig, axs = plt.subplots(1, 1, figsize=(6.5, 3.5))
    plt.plot(df_y.index, v_prop_sig, color='red', marker='.',)
    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024, ], fontsize=11)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.ylabel('Insignificant p-values (%)', fontsize=13)
    plt.xlabel('Year', fontsize=13)
    plt.title('Percentage of p-values that are insignificant (p > .05)', fontsize=13)
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.75)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.yticks([.12, .14, .16, .18, .20, .22], fontsize=12)
    plt.ylim(.116, .224)
    plt.xlim(2003.5, 2024.5)
    SE_low = df_y['prop_sig'] - df_y_se['prop_sig']
    SE_high = df_y['prop_sig'] + df_y_se['prop_sig']
    plt.fill_between(df_y.index, SE_low, SE_high, alpha=0.2,
                     color='r', linewidth=0)

    plt.savefig('../figs_and_tables/Figure_S3_insignif_pvals.png', dpi=600)
    plt.show()


def plot_all_p_insig():
    fp = fr'../dataframes/df_combined_w_no_sig_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    df['sig_zero'] = df['sig'] == 0
    df['sig_zero_exact'] = (df['sig'] == 0) & (df['insig_exact'] > 1)
    df = df[df['num_ps_any'] > 1]

    print(f'Papers with at least two p-values: {len(df)}')

    # regress out num_ps from sig_zero
    X = df['num_ps']
    y = df['sig_zero']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    df['sig_zero_reg'] = model.resid
    df['sig_zero_reg'] += df['sig_zero'].mean()

    df['prop_sig'] = 1 - df['sig'] / df['num_ps']
    df_y = df.groupby('year')[['prop_sig', 'sig_zero',
                               'num_ps', 'sig_zero_reg',
                               'sig_zero_exact']].mean()
    # print(df_y)
    df_y_se = df.groupby('year')[['prop_sig', 'sig_zero',
                                  'num_ps', 'sig_zero_reg',
                                  'sig_zero_exact']].sem()
    # print(df_y_se)
    # quit()
    df_y.sort_index(inplace=True)

    fig, axs = plt.subplots(2, figsize=(6.5, 6))
    plt.sca(axs[0])
    v_sig_zero = df_y['sig_zero'].values
    plt.plot(df_y.index, v_sig_zero, color='dodgerblue', marker='.', )
    SE_low = df_y['sig_zero'] - df_y_se['sig_zero']
    SE_high = df_y['sig_zero'] + df_y_se['sig_zero']
    plt.fill_between(df_y.index, SE_low, SE_high, alpha=0.2,
                     color='dodgerblue', linewidth=0)
    plt.ylabel('Papers (%)', fontsize=13)
    plt.title('Percentage of papers reporting only insignificant p-values\n'
              '(papers with 2+ p-values)',
              fontsize=13)
    plt.yticks(fontsize=12)
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.75)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.1%}'))
    plt.xlim(2003.5, 2024.5)
    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024], fontsize=11)

    plt.sca(axs[1])
    plt.title('Percentage of papers reporting only insignificant p-values\n'
              '(papers with only 2+ p-values)', fontsize=13)
    plt.xlabel('Year', fontsize=13)

    v_sig_zero = df_y['sig_zero_exact'].values
    plt.plot(df_y.index, v_sig_zero, color='green', marker='.', )
    SE_low = df_y['sig_zero_exact'] - df_y_se['sig_zero_exact']
    SE_high = df_y['sig_zero_exact'] + df_y_se['sig_zero_exact']
    print(SE_low)
    plt.fill_between(df_y.index, SE_low, SE_high, alpha=0.2,
                     color='green', linewidth=0)
    plt.ylabel('Papers (%)', fontsize=13)
    plt.title('Percentage of papers reporting only insignificant p-values\n'
              '(papers with 2+ exact p-values)',
              fontsize=13)
    plt.yticks(fontsize=12)
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.75)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.1%}'))
    plt.xlim(2003.5, 2024.5)
    plt.xticks([2004, 2008, 2012, 2016, 2020, 2024], fontsize=11)

    plt.subplots_adjust(left=0.17, bottom=0.10,
                        right=0.95, top=0.9,
                        wspace=0.24, hspace=.44)
    plt.savefig('../figs_and_tables/Figure_S4_insignif_papers.png', dpi=600)

    plt.show()


def test_sample_x_p_val():
    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)
    y = df['p_fragile']
    x0 = df['t_N_sig']
    x1 = df['d_sig']
    # regression:
    X = np.column_stack((x0, x1))
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    print(model.summary())


if __name__ == '__main__':
    plot_percent_insig()
