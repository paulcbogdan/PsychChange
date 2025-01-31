from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker as mtick, lines
from scipy import stats as stats
from statsmodels.formula import api as smf

from utils import read_csv_fast

pd.options.mode.chained_assignment = None


def proc_country(country):
    euro = {'Luxembourg', 'Greece', 'Portugal', 'Belgium',
            'Lithuania', 'Norway', 'Austria', 'Malta', 'Hungary', 'Iceland',
            'Spain', 'Latvia', 'Finland', 'France', 'Italy', 'Sweden',
            'Slovakia', 'Cyprus', 'Czech Republic', 'Switzerland',
            'Croatia', 'Denmark', 'Turkey', 'Romania', 'Germany',
            'Poland', 'Northern Cyprus', 'Ireland',
            'Slovenia', 'Serbia', 'Netherlands', 'Estonia',
            'Russian Federation'}

    anglo = {'United States', 'Canada', 'United Kingdom', 'Australia',
             'New_Zealand', }

    asia = {'Hong Kong', 'Macao', 'Japan', 'South Korea', 'Vietnam',
            'Singapore', 'Pakistan', 'India', 'Taiwan',
            'Philippines', 'Malaysia', 'Indonesia', 'Brunei Darussalam',
            'China'}

    if country in euro:
        return 'Europe'
    elif country in anglo:
        return 'Anglo'
    elif country in asia:
        return 'Asia'
    else:
        return 'Other'


def control_vars(df, covs, target='p_fragile'):
    formula = fr'{target} ~ 1 + ' + ' + '.join(covs)
    mod = smf.ols(formula=formula, data=df)
    res = mod.fit()
    df[target] = res.resid + df[target].mean()
    return df


def plot_scatter(key='p_fragile', bias_adjustment=.023):
    fp = fr'../dataframes/df_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False)

    df['region'] = df['country'].apply(proc_country)
    df.dropna(subset=[key], inplace=True)

    if 'fragile' in key:
        df = control_vars(df, ['year'], target=key)
    if key == 'p_fragile':
        df['p_fragile'] -= bias_adjustment

    median_rank = df['target_rank'].median()
    mean_rank = df['target_rank'].mean()
    print(f'{median_rank=}, {mean_rank=:.1f}')

    sphere2color = {'Europe': 'green',
                    'Anglo': 'royalblue',
                    'Asia': 'r',
                    'Other': 'purple'}
    sphere2marker = {'Europe': 'o',
                     'Anglo': 's',
                     'Asia': '^',
                     'Other': 'X'}

    pd.set_option('display.max_rows', 1000)

    plt.figure(figsize=(6, 5))

    colors = []
    markers = []
    y = []
    x = []
    num_schools = 0
    for school, df_school in df.groupby('school'):
        assert df_school['target_rank'].nunique() == 1, \
            f'Too much: {school}: {df_school["target_rank"].unique()=}'
        rank = df_school['target_rank'].iloc[0]
        df_school = df_school

        M_score = df_school[key].mean()
        if 'fragile' in key:
            if M_score < .10: continue
            SE_score = df_school[key].std() / np.sqrt(len(df_school))
        else:  # proportions
            SE_score = np.sqrt((M_score * (1 - M_score)) / len(df_school))
        if SE_score > .04:
            continue
        if len(df_school) < 10:
            continue
        if np.isnan(M_score):
            continue
        y.append(M_score)
        x.append(rank)
        num_schools += 1
        sphere = df_school['region'].iloc[0]
        colors.append(sphere2color[sphere])
        markers.append(sphere2marker[sphere])

    for x_, y_, c_, m_ in zip(x, y, colors, markers):
        plt.scatter(x_, y_, color=c_, marker=m_, alpha=.4)
    print(f'{num_schools=}')
    r, p = stats.spearmanr(x, y)

    plt.title('University ranking', fontsize=14)
    if key == 'p_fragile':
        plt.ylabel('Fragile p-value (%)', fontsize=14, labelpad=8)
    elif key == 'p_fragile_implied':
        plt.ylabel('Fragile implied p-value (%)', fontsize=14, labelpad=8)
    elif key == 'school_M_baye':
        plt.ylabel('Bayesian papers (%)', fontsize=14, labelpad=8)
    elif key == 'school_M_ML':
        plt.ylabel('Machine learning papers (%)', fontsize=14, labelpad=8)
    else:
        raise ValueError
    plt.yticks(fontsize=12)
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.gca().set_facecolor('whitesmoke')
    plt.grid(color='lightgray', linestyle='-', linewidth=0.5, alpha=0.4)

    if key == 'p_fragile':
        pos = (850, .399)
    elif key == 'p_fragile_implied':
        pos = (850, .372)
    elif key == 'school_M_baye':
        pos = (850, .11)
        plt.ylim(-.0038, None)
    elif key == 'school_M_ML':
        pos = (850, .068)
        plt.ylim(-.0025, None)

    plt.text(pos[0], pos[1], '$r_{' + str(len(x) - 2) + '}$ = ' +
             f'{-r:.2f}', fontsize=14,
             va='top', ha='left')

    df_grp = pd.DataFrame({key: y, 'target_rank': x})
    mod = smf.ols(formula=f'{key} ~ 1 + target_rank', data=df_grp)
    res = mod.fit()
    df_pred = pd.DataFrame({'target_rank': np.arange(1, 1001)})
    pred = res.predict(exog=df_pred)
    plt.plot(np.arange(1, 1001), pred, color='k',
             linestyle='--', alpha=.8)

    plt.xticks([1000, 750, 500, 250, 1],
               labels=['#1000', '#750', '#500', '#250', '#1'], fontsize=12)
    plt.xlim(1020, -20)
    plt.gca().spines[['right', 'top', ]].set_visible(False)

    s = 8
    a = .5
    blue_sq = lines.Line2D([], [], color='blue',
                           marker='s', linestyle='None',
                           markersize=s, label='Anglo',
                           alpha=a)
    green_o = lines.Line2D([], [], color='green',
                           marker='o', linestyle='None',
                           markersize=s, label='Europe',
                           alpha=a)
    red_tri = lines.Line2D([], [], color='red',
                           marker='^', linestyle='None',
                           markersize=s, label='Asia',
                           alpha=a)
    purple_x = lines.Line2D([], [], color='purple',
                            marker='X', linestyle='None',
                            markersize=s, label='Other',
                            alpha=a)

    plt.figlegend(handles=[blue_sq, green_o, red_tri, purple_x],
                  ncol=4, frameon=False, columnspacing=0.7,
                  handletextpad=0.05,
                  bbox_to_anchor=[0.562, 0.0],
                  loc='lower center',
                  fontsize=13)

    plt.subplots_adjust(left=0.15,
                        bottom=0.15,
                        right=0.985,
                        top=0.925,
                        wspace=0.24,
                        hspace=.36)
    Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)
    if key == 'school_M_baye':
        plt.savefig('../figs_and_tables/Figure_S8A_corr_Baye.png', dpi=600)
    elif key == 'school_M_ML':
        plt.savefig('../figs_and_tables/Figure_S8B_corr_ML.png', dpi=600)
    elif key == 'p_fragile':
        plt.savefig('../figs_and_tables/Figure_5C_prestige.png', dpi=600)
    elif key == 'p_fragile_implied':
        plt.savefig('../figs_and_tables/Figure_S11C_prestige_p_implied.png',
                    dpi=600)
    else:
        raise ValueError

    plt.show()


if __name__ == '__main__':
    for key in ['p_fragile', 'p_fragile_implied', 'school_M_baye',
                'school_M_ML']:
        plot_scatter(key=key)
