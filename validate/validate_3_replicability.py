from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from utils import read_csv_fast


def classify_non_5050(df, col):
    # Not used. Performs the classification without stratifying to 50/50
    if isinstance(col, list):
        df = df.dropna(subset=col)
    else:
        df = df.dropna(subset=[col])
        col = [col]

    if isinstance(col, list):
        X = df[col].values
    else:
        X = df[col].values.reshape(-1, 1)
    y = df['replicated'].values
    loo = LeaveOneOut()
    clf = LogisticRegression()
    scores = cross_val_score(clf, X, y, cv=loo)
    acc = scores.mean()
    print(f'\t{acc=:.1%}, N = {len(df)} | {col}')


def classify_binary(df, col):
    # fit svm import as needed, use cross val score
    np.random.seed(0)
    if isinstance(col, list):
        df = df.dropna(subset=col)
    else:
        df = df.dropna(subset=[col])
        col = [col]

    accs_all = []
    doi_prev = None
    print()
    for _ in tqdm(range(1_000), desc='clf', position=0, leave=True):
        # stratify then run
        df_yes = df[df['replicated'] == 1]
        df_no = df[df['replicated'] == 0]

        assert len(df_yes) < len(df_no)
        df_ = pd.concat([df_yes, df_no.sample(len(df_yes))])
        assert len(df_) == 2 * len(df_yes)
        if doi_prev is None:
            doi_prev = set(df_['doi_str'])
        else:
            doi_new = set(df_['doi_str'])
            overlap = len(doi_prev.intersection(doi_new)) / len(doi_prev)
            assert overlap < .999
            doi_prev = doi_new

        if isinstance(col, list):
            X = df_[col].values
        else:
            X = df_[col].values.reshape(-1, 1)
        y = df_['replicated'].values
        M_y = np.mean(y)
        assert M_y == .5
        skf = StratifiedKFold(n_splits=len(df_yes))
        clf = LogisticRegression()
        scores = cross_val_score(clf, X, y, cv=skf)
        acc = scores.mean()
        accs_all.append(acc)

    acc = np.mean(accs_all)
    print(f'Cross-validated accuracy: {acc:.1%}, N = {len(df)} | {col}')


def get_Uzzi_true_replications():
    df_r = pd.read_csv('../dataframes/Youyou_etal_replications.csv',
                       encoding='unicode_escape')
    df_r['doi_str'] = df_r['doi'].str.replace('/', '_').str.replace('.', '-')
    df_r['replicated'] = df_r['replicated_binary'] == 'yes'
    df_r['replicated'] = df_r['replicated'].astype(int)
    return df_r


def get_df_ReD(drop_dup=True):
    df_r = pd.read_csv('../dataframes/FORRT_2024-08-10.csv',
                       encoding='unicode_escape')
    df_r.rename(columns={'doi_original': 'doi'}, inplace=True)
    df_r['doi_str'] = df_r['doi'].str.replace('/', '_').str.replace('.', '-')
    renamer = {'informative failure to replicate': 0,
               'success': 1}
    df_r['replicated'] = df_r['result'].apply(
        lambda x: renamer[x] if x in renamer else pd.NA)
    df_r.dropna(subset=['replicated'], inplace=True)
    if drop_dup:
        df_r.drop_duplicates('doi_str', inplace=True, keep='first')  # most recent
    return df_r


def get_df_replicated():
    df_Uzzi = get_Uzzi_true_replications()
    df_ReD = get_df_ReD(drop_dup=True)
    df_r = pd.concat([df_Uzzi, df_ReD])
    df_r.drop_duplicates('doi_str', inplace=True)
    print(f'Total replication dataset size (duplicates dropped): {len(df_r)}')
    print(f'\tUzzi et al. replication dataset size: {len(df_Uzzi)}')
    print(f'\tReD replication dataset size: {len(df_ReD)}')

    return df_r


def get_cutoff_plane(df, col):
    # Note that the dataset for here is intentionally not stratified
    formula = 'replicated ~ ' + col
    prop_rep = df['replicated'].mean()
    ratio_rep = (1 - prop_rep) / prop_rep
    weights = df['replicated'].apply(
        lambda x: ratio_rep if x == 1 else 1)
    weights = weights / weights.sum()
    weights = [1] * len(weights)
    glm = smf.glm(formula, data=df, family=sm.families.Binomial(),
                  freq_weights=weights)
    res = glm.fit(disp=0)

    df_pred = pd.DataFrame({col: df[col]})
    df_pred['replicated_pred'] = res.predict(df) > .5
    df_pred['acc'] = df_pred['replicated_pred'] == df['replicated']
    accuracy = df_pred['acc'].mean()

    cutoff = -res.params['Intercept'] / res.params[col]
    print(f'Logistic regression cutoff plane: {cutoff=:.3f} '
          f'(non-CV accuracy: {accuracy:.1%})')
    return cutoff


def plot_jitterplot(df_y, df_n, col):
    plt.figure(figsize=(3.2, 4))

    jitter = np.random.uniform(-0.1, 0.1, size=len(df_y))
    plt.scatter(jitter + 0.5, df_y[col], alpha=.5, color='dodgerblue',
                linewidth=0.5)
    jitter = np.random.uniform(-0.1, 0.1, size=len(df_n))
    plt.scatter(jitter + 1.5, df_n[col], alpha=.5, color='red',
                linewidth=0.5)
    plt.xlim(0, 2)
    plt.ylim(-.05, 1.02)
    plt.xticks([0.5, 1.5], ['Replicated', 'Not\nreplicated'],
               fontsize=12.5)
    plt.yticks(fontsize=12.5)
    plt.gca().spines[['right', 'top', 'bottom']].set_visible(False)
    df = pd.concat([df_y, df_n])
    plane = get_cutoff_plane(df, col)

    assert (df_n[col] == plane).sum() == 0 and (df_y[col] == plane).sum() == 0

    num_yes_above = len(df_y[df_y[col] > plane])
    num_yes_below = len(df_y[df_y[col] < plane])
    num_no_above = len(df_n[df_n[col] > plane])
    num_no_below = len(df_n[df_n[col] < plane])

    prop_below_yes = num_yes_below / (num_yes_below + num_no_below)
    prop_above_no = num_no_above / (num_no_above + num_yes_above)

    plt.plot([0.2, 1.8], [plane, plane], color='k', linestyle='--',
             linewidth=1.25)
    plt.text(1.82, plane,
             f'{plane:.0%}', fontsize=12.5, ha='left', va='center')

    plt.text(1, plane + 0.013, f'{1 - prop_above_no:.1%}', fontsize=12.5, ha='center',
             va='bottom', color='red')

    plt.text(1, plane - 0.025, f'{prop_below_yes:.1%}', fontsize=12.5,
             ha='center', va='top', color='dodgerblue')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    Path(r'../figs_and_tables').mkdir(parents=True, exist_ok=True)

    if col == 'p_fragile_implied':
        plt.ylabel('Fragile implied p-values (%)', fontsize=12.5)
        plt.tight_layout()
        plt.savefig('../figs_and_tables/Figure_S9_replicability_implied.png',
                    dpi=600)
    else:
        plt.ylabel('Fragile p-values (.01 < p < .05)', fontsize=12.5)
        plt.tight_layout()
        plt.savefig('../figs_and_tables/Figure_S5_replicability.png', dpi=600)
    plt.show()


def report_ttest(df_y, df_n, col):
    My = df_y[col].mean()
    SDy = df_y[col].std()
    SEy = SDy / np.sqrt(len(df_y))
    Mn = df_n[col].mean()
    SDn = df_n[col].std()
    SEn = SDn / np.sqrt(len(df_n))
    t, p = stats.ttest_ind(df_y[col], df_n[col], nan_policy='omit')
    N = len(df_n) + len(df_y)
    mult = np.sqrt(1 / len(df_y) + 1 / len(df_n))
    d = t * mult

    print(f't-test, {col}: t[{N - 1}] = {t:.2f}, {p=:.4f}, {d=:.2f}\n'
          f'\tMean replicated = {My:.2f} [SE = {SEy:.2f}], '
          f'\tMean not replicated = {Mn:.2f} [SE = {SEn:.2f}]')


def validate_replicability(col='p_fragile'):
    fp = fr'../dataframes/df_combined_Jan21.csv'
    df = read_csv_fast(fp, easy_override=False, check_dup=False)
    df_r = get_df_replicated()
    doi_strs = set(df['doi_str'])
    df_r['is_in'] = df_r['doi_str'].apply(lambda x: x in doi_strs)

    df_r = df_r[df_r['is_in']]
    doi_strs_r = set(df_r['doi_str'])
    df = df[df['doi_str'].apply(lambda x: x in doi_strs_r)]
    df = df_r.merge(df, on='doi_str', how='left')
    df = df[df['has_results'] == True]

    df['replicated'] = df['replicated'].astype(int)
    df = df.dropna(subset=[col])
    print(f'Overlapping dataset size: {len(df)}')
    df_y = df[df['replicated'] == 1]
    print(f'\tYes replicated: {len(df_y)} ({len(df_y) / len(df):.1%})')
    df_n = df[df['replicated'] == 0]
    print(f'\tNot replicated: {len(df_n)} ({len(df_n) / len(df):.1%})')

    plot_jitterplot(df_y, df_n, col)
    report_ttest(df_y, df_n, col)
    classify_binary(df, col)
    print('-*-' * 10)


if __name__ == '__main__':
    validate_replicability(col='p_fragile')
    validate_replicability(col='p_fragile_implied')
