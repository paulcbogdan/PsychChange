import re
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import pickle_wrap, save_csv_pkl, read_csv_fast

# 'AIC', 'BIC', 'stan' would all be portions of other words (e.g., 'mosaic')
# 'PyMC' dropped due to being in < .01% of papers
PATTERN_BAYE = ('(Bayesian|Information Criterion|Bayes|'
                'log-likelihood|credible interval|'
                'Markov chain Monte Carlo|Gibbs sampling|'
                'Metropolis-Hastings algorithm|'
                'hierarchical modeling|conjugate prior|'
                'information criterion|Beta distribution|'
                'Dirichlet distribution|Gaussian distribution|'
                'Bayes factor|marginal likelihood|'
                'posterior distribution|No-U-Turn sampler|'
                'Hamiltonian Monte Carlo|Gaussian process|'
                'Kullback-Leibler|KL divergence|Jeffrey\'s prior|'
                'Maximum a posteriori|Variational inference|'
                'informative prior|'
                'Posterior likelihood|'
                'Convergence diagnostics|Posterior odds|Prior odds|'
                'Posterior probability|Prior probability)'
                ).lower()
BAYE_WORDS = list(PATTERN_BAYE.replace('(', '').replace(')', '').split('|'))

PATTERN_FREQ = ('(ANOVA|t-test|F-test|chi-square test|linear regression|'
                'null hypothesis|alternative hypothesis|'
                'Structural equation modeling|Analysis of variance|'
                'Analysis of Covariance|MANOVA|Nonparametric tests|Z-test|'
                'Mann-Whitney U test|Wilcoxon signed-rank test|'
                'Kolmogorov-Smirnov test|Shapiro-Wilk test|'
                'Bonferroni correction|False Discovery Rate|Effect size|'
                'Cohen\'s d|Cohen\'s f|Cohen\'s kappa|Fisher\'s exact test|'
                'Levene\'s test|Tukey\'s Honest Significant Difference|'
                'Tukey\'s HSD|Spearman\'s rank|Spearman rank|'
                'Spearman\'s correlation|Spearman correlation|'
                'Factor Analysis|Multicollinearity|Jackknife resampling|'
                'multilevel modeling|mixed effects model|'
                'multilevel regression)'
                ).lower()
FREQ_WORDS = list(PATTERN_FREQ.replace('(', '').replace(')', '').split('|'))

# BERT, GRU, GAN are parts of =words like AIC above
# 'reinforcement learning' is also a psychology term
# 'overfitting', 'underfitting', 'dropout' don't seem specific to ML enough
PATTERN_ML = ('(machine learning|deep learning|neural network|'
              'artificial intelligence|support vector machine|'
              'random forest|gradient boosting|XGBoost|K-means clustering|'
              'K-nearest neighbors|principal component analysis|'
              'natural language processing|convolutional neural network|'
              'recurrent neural network|long short-term memory|'
              'Gated Recurrent Unit|transformer network|autoencoder|'
              'generative adversarial network|unsupervised learning|'
              'supervised learning|semi-supervised learning|'
              'transfer learning|ensemble learning|hyperparameter tuning|'
              'Confusion matrix|'
              'Distributed Stochastic Neighbor Embedding|t-SNE|Word embedding|'
              'Ensemble methods|Training set|Test set|Validation set|'
              'Feature selection|F1 score|Testing set|Train set)'
              ).lower()
ML_WORDS = list(PATTERN_ML.replace('(', '').replace(')', '').split('|'))


def detect_text_paper(fp_text):
    with open(fp_text, 'r', encoding='utf-8') as file:
        plain_text = file.read()
    plain_text = plain_text.lower()

    baye_matches = re.finditer(PATTERN_BAYE, plain_text)
    baye_cnt = defaultdict(int)
    for m in baye_matches:
        baye_cnt[m[0]] += 1

    freq_matches = re.finditer(PATTERN_FREQ, plain_text)
    freq_cnt = defaultdict(int)
    for m in freq_matches:
        freq_cnt[m[0]] += 1

    ML_matches = re.finditer(PATTERN_ML, plain_text)
    ML_cnt = defaultdict(int)
    for m in ML_matches:
        ML_cnt[m[0]] += 1

    return baye_cnt, freq_cnt, ML_cnt


def detect_whole_text():
    fp_in = fr'../dataframes/df_has_results_Aug24.csv'
    df = read_csv_fast(fp_in, easy_override=False)
    df = df[df['has_results'] == True]

    l = []
    pre_cols = list(df.columns)
    for row in tqdm(df.itertuples(), desc='running papers_df',
                    total=df.shape[0]):
        fp_text = (fr'..\data\plaintexts_res_Aug24\{row.publisher}\{row.journal}' +
                   fr'\{row.year}\{row.doi_str}.txt')
        fp_pkl = f'../cache/text_detect/res_{row.doi_str}.pkl'
        baye_cnt, freq_cnt, ML_cnt = pickle_wrap(
            detect_text_paper, fp_pkl, kwargs={'fp_text': fp_text, },
            verbose=-1, easy_override=False,
            dt_max=datetime(2024, 8, 24, 10, 0, 0, 0))

        d = row._asdict()
        d.update(baye_cnt)
        d.update(freq_cnt)
        d.update(ML_cnt)
        l.append(d)
    df = pd.DataFrame(l)

    baye_cols = [col for col in df.columns if col in BAYE_WORDS]
    freq_cols = [col for col in df.columns if col in FREQ_WORDS]
    ML_cols = [col for col in df.columns if col in ML_WORDS]
    df.fillna(0, inplace=True)
    df = df[pre_cols + baye_cols + freq_cols + ML_cols]

    df[baye_cols] = (df[baye_cols] > 0).astype(int)
    df[freq_cols] = (df[freq_cols] > 0).astype(int)
    df[ML_cols] = (df[ML_cols] > 0).astype(int)
    df['baye_paper'] = df[baye_cols].sum(axis=1) > 0
    df['freq_paper'] = df[freq_cols].sum(axis=1) > 0
    df['ML_paper'] = df[ML_cols].sum(axis=1) > 0
    text_cols = (baye_cols + ['baye_paper'] + freq_cols + ['freq_paper'] +
                 ML_cols + ['ML_paper'])
    N = len(df)
    fp_out = fr'../dataframes/df_text_Aug24.csv'
    save_csv_pkl(df, fp_out)

    num_baye_cols = len(baye_cols)
    num_freq_cols = len(freq_cols)
    print(f'{num_baye_cols=}, {num_freq_cols=}')

    print('--')
    for i in range(len(text_cols)):
        if text_cols[i] in ['baye_paper', 'freq_paper', 'ML_paper']:
            print('-')
        M = df[text_cols[i]].mean()
        print(f'{i:<2} ({M=:.2%}) {text_cols[i]=}')
        if text_cols[i] in ['baye_paper', 'freq_paper', 'ML_paper']:
            print('--')

    corr = df[text_cols].corr()
    print(corr)
    corr = corr.values
    corr[np.diag_indices_from(corr)] = np.nan
    plt.matshow(corr, interpolation='none')
    plt.colorbar()
    plt.show()

    print(df[['ML_paper', 'baye_paper', 'freq_paper']].value_counts())


if __name__ == '__main__':
    detect_whole_text()
