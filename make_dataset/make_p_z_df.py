import os.path
import re
import string
from datetime import datetime

import numpy as np
import pandas as pd
import unicodedata
from scipy import stats
from tqdm import tqdm

from utils import pickle_wrap, read_csv_fast, save_csv_pkl

LETTERS = set(string.ascii_lowercase + string.ascii_uppercase)


def get_parentheses(s, max_len=255):
    openers = {'(': ')', '[': ']'}
    closers = {v: k for k, v in openers.items()}
    stack = []
    result = []

    for i, c in enumerate(s):
        if c in openers:
            stack.append([c, i])
        elif c in closers:
            if not stack:
                stack = []
                continue

            pair, idx = stack.pop()
            result.append([idx, i])

    results_idxs = list(filter(lambda x: x[1] - x[0] < max_len, result))
    results_s = [s[tup[0]:tup[1] + 1] for tup in results_idxs]

    return results_idxs, results_s


def find_period_non_parentheses(sentence_pre, which='last'):
    prev_period_l = [x for x in
                     re.finditer(r'[.!?]\s', sentence_pre)]  # .end()
    prev_period_l = [x.end() for x in prev_period_l]

    # will make sure that sentence_pre = '... (Fig. 6; p < .001'
    #   is not triggered by the '.' in fig
    if which == 'last':
        sentence_pre_paren = sentence_pre + ')'
        parens_idxs, _ = get_parentheses(sentence_pre_paren)
        sentence_pre_brack = sentence_pre + ']'
        parens_idxs += get_parentheses(sentence_pre_brack)[0]
    else:
        sentence_pre_paren = '(' + sentence_pre
        parens_idxs, _ = get_parentheses(sentence_pre_paren)
        parens_idxs = [(x[0] - 1, x[0] - 1) for x in parens_idxs]
        sentence_pre_brack = '[' + sentence_pre
        parens_idxs += get_parentheses(sentence_pre_brack)[0]
        parens_idxs = [(x[0] - 1, x[0] - 1) for x in parens_idxs]

    sentence_start = None
    if which == 'last':
        prev_period_l = prev_period_l[::-1]
    for x in prev_period_l:
        for paren_tup in parens_idxs:
            if paren_tup[0] < x < paren_tup[1]:
                # avoid like '(Fig. 2)'
                break
        else:
            sentence_start = x
            break
    return sentence_start


def get_sentence(tup, prev_end, stat_range, plain_text,
                 sentence_range, look_future=False):
    start_idx = tup.start()

    sentence_idx = max(0, start_idx - sentence_range, prev_end)
    sentence_pre = plain_text[sentence_idx:start_idx]
    sentence_start = find_period_non_parentheses(sentence_pre,
                                                 which='last')
    if sentence_start is None:
        sentence_start = sentence_idx

    if look_future:  # takes words ahead not just behind
        sentence_end_idx = min(len(plain_text), start_idx + sentence_range)
        sentence_post = plain_text[start_idx:sentence_end_idx]
        sentence_end = find_period_non_parentheses(sentence_post,
                                                   which='first')
        if sentence_end is None:
            sentence_end = sentence_end_idx
        sentence = (sentence_pre[sentence_start:] +
                    sentence_post[:sentence_end])
    else:
        sentence_end = start_idx
        sentence = sentence_pre[sentence_start:sentence_end]

    parens_idxs, parens_s = get_parentheses(sentence)  # remove parentheticals
    for tup in parens_idxs[::-1]:
        st, end = tup
        sentence = sentence[:st] + sentence[end + 1:]
    if len(sentence) == 0:
        return None
    # remove unopen parentheticals at start,
    #   e.g., if sentence starts 'd = .05) ...' then this drops 'd = .05)
    sentence = '(' + sentence
    parens_idxs, parens_s = get_parentheses(sentence)
    for tup in parens_idxs[::-1]:
        st, end = tup
        sentence = sentence[:st] + sentence[end + 1:]
    if len(sentence) == 0:
        return None
    if sentence[0] == '(':
        sentence = sentence[1:]
    # remove unclosed parentheticals at end
    sentence = sentence + ')'
    parens_idxs, parens_s = get_parentheses(sentence)
    for tup in parens_idxs[::-1]:
        st, end = tup
        sentence = sentence[:st] + sentence[end + 1:]
    if len(sentence) == 0:
        return None
    if sentence[-1] == ')':
        sentence = sentence[:-1]

    # keep only alpha numeric
    sentence = re.sub(r'[^A-Za-z0-9,\- ×]+', '', sentence)  # still has, e.g., r06, p001

    # remove numbers
    nums = re.finditer(r'\d+', sentence)
    nums = list(nums)[::-1]
    for num in nums:
        st, end = num.span()
        sentence = sentence[:st] + sentence[end:]

    # remove statistics, e.g., ' t.' (numerics will already have been removed)
    sl = '(p|P|t|f|F|r|d|ns|NS|MS|b|β|z|M|SD|ρ|rho|CI|SE|OR|MSE)'
    sentence = re.sub(rf'(\s{sl}[.!?]|\s{sl},|\s{sl}\s)', '', sentence)

    pre_len = len(sentence)
    while True:
        sentence = re.sub('  ', ' ', sentence)
        sentence = re.sub(' ,', ',', sentence)
        sentence = re.sub(',,', ',', sentence)
        if len(sentence) == pre_len:
            break
        else:
            pre_len = len(sentence)
    sentence = sentence[:-1] + '.'  # replace end white space at the end with a period
    sentence = sentence.replace(',.', '.')  # remove comma period ending
    return sentence


def extract_stat(tup, prev_end, stat_range, plain_text, doi):
    # sometimes t could trigger the double type if huge sample size like t(1,010)
    start_idx = tup.start()
    lower_idx = max(prev_end, start_idx - stat_range)
    stat_subset = plain_text[lower_idx:start_idx]
    stat_subset = re.sub('Δ', '', stat_subset)

    space_opt = r"(.{0}|\s+)"  # \s+ captures multiples of any type of space
    space_paren = r"( |\s+|,|\(|\[|\{)"
    cat1 = r'(t|ρ|rho|χ²|χ2|chi\-square|chi\-squared)'  # note: the "ρ" is a rho
    chi = r'(χ²|χ2|chi\-squared|chi\-square)'

    int_or_decimal = r'(\d+(?:\.\d+)?)'  # note: a df can be a decimal
    pos_neg_int_or_decimal = r'(-?\d+(?:\.\d+)?)'  # pos or neg

    open_parentheses_opt = r'(\(|\[|.{0})'  # '(', '[', ')', ']', or ''
    # I tried just taking blank space but it
    #  caused unwanted text interpreted as stats

    # black space here captures df as subscripts
    close_parentheses_opt = r'(\]|\)|.{0})'

    t_pattern = (f'{space_paren}{cat1}{space_opt}{open_parentheses_opt}'
                 f'{int_or_decimal}'
                 f'{close_parentheses_opt}{space_opt}={space_opt}'
                 f'{pos_neg_int_or_decimal}')
    t_matches = list(re.finditer(t_pattern, stat_subset))

    r_cat = '(r)'  # require parentheses otherwise capture r^2
    open_parentheses = r'(\(|\])'
    close_parentheses = r'(\)|\])'
    r_pattern = (f'{space_paren}{r_cat}{space_opt}{open_parentheses}'
                 f'{int_or_decimal}'
                 f'{close_parentheses}{space_opt}={space_opt}'
                 f'{pos_neg_int_or_decimal}')
    r_matches = list(re.finditer(r_pattern, stat_subset))
    t_matches += r_matches

    found_multiple = False
    single_stat = (None, None, None)
    if len(t_matches) > 1:  # it's pretty rare that more than 1 match is found
        # in that case, just take the first one
        print(f'Multiple one-df matches: {stat_subset}')
        print(f'Extended: {plain_text[start_idx - 100:start_idx]}')
        print(f'- ({doi}) -\n')
        found_multiple = len(t_matches)
    for m in t_matches:
        # replace all the chi variants to just "chi"
        stat_type = re.sub(chi, 'chi', m[2])
        stat_type = re.sub('(ρ|rho)', 'r', stat_type)  # rho treated as r
        single_stat = (stat_type, m[5], m[9])
        # if len(t_matches):
        return (single_stat, (None, None, None, None), (None, None), start_idx,
                found_multiple)
    double_stat = (None, None, None, None)
    cat2 = '(F)'

    F_pattern = (f'{space_paren}{cat2}{space_opt}{open_parentheses_opt}'
                 f'{int_or_decimal},{space_opt}{int_or_decimal}'
                 f'{close_parentheses_opt}{space_opt}={space_opt}'
                 f'{pos_neg_int_or_decimal}')
    F_matches = list(re.finditer(F_pattern, stat_subset))
    if len(F_matches) > 1:
        print(f'Multiple two-df matches: {stat_subset}')
        print(f'Extended: {plain_text[start_idx - 100:start_idx]}')
        print(f'- ({doi}) -\n')
        found_multiple = len(F_matches)

    for m in F_matches:
        double_stat = (re.sub(chi, 'chi', m[2]), m[5], m[7], m[11])
        return single_stat, double_stat, (None, None), start_idx, found_multiple

    no_stat = (None, None)

    # better no-df statistics ones are earlier,
    cat_tiers = ['(z)',  # z can be interpreted fine without df
                 '(t|F)',  # t doesn't depend much on df if df is reasonably high
                 '(r|ρ|χ²|χ2|X²|X|χ|X2|R|R²|R2|β|ρ|d|Wald|wald|'
                 r'chi\-square|chi\-squared)',
                 # some of these latter ones may be considered effect sizes
                 '(OR|b|B)'  # unstandardized regression coefficients
                 ]

    for cat0 in cat_tiers:
        no_df_pattern = (f'{space_paren}{cat0}{space_opt}={space_opt}'
                         f'{pos_neg_int_or_decimal}')
        no_df_matches = list(re.finditer(no_df_pattern, stat_subset))
        if len(no_df_matches) > 1:
            print(f'Multiple no-df matches: {stat_subset}')
            print(f'Extended: {plain_text[start_idx - 100:start_idx]}')
            print(f'- ({doi}) -\n')
            found_multiple = len(no_df_matches)
        for m in no_df_matches:
            stat_type = re.sub(chi, 'chi', m[2])
            no_stat = (stat_type, m[5])
            return single_stat, double_stat, no_stat, start_idx, found_multiple
    return single_stat, double_stat, no_stat, start_idx, found_multiple


def extract_CI(tup, prev_end, stat_range, plain_text):
    start_idx = tup.start()

    stat_subset = plain_text[start_idx:start_idx + stat_range]

    space_opt = r"(.{0}|\s+)"  # \s+ captures multiples of any type of space
    pos_neg_int_or_decimal = r'(-?\d+(?:\.\d+)?)'  # pos or neg
    int_or_decimal = r'(\d+(?:\.\d+)?)'  # note: a df can be a decimal

    # stat_subset = '95% CI = [0.07, 0.27]'
    # print(stat_subset)
    gap = r'((=|:)(.{0}|\s+))'
    CI_pattern = (rf'{int_or_decimal}%{space_opt}CI{space_opt}{gap}'
                  rf'\[{pos_neg_int_or_decimal},{space_opt}'
                  rf'{pos_neg_int_or_decimal}\]')
    matches = list(re.finditer(CI_pattern, stat_subset))

    if len(matches):
        m = matches[0]
        CI_size = m[1]
        CI_low = m[7]
        CI_high = m[9]
        return CI_size, CI_low, CI_high
    else:
        return None, None, None


def parse_ps(plain_text, doi, stat_range=64, sentence_range=512):
    plain_text = plain_text.replace('Fig.', '')  # helps with sentence parsing
    plain_text = plain_text.replace('Full size image', '')  # helps with sentence parsing
    plain_text = plain_text.replace('COVID-19', 'COVID')  # helps with sentence parsing
    plain_text = plain_text.replace('COVID19', 'COVID')  # helps with sentence parsing

    has_tail = False
    for keyword in ['-tail', ' tail', ' directional', '-sided', ' sided']:
        if keyword in plain_text:
            has_tail = True
            break

    p_code = r'(p|P|p\-value|P\-value|p‐value|P‐value|p value|P value|p‐val)'
    space_opt = r"(.{0}|\s+)"  # \s+ captures multiples of any type of space

    space_paren = r"( |\s+|,|\(|\[|\{)"

    sign = '(=|<|>|≤|≥)'
    lead_zero = '(0|.{0})'
    # NOTE: updated to include a space before the p to rule out Frontiers'
    #   partial-eta squared reporting will ignore "(p = .01" because there is
    #   no space

    # Matches scientific notation: ([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+))
    #   Copied from: https://stackoverflow.com/questions/41668588/regex-to-match-scientific-notation
    # plain_text = r'Z-score 2.019, p-value = .0218'
    pattern = (rf"{space_paren}{p_code}{space_opt}{sign}{space_opt}" +
               rf"({lead_zero}\.{space_opt}\d+)")  # not checking plural "ps ..."
    p_val_texts = list(re.finditer(pattern, plain_text))

    # Matches scientific notation
    #   Copied from: https://stackoverflow.com/questions/41668588/regex-to-match-scientific-notation
    # sci_num = r'([+-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-\−]?\d+))'

    sci_num2 = r'(\d+(?:\.\d+)?)\s(×|x)\s(10[−-]\d+)'
    pattern_sci = (rf"{space_paren}{p_code}{space_opt}{sign}{space_opt}" +
                   rf"{sci_num2}")

    p_val_texts_sci = list(re.finditer(pattern_sci, plain_text))
    is_scientific = [False] * len(p_val_texts) + [True] * len(p_val_texts_sci)
    p_val_texts += p_val_texts_sci

    p_val_l = []
    prev_end = 0
    for i, (tup, is_sci) in enumerate(zip(p_val_texts, is_scientific)):
        sentence = get_sentence(tup, prev_end, stat_range, plain_text,
                                sentence_range)
        if sentence is not None:
            sentence = remove_diacritics_fast(sentence)
        single_stat, double_stat, no_stat, start_idx, found_multiple = (
            extract_stat(tup, prev_end, stat_range, plain_text, doi))

        CI_size, CI_low, CI_high = extract_CI(tup, prev_end, stat_range,
                                              plain_text)
        prev_end = tup.end()

        sign = tup[4]
        if is_sci:
            # You may be wondering, why am I replacing a negative sign with
            #   another negative sign... well, the former is actually some
            #   variant of a negative sign ('−') that won't work. See how it
            #   is positioned slightly lower than a normal negative sign
            #   ('−' vs '-')... unicode was a mistake
            base_to_the_exponent = tup[8].replace('−', '-')
            if '-' not in base_to_the_exponent:
                continue

            base = int(base_to_the_exponent.split('-')[0])
            exponent = int(base_to_the_exponent.split('-')[1])
            p_val = float(tup[6]) * (base ** -exponent)
        else:
            try:
                p_val = float(tup[6].replace('\u2009', '').replace('\xa0', '').
                              replace(' ', ''))
            except ValueError as e:
                print(f'BAD FLOAT! ValueError (!): {e=}')
                print(f'{tup=}')
                print(fr'https://doi.org/{doi}')
                continue

        d = {'sign': sign, 'p_val': p_val, 'sentence': sentence,
             'stat_type': single_stat[0] or double_stat[0] or no_stat[0],
             'df1': single_stat[1] or double_stat[1],
             'df2': double_stat[2],
             'stat': single_stat[2] or double_stat[3] or no_stat[1],
             'multiple_stats': found_multiple,
             'CI_size': CI_size, 'CI_low': CI_low, 'CI_high': CI_high,
             'p_is_scientific': is_sci,
             'has_tail': has_tail,
             }
        d['valid_stat'] = (not pd.isna(d['df1'])) or (d['stat_type'] == 'z')

        p_val_l.append(d)
    return p_val_l


def add_row_to_df(df, row):
    df['doi_str'] = row.doi_str
    df['publisher'] = row.publisher
    df['journal'] = row.journal
    df['year'] = row.year
    df['link'] = fr'https://doi.org/{row.doi}'
    return df


def get_paper_df(fp_text, row):
    with open(fp_text, 'r', encoding='utf-8') as file:
        plain_text = file.read()

    p_val_l = parse_ps(plain_text, row.doi)
    df_paper = pd.DataFrame(p_val_l)
    df_paper = add_row_to_df(df_paper, row)
    return df_paper


def stat2p_z_single(stat_type, stat, df1, df2, p_val):
    if stat_type == 't':
        p_implied = stats.t.sf(np.abs(stat), df1) * 2
        # df.loc[df_stat.index, 'p_implied'] = (
        #         stats.t.sf(df_stat['stat'].abs(), df_stat['df1']) * 2)
    elif stat_type == 'chi':
        p_implied = stats.chi2.sf(stat, df1)
        is_flip = np.abs(p_implied - 1 + p_val) < .1
        is_flip = is_flip and (np.abs(0.5 - p_val) > .1)
        if is_flip:
            p_implied = 1 - p_implied
    elif stat_type == 'r':
        t_implied = (stat / np.sqrt((1 - stat ** 2) / df1))
        p_implied = stats.t.sf(t_implied, df1) * 2
    elif stat_type == 'F':
        p_implied = stats.f.sf(stat, df1, df2)
        is_flip = np.abs(p_implied - 1 + p_val) < .1
        is_flip = is_flip and np.abs((0.5 - p_val) > .1)
        if is_flip:
            p_implied = 1 - p_implied
    elif stat_type == 'z':
        p_implied = stats.norm.sf(np.abs(stat)) * 2
    else:
        raise ValueError(f'Bad stat_type: {stat_type=}')
    return p_implied


def stat2p_z_all(df):
    df_ = df[df['valid_stat']]
    for stat_type, df_stat in df_.groupby('stat_type'):
        print(f'{stat_type}: {len(df_stat)=}')
        if stat_type == 'z':
            df.loc[df_stat.index, 'p_implied'] = (
                    stats.norm.sf(df_stat['stat'].abs()) * 2)
        elif stat_type == 't':
            df.loc[df_stat.index, 'p_implied'] = (
                    stats.t.sf(df_stat['stat'].abs(), df_stat['df1']) * 2)
        elif stat_type == 'chi':
            df_stat['p_implied'] = stats.chi2.sf(df_stat['stat'],
                                                 df_stat['df1'])
            # Sometimes a chi-square is reported such that the p-value output
            #   is flipped (e.g., the above equation yields p = .99, but the
            #   paper reports and interprets it at p = .01).
            # This is fixed here. This fixes around 750 cases

            # check if p_implied is vaguely close to (1 - p_val)
            #   The wide range will catch "<" sign p-values up to p < .10
            is_flip = (df_stat['p_implied'] - 1 + df_stat['p_val']).abs() < .1
            # ignore .4-.6
            is_flip = is_flip & ((0.5 - df_stat['p_val']).abs() > .1)

            flipped_chi = df_stat[is_flip &
                                  df_stat['sign'].isin(['=', '<'])]
            num_flipped_chi = len(flipped_chi)
            prop_flipped = num_flipped_chi / len(df_stat)
            print(f'{num_flipped_chi=} ({prop_flipped:.1%})')
            df_stat.loc[flipped_chi.index, 'p_implied'] = (
                    1 - flipped_chi['p_implied'])
            df.loc[df_stat.index, 'p_implied'] = df_stat['p_implied']
        elif stat_type == 'r':
            t_implied = (df_stat['stat'] /
                         np.sqrt((1 - df_stat['stat'] ** 2) / df_stat['df1']))
            df.loc[df_stat.index, 'p_implied'] = stats.t.sf(t_implied,
                                                            df_stat['df1']) * 2
        elif stat_type == 'F':
            # The above chi-square flipping also applies to F-values but is
            #   much rarer, aplying to only a few dozen cases. Nonetheless,
            #   it is fixed here.
            df_stat['p_implied'] = stats.f.sf(df_stat['stat'], df_stat['df1'],
                                              df_stat['df2'])

            is_flip = (df_stat['p_implied'] - 1 + df_stat['p_val']).abs() < .1
            # ignore .4-.6
            is_flip = is_flip & ((0.5 - df_stat['p_val']).abs() > .1)

            flipped_F = df_stat[is_flip &
                                df_stat['sign'].isin(['=', '<'])]
            num_flipped_F = len(flipped_F)
            prop_flipped = num_flipped_F / len(df_stat)
            print(f'{num_flipped_F=} ({prop_flipped:.1%})')

            df_stat.loc[flipped_F.index, 'p_implied'] = (
                    1 - flipped_F['p_implied'])
            df.loc[df_stat.index, 'p_implied'] = df_stat['p_implied']

    df['z_implied'] = stats.norm.isf(df['p_implied'] / 2)


def make_p_z_df():
    fp_in = fr'../dataframes/df_has_results_Aug24.csv'
    df = read_csv_fast(fp_in, easy_override=False)
    df = df[df['has_results'] == True]
    df_pz_as_l = []

    for row in tqdm(df.itertuples(), desc='extracting p-values/stats/CIs',
                    total=df.shape[0]):
        fp_text = (fr'..\data\plaintexts_res_Aug24\{row.publisher}' +
                   fr'\{row.journal}\{row.year}\{row.doi_str}.txt')
        fp_pkl = f'../cache/p_val_l/res_{row.doi_str}.pkl'

        if not os.path.exists(fp_text): continue
        df_paper = pickle_wrap(get_paper_df, fp_pkl,
                               kwargs={'fp_text': fp_text, 'row': row},
                               verbose=-1, easy_override=True,
                               dt_max=datetime(2024, 8, 24, 10, 0, 0, 0))
        df_pz_as_l.append(df_paper)

    df_pz = pd.concat(df_pz_as_l).reset_index()
    keys = ['stat', 'df1', 'df2']
    for key in keys:
        df_pz[key] = pd.to_numeric(df_pz[key], errors='coerce')

    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.width', 200)

    stat2p_z_all(df_pz)

    num_ps = df_pz['p_val'].count()
    num_implieds = df_pz['p_implied'].count()
    p_coverage = num_implieds / num_ps
    print(f'coverage ({num_implieds:,} / {num_ps:,}): {p_coverage:.1%}')

    fp_out = fr'../dataframes/df_by_pval_Jan21.csv'
    save_csv_pkl(df_pz, fp_out, check_dup=False)


def remove_diacritics_fast(text):
    # sometimes removes apostrophe's (ones that were mis-entered)
    #   Will kill chi (χ) though
    return (unicodedata.normalize('NFKD', text).
            encode('ASCII', 'ignore').decode('ASCII'))


if __name__ == '__main__':
    # make_df_w_results()
    make_p_z_df()
