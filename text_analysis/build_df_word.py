import os.path
from collections import defaultdict
from copy import deepcopy
from functools import cache

import numpy as np
import pandas as pd
from colorama import Fore
from textblob.en.inflect import singularize
from tqdm import tqdm

from make_dataset.make_processed_p_dfs import smol
from utils import read_csv_fast, pickle_wrap, save_csv_pkl

pd.options.mode.chained_assignment = None


def clean_words(word_counter, words_above):
    import keyword
    # These are (e.g., 'False', 'None'), which could cause errors
    #   the list also includes words used in the main dataframe (e.g., 'school')
    #   Hence, we add an underscore to the end of each word
    naughty_words = list(keyword.kwlist) + ['school', 'country', 'year']
    syntax_mapper = {word: f'{word}_' for word in naughty_words}
    words_above_ = [syntax_mapper[word] for word in words_above
                    if word in naughty_words]
    words_above = [word for word in words_above if word not in naughty_words]
    words_above += words_above_
    for word, word_ in syntax_mapper.items():
        try:
            word_counter[word_] = word_counter.pop(word)
        except KeyError:  # Word not in, not necessary to override
            pass
            # print(f'Word not in, not necessary to override: {word}')

    return words_above, word_counter


def singularize_sentence(sentence_set):
    return {try_singular(word) for word in sentence_set}


def get_is_fragile(p_val, sign):
    if p_val > .05: raise ValueError
    if sign == '=':
        if p_val > .01 - smol:
            return True
        else:
            return False
    else:
        assert sign == '<'
        if p_val > .01 + smol:
            return True
        else:
            return False


def get_fragile_implied(p_implied):
    if pd.isna(p_implied):
        return np.nan
    elif p_implied > 0.05:
        return np.nan
    elif p_implied < .01:
        return False
    else:
        return True


def clean_sentence_rows(df, stat_type, p_implied):
    df = prune_to_stat_type(df, stat_type)
    print(f'Number of instances of stat ({stat_type}): {len(df):,}')
    df.dropna(subset=['sentence'], inplace=True)
    print(f'\tNumber after dropping null sentences: {len(df):,}')
    pd.set_option('display.max_rows', 5000)

    pre_len = len(df)
    df = df[df['sig'] == True]

    assert len(df) < pre_len
    pre_len = len(df)
    df = df[df['cond'] != 'all_less0.05']
    assert len(df) <= pre_len
    print(f'\tNumber after dropping non-significant: {len(df):,}')

    df.drop_duplicates(subset=['sentence'], inplace=True, keep='first')
    print(f'\tNumber after dropping null duplicates: {len(df):,}')

    if p_implied:
        df = df.dropna(subset=['p_implied'])
        print(f'\tNumber after dropping no implieds: {len(df):,}')
    df['sentence'] = (df['sentence'].str.replace(',', '').str.replace('.', '')
                      .str.replace('×', ' '))
    df['sentence'] = df['sentence'].str.lower()
    return df


def make_paper_word_df(num_words=2500, stat_type='all', p_implied=False,
                       subject=None):
    def fix_space2hyphen(sentence):
        sentence = ' ' + sentence + ' '
        for old, new in hyphenwords.items():
            sentence = sentence.replace(f' {old} ', f' {new} ')
        return sentence

    nwords_str = f'_nw{num_words}'
    stat_type_str = f'_{stat_type}' if stat_type != 'all' else ''
    p_implied_str = f'_p_implied' if p_implied else ''
    subj_str = '' if subject is None else f'_{subject}'
    fp_words = (fr'../dataframes/df_words/df_'
                fr'{nwords_str}{stat_type_str}{p_implied_str}{subj_str}'
                fr'_Jan21.csv')
    if os.path.exists(fp_words):
        print(f'Already done: {fp_words=}')
        return

    print(f'Working on: {fp_words=}')

    fp_top2500 = (f'../cache/get_word_lists'
                  f'{stat_type_str}{p_implied_str}{nwords_str}{subj_str}.pkl')

    word_counter, words_above, hyphenwords = (
        pickle_wrap(get_word_lists, filepath=fp_top2500,
                    kwargs={'num_words': num_words, 'stat_type': stat_type,
                            'p_implied': p_implied, 'subject': subject},
                    easy_override=True))
    assert len(words_above)

    fp_p_by_p = fr'../dataframes/df_by_pval_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp_p_by_p, easy_override=False, check_dup=False)
    if subject:
        df = df[df[subject] == True]
    df = clean_sentence_rows(df, stat_type, p_implied)

    df['sentence'] = df['sentence'].apply(fix_space2hyphen)
    df['sentence'] = df['sentence'].str.split(' ')
    df['sentence'] = df['sentence'].apply(singularize_sentence)
    df['sentence'] = df['sentence'].apply(set)

    df['p_is_fragile'] = df.apply(lambda x: get_is_fragile(x['p_val'],
                                                           x['sign']), axis=1)
    if p_implied:
        df['p_is_fragile_implied'] = df['p_implied'].apply(get_fragile_implied)

    d_sentence = {}
    for word in tqdm(words_above, desc=f'making paper-level word df '
                                       f'({stat_type})'):
        try:
            out = (
                df['sentence'].apply(lambda x: word in x).astype(int).values)
            d_sentence[word] = out
        except TypeError as e:
            print(f'TYPE ERROR: {word=}')
            print(e)
    d = {'doi_str': df['doi_str'].values,
         'stat_type': df['stat_type'].values
         }
    d = {**d, **d_sentence}
    df_ = pd.DataFrame(d)
    df_['p_is_fragile'] = df['p_is_fragile'].values
    if p_implied:
        df_['p_is_fragile_implied'] = df['p_is_fragile_implied'].values
    df = df_

    cols = ['doi_str', 'stat_type'] + words_above + ['p_is_fragile', ]
    if p_implied:
        cols += ['p_is_fragile_implied']
    df_result_words = df[cols]
    print(f'{fp_words=}')
    # no .csv, dfs too big
    save_csv_pkl(df_result_words, fp_words, no_csv=True, check_dup=False)
    print(f'\tMade single-result word df: n = {len(df_result_words)}')


@cache
def get_EN_dict():
    with open(r'../text_analysis/EN_dict.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip().replace('-', '').replace(',', '.').replace('.', '')
             for line in lines]
    words = set(lines)
    return words


def get_word_lists(num_words, stat_type='all', p_implied=False,
                   subject=None):
    fp_p_by_p = fr'../dataframes/df_by_pval_combined_semi_pruned_Jan21.csv'
    df = read_csv_fast(fp_p_by_p, easy_override=False, check_dup=False)
    if subject:
        df = df[df[subject] == True]
    df = clean_sentence_rows(df, stat_type, p_implied)

    assert len(df)

    # LETTERS is a set of letters (single characters). These aren't words.
    word_counter = defaultdict(lambda: 0)
    hyphenwords = {}
    for sentence in tqdm(df['sentence'],
                         desc=f'Looping sentences to make word list '
                              f'({stat_type}; {p_implied=})'):
        # Working with language is dreadful. There are so many ways people can
        #   write a sentence and further ways an editor can format them
        words = set(sentence.split(' '))
        # eliminate 1-character or '' items
        words_ = list(filter(lambda x: len(x) > 2, words))

        words = []
        for i, word in enumerate(words_):
            if '-' not in word:
                words.append(word)
                continue
            if '----' in word:
                continue
            elif '---' in word:
                word = word.replace('---', '-')
            elif '--' in word:
                word = word.replace('--', '-')
            if len(word) < 2: continue
            if word[0] == '-':
                word = word[1:]
            if word[-1] == '-':
                word = word[:-1]
            if len(word) < 2: continue
            if '-' in word:
                words.append(word.replace('-', ''))
                word_space = word.replace('-', ' ')
                hyphenwords[word_space] = try_singular(words[-1])

        words = [try_singular(word) for word in words]  # TextBlob package
        for word in words:
            word_counter[word] += 1

    words_above = list(word_counter)
    try:
        words_above.remove('result')  # sometimes a section title gets in
    except ValueError:
        pass
    try:
        words_above.remove('discussion')  # sometimes a section title gets in
    except ValueError:
        pass
    words_above = sorted(words_above, key=word_counter.get, reverse=True)
    words_above = words_above[:num_words]

    word_counter = {word: word_counter[word] for word in words_above}
    hyphenwords = {space: removed for space, removed in hyphenwords.items()
                   if removed in word_counter}

    return word_counter, words_above, hyphenwords


def ef2color(ef):
    if ef > 5:
        return Fore.LIGHTGREEN_EX
    elif ef < -5:
        return Fore.LIGHTRED_EX
    else:
        return Fore.RESET


@cache
def try_singular(word):
    singular_attempt = singularize(word)
    EN_words = get_EN_dict()
    if singular_attempt in EN_words:
        return singular_attempt
    else:
        return word


def prune_to_stat_type(df_result_words, stat_type):
    if stat_type == 'any':
        df_result_words = df_result_words[~df_result_words['stat_type'].isna()]
    elif stat_type != 'all':
        if stat_type == 'ft':
            df_result_words = df_result_words[
                df_result_words['stat_type'].isin(['t', 'F'])]
        elif stat_type == 'rB':
            df_result_words = df_result_words[
                df_result_words['stat_type'].isin(['r', 'β', 'b', 'B'])]
        else:
            df_result_words = df_result_words[
                df_result_words['stat_type'] == stat_type]
    else:
        df_result_words = deepcopy(df_result_words)
    return df_result_words


if __name__ == '__main__':
    P_IMPLIED = False
    STAT_TYPES = ['t', 'F', 'chi', 'rB', 'all', ]
    SUBJECTS = ['Developmental_and_Educational_Psychology',
                'General_Psychology', 'Social_Psychology',
                'Applied_Psychology', 'Clinical_Psychology',
                'Experimental_and_Cognitive_Psychology',
                'Psychology_Miscellaneous']
    SUBJECTS = [None]
    for SUBJECT in SUBJECTS:
        for STAT_TYPE in STAT_TYPES:
            make_paper_word_df(stat_type=STAT_TYPE, p_implied=P_IMPLIED,
                               num_words=2500, subject=SUBJECT)
