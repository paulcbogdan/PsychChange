import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import api as sm
from statsmodels.formula import api as smf
from tqdm import tqdm

from text_analysis.build_df_word import get_word_lists, prune_to_stat_type, clean_words
from text_analysis.plot_Figure6_language import get_base_kwargs
from utils import pickle_wrap, read_csv_fast, save_csv_pkl, PSYC_SUBJECTS_


def clean_df_words(df_words):
    import keyword
    naughty_words = list(keyword.kwlist) + ['school', 'country', 'year']

    syntax_mapper = {word: f'{word}_' for word in naughty_words}
    df_words.rename(columns=syntax_mapper, inplace=True)
    return df_words


def make_paper_word_regr_df(stat_type='all', paper_any=False, do_logistic=True,
                            num_words=2000, include_SNIP=False,
                            single_regr=False, regress_specific=None,
                            control_subject=False, single_frag=False,
                            p_implied=False, whole_paper_stats=True,
                            subject=None):
    signif_str = '_signif' if True else ''
    nwords_str = f'_nw{num_words}'
    drop_dup_str = '_drop_dup' if True else ''
    stat_type_str = f'_{stat_type}' if stat_type != 'all' else ''
    p_implied_str = f'_p_implied' if p_implied else ''

    w_logit = '_logit' if do_logistic else ''
    jif_str = '_SNIP' if include_SNIP else ''
    single_str = '_single' if single_regr else ''
    paper_any_str = '_paper_any' if paper_any else ''
    is_frag_str = '_single_frag' if single_frag else ''
    ctrl_subject_str = '_ctrl_subj' if control_subject else ''
    regress_specific_str = f'_R{regress_specific}' if regress_specific else ''
    whole_paper_str = '_whole' if whole_paper_stats else ''
    subj_str = '' if subject is None else f'_{subject}'

    fp_out = (fr'../dataframes/word_regr/df_word_regr_'
              fr'{w_logit}{nwords_str}{signif_str}{jif_str}{single_str}'
              fr'{paper_any_str}{stat_type_str}{is_frag_str}{drop_dup_str}'
              fr'{ctrl_subject_str}{regress_specific_str}{p_implied_str}'
              fr'{whole_paper_str}{subj_str}_Jan21.csv')
    print(f'{fp_out=}')

    if regress_specific: assert single_regr

    fp_top2500 = (f'../cache/get_word_lists'
                  f'{stat_type_str}{p_implied_str}{nwords_str}{subj_str}.pkl')

    word_counter, words_above, hyphenwords = (
        pickle_wrap(get_word_lists, filepath=fp_top2500,
                    kwargs={'num_words': num_words, 'stat_type': stat_type,
                            'p_implied': p_implied, 'subject': subject},
                    easy_override=False))
    words_above, word_counter = clean_words(word_counter, words_above)

    # Load a dataframe with all non-word variables (e.g., affiliation, p-vals)
    fp_combined = fr'../dataframes/df_combined_pruned_Jan21.csv'
    df = read_csv_fast(fp_combined, easy_override=False)
    if subject:
        df = df[df[subject] == True]
    df = df[df['cond'] != 'all_less0.05']
    analysis_cols = ['year', 'target_score', 'log_cites_year_z', ]
    if control_subject:
        analysis_cols += PSYC_SUBJECTS_
    if include_SNIP: analysis_cols.append('SNIP')
    df.dropna(subset=analysis_cols, inplace=True)

    df = df[analysis_cols + ['doi_str', 'num_ps', 'p_fragile',
                             'p_fragile_implied', 'doi']]

    fp_words = (fr'../dataframes/df_words/df_'
                fr'{nwords_str}{stat_type_str}{p_implied_str}{subj_str}'
                fr'_Jan21.csv')

    # Load the dataframe created by make_paper_word_df, containing info
    #   on the words each result used (encoded where each column represents the
    #   whether a result sentence used a given word)
    df_result_words = read_csv_fast(fp_words, easy_override=False,
                                    check_dup=False)

    df_result_words = prune_to_stat_type(df_result_words, stat_type)
    df_result_words = clean_df_words(df_result_words)

    assert np.sum(df_result_words[words_above].isna().sum()) == 0

    if do_logistic and not paper_any:
        assert not whole_paper_stats
        df_result_words['num_words'] = df_result_words[words_above].sum(axis=1)
        df = df_result_words.merge(df[analysis_cols + ['doi_str', 'num_ps']],
                                   on='doi_str')
        df['p_key'] = df['p_is_fragile_implied'] if p_implied else (
            df)['p_is_fragile']

        print(f'Merged... : {len(df)=}')

        df['i_num_ps'] = 1 / df['num_ps']
        assert np.sum(df['i_num_ps'].isna().sum()) == 0
    else:
        if paper_any:
            assert do_logistic
            df_paper_words = df_result_words.groupby('doi_str').any()
            df_paper_words['num_words'] = df_paper_words[words_above].sum(
                axis=1)
        else:
            df_paper_words = df_result_words.groupby('doi_str').sum()
            df_paper_words['num_words'] = df_paper_words[words_above].sum(
                axis=1)
            normed = (df_paper_words[words_above].values /
                      df_paper_words['num_words'].values[:, None])
            df_paper_words[words_above] = normed
        imp_str = '_implied' if p_implied else ''
        if not whole_paper_stats:
            df_result_words['p_key'] = df_result_words[f'p_is_fragile{imp_str}']
            df_p = df_result_words.groupby('doi_str')[['p_key']].mean()
            df_p.reset_index(inplace=True)
            df = df.merge(df_p, on='doi_str')

        df = df_paper_words.merge(df[analysis_cols + ['doi_str', 'p_fragile',
                                                      'p_fragile_implied',
                                                      'doi'] +
                                     (['p_key'] if 'p_key' in
                                                   df.columns else [])],
                                  on='doi_str')
        if whole_paper_stats:
            df['p_key'] = df[f'p_fragile{imp_str}']
        pre_len = len(df)
        df.dropna(subset=['p_key'], inplace=True)  # for p_implied
        if not p_implied: assert len(df) == pre_len
        df['p_key'] = df['p_key'].astype(float)

    analysis_cols += ['p_key']

    word2prop = {}
    word2total = {}
    word2total_normed = {}
    for word in words_above:
        prop = df[word].mean()
        word2prop[word] = prop
        total = df_result_words[word].sum()
        word2total[word] = total
        total_normed = df[word].sum()
        word2total_normed[word] = total_normed

    no_words_rows = len(df[df['num_words'] == 0])
    print(f'Number of rows with no matching words: {no_words_rows}')
    df = df[df['num_words'] > 0]
    df['SNIP'] = df['SNIP'].astype(float)

    for col in analysis_cols:
        if col in PSYC_SUBJECTS_:
            df[col] = df[col].astype(int)
        df[col] = stats.zscore(df[col], nan_policy='omit')

    df_word_scores = []
    for word in tqdm(words_above, desc=f'Running regressions ({stat_type})'):
        if word not in df.columns:
            print(f'Missing: {word}')
            continue
        cols = [word] + analysis_cols
        if 'i_num_ps' in df.columns: cols += ['i_num_ps']
        df_ = df[cols]
        assert np.sum(df[word].isna().sum()) == 0
        assert len(df_) > 0

        prop = word2prop[word]
        total = word2total[word]
        total_normed = word2total_normed[word]
        papers_analyzed = len(df_)
        print(f'{word}: {total_normed=:.5f}, {prop=:.5%} ({papers_analyzed=})')
        # Here, the word is not a boolean but instead reflects the proportion
        #   of a paper's result sentences that contain the word
        # Checking for .any() at the paper level would lead to a strong bias
        #   linked to the number of results per paper

        d = {'word': word, 'prop': prop, 'total': total,
             'total_normed': total_normed, 'papers_analyzed': papers_analyzed}

        if single_regr:
            for col in analysis_cols:
                if col in PSYC_SUBJECTS_: continue
                formula = f'{word} ~ {col}'
                if regress_specific and col != regress_specific:
                    formula += f'+ {regress_specific}'
                if do_logistic and not paper_any:
                    df_[word] = df_[word].astype(int)
                    glm = smf.glm(formula, data=df_,
                                  family=sm.families.Binomial(),
                                  freq_weights=df_['i_num_ps'])
                    res = glm.fit(disp=0)
                elif do_logistic:
                    df_[word] = df_[word].astype(int)
                    try:
                        res = smf.logit(formula=formula, data=df_).fit(disp=0)
                    except np.linalg.LinAlgError as e:
                        print(f'{word}, {prop=:.1%} ({len(df_)=}): {e=}')
                        print(df_)
                        continue
                else:
                    res = smf.ols(formula=formula, data=df_).fit(disp=0)
                d[f'coef_{col}'] = res.params[col]
                d[f't_{col}'] = res.tvalues[col]
                if regress_specific:
                    d[f'coef_{col}_r{regress_specific}'] = \
                        (res.params[regress_specific])
                    d[f't_{col}_r{regress_specific}'] = \
                        (res.tvalues[regress_specific])
            df_word_scores.append(d)
        else:
            formula = f'{word} ~ p_key + year + target_score + log_cites_year_z'
            if include_SNIP: formula += ' + SNIP'
            if do_logistic and not paper_any:
                df_[word] = df_[word].astype(int)
                glm = smf.glm(formula, data=df_, family=sm.families.Binomial(),
                              freq_weights=df_['i_num_ps'])
                res = glm.fit(disp=0)
            elif do_logistic:
                try:
                    df_[word] = df_[word].astype(int)
                    res = smf.logit(formula=formula, data=df_).fit(disp=0)
                except np.linalg.LinAlgError as e:
                    print(f'{word} ({len(df_)=}): {e=}')
                    print(df_)
                    continue
            else:
                res = smf.ols(formula=formula, data=df_).fit(disp=0)
            ts = res.tvalues.add_prefix('t_').to_dict()
            efs = res.params.add_prefix('coef_').to_dict()
            d.update(ts)
            d.update(efs)
            df_word_scores.append(d)

    df_regr = pd.DataFrame(df_word_scores)

    print(f'{fp_out=}')
    save_csv_pkl(df_regr, fp_out, check_dup=False)


if __name__ == '__main__':
    P_IMPLIED = False
    # Can also look at just 't' or 'F' with 'ft'
    # The overlap analysis yields far fewer results if you do, likely because
    #   you are reducing the heterogeneity in the types of research sampled
    STAT_TYPES = ['t', 'F', 'chi', 'rB', 'all']
    SINGLE_REGR = True
    NUM_WORDS = 2500
    # SUBJECTS = ['Developmental_and_Educational_Psychology',
    #             'General_Psychology', 'Social_Psychology',
    #             'Applied_Psychology', 'Clinical_Psychology',
    #             'Experimental_and_Cognitive_Psychology',
    #             'Psychology_Miscellaneous']
    SUBJECTS = [None]
    for SUBJECT in SUBJECTS:
        for STAT_TYPE in STAT_TYPES:
            if STAT_TYPE == 'all':
                kwargs = get_base_kwargs(analysis_name='paper_sum_reg',
                                         include_SNIP=True, single_regr=SINGLE_REGR,
                                         p_implied=P_IMPLIED,
                                         whole_paper_stats=True,
                                         num_words=NUM_WORDS,
                                         regress_specific='p_key',
                                         )
            else:
                kwargs = get_base_kwargs(analysis_name='paper_sum_reg',
                                         include_SNIP=True, single_regr=SINGLE_REGR,
                                         p_implied=P_IMPLIED, whole_paper_stats=False,
                                         num_words=NUM_WORDS
                                         )
            kwargs['control_subject'] = False
            kwargs['stat_type'] = STAT_TYPE
            kwargs['subject'] = SUBJECT
            make_paper_word_regr_df(**kwargs)
