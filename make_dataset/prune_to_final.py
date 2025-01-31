import pandas as pd

from utils import read_csv_fast, save_csv_pkl

pd.options.mode.chained_assignment = None

NEURO_W_PSYCH_JOURNALS = {
    'cognition', 'memory_amp_cognition', 'cognitive_science',
    'journal_of_memory_and_language', 'human_factors',
    'journal_of_communication_disorders',
    'journal_of_contextual_behavioral_science',
    'journal_of_the_experimental_analysis_of_behavior',
    'international_journal_of_behavioral_development',
    'learning_and_motivation',
    'cognitive_processing', 'cognitive_psychology',
    'learning_amp_behavior',
    'topics_in_cognitive_science',
    'nature_human_behaviour',
    'the_quarterly_journal_of_experimental_psychology',
    'journal_of_behavioral_and_cognitive_therapy',
    'quarterly_journal_of_experimental_psychology',
    'asian_journal_of_sport_and_exercise_psychology',
    'journal_of_research_on_adolescence',
    'evolutionary_psychology',
    'adaptive_behavior',
    'british_journal_of_developmental_psychology',
    'journal_of_behavioral_and_cognitive_therapy',
    'human_factors_the_journal_of_the_human_factors_and_ergonomics_society',
    'child_development_research',
    'cognitive_systems_research',
    'sleep_medicine_clinics',
    'journal_of_fluency_disorders',
    'chronic_stress',
    'developmental_science'}

NEURO_JOURNALS = \
    {'psychophysiology',
     'current_developmental_disorders_reports',
     "american_journal_of_alzheimer's_disease_amp_other_dementiasr",
     'mind,_brain,_and_education',
     "american_journal_of_alzheimer's_disease_amp_other_dementias®",
     'adaptive_human_behavior_and_physiology',
     'applied_psychophysiology_and_biofeedback',
     'brain_research._cognitive_brain_research',
     'archives_of_clinical_neuropsychology__the_official_journal_of_'
     'the_national_academy_of_neuropsychologists',
     'computational_brain_amp_behavior', 'neurobiology_of_learning_and_memory',
     'cortex;'
     '_a_journal_devoted_to_the_study_of_the_nervous_system_and_behavior',
     'trends_in_cognitive_sciences',
     'biologically_inspired_cognitive_architectures',
     'brain_and_cognition', 'brain_and_language', 'developmental_psychobiology',
     'physiology_amp_behavior',
     'cognitive_research_principles_and_implications', 'cortex',
     'journal_of_neurolinguistics', 'neuropsychologia',
     'evolutionary_psychology__an_international_journal_of_evolutionary_'
     'approaches_to_psychology_and_behavior',
     }


def make_pruned_df(fp):
    assert 'combined' in fp
    df = read_csv_fast(fp, verbose=0)

    df = df[df['journal'] != 'journal_of_fluorescence']  # lens.org error entry?
    df = df[(df['is_neuro'] != True) |
            (df['journal'].isin(NEURO_W_PSYCH_JOURNALS))]

    df = df[df['year'] >= 2004]

    print(f'Number of papers in year from 2004-2024: {len(df):,}')

    len_2004 = len(df)
    df = df[df['has_results'] == True]
    print(f'\tNumber of papers with results: {len(df):,} '
          f'({len(df) / len_2004:.1%})')
    len_pre = len(df)

    df = df[df['jrl_cnt'] >= 5]
    num_dropped = len_pre - len(df)
    print(f'\tNumber of papers with >= 5 papers in journal (true has_results): '
          f'{len(df):,} ({num_dropped=})')

    # Drop papers with bad journals, years, no SNIP, no school
    #   but with a Results section
    if 'all_aff' not in fp:
        df_emp = df[(df['SNIP'].notna()) & (df['school'].notna())]
        fp_out = fp.replace('combined', 'combined_all_empirical')
        print(f'Number of papers with Results and SNIP and school: '
              f'{len(df_emp):,}')
        save_csv_pkl(df_emp, fp_out, check_dup=False, verbose=0)

    df['has_ps'] = df['num_ps_any'] > 0
    pre_len = len(df)
    df = df[df['has_ps'] == True]
    print(f'\tNumber of papers with p-values: {len(df):,} '
          f'({len(df) / pre_len:.1%})')
    if df['sig'].max() == 1:  # i.e., a _by_pval dataframe
        ps_per_paper = df.groupby('doi_str')['sig'].sum()
        df['num_sig_in_paper'] = df['doi_str'].map(ps_per_paper)
        df['has_signif_ps'] = df['num_sig_in_paper'] > 1
        df = df[df['has_signif_ps'] == True]
        df.drop(columns=['has_signif_ps'], inplace=True)
        print(f'\tNumber of ps after dropping papers with under 2 sig: '
              f'{len(df):,}')
    else:
        print('-*-')
        fp_out = fp.replace('combined', 'combined_w_no_sig')
        all_insig = ((df['num_ps_any'] > 0) & (df['sig'] == 0)).sum()

        print(f'\tNumber of papers with only insignificant results: {all_insig:}')
        all_insig = ((df['num_ps_any'] > 1) & (df['sig'] == 0)).sum()
        print(f'\tNumber of papers with only insignificant results (2+ p-values): {all_insig:}')
        save_csv_pkl(df, fp_out, check_dup=False, verbose=0)
        df['has_signif_ps'] = df['sig'] > 0
        df = df[df['has_signif_ps'] == True]
        print(f'\tNumber of papers with significant p-values: {len(df):,}')
        df['has_signif_ps'] = df['sig'] > 1
        df = df[df['has_signif_ps'] == True]
        df.drop(columns=['has_signif_ps'], inplace=True)
        print(f'\tNumber of papers with 2+ signif. p-values: {len(df):,}')
    n_nan_snip = df['SNIP'].isna().sum()
    print(f'\tNumber of papers with NaN SNIP: {n_nan_snip:,}')
    if 'all_aff' in fp:
        n_nan_affil = df['Random_school'].isna().sum()
        print(f'\tNumber of papers with NaN affiliation: {n_nan_affil:,}')
        df_pruned = df[df['SNIP'].notna() & df['Random_school'].notna()]
        aff_types = ['Random', 'TargetMax', 'TargetMin', 'TargetMed',
                     'YearMax', 'YearMin', 'YearMed',
                     'Mean', 'Mode']
        for aff_type in aff_types:
            assert df_pruned[f'{aff_type}_target_score'].notna().all()
            assert df_pruned[f'{aff_type}_year_score'].notna().all()
    else:
        n_nan_affil = df['school'].isna().sum()
        print(f'\tNumber of papers with NaN affiliation: {n_nan_affil:,}')
        df_pruned = df[df['SNIP'].notna() & df['school'].notna()]
    print(f'Final dataset size if dropping SNIP or no-university: '
          f'{len(df_pruned):,}')
    fp_out = fp.replace('combined', 'combined_pruned')
    save_csv_pkl(df_pruned, fp_out, check_dup=False, verbose=0)
    print('-*-')

    fp_out = fp.replace('combined', 'combined_semi_pruned')
    save_csv_pkl(df, fp_out, check_dup=False, verbose=0)
    print('-*-')


if __name__ == '__main__':
    # One row = one paper. Used for most of the multilevel regressions)
    make_pruned_df(r'..\dataframes\df_combined_Jan21.csv')

    # One row = one reported p-value (used for the language analysis)
    make_pruned_df(r'..\dataframes\df_by_pval_combined_Jan21.csv')

    # This one includes alternative university assignments (e.g., max or mean)
    make_pruned_df(r'..\dataframes\df_combined_all_aff_Jan21.csv')
