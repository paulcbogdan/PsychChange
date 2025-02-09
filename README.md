
# How Have Psychological Results Changed?

This repository contains the code for the manuscript: Bogdan, P.C. (2025). "One decade into the Replication Crisis, How have Psychological Results Changed?" _Advances in Methods and Practices in Psychological Science_. The downloaded data and preprocessed datasets have been added to the associated [OSF/Box repository](https://osf.io/mxs47/). A preprint form of the paper has been uploaded here as a PDF.

The code can be divided into two parts: (1) "Dataset construction" and (2) "Dataset use".

(1) "Dataset construction" entails (1.1) downloading full text articles and converting them into usable plaintexts. Then, (1.2) parsing them and building a dataset which organized the parsed data alongside other variables, such as SNIP, citations, and university prestige. This ultimately produces a some spreadsheets. 

These spreadsheets have all been uploaded to the associated OSF repository and placed in the "dataframes" folder. Hence, anyone who wants to use the dataset could just skip all of these parts. You can find "4. Final spreadsheet columns" further below for details on which spreadsheet you should pick for your work, and for instructions on what the spreadsheet columns represent. 

Alternatively, based on this code, anyone (with a suitable university library subscription) could in reproduce the spreadsheets themselves. This would, notably, involve you running the scripts to scrape the fulltexts. I have uploaded as much as I can, but I cannot upload the fulltexts themselves for copyright reasons.

(2) "Dataset use" entails actually using the dataset and, based on the uploaded spreadsheets, generating the results in the manuscript. This is divided into (2.1) running the three forms of validation, (2.2) producing the figures along with all of the various counts, e.g., counting the number of papers or running the Table S1 p-value survey, (2.3) running the multilevel regressions reported in either the main text or the supplemental materials, and (2.4) running the word-usage analysis. 

The multilevel regression step may be of the most interest to readers. With it, you can reproduce the exact numbers reported for, arguably, the most critical part of the paper (the multilevel regression findings featured in Results Section 3.2). In addition, if you would like to run your own analyses with the dataset, these scripts can serve as a starting point. Of note, the multilevel regressions were all done in R. I expect that most readers familiar with multilevel regressions would prefer this. 

If you would like to just run analyses, then you should make a folder named "dataframes" at the outermost level of the project (i.e., at the same level as the `R_multilevel_reg` folder). To reproduce Results Section 3.2, run `R_multilevel_reg\Results_3.3_main.R`

There are a few parts to the code that I have included for any very curious reader, such as implementing something to fit a "z-curve" for each paper (see the Z-Curve 2.0 by Bartoš, F., & Schimmack, U., 2022, Meta-Psychology). Many functions also have features that ultimately weren't used for the manuscript, many of which I built but didn't have the time to fully explore myself.

To reproduce the multilevel regression results, you just need R, although everything else is done with Python.

The .csv files needed to reproduce the results, figures, and tables of the report have been added to the OSF repository in the "dataframes" folder. If you just download that, you should be able to reproduce them. However, rebuilding the dataset from its components will require downloading the data from the OSF Box add-on.

NOTE FOR PEER REVIEW: Many of the uploaded spreadsheets have dates in the filenames (e.g., "Jan21"). This reflects the date in which they were made and considers the possibility that they may be changed further (e.g., if a Reviewer request to change the exclusion criteria or add analyses requiring new columns). The final version will not have this date signifier.

UPDATE (October 3, 2024): I recently reinstalled my operating system, and I attempted to rerun many of the scripts based solely on downloading them from the present OSF repository. The only issue I encountered was that the project requires an empty "cache" folder to function, and I discovered that Chrome may need to be installed for downloading the full-text articles (Chrome may need to be installed along with the ChromeDriver, see 1.1.2, which I just updated). With these corrections, the scripts seem to be reproducible.

# 1. Dataset construction

## 1.1. Organizing the full text articles

The `prepare_fulltexts` folder contains the code needed to prepare the article full-texts as .txt files, which can be latter parsed by the scripts in `make_dataset`. If you want to run this yourself, you will need to download the .csvs from the Box OSF add-on.

1. `make_lens_df.py` prepares the Lens.org metadata dataset, which the scraping script (`download_fulltexts.py`) uses as the input. The data freely downloaded from Lens.org is available in `data\lens_org` (downloaded on August 3rd, 2024). `make_lens_df.py` will combine the many different Lens.org .csv into a single .csv. `make_lens_df.py` will also do some minor preprocessing. For instance, it fixes some journals names (e.g., sometimes the same journal name is saved with "&" but othertimes with "and"), such that the ISSN is ultimately what determines the name. `make_lens_df.py` will also handle the Taylor & Francis and APA data, even though that is only used for calculating authors' ages.
2. `download_fulltexts.py`downloads web versions of fulltext articles either via Elsevier's API or the other publishers websites. As of early August 2024, the publishers' websites Robots.txt do not block reading from the pages accessed (publishers' search features were NOT used; collection is targeted and based on the Lens.org data prepared with `make_lens_df.py`). To run `download_fulltexts.py` in full, you must have signed up for the Elsevier API (it's free), download Selenium, and download  ChromeDriver.exe (a version of Chrome usable with Selenium). I got this running on a Windows PC, using Python 3.12, and with packages installed using Anaconda. You will need to specify the API key and ChromeDriver.exe path in the script yourself. `download_fulltexts.py` will likely be the most difficult script to run if you have not scraped a website before, as it is more involved than just installing Python packages. Fulltexts are downloaded as XML/HTML files. It will likely take over a day to collect all the fulltexts from 2004-2024. Critically, you will likely need to be connected to a university VPN for the downloading to work, as most of the articles are subscription gated. If you are not affiliated with a university or your university's IT infrastructure doesn't support this, then the scripts will probably not generate fulltext articles for you. You may also need to install Chrome (and possibly also modify your system environment variables so that it is accessible). Note, I have also tested the FireFox driver, and it does not work as well for some journals (the site rejects the attempt to download an article).
3. `make_plaintexts.py` takes the XML/HTML fulltexts downloaded and processes them with BeautifulSoup. `make_plaintexts.py`  produces one version of the article with all of the codes removed (including Results sections), which is later used for identifying papers' affiliations. `make_plaintexts.py` will also make a version of the article while attempting to extract only the Results section and to omit figures, tables, and captions. As you will see, `make_plaintexts.py` is quite elaborate and involves a lot of things manually coded to handle specific publishers. You may want to set up the paths to use an SSD or NVME drive if available, as they may increase processing speed considerably. It will likely take over a day to process all the fulltexts from 2004-2024.
4. `make_has_results_df.py` produces a dataframe describing whether each paper has a valid Results section or not.

## 1.2. Making the dataset for analysis

The `make_dataset` takes the fulltext .txt files made using `prepare_fulltexts` along with data from other sources (e.g., THE rankings) to produce the dataset used for statistical analysis. If you want to run this yourself, you will need to download the .csvs from the Box OSF add-on.

1. `make_affiliation_df.py` parses the first half of fulltext .txt files (all sections), and search for university names and assign papers university prestige scores. It can do this both for papers only with Results sections or all papers. The former is more relevant to the analysis but the latter was used to show that 978 universities across all downloaded documents can be found (reported in Supplemental Materials 7.3), and organizes these as a .csv (one row = one paper).
2. `make_author_age_df.py` uses the lens.org .csv created by `make_lens_df.py` to calculate academic age (see Supplemental Materials 9.3), and organizes these as a .csv (one row = one paper).
3. `make_Bayesian_ML_word_df.py` identifies Bayesian/ML words in Results sections (see Supplemental Materials 9.4), and organizes these as a .csv (one row = one paper).
4. `make_cites_df.py` calculates the log-transformed and year-normalized citation scores (see Supplemental Materials 7.2), and organizes these as a .csv (one row = one paper).
5. `make_journal_d_scopus.py` uses the Scopus API to gather information on journals (Scopus journal categories, e.g,. Social psychology & SNIP scores). It uses ISSNs. You need to signed up for the Elsevier API and configured it for the pybliometrics package (https://pybliometrics.readthedocs.io/en/stable/configuration.html).
6. `make_journals_df.py` uses the Scopus API data gathered via `make_journal_d_scopus.py` and papers a .csv representing this for each paper in the dataset.
7. `make_p_z_df.py` is a big kahuna. It does all of the p-value, statistic and test statistic, and sentence extraction. It contains much delicate regex code. It produces a .csv where one row = one p-value (with colums for statistics, degrees of freedom, sentences, etc.)
8. `make_processed_p_dfs.py` takes the .csv prepared by `make_p_z_df.py` and processes it, categorizing each p-value (e.g., .01 <= p < .05), calculates fragile p-values, categorizes a paper's reporting style, etc. `make_processed_p_dfs.py` produces "df_p_process_Jan21.csv" where one row = one paper and "df_p_processed_by_p_Jan21.csv" where one row = one p-value.
9. `prepare_combined.py` will combine everything from the prior scripts into and produce three .csvs (pre-pruning, many papers here were not in the final dataset). The .csvs are as follows "df_combined_Jan21.csv" (each row = one paper), "df_by_pval_combined_Jan21.csv" (each row = one p-value), and "df_combined_all_aff_Jan21.csv" (each row = one paper, but this one also has many columns corresponding to the many ways that one affiliation and prestige score can be assigned to each paper).
10. `prune_to_final.py` will apply the exclusion criteria listed in Methods 2.2 to the dataframes. It produces multiple versions: `empirical` (keep all papers with a Results section but dropping ones without SNIP or an identified university), `semi_pruned` (i.e,. keeping papers with 2+ significant p-values but missing SNIP or a no identified university) and `pruned` (i.e,. keeping papers with 2+ significant p-values and missing SNIP and an identified university)
11. `UNUSED_make_zcurve_df.py` some code related to the z-curve 2.0 paper (F Bartoš, U Schimmack; Meta-Psychology, 2022), which was ultimately not used for the final report

# 2. Dataset use

## 2.1. Validation

`validate` contains the code needed to run the three validations. To run these, you can just download the .csvs from the `dataframes` folder. You shouldn't need to download the full set of data from the Box add-on.

1. `validate_1_manual.py` compares two of the output .csvs generated using the `make_dataset` scripts and compares them to the p-value counts established via manual inspection (manual counts with DOIs provided saved in `validate/manual_validation_sheet.csv`)
2. `validate_2_p_x_p_implied.py` cross-checks p-values and p-values implied from test statistics
3. `validate_3_replicability.py` makes the replication data figures, runs the t-tests, and runs the classification based on the fragile p-value percentage calculated with p-values or p-values implied from test statistics
 
## 2.2. Counting and plotting

`plot_and_count` contains the code needed to produce the figures and produces counts used in different parts of the manuscript. For the most part, the scripts' outputs are in the filename, and descriptions are only given in cases where isn't the case and nothing needs to be noted. You should be able to run these if you just .csvs from the `dataframes` folder. You shouldn't need to download the full set of data from the Box add-on. 

1. `calc_pfrag_sample_effect_corrs.py` compute the paper-by-paper correlations between fragile p-values, sample sizes, and effect sizes reported in Results 3.1
2. `calc_pfragile_bias.py` calculates the 2.3% bias reported 
3. `calc_power_lines.py` simulated the expected fragile p-value percentage based on different levels of statistical power, while using the number of p-values actually seen in the dataset (e.g., a p-fragile percentage of 26% is expected from a study with 80% power for every result)
4. `count_many_things.py` produces most of the counts reported in the Methods (e.g., the number of papers meeting each criteria, the number of universities identified across the whole dataset, etc.)
5. `plot_Figure1_descriptives.py` plots Figure 1
6. `plot_Figure2_4_temporal_trends.py` plots Figures 2 and 3
7. `plot_Figure3_ridgeplot.py` plots Figure 3
8. `plot_Figure5A_cites.py` plots Figure 5A and the equivalent p-implied figure (Figure S11A)
9. `plot_Figure5B_journal.py` plots Figure 5B and the equivalent p-implied figure (Figure S11B)
10. `plot_Figure5C_university.py` lots Figure 5C and the equivalent p-implied figure (Figure S11C) along with Figure S8 on the university ranking x Bayes/ML usage scatterplots
11. `plot_FigureS12_misreport.py` plots Figures S1 and S2
12. `plot_FigureS34_insig.py` plots Figures S3 and S4
13. `plot_FigureS6_THE_matrix.py` plots Figure S6
14. `plot_FigureS7_age_histogram.py` plots Figure S7
15. `tally_TableS1_pval_style.py` produces the Table S1 counts of p-value reporting style. P-value reporting style is mainly calculated in `make_dataset\make_processed_p_dfs.py` but is also slightly processed here to make the .001, .01, etc. cutoffs (e.g., .005 "all less" papers, which are rare, are here converted to .01 "all less" papers)


## 2.3. Multilevel regressions

`R_multilevel_reg` contains the code needed to reproduce multilevel regression results. These are all done using  R and their filenames describe what portion of the manuscript or supplemental materials that they correspond to. To run these, you can just download the .csvs from the `dataframes` folder. You shouldn't need to download the full set of data from the Box add-on.

1. `Results_3.2_main.R`
2. `SuppMat_9.1_alt_university.R`
3. `SuppMat_11.2_num_fragile.R`
4. `SuppMat_11.3_ages.R`
5. `SuppMat_11.4_has_pvals.R`
6. `SuppMat_11.5_Baye_ML.R`
7. `SuppMat_13.2_p_implied.R`

## 2.4. Word use analysis

`text_analysis` performs the text analysis whose results are reported in Results Section 3.3. It is conducted sequentially, build, mass rergess, then plot and make tables. If you want to reproduce figure 4 and tables, you should run `produce_plots_tables.py`. It will leverage the stored regression from the `dataframes/word_regr` folder. If you want to do more than that (e.g,. running the regressions), you will need to download the full set of data from the Box add-on to OSF. `EN_dict.txt` is used by the code to convert plural nouns to singular ones

1. `build_df_word.py` finds the top 2500 words for each category (e.g., t-values, F-values, all) and makes a dataframe representing where each row represents one sentence's usage of each of the top 2500 words (words each encoded in a separate column where 1 = sentence has word, 0 = does not). Given all these columns, the dataframes are massive. They are hence saved as .pkl files in the `cache` folder (see also `utils.py` below)
2. `mass_regressions.py` uses the .csv produced with `build_df_word.py` to run the many regressions. The regression coefficients are saved in a .csv in `dataframes/word_regr`
3. `plot_Figure6_language.py` reads the regression .csv(s) made with `mass_regression.py` and produces Figure 6
4. `regress_control_language.py` performs the regression of p-fragile ~ university rankings while controlling for the usage of the 2500 words. 
5. `tall_Table1_language_overlap.py` reads the regression .csv(s) made with `mass_regression.py` and produces Table 1 along with Supplemental Tables S2-12

## 2.5. Minor utilities

`utils.py` contains some functions that are helpful elsewhere. Of note, `pickle_wrap` helps with caching the output of functions, `save_csv_pkl` is used to save .pkl versions of each .csv as .pkl files are faster to save/load, and `read_csv_fast` will load a .pkl version of a .csv if available (otherwise it will use the .csv to create one). The .pkl versions of files are saved in a `cache` folder. If you do not care at all about having available .csvs (e.g., you don't plan to open them in excel or R), you could change the code to operate entirely with .pkls. This is what `build_df_word.py` is doing when dealing with those massive 2500+ colums dataframes. 

# 3. Final spreadsheet columns

These are all the columns in the "_combined" dataframes. All of the analyses are based on these "combined" spreadsheets.

They were based on considering the version where each row corresponds to one paper (e.g., "df_combined_pruned_Jan21.csv"). The main relevant difference to the version with "_by_pval" is that the by_pval version uses a 1 or 0 for the columns representing the number of p-values of different types (e.g., for the paper-wise sheet, n05_exact represents the number of "=" p-values between .01 to .05 but for the p-val-wise sheet, it is a 1 if the p-value is in that range and 0 otherwise). The difference relative to the "_all_aff" sheet is that the latter includes many more columns representing the different ways in which one university and/or prsetige score can be assigned to a one paper (only used by `SuppMat_9.1_alt_university.R`).

Some spreadsheets necessary for some of the scripts (e.g., "df_by_pval_combined_semi_pruned_Jan21.csv") are too large to upload to OSF/Box (5 GB max filesize). To produce these, download all of the other .csvs and combine them with `prepare_combined.py` then drop not-analyzed papers with `prune_to_final.py`.

The columns are as follows: 

* "school": a paper's affiliated university (selected via paper name that appears most often in the fulltext, ties broken randomly)
* "country": school's university
* "year_rank": THE rank based on publication year
* "target_rank": THE rank based on 2024
* "year_score": THE research score based on publication year 
* "target_score": THE research score based on 2024
* "doi_str": doi formatted to remove “/” and “.”
* "num_affil": number of matched affiliations, even though only one is listed in the “school” column
* "ISSN": ISSN (number representation of a journal)
* "ISSNs": ISSN content downloaded from lens.org (“ISSN” column is a clean version of this)
* "age_first": academic age calculated based on the first author’s first time in the first author position (not used for manuscript)
* "age_last": academic age calculated based on the last author’s first time in the last author position (not used for manuscript)
* "author_first": first author
* "author_last": last author 
* "authors_str": full author info from lens.org
* "backup_ISSN": secondary ISSN
* "date": from Lens.org
* "doi": from Lens.org
* "external_url": from Lens.org 
* "is_open_access": from lens.org
* "journal": from Lens.org, the journal names were cleaned to make sure that they are consistent across small changes (e.g., "and" vs. "&"). See the "journal_clean" column
* "mesh_terms": some information on paper content from Lens.org (not investigated)
* "publisher": from Lens.org
* "source_url": from Lens.org 
* "title": from Lens.org 
* "year": from Lens.org 
* "Applied_Psychology": 1 or 0 representing whether a paper’s journal is given this label by scopus
* "Clinical_Psychology": 1 or 0 representing whether a paper’s journal is given this label by scopus
* "Developmental_and_Educational_Psychology": 1 or 0 representing whether a paper’s journal is given this label by scopus
* "Experimental_and_Cognitive_Psychology": 1 or 0 representing whether a paper’s journal is given this label by scopus
* "General_Psycholog": 1 or 0 representing whether a paper’s journal is given this label by scopus 
* "Psychology_Miscellaneous": 1 or 0 representing whether a paper’s journal is given this label by scopus 
* "Social_Psychology": 1 or 0 representing whether a paper’s journal is given this label by scopus
* "is_neuro": 1 if scopus labeled as “Cognitive Neuroscience” otherwise 0
* "SNIP": from scopus
* "baye_paper": 1 if Results section contains a Bayesian word otherwise 0
* "freq_paper": 1 if Results section contains a frequentist word otherwise 0
* "ML_paper": 1 if Results section contains a machine learning word otherwise 0
* "school_M_baye": percentage of papers from a paper's school that have a Bayesian word 
* "school_M_freq": percentage of papers from a paper's school that have a frequentist word
* "school_M_ML": percentage of papers from a paper's school that have a machine learning  word
* "has_results": True if has results else False 
* "cites": from Lens.org
* "cites_year": cites divided by (2024 - year), 
* "log_cites_year": log of cites_years, 
* "log_cites_year_z": log_cites_year z-scored separately for each year, 
* "cond": p-value reporting style
* "insig": number of insignificant p-values (this and for all of the "number of p-value predictors" are 0 or 1 for the by_p .csv)
* "insig_exact": number of insignificant p-value repoterd with "="
* "insig_implied": number of insignificant implied p-values
* "insig_less": number of insig p-values with "<" sign
* "insig_over": number of insig p-values with ">" sign
* "lowest_p_implied": lowest implied p-value
* "lowest_p_val": lowest p-value
* "n001": number of p-values below .001 with "=" or "<"
* "n001_exact": number of p-values below .001 with "=" or "<" (redundant with n001; this is the only case where "exact" includes "p < .001" because this matches APA style)
* "n001_implied": number of implied p-values below .001 
* "n001_less":  number of p-values below .001 with "<"
* "n005_h": number of p-values between .005 and .01 with "=" or "<"
* "n005_h_exact": number of p-values between .005 and .01 with "="
* "n005_h_implied": number of implied p-values between .005 and .01 
* "n005_h_less": number of p-values between .005 and .01 with "<"
* "n005_l": number of p-values between .001 and .005 with "=" or "<"
* "n005_l_exact": number of p-values between .001 and .005 with "="
* "n005_l_implied": number of p-values between .001 and .005 
* "n005_l_less": number of p-values between .001 and .005 with "<"
* "n01": n005_h + n005_l
* "n01_001": n01 + n001 
* "n01_001_implied": n01_implied + n001_implied 
* "n01_implied": n005_h_implid + n005_l_implid
* "n05": number of p-values between .01 and .05 with "=" or "<"
* "n05_exact": number of p-values between .01 and .05 with "="
* "n05_implied": number of implied p-values between .01 and .05
* "n05_less": number of p-values between .01 and .05 with "="
* "n_exact05": number of p-values where p = .05 (as in, exactly p = .05; these were excluded because it is ambiguous whether they are considered significant or insignificant)
* "num_ps": number of significant "=" p-values except for "p = .05" + number of "<" p-values
* "num_ps_any": number of significant or insignificant p-values, including p = .05
* "num_ps_any_implied": number of significant or insignificant implied p-values
* "num_ps_exact": number of p-values with "="
* "num_ps_implied": number of significant implied p-values
* "num_ps_less": number of p-values with "<"
* "p_fragile": percentage of p-values in a manuscript that are fragile (the main dependent variable of the research)
* "p_fragile_implied":  percentage of implied p-values in a manuscript that are fragile , 
* "p_fragile_orig": percentage of p-values in a manuscript that are fragile while not setting pure "p < .05" papers to 51%
* "p_fragile_w_exact05": p-fragile while counting "p = .05" as a fragile p-value rather than dropping it (not used for manuscript)
* "prop_implied": num_ps_implid / num_ps
* "sig": number of significant p+values
* "sig_exact": number of significant exact p-values or number of "p < x" p-values where x <= .001
* "sig_implied": number of significant implied p-values
* "sig_less": number of significant exact p-values reported with "<"
* "d": cohen's d implied from t-value and degrees of freedom + 1
* "t_N": sample size implied from t-stat degrees of freedom + 1
* "journal_clean": journal name from Lens.org but cleaned up such that the same journal always has the same name (e.g., fix "and" vs. "&" reporting)
* "p_fragile_j": percentage of fragile p-values while extrapolating for empirical papers without p-values by using the journal-wide data
* "jrl_cnt": count of the number of papers with Results sections that a journal published in the row's publication year
* "has_results_j": has_results and jrl_cnt >= 5 (manually set papers from journal years with few empirical reports to not being considered to have a Results section)
* "country_school_count": for a given paper's country, this is how many top 1000 identified universities are from that country countries (e.g., if the USA has 150 schools, then all USA papers have 150 for this value)
* "has_ps": num_ps > 0 