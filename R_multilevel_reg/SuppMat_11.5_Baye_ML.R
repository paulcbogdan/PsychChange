# ----------
require('lme4')
require('lmerTest')
require('dplyr')
require('rstudioapi')

# ----------
# Set working directory to the parent folder of the current file's folder
setwd(dirname(dirname(getSourceEditorContext()$path)))

# ----------
# Not pruned to only papers with p_values
df_ = read.csv('dataframes/df_combined_all_empirical_Jan21.csv') 
df = df_

# -----------
# Convert 'True' and 'False' to TRUE and FALSE, respectively
df$baye_paper = toupper(df$baye_paper) == TRUE
df$ML_paper = toupper(df$ML_paper) == TRUE
df$freq_paper = toupper(df$freq_paper) == TRUE

# -----------
# Counts
print(mean(df$baye_paper))
print(mean(df$ML_paper))
print(mean(df$freq_paper))

# -----------
# z-standardize
df['research'] = df['target_score']

df$research = scale(df$research)
df$num_ps = scale(df$num_ps)
df$age_last = scale(df$age_last)
df$year = scale(df$year)
df$p_fragile = scale(df$p_fragile)
df$p_fragile_implied = scale(df$p_fragile_implied)
df$SNIP = scale(df$SNIP)
df$log_cites_year_z = scale(df$log_cites_year_z)
# -----------

formula = 'baye_paper ~ SNIP * year + log_cites_year_z * year + 
                        research * year + 
                        (1 + SNIP | journal_clean) + (1 | school) + 
                        (1 | country)'
m = glmer(formula, data=df,  control=glmerControl('bobyqa'), family=binomial)
print(summary(m))

# -----------

formula = 'ML_paper ~ SNIP * year + log_cites_year_z * year + 
                      research * year + 
                      (1 | journal_clean) + (1 | school) + (1 | country)'
m = glmer(formula, data=df,  control=glmerControl('bobyqa'), family=binomial,)
print(summary(m))


# -----------
# Main text multilevel regression predicting p-fragile but now accounting for
#   has_baye and the like
df_ = read.csv('dataframes/df_combined_pruned_Jan21.csv') 

# -----------
df = df_

df['research'] = df['target_score']

df$research = scale(df$research)
df$num_ps = scale(df$num_ps)
df$age_last = scale(df$age_last)
df$year = scale(df$year)
df$p_fragile = scale(df$p_fragile)
df$p_fragile_implied = scale(df$p_fragile_implied)
df$SNIP = scale(df$SNIP)
df$log_cites_year_z = scale(df$log_cites_year_z)

df$baye_paper = toupper(df$baye_paper) == TRUE
df$ML_paper = toupper(df$ML_paper) == TRUE
df$freq_paper = toupper(df$freq_paper) == TRUE

# -----------

formula = 'p_fragile ~ research * year +
                       baye_paper + ML_paper + school_M_baye + school_M_ML +
                       (1 | journal_clean) + (1 | school) + 
                       (1 | country)'
m = lmer(formula, data=df,  REML=F,
         control=lmerControl('bobyqa'),)
print(summary(m))
