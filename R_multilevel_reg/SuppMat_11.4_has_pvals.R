# ----------
require('lme4')
require('lmerTest')
require('dplyr')
require('rstudioapi')

# ----------
# Set working directory to the parent folder of the current file's folder
setwd(dirname(dirname(getSourceEditorContext()$path)))

# -----------
# Not pruned to only papers with p_values
df_ = read.csv('dataframes/df_combined_all_empirical_Jan21.csv') 

# -----------
# z-standardize
df = df_
df$has_ps = df$num_ps > 0

df['research'] = df['target_score']

df$research = scale(df$research)
df$num_ps = scale(df$num_ps)
df$age_last = scale(df$age_last)
df$year = scale(df$year)
df$p_fragile = scale(df$p_fragile)
df$p_fragile_j = scale(df$p_fragile_j)
df$SNIP = scale(df$SNIP)
df$log_cites_year_z = scale(df$log_cites_year_z)

# -----------

formula = 'has_ps ~ SNIP * year + log_cites_year_z * year + research * year + 
                    (1 + SNIP | journal_clean) + (1 | school) + (1 | country)'
m = glmer(formula, data=df,  control=glmerControl('bobyqa'), family=binomial,)
print(summary(m))

# -----------

# some nans. Can occur if a journal has 5+ papers with a results section in a 
#   year but none have p-values

formula = 'p_fragile_j ~ research * year + 
                         (1 | journal_clean) + (1 | school) + 
                         (1 | country)'
m = lmer(formula, data=df, control=lmerControl('bobyqa'))
print(summary(m))

