# ----------
require('lme4')
require('lmerTest')
require('dplyr')
require('rstudioapi')

# ----------
# Set working directory to the parent folder of the current file's folder
setwd(dirname(dirname(getSourceEditorContext()$path)))

df_ = read.csv('dataframes/df_combined_pruned_Aug24.csv') 
# -----------
# z-standardize
df['research'] = df['target_score']
df$age_last = -df$age_last # now, older = higher

df$research = scale(df$research)
df$age_last = scale(df$age_last)
df$SNIP = scale(df$SNIP)
df$log_cites_year_z = scale(df$log_cites_year_z)
df$year = scale(df$year)
df$p_fragile = scale(df$p_fragile)

# ------------

formula = 'age_last ~ SNIP * year + log_cites_year_z * year + 
                      research * year +
                      (1 + SNIP | journal_clean) + (1 | school) + (1 | country)'
m = lmer(formula, data=df,  REML=F,
         control=lmerControl('bobyqa'),)
print(summary(m))

# ------------

# Yields null effect of age
formula = 'p_fragile ~ age_last + year + 
                       (1 | journal_clean) +
                       (1 | school) + 
                       (1 | country)'
m = lmer(formula, data=df,  REML=F,
         control=lmerControl('bobyqa'),)
print(summary(m))



