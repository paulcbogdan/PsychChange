# ----------
require('lme4')
require('lmerTest')
require('dplyr')
require('rstudioapi')

# ----------
# Set working directory to the parent folder of the current file's folder
setwd(dirname(dirname(getSourceEditorContext()$path)))

df_ = read.csv('dataframes/df_combined_pruned_Jan21.csv') 
# -----------
# z-standardize
df = df_
df['research'] = df['target_score'] # just renaming
df$research = scale(df$research)
df$num_ps = scale(df$num_ps)
df$age_last = scale(df$age_last)
df$year = scale(df$year)
df$p_fragile = scale(df$p_fragile)
df$p_fragile_implied = scale(df$p_fragile_implied)
df$SNIP = scale(df$SNIP)
df$log_cites_year_z = scale(df$log_cites_year_z)

df$n05_ = scale(df$n05)
df$n001_ = scale(df$n001)
df = dplyr::filter(df, n05_ < 3)

# ------------

# formula = 'n05 ~ SNIP * year + log_cites_year_z * year + 
#                        research * year +
#                        (1 + SNIP | journal_clean) + (1 | school) + 
#                        (1 | country)'
# m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)
# 
# print(summary(m))

# ------------
# df = dplyr::filter(df, n05_ < 3) # outliers 
df = dplyr::filter(df, n05_ < 3)

formula = 'SNIP ~ n05_ * year + 
                  (1 + n05_ | journal_clean) + 
                  (1 + n05_ | school) + 
                  (1 | country)'
m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)
print(summary(m))

# ------------
formula = 'log_cites_year_z ~ n05_ * year + SNIP * year +
                       (1 + n05_ | journal_clean) + 
                       (1 + n05_ | school) + 
                       (1 | country)'
m = lmer(formula, data=df,  REML=F, 
         control=lmerControl(optimizer="optimx", optCtrl=list(method="nlminb")),)
print(summary(m))

# ------------
formula = 'n05_ ~ research * year + 
                       (1 | journal_clean) + 
                       (1 | school) + 
                       (1 | country)'
m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)
print(summary(m))


# ------------

df = dplyr::filter(df, n05_ < 3)
formula = 'num_ps ~ SNIP * year + log_cites_year_z * year + 
                       research * year +
                       (1 + SNIP | journal_clean) + (1 | school) + 
                       (1 | country)'
m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)

print(summary(m))

