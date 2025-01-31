# ----------
require('lme4')
require('lmerTest')
require('dplyr')
require('rstudioapi')

# ----------
# Set working directory to the parent folder of the current file's folder
setwd(dirname(dirname(getSourceEditorContext()$path)))

df_ = read.csv('dataframes/df_combined_pruned_all_aff_Jan21.csv') 
# -----------
# z-standardize
df = df_

df['research'] = df['TargetMax_target_score']
df['school'] = df['TargetMax_school']
df['country'] = df['TargetMax_country']

df$research = scale(df$research)
df$num_ps = scale(df$num_ps)
df$age_last = scale(df$age_last)
df$year = scale(df$year)
df$p_fragile = scale(df$p_fragile)
df$p_fragile_implied = scale(df$p_fragile_implied)
df$SNIP = scale(df$SNIP)
df$log_cites_year_z = scale(df$log_cites_year_z)

# ------------
formula = 'p_fragile ~ research * year +
                      (1 | journal_clean) + (1 | school) + (1 | country)'
m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)
print(summary(m))

# ------------
df['research'] = df['TargetMin_target_score']
df['school'] = df['TargetMin_school']
df['country'] = df['TargetMin_country']
df$research = scale(df$research)
formula = 'p_fragile ~ research * year +
                      (1 | journal_clean) + (1 | school) + (1 | country)'
m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)
print(summary(m))

# ------------
df['research'] = df['TargetMed_target_score']
df['school'] = df['TargetMed_school']
df['country'] = df['TargetMed_country']
df$research = scale(df$research)
formula = 'p_fragile ~  research * year +
                      (1 | journal_clean) + (1 | school) + (1 | country)'
m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)
print(summary(m))

# ------------
df['research'] = df['Mean_target_score']
df$research = scale(df$research)
formula = 'p_fragile ~ research * year +
                       (1 | journal_clean)'
m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)
print(summary(m))

# ------------
# yearly
df['research'] = df['Mode_year_score']
df$research = scale(df$research)
formula = 'p_fragile ~ research * year +
                       (1 | journal_clean) + (1 + research | school) + (1 | country)'
m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)
print(summary(m))
