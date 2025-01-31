# ----------
require('lme4')
require('lmerTest')
require('dplyr')
require('rstudioapi')
require('optimx')

# ----------
# Set working directory to the parent folder of the current file's folder
setwd(dirname(dirname(getSourceEditorContext()$path)))

# ----------
df_ = read.csv('dataframes/df_combined_pruned_Jan21.csv') 
df = df_
# -----------
# z-standardize
df['research'] = df['target_score'] # just renaming
df$research = scale(df$research)
df$num_ps = scale(df$num_ps)
df$age_last = scale(df$age_last)
df$year = scale(df$year)
df$p_fragile = scale(df$p_fragile)
df$p_fragile_implied = scale(df$p_fragile_implied)
df$SNIP = scale(df$SNIP)
df$log_cites_year_z = scale(df$log_cites_year_z)

# -----------
# unstandardized
df = df_
df$year = df$year - 2016.537
df['research'] = df['target_score']

# ------------
formula = 'SNIP ~ p_fragile * year + 
                  (1 + p_fragile | journal_clean) + 
                  (1 + p_fragile | school) + 
                  (1 | country)'
m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)
print(summary(m))

# ------------
df$SNIP = scale(df$SNIP)
formula = 'log_cites_year_z ~ p_fragile * year + SNIP * year +
                       (1 + p_fragile | journal_clean) + 
                       (1 + p_fragile | school) + 
                       (1 | country)'
m = lmer(formula, data=df,  REML=F, 
         control=lmerControl(optimizer="optimx", optCtrl=list(method="nlminb")),)
print(summary(m))

# ------------
formula = 'p_fragile ~ research * year + 
                       (1 | journal_clean) + 
                       (1 | school) + 
                       (1 | country)'
m = lmer(formula, data=df,  REML=F, control=lmerControl('bobyqa'),)
print(summary(m))




