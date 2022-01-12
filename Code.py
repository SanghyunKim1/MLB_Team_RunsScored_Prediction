import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy import stats
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

# Purpose of Analysis: predicting a team's runs scored based on various team batting statistics

# load datasets
path = "/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/"
data_files = sorted([file for file in os.listdir(path)], reverse = True)

# empty dataframe
batting_df = pd.DataFrame()

# load data files one at a time and merge them into a single data frame
for file in data_files:
    df = pd.read_csv(path + file)
    batting_df = pd.concat([batting_df, df])

print(batting_df.head().to_string())

# 1. Data Cleaning
# rename specific column names
batting_df.rename(columns = {'R': 'RS'}, inplace = True)

# check missing data
print("Total number of missing values in each column:")
print(batting_df.isnull().sum())

# check duplicated data
print("Total number of duplicates in batting data: {}".format(batting_df.duplicated().sum()))

# create a new column for "Total Base (TB)" data
batting_df["TB"] = batting_df["1B"] + (2 * batting_df["2B"]) + (3 * batting_df["3B"]) + (4 * batting_df["HR"])

# export league data: National League (NL) / American League (AL)
nl_teams = ["ATL", "CHC", "CIN", "COL", "LAD",
            "MIL", "NYM", "PHI", "PIT", "SDP",
            "SFG", "STL", "ARI", "WSN", "FLA",
            "MIA", "MON"]
al_teams = ["MIN", "CHW", "CLE", "DET", "KCR",
            "TOR", "BAL", "BOS", "NYY", "OAK",
            "SEA", "TEX", "LAA", "TBR", "TBD",
            "ANA"]

# define leagues: NL / AL
def get_league(data):
    if data["Team"] in nl_teams:
        return "NL"
    elif data["Team"] in al_teams:
        return "AL"
    elif data["Team"] == "HOU" and data["Season"] <= 2012:
        return "NL"
    else:
        return "AL"

batting_df["League"] = batting_df.apply(lambda x: get_league(x), axis = 1)

# check data types
print(batting_df.dtypes)

# check memory usage
print("Total Memory Usage Before: {}".format(batting_df.memory_usage(deep = True).sum()))

# to save memory usage, change "League" and "Team" data type
batting_df["Team"] = batting_df["Team"].astype("category")
batting_df["League"] = batting_df["League"].astype("category")

print("Total Memory Usage After: {}".format(batting_df.memory_usage(deep = True).sum()))

# reorder data columns
cols = ["Season", "League"] + list(batting_df.columns)[1:-1]
batting_df = batting_df.reindex(columns = cols)



# 2. EDA (Exploratory Data Analysis)
# 2-1. RS Analysis: How did the league average runs scored change over time?: One-way ANOVA
print("------- Team Runs Scored Descriptive Summary -------")
print(batting_df["RS"].describe())

season_df = batting_df.groupby("Season")
lg_avg_rs = season_df["RS"].mean().round(1).reset_index()
print("------- Yearly Changes in League Average Runs Scored -------")
print(lg_avg_rs)

# bar plot
values = np.array(lg_avg_rs["RS"])
idx = np.array(lg_avg_rs["Season"])
c1 = mpatches.Patch(color = "darkred", label = "Steroid Era")
c2 = mpatches.Patch(color = "lightcoral", label = "Post-steroid Era")
c3 = mpatches.Patch(color = "red", label = "Fly-ball Revolution Era")

fig, ax = plt.subplots(figsize = (12, 8))

plt.bar(idx, values, edgecolor = "darkgrey", linewidth = 0.6,
        color = ["darkred"] * 7 + ["lightcoral"] * 8 + ["red"] * 7,
        alpha = 0.7, zorder = 3)
plt.xticks(lg_avg_rs["Season"], rotation = 45)
plt.xlabel("Season")
plt.ylabel("League Average Runs Scored")
plt.title("Yearly Changes in League Average Runs Scored", fontsize = 18)
plt.legend(handles = [c1, c2, c3], ncol = 3,
           bbox_to_anchor= (0.72, -0.12), loc = "upper center")
plt.grid(zorder = 0)
fig.subplots_adjust(bottom = 0.15)
plt.show()

# box plot
fig, ax = plt.subplots(figsize = (10, 9))

sns.boxplot(x = "Season", y = "RS", data = batting_df, ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
ax.set_title("Team RS Distribution in each Season", fontsize = 20)
plt.show()

# divide seasons into "Era" groups
def get_era(data):
    if data["Season"] <= 2006:
        return "Steroid Era"
    elif data["Season"] > 2006 and data["Season"] <= 2014:
        return "Post-steroid Era"
    else:
        return "Fly-ball Revolution Era"

batting_df["Era"] = batting_df.apply(lambda x: get_era(x), axis = 1)
batting_df["Era"] = batting_df["Era"].astype("category")

# check one-way ANOVA assumptions
# normality
eras = batting_df["Era"].unique()
fig = plt.figure(figsize = (12, 12))

for era, i in zip(eras, range(1, 5)):
    ax = fig.add_subplot(2, 2, i)
    stats.probplot(batting_df.loc[batting_df["Era"] == era]["RS"], plot = plt)
    ax.set_title("{}".format(era))
    ax.set
fig.suptitle("Team RS QQ Plot in each Era", fontsize = 24, y = 0.95)
plt.show()

# since Fly-ball Era "RS" are not normally distributed due to 2020 season data,
# drop 2020 season data (abnormal 60-game season = outliers)
batting_df = batting_df.loc[batting_df["Season"] != 2020]

# check the normality assumption again
eras = batting_df["Era"].unique()
fig = plt.figure(figsize = (12, 12))

for era, i in zip(eras, range(1, 5)):
    ax = fig.add_subplot(2, 2, i)
    stats.probplot(batting_df.loc[batting_df["Era"] == era]["RS"], plot = plt)
    ax.set_title("{}".format(era))
fig.suptitle("Team RS QQ Plot in each Era without 2020 Season Data", fontsize = 24, y = 0.95)
plt.show()

# equal-variance
# descriptive summary
era_df = batting_df.groupby("Era")
rs_sum_era = era_df["RS"].describe()
print("------- Team Runs Scored Data Descriptive Summary by Era -------")
print(rs_sum_era.to_string())

# box plot
fig, ax = plt.subplots(figsize = (10, 9))

sns.boxplot(x = "Era", y = "RS", data = batting_df, ax = ax,
            palette = "BuPu", boxprops = dict(alpha = 0.5))
ax.set_xticklabels(ax.get_xticklabels())
ax.set_title("Team RS Distribution in each Era", fontsize = 20)
plt.show()

# one-way ANOVA F-test
model = ols("RS ~ C(Era)", data = batting_df).fit()
one_aov_table = sm.stats.anova_lm(model, typ = 1)
print("------- One-way ANOVA F-test Result -------")
print(one_aov_table.round(3))
# since the p-value is approximately 0,
# we have significant evidence that is at least one pairwise group mean difference in "RS"

# one-way ANOVA post-hoc test with a Bonferroni correction
compar = mc.MultiComparison(batting_df['RS'], batting_df['Era'])
table, a1, a2 = compar.allpairtest(stats.ttest_ind, method = "bonf")
print("------- One-way ANOVA Post-hoc Test Result -------")
print(table)



# 2-2. RS Analysis: In which League (NL vs AL) did teams scored more?: Two-sample t-test
# group data based on "League" data values
lg_df = batting_df.groupby("League")
leagues = batting_df["League"].unique()
print("------- Team Runs Scored Data Descriptive Summary by League -------")
print(lg_df["RS"].describe())

nl_rs = batting_df.loc[batting_df["League"] == "NL"]["RS"]
al_rs = batting_df.loc[batting_df["League"] == "AL"]["RS"]

# check two-sample t-test assumptions
# normality
nl_rs = batting_df.loc[batting_df["League"] == "NL"]["RS"]
al_rs = batting_df.loc[batting_df["League"] == "AL"]["RS"]

fig, axes = plt.subplots(1, 2, figsize = (18, 8))
stats.probplot(nl_rs, plot = axes[0])
stats.probplot(al_rs, plot = axes[1])
axes[0].set_title("National League Runs Scored QQ Plot without 2020 Season", fontsize = 14)
axes[1].set_title("American League Runs Scored QQ Plot without 2020 Season", fontsize = 14)
plt.show()

# an equal-variance assumption
print("NL Runs Scored Variance: {}".format(nl_rs.var()))
print("AL Runs Scored Variance: {}".format(al_rs.var()))

# box plot
fig, ax = plt.subplots(figsize = (8, 8))
sns.boxplot(x = "League", y = "RS", data = batting_df, palette = "Set1", boxprops = dict(alpha = 0.5))
ax.set(title = "Runs Scored Distribution by League")
plt.show()

# Pooled two-sample t-test
test_result = pg.ttest(al_rs, nl_rs, paired = False, alternative = 'greater', correction = False).round(3)
print("------- Pooled two-sample t-test result -------")
print(test_result.to_string())
# given the p-value is approximately 0,
# we reject H0 and have a strong evidence that AL teams scored more than NL teams on average

# 'RS' histogram and QQ plot
fig, axes = plt.subplots(1, 2, figsize = (20, 8))

sns.histplot(batting_df['RS'], kde = True, ax = axes[0], color = "navy")
axes[0].set_title('Team RS Histogram')
axes[1] = stats.probplot(batting_df['RS'], plot = plt)
plt.title('Team RS QQ Plot')
plt.show()

# 2-3. RS Analysis: Do both "Era" and "League" affect the league average team "RS"?
# two-factor ANOVA F-test
# factor 1: "Era" and factor 2: "League"
model = ols("RS ~ C(Era) + C(League) + C(Era):C(League)", data = batting_df).fit()
two_aov_table = sm.stats.anova_lm(model, typ = 2)
print("------- Two-factor ANOVA Table -------")
print(two_aov_table.round(3))

# interaction plot
fig = interaction_plot(x = batting_df["League"], trace = batting_df["Era"], response = batting_df["RS"],
                       colors = ['#4c061d','#d17a22', '#b4c292'])
plt.title("Interaction Plot")
plt.ylabel("Mean RS")
plt.show()

# check ANOVA assumptions
# normality
fig = sm.qqplot(model.resid, line = "s")
plt.title("Two-factor ANOVA QQ Plot")
plt.show()

# equal-variance
g = sns.FacetGrid(batting_df, col = "Era", row = "League", height = 4, aspect = 1)
g.map_dataframe(sns.boxplot, y = "RS", data = batting_df,
                palette = "BuPu", boxprops = dict(alpha = 0.5))
g.set_axis_labels(y_var = "Team RS", labelpad = -2)
plt.show()



# 3. Feature Selection
# correlation matrix
corrMatrix = batting_df.corr()
fig, ax = plt.subplots(figsize = (8, 8))

sns.heatmap(corrMatrix, square = True, linewidths = 0.3)
plt.title('Correlation Matrix')
plt.show()

# 3-1. first drop variables that have lower (absolute) correlations with 'RS' than 0.65
corrMatrix = abs(batting_df.corr())
cols = list(corrMatrix.columns)
vars_to_drop = []
for col in cols:
    if corrMatrix[col]['RS'] < 0.65:
        vars_to_drop.append(col)

filtered_df = batting_df.loc[:, ~batting_df.columns.isin(vars_to_drop)]
filtered_vars = list(filtered_df.columns)

# 3-2. drop any independent variables if its pairwise correlation between other independent variables is higher than 0.9
ind_vars_df = filtered_df.iloc[:, filtered_df.columns != 'RS']

corrMatrix = abs(filtered_df.corr())
upperTri = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k = 1).astype(np.bool))
vars_to_drop = [col for col in upperTri.columns if any(upperTri[col] >= 0.9)]
filtered_df.drop(vars_to_drop, axis = 1, inplace = True)

# new correlation matrix for filtered data features
fig, ax = plt.subplots(figsize = (10, 10))

corrMatrix = filtered_df.corr()
sns.heatmap(corrMatrix, square = True, linewidths = 0.5, annot = True, annot_kws = {'size': 10},
            xticklabels = corrMatrix.columns, yticklabels = corrMatrix.columns)
plt.title('Correlation Matrix')

plt.show()

# 3-3. Recursive Feature Elimination
# select numerical data
num_df = filtered_df.select_dtypes(exclude = "category")
x = num_df.loc[:, num_df.columns != 'RS']
y = num_df['RS']
cols = list(x.columns)

model = LinearRegression()

rfe = RFE(estimator = model, n_features_to_select = 2)
x_rfe = rfe.fit_transform(x, y)
model.fit(x_rfe, y)

temp = pd.Series(rfe.support_, index = cols)
selected_vars = list(temp[temp == True].index)
print('RFE Features: {}'.format(selected_vars))

# check VIF
x = num_df[selected_vars]
x = sm.add_constant(x)
y = num_df['RS']

lm = sm.OLS(y, x)
result_rs = lm.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm.exog_names
vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
print(vif[vif['Feature'] != 'const'])

# 3-4. backward selection
lm = LinearRegression()
num_df = batting_df.select_dtypes(exclude = "category")
x = num_df.loc[:, num_df.columns != 'RS']
y = batting_df['RS']
sfs = SFS(lm, k_features = 2, forward = False, verbose = 2,
          scoring = "r2", cv = 0, n_jobs = -1)
sfs.fit(x, y)
print("\nBackward Selection Features: {}\n".format(sfs.k_feature_names_))
# from both Recursive Feature Elimination and Backward Selection methods,
# it turns out that the two most significant independent variables are: "OBP" and "ISO"

# select final features for multiple linear regression model
mlr_df = batting_df.loc[:, ["RS", "OBP", "ISO"]]



# 4. Multiple Linear Regression with feature selection
# split data into training and test data and build a multiple linear regression model
# multiple linear regression (x:'OBP', 'ISO' / y:'RS')
x = mlr_df[selected_vars]
y = mlr_df['RS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

mlr = linear_model.LinearRegression().fit(x_train, y_train)
y_predict = mlr.predict(x_test)

# 4-1. Assumption checking
# linearity
# scatter plots
cols = list(mlr_df.drop('RS', axis = 1).columns)

fig, axes = plt.subplots(1, 2, figsize = (20, 10))

for col, ax in zip(cols, axes.flatten()[:2]):
    sns.regplot(x = col, y = 'RS', data = mlr_df, scatter_kws = {'color': 'navy'},
                line_kws = {'color': 'red'}, ax = ax)
    ax.set_title('Correlation between {} and RS'.format(col), fontsize = 18)
    ax.set_xlabel(col, fontsize = 14)
    ax.set_ylabel('RS', fontsize = 14)

plt.show()

# homoscedasticity
# residual plot
model = linear_model.LinearRegression().fit(x, y)
fitted_y = model.predict(x)
resid = fitted_y - y

fig = plt.subplots(figsize = (12, 8))
sns.residplot(fitted_y, "RS", data = mlr_df, lowess = True,
              scatter_kws = {"alpha": 0.5}, line_kws = {"color": "red", "lw": 1})
plt.xlabel("Fitted values", fontsize = 14)
plt.ylabel("Residuals", fontsize = 14)
plt.title("Residuals vs Fitted", fontsize = 20)
plt.show()

# normality
# QQ plot
fig = plt.subplots(figsize = (8, 8))
stats.probplot(resid, dist = "norm", plot = plt)
plt.title("Multiple Linear Regression QQ Plot")
plt.show()

# independence
# given that data observations are independent of each other,
# the independence assumption is satisfied

# multiple linear regression results
print('------- Multiple Linear Regression -------')
print("Intercept: {}".format(mlr.intercept_))

print("Coefficients: {}".format(mlr.coef_))

mse = metrics.mean_squared_error(y_test, y_predict)
print("RMSE: {}".format(sqrt(mse)))

print("R-squared: {}".format(metrics.r2_score(y_test, y_predict)))



# 5. Simple Linear Regression
# univariate feature selection
num_df = batting_df.select_dtypes(exclude = "category")
x = num_df.loc[:, num_df.columns != 'RS']
y = num_df['RS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

selector = SelectKBest(score_func = f_regression, k = 1)
selected_xTrain = selector.fit_transform(x_train, y_train)
selected_xTest = selector.transform(x_test)

all_cols = x.columns
selected_mask = selector.get_support()
selected_var = all_cols[selected_mask].values

print('Simple Linear Regression Selected Feature: {}'.format(selected_var))

# simple linear regression (x:'OPS' / y:'RS')
x = np.array(batting_df[selected_var]).reshape(-1, 1)
y = batting_df['RS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

lm = linear_model.LinearRegression().fit(x_train, y_train)

y_predicted = lm.predict(x_test)

print('------- Simple Linear Regression -------')
print("Intercept: {}".format(lm.intercept_))

print("Coefficients: {}".format(lm.coef_))

mse = metrics.mean_squared_error(y_test, y_predicted)
print("RMSE: {}".format(sqrt(mse)))

print("R-squared: {}".format(metrics.r2_score(y_test, y_predicted)))



# 6. Model Validation
# 10-Fold Cross-validation for the multiple linear regression model
model = LinearRegression()
x = filtered_df[['OBP', 'ISO']]
y = filtered_df['RS']

cv_r2 = cross_val_score(model, x, y, scoring = 'r2', cv = 10)
cv_mse = cross_val_score(model, x, y, scoring = 'neg_mean_squared_error', cv = 10)
cv_rmse = np.sqrt(-1 * cv_mse)

print('------- Multiple Linear Regression Validation -------')
print('Mean R-squared: {}'.format(cv_r2.mean()))
print('Mean RMSE: {}'.format(cv_rmse.mean()))

# 10-Fold Cross-validation for the simple linear regression model
model = LinearRegression()
x = np.array(batting_df['OPS']).reshape(-1, 1)
y = batting_df['RS']

cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
cv_rmse = np.sqrt(-1 * cv_mse)

print('------- Simple Linear Regression Validation -------')
print('Mean R-squared: {}'.format(cv_r2.mean()))
print('Mean RMSE: {}'.format(cv_rmse.mean()))



# 7. Cross-era comparison
# compare how league median "OBP", "ISO", and "OPS" changed over time
median_metrics = season_df[["OBP", "ISO", "OPS"]].median().reset_index()
melted_df = median_metrics.melt("Season", var_name = "Metrics", value_name = "League Median")

fig, ax = plt.subplots(figsize = (12, 8))

sns.lineplot(x = "Season", y = "League Median", hue = "Metrics", data = melted_df,
             palette = ["royalblue", "lightskyblue", "midnightblue"])
plt.xticks(melted_df["Season"], rotation = 45)
plt.title("Yearly Changes in League Median Metrics", fontsize = 18)
plt.legend(bbox_to_anchor= (0.86, -0.10), loc = "upper center", ncol = 3)
plt.axvline(x = 2006, color='red', linestyle = "--")
plt.axvline(x = 2014, color='red', linestyle = "--")
plt.grid()
fig.subplots_adjust(bottom = 0.15)
plt.show()

# get different eras
eras = batting_df["Era"].unique()
leagues = batting_df["League"].unique()

# start with data of which some features have already been filtered
# based on pairwise correlations between independent variables above
# for each era, find the best two features to build a multiple linear regression model
for era in eras:
    data = filtered_df.loc[filtered_df["Era"] == era]
    num_df = data.select_dtypes(exclude = "category")
    x = num_df.loc[:, num_df.columns != 'RS']
    y = num_df['RS']
    cols = list(x.columns)

    model = LinearRegression()

    rfe = RFE(estimator = model, n_features_to_select = 2)
    x_rfe = rfe.fit_transform(x, y)
    model.fit(x_rfe, y)

    temp = pd.Series(rfe.support_, index = cols)
    selected_vars = list(temp[temp == True].index)
    print('{} RFE Features: {}'.format(era, selected_vars))

    x = num_df[selected_vars]
    x = sm.add_constant(x)
    y = num_df['RS']

    lm = sm.OLS(y, x)
    result_rs = lm.fit()
    print("{} OLS Summary".format(era))
    print(result_rs.summary())

    vif = pd.DataFrame()
    vif['Feature'] = lm.exog_names
    vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
    print("{} VIF".format(era))
    print(vif[vif['Feature'] != 'const'])