import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pingouin as pg
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import statsmodels.api as sm
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

# create league data: National League (NL) / American League (AL)
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
print(batting_df.memory_usage(deep = True))

# to save memory usage, change "League" and "Team" data type
batting_df["Team"] = batting_df["Team"].astype("category")
batting_df["League"] = batting_df["League"].astype("category")

print(batting_df.memory_usage(deep = True))

# reorder data columns
cols = ["Season", "League"] + list(batting_df.columns)[1:-1]
batting_df = batting_df.reindex(columns = cols)


# 2. EDA (Exploratory Data Analysis)
# 2-1. RS Analysis: How did the league average runs scored change over time?
print("------- Runs Scored Data Descriptive Summary (Seasonal) -------")
print(batting_df["RS"].describe())

season_df = batting_df.groupby("Season")
lg_avg_rs = season_df["RS"].mean().round(1).reset_index()
print("------- Changes in League Average Runs Scored -------")
print(lg_avg_rs)

# bar plot
values = np.array(lg_avg_rs["RS"])
idx = np.array(lg_avg_rs["Season"])
colors = ["navy" if (x < 730) else "red" for x in values]
red_bar = mpatches.Patch(color = 'red', label = "RS >= 730")
navy_bar = mpatches.Patch(color = 'navy', label = "RS < 730")

fig, ax = plt.subplots(figsize = (10, 7))

plt.bar(idx, values, color = colors, zorder = 3)
plt.xticks(lg_avg_rs["Season"], rotation = 45)
plt.xlabel("Season")
plt.ylabel("League Average Runs Scored")
plt.title("Changes in League Average Runs Scored", fontsize = 14, fontweight = "bold")
plt.legend(handles = [red_bar, navy_bar], loc = "upper right")
plt.grid(zorder = 0)
plt.show()

# box plot
fig, ax = plt.subplots(figsize = (10, 9))

sns.boxplot(x = "Season", y = "RS", data = batting_df, ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
ax.set(title = "Yearly Changes in Team Runs Scored")
plt.show()

# 2-2. RS Analysis: In which League (NL vs AL) did teams scored runs more?: Two-sample t-test
# group data based on "League" data values
lg_df = batting_df.groupby("League")
print("------- Runs Scored Data Descriptive Summary (League) -------")
print(lg_df["RS"].describe())

nl_rs = batting_df.loc[batting_df["League"] == "NL"]["RS"]
al_rs = batting_df.loc[batting_df["League"] == "AL"]["RS"]

# check two-sample t-test assumptions
# check normality
fig, axes = plt.subplots(1, 2, figsize = (20, 8))
sns.histplot(nl_rs, kde = True, ax = axes[0], color = "blue")
sns.histplot(al_rs, kde = True, ax = axes[1], color = "red")
axes[0].set_title('National League Runs Scored Histogram')
axes[1].set_title('American League Runs Scored Histogram')
plt.show()

# since the data is not normally distributed due to 2020 short-season data (low RS = outliers),
# remove 2020 season data
batting_df = batting_df.loc[batting_df["Season"] != 2020]

nl_rs = batting_df.loc[batting_df["League"] == "NL"]["RS"]
al_rs = batting_df.loc[batting_df["League"] == "AL"]["RS"]

fig, axes = plt.subplots(1, 2, figsize = (20, 8))
stats.probplot(nl_rs, plot = axes[0])
stats.probplot(al_rs, plot = axes[1])
axes[0].set_title("National League Runs Scored Q-Q Plot without 2020 Season")
axes[1].set_title("American League Runs Scored Q-Q Plot without 2020 Season")
plt.show()

# check an equal-variance assumption
print("NL Runs Scored Variance: {}".format(nl_rs.var()))
print("AL Runs Scored Variance: {}".format(al_rs.var()))

# box plot
fig, ax = plt.subplots(figsize = (8, 8))
sns.boxplot(x = "League", y = "RS", data = batting_df, palette = "Set1")
ax.set(title = "Runs Scored Distribution by League")
plt.show()

# Welch's two-sample t-test
test_result = pg.ttest(al_rs, nl_rs, paired = False, alternative='greater', correction = True).round(3)
print(test_result.to_string())

# # 'RS' histogram and Q-Q plot
# fig, axes = plt.subplots(1, 2, figsize=(20, 8))
#
# sns.histplot(batting_df['RS'], kde = True, ax = axes[0])
# axes[0].set_title('Team RS Histogram')
#
# axes[1] = stats.probplot(batting_df['RS'], plot=plt)
# plt.title('Team RS Q-Q Plot')
#
# plt.show()
#
# print('------- Team RS Data Distribution -------')
# print('Mean RS: {}'.format(batting_df['RS'].mean()))
# print('Median RS: {}'.format(batting_df['RS'].median()))
# print('RS Standard Deviation: {}'.format(batting_df['RS'].std()))
# print('RS Skewness: {}'.format(batting_df['RS'].skew()))
# print('RS Kurtosis: {}'.format(batting_df['RS'].kurt()))
#
# # yearly changes in RS
# yearly_rs = pd.concat([batting_df['YEAR'], batting_df['RS']], axis=1)
# fig, ax = plt.subplots(figsize=(10, 10))
#
# sns.boxplot(x='YEAR', y='RS', data=yearly_rs, ax=ax)
# ax.set(title='Yearly Changes in Team Runs Scored')
#
# plt.show()
#
# # correlation matrix
# corrMatrix = batting_df.corr()
# fig, ax = plt.subplots(figsize=(8, 8))
#
# sns.heatmap(corrMatrix, square=True)
# plt.title('Correlation Matrix')
#
# plt.show()
#
# # multicollinearity detection
# # 1. drop independent variables if its correlation between other independent variables is higher than 0.95
# no_target = batting_df.iloc[:, batting_df.columns != 'RS']
#
# corrMatrix = abs(no_target.corr())
# upperTri = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))
# vars_drop = [col for col in upperTri.columns if any(upperTri[col] > 0.95)]
#
# df = batting_df.drop(vars_drop, axis=1)
#
# # 2. drop variables that have lower correlations than 0.60 with 'RS'
# corrMatrix = abs(df.corr())
# cols = list(corrMatrix.columns)
#
# for col in cols:
#     if corrMatrix[col]['RS'] < 0.65:
#         vars_drop = col
#         df.drop(col, axis=1, inplace=True)
#
# filtered_vars = list(df.columns)
# print('Filtered Variables: {}'.format(filtered_vars))
#
# df = df[filtered_vars]
#
# # new correlation matrix for filtered data features
# fig, ax = plt.subplots(figsize=(10, 10))
#
# corrMatrix = df.corr()
# sns.heatmap(corrMatrix, square=True, annot=True, annot_kws={'size':10},
#             xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns)
# plt.title('Correlation Matrix')
#
# plt.show()
#
# # independent variables EDA
# cols = list(df.drop('RS', axis=1).columns)
# # histograms
# fig, axes = plt.subplots(3, 3, figsize=(15, 15))
#
# for col, ax in zip(cols, axes.flatten()[:8]):
#     sns.histplot(df[col], kde=True, color='red', ax=ax)
#     ax.set_title('Team {} Histogram'.format(col))
#
# plt.show()
#
# # Q-Q plots
# fig, axes = plt.subplots(3, 3, figsize=(17, 17))
#
# for col, ax in zip(cols, axes.flatten()[:8]):
#     stats.probplot(df[col], plot=ax)
#     ax.set_title('{} Q-Q Plot'.format(col))
#
# plt.show()
#
# # scatter plots
# fig, axes = plt.subplots(3, 3, figsize=(15, 15))
#
# for col, ax in zip(cols, axes.flatten()[:8]):
#     sns.regplot(x=col, y='RS', data=df, scatter_kws={'color': 'navy'},
#                 line_kws={'color': 'red'}, ax=ax)
#     ax.set_title('Correlation between {} and RS'.format(col))
#     ax.set_xlabel(col)
#     ax.set_ylabel('RS')
#
# plt.show()
#
#
#
# ### 3. Feature Scaling ###
# # check independent data ranges
# print(df.describe().to_string())
#
# # StandardScaler
# scaled_df = df.drop(['RS'], axis=1)
# cols = list(scaled_df.columns)
#
# std_scaler = StandardScaler()
# scaled_data = std_scaler.fit_transform(scaled_df)
# scaled_df = pd.DataFrame(scaled_data, columns=cols)
#
# # KDE plot after scaling
# scaled_cols = list(scaled_df.columns)
# fig, ax = plt.subplots(figsize=(10, 10))
#
# for col in scaled_cols:
#     sns.kdeplot(scaled_df[col], label=col, ax=ax)
#     ax.set_title('After StandardScaler')
#     ax.set_xlabel('Data Scale')
#     ax.legend(loc=1)
#
# plt.show()
#
#
#
# ### 4. Multiple Linear Regression with feature selection ###
# # Recursive Feature Elimination
# df = pd.concat([df['RS'], scaled_df], axis=1)
#
# x = df.loc[:, df.columns != 'RS']
# y = df['RS']
# cols = list(x.columns)
#
# lm = LinearRegression()
#
# rfe = RFE(lm, 2)
# x_rfe = rfe.fit_transform(x, y)
# lm.fit(x_rfe, y)
#
# temp = pd.Series(rfe.support_, index=cols)
# selected_vars = list(temp[temp == True].index)
# print('Selected Features: {}'.format(selected_vars))
#
# # check VIF
# x = df[selected_vars]
# x = sm.add_constant(x)
# y = df['RS']
#
# lm = sm.OLS(y, x)
# result_rs = lm.fit()
# print(result_rs.summary())
#
# vif = pd.DataFrame()
# vif['Feature'] = lm.exog_names
# vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
# print(vif[vif['Feature'] != 'const'])
#
#
# # split data into training and test data and build a multiple linear regression model
# # multiple linear regression (x:'TB', 'OBP' / y:'RS')
# x = df[selected_vars]
# y = df['RS']
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#
# lm = linear_model.LinearRegression().fit(x_train, y_train)
#
# y_predict = lm.predict(x_test)
#
# print('------- Multiple Linear Regression -------')
# print('------- Intercept -------')
# print(lm.intercept_)
#
# print('------- Coefficient -------')
# print(lm.coef_)
#
# print('------- RMSE -------')
# mse = metrics.mean_squared_error(y_test, y_predict)
# print(sqrt(mse))
#
# print('------- R-squared -------')
# print(metrics.r2_score(y_test, y_predict))
#
#
#
# ### 5. Simple Linear Regression ###
# # univariate feature selection
# x = batting_df.loc[:, batting_df.columns != 'RS']
# y = batting_df['RS']
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#
# selector = SelectKBest(score_func=f_regression, k=1)
# selected_xTrain = selector.fit_transform(x_train, y_train)
# selected_xTest = selector.transform(x_test)
#
# all_cols = x.columns
# selected_mask = selector.get_support()
# selected_var = all_cols[selected_mask].values
#
# print('Selected Feature: {}'.format(selected_var))
#
# # simple linear regression (x:'OPS' / y:'RS')
# x = np.array(batting_df[selected_var]).reshape(-1, 1)
# y = batting_df['RS']
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#
# lm = linear_model.LinearRegression().fit(x_train, y_train)
#
# y_predicted = lm.predict(x_test)
#
# print('------- Simple Linear Regression -------')
# print('------- Intercept -------')
# print(lm.intercept_)
#
# print('------- Coefficient -------')
# print(lm.coef_)
#
# print('------- RMSE -------')
# mse = metrics.mean_squared_error(y_test, y_predicted)
# print(sqrt(mse))
#
# print('------- R-squared -------')
# print(metrics.r2_score(y_test, y_predicted))
#
#
#
# ### 6. Model Validation ###
#
# # 10-Fold Cross Validation for the multiple linear regression model
# model = LinearRegression()
# x = df[['TB', 'OBP']]
# y = df['RS']
#
# cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
# cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
# cv_rmse = np.sqrt(-1 * cv_mse)
#
# print('------- Multiple Linear Regression Validation -------')
# print('Mean R-squared: {}'.format(cv_r2.mean()))
# print('Mean RMSE: {}'.format(cv_rmse.mean()))
#
# # 10-Fold Cross Validation for the simple linear regression model
# model = LinearRegression()
# x = np.array(batting_df['OPS']).reshape(-1, 1)
# y = batting_df['RS']
#
# cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
# cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
# cv_rmse = np.sqrt(-1 * cv_mse)
#
# print('------- Simple Linear Regression Validation -------')
# print('Mean R-squared: {}'.format(cv_r2.mean()))
# print('Mean RMSE: {}'.format(cv_rmse.mean()))