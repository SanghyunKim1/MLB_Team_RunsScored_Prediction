import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats
from math import sqrt
from functools import reduce
import warnings
warnings.filterwarnings('ignore')
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

### Purpose of Work: predicting a team's runs scored based on various batting statistics

# load datasets
batting_2019 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/batting_2019.csv')
batting_2018 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/batting_2018.csv')
batting_2017 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/batting_2017.csv')
batting_2016 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/batting_2016.csv')
batting_2015 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/batting_2015.csv')
batting_2014 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/batting_2014.csv')
batting_2013 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/batting_2013.csv')
batting_2012 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/batting_2012.csv')
batting_2011 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/batting_2011.csv')
batting_2010 = pd.read_csv('/Users/sanghyunkim/Desktop/Data Science Project/MLB Analysis/MLB_Team_RunsScored_Prediction/data/batting_2010.csv')

# merge datasets
batting_dfs = [batting_2019, batting_2018, batting_2017, batting_2016, batting_2015,
               batting_2014, batting_2013, batting_2012, batting_2011, batting_2010]
batting_df = reduce(lambda x, y: pd.merge(x, y, how='outer'), batting_dfs)
print(batting_df.head().to_string())


### 1. Data Cleaning ###
# drop unnecessary columns
batting_df.drop(['#'], axis=1, inplace=True)

# rename specific column names
batting_df.rename(columns={'R': 'RS'}, inplace=True)

# check missing values
print("Total Number of Missing Values in Batting Data:")
print(batting_df.isnull().sum())

# check duplicates
print("Total Number of Duplicates in Batting Data: {}".format(batting_df.duplicated().sum()))

# check data types
print(batting_df.dtypes)

# categorical data
obj_cols = list(batting_df.select_dtypes(include='object').columns)
print(batting_df[obj_cols].head())

# eliminate commas
comma_cols = ['PA', 'AB', 'H', 'SO']
batting_df[comma_cols] = batting_df[comma_cols].replace(',', '', regex=True)

# change data types
batting_df[comma_cols] = batting_df[comma_cols].apply(pd.to_numeric)

# check new data types
print(batting_df.dtypes)



### 2. EDA (Exploratory Data Analysis) ###

# dependent variable, 'RS' EDA
print("------- Batting Data Descriptive Summary -------")
print(batting_df.describe().to_string())

# 'RS' histogram and Q-Q plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.histplot(batting_df['RS'], kde=True, ax=axes[0])
axes[0].set_title('Team RS Histogram')

axes[1] = stats.probplot(batting_df['RS'], plot=plt)
plt.title('Team RS Q-Q Plot')

plt.show()

print('------- Team RS Data Distribution -------')
print('Mean RS: {}'.format(batting_df['RS'].mean()))
print('Median RS: {}'.format(batting_df['RS'].median()))
print('RS Standard Deviation: {}'.format(batting_df['RS'].std()))
print('RS Skewness: {}'.format(batting_df['RS'].skew()))
print('RS Kurtosis: {}'.format(batting_df['RS'].kurt()))

# yearly changes in RS
yearly_rs = pd.concat([batting_df['YEAR'], batting_df['RS']], axis=1)
fig, ax = plt.subplots(figsize=(10, 10))

sns.boxplot(x='YEAR', y='RS', data=yearly_rs, ax=ax)
ax.set(title='Yearly Changes in Team Runs Scored')

plt.show()

# correlation matrix
corr = batting_df.corr()
fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(corr, square=True)
plt.title('Correlation Matrix')

plt.show()

# drop variables that have lower correlations with 'RS' than 0.65
corr = abs(batting_df.corr())
corr_df = corr['RS'].to_frame(name='Correlation with RS').T
corr_cols = corr_df.columns

corr_df.drop(columns=corr_cols[(corr_df < 0.65).any()], inplace=True)
print(corr_df.to_string())

cols = list(corr_df.columns)

batting_df = batting_df[cols]
print(batting_df.head())

# new correlation matrix for selected data features
fig, ax = plt.subplots(figsize=(10, 10))

corr = batting_df.corr()
sns.heatmap(corr, square=True, annot=True, annot_kws={'size':10},
            xticklabels=corr.columns, yticklabels=corr.columns)
plt.title('Correlation Matrix')

plt.show()


# independent variables EDA
# histograms
cols = list(batting_df.drop(['RS'], axis=1).columns)

fig, axes = plt.subplots(4, 3, figsize=(20, 24))

for col, ax in zip(cols, axes.flatten()[:11]):
    sns.histplot(batting_df[col], kde=True, color='red', ax=ax)
    ax.set_title('Team {} Histogram'.format(col))

plt.show()

# Q-Q plots
fig, axes = plt.subplots(4, 3, figsize=(21, 21))

for col, ax in zip(cols, axes.flatten()[:11]):
    stats.probplot(batting_df[col], plot=ax)
    ax.set_title('{} Q-Q Plot'.format(col))

plt.show()

# scatter plots
fig, axes = plt.subplots(4, 3, figsize=(20, 20))

for col, ax in zip(cols, axes.flatten()[:11]):
    sns.regplot(x=col, y='RS', data=batting_df, scatter_kws={'color': 'navy'},
                line_kws={'color': 'red'}, ax=ax)
    ax.set_title('Correlation between {} and RS'.format(col))
    ax.set_xlabel(col)
    ax.set_ylabel('RS')

plt.show()



### 3. Feature Scaling ###
# check independent data ranges
print(batting_df.describe().to_string())

# StandardScaler
df = batting_df.drop(['RS'], axis=1)
cols = list(df.columns)

std_scaler = StandardScaler()
scaled_data = std_scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=cols)

# KDE plot after scaling
scaled_cols = list(scaled_df.columns)
fig, ax = plt.subplots(figsize=(10, 10))

for col in scaled_cols:
    sns.kdeplot(scaled_df[col], label=col, ax=ax)
    ax.set_title('After StandardScaler')
    ax.set_xlabel('Data Scale')
    ax.legend(loc=1)

plt.show()



### 4. Multiple Linear Regression with feature selection

# check multicollinearity
df = pd.concat([batting_df['RS'], scaled_df], axis=1)
x = df.drop(['RS'], axis=1)
x = sm.add_constant(x)
y = df['RS']

lm_rs = sm.OLS(y, x)
result_rs = lm_rs.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm_rs.exog_names
vif['VIF'] = [variance_inflation_factor(lm_rs.exog, i) for i in range(lm_rs.exog.shape[1])]
print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# exclude 'OPS'
x = df.drop(['RS', 'OPS'], axis=1)
x = sm.add_constant(x)
y = df['RS']

lm_rs = sm.OLS(y, x)
result_rs = lm_rs.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm_rs.exog_names
vif['VIF'] = [variance_inflation_factor(lm_rs.exog, i) for i in range(lm_rs.exog.shape[1])]
print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# exclude 'HRr', 'PA', 'TB', and 'SLG'
x = df.drop(['RS', 'OPS', 'HRr', 'PA', 'TB', 'SLG'], axis=1)
x = sm.add_constant(x)
y = df['RS']

lm_rs = sm.OLS(y, x)
result_rs = lm_rs.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm_rs.exog_names
vif['VIF'] = [variance_inflation_factor(lm_rs.exog, i) for i in range(lm_rs.exog.shape[1])]
print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# exclude 'HR', 'DRAA', 'BWARP'
x = df.drop(['RS', 'OPS', 'HRr', 'PA', 'TB', 'SLG', 'HR', 'DRAA', 'BWARP'], axis=1)
x = sm.add_constant(x)
y = df['RS']

lm_rs = sm.OLS(y, x)
result_rs = lm_rs.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm_rs.exog_names
vif['VIF'] = [variance_inflation_factor(lm_rs.exog, i) for i in range(lm_rs.exog.shape[1])]
print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# include 'OPS' again
x = df.drop(['RS', 'HRr', 'PA', 'TB', 'SLG', 'HR', 'DRAA', 'BWARP'], axis=1)
x = sm.add_constant(x)
y = df['RS']

lm_rs = sm.OLS(y, x)
result_rs = lm_rs.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm_rs.exog_names
vif['VIF'] = [variance_inflation_factor(lm_rs.exog, i) for i in range(lm_rs.exog.shape[1])]
print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# exclude 'OBP'
x = df.drop(['RS', 'HRr', 'PA', 'TB', 'SLG', 'HR', 'DRAA', 'BWARP', 'OBP'], axis=1)
x = sm.add_constant(x)
y = df['RS']

lm_rs = sm.OLS(y, x)
result_rs = lm_rs.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm_rs.exog_names
vif['VIF'] = [variance_inflation_factor(lm_rs.exog, i) for i in range(lm_rs.exog.shape[1])]
print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# exclude 'OPS' and include 'OBP', TB' again
x = df.drop(['RS', 'HRr', 'PA', 'SLG', 'HR', 'DRAA', 'BWARP', 'OPS'], axis=1)
x = sm.add_constant(x)
y = df['RS']

lm_rs = sm.OLS(y, x)
result_rs = lm_rs.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm_rs.exog_names
vif['VIF'] = [variance_inflation_factor(lm_rs.exog, i) for i in range(lm_rs.exog.shape[1])]
print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# exclude 'TB'
x = df.drop(['RS', 'HRr', 'PA', 'SLG', 'HR', 'DRAA', 'BWARP', 'OPS', 'TB'], axis=1)
x = sm.add_constant(x)
y = df['RS']

lm_rs = sm.OLS(y, x)
result_rs = lm_rs.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm_rs.exog_names
vif['VIF'] = [variance_inflation_factor(lm_rs.exog, i) for i in range(lm_rs.exog.shape[1])]
print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))

# exclude 'DRC+'
x = df.drop(['RS', 'HRr', 'PA', 'SLG', 'HR', 'DRAA', 'BWARP', 'OPS', 'TB', 'DRC+'], axis=1)
x = sm.add_constant(x)
y = df['RS']

lm_rs = sm.OLS(y, x)
result_rs = lm_rs.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm_rs.exog_names
vif['VIF'] = [variance_inflation_factor(lm_rs.exog, i) for i in range(lm_rs.exog.shape[1])]
print(vif[vif['Feature'] != 'const'].sort_values('VIF', ascending=False))
# conclusion: the best multiple liear regression model is when independent variables are 'OBP and 'ISO'

# split data into training and test data and build a multiple linear regression model
# multiple linear regression (x:'OBP', 'ISO' / y:'RS')
x = df[['OBP', 'ISO']]
y = df['RS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

lm = linear_model.LinearRegression().fit(x_train, y_train)

y_predict = lm.predict(x_test)

print('------- Multiple Linear Regression -------')
print('------- Intercept -------')
print(lm.intercept_)

print('------- Coefficient -------')
print(lm.coef_)

print('------- RMSE -------')
mse = metrics.mean_squared_error(y_test, y_predict)
print(sqrt(mse))

print('------- R-squared -------')
print(metrics.r2_score(y_test, y_predict))



### 5. Simple Linear Regression ###

# correlation
print('------- Correlations with RS -------')
print(batting_df.corr().to_string())

# select 'OPS', which has the highest correlation with 'RS' (0.950), as an independent variable
# simple linear regression (x:'OPS' / y:'RS')
x = np.array(batting_df['OPS']).reshape(-1, 1)
y = batting_df['RS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

lm_rs2 = linear_model.LinearRegression().fit(x_train, y_train)

y_predicted = lm_rs2.predict(x_test)

print('------- Simple Linear Regression -------')
print('------- Intercept -------')
print(lm_rs2.intercept_)

print('------- Coefficient -------')
print(lm_rs2.coef_)

print('------- RMSE -------')
mse = metrics.mean_squared_error(y_test, y_predicted)
print(sqrt(mse))

print('------- R-squared -------')
print(metrics.r2_score(y_test, y_predicted))



### 6. Model Validation ###

# 10-Fold Cross Validation for the multiple linear regression model
model = LinearRegression()
x = df[['OBP', 'ISO']]
y = df['RS']

cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
cv_rmse = np.sqrt(-1 * cv_mse)

print('------- Multiple Linear Regression Validation -------')
print('Mean R-squared: {}'.format(cv_r2.mean()))
print('Mean RMSE: {}'.format(cv_rmse.mean()))

# 10-Fold Cross Validation for the simple linear regression model
model = LinearRegression()
x = np.array(batting_df['OPS']).reshape(-1, 1)
y = batting_df['RS']

cv_r2 = cross_val_score(model, x, y, scoring='r2', cv=10)
cv_mse = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
cv_rmse = np.sqrt(-1 * cv_mse)

print('------- Simple Linear Regression Validation -------')
print('Mean R-squared: {}'.format(cv_r2.mean()))
print('Mean RMSE: {}'.format(cv_rmse.mean()))