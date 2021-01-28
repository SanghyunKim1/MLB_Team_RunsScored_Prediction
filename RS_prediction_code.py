import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
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

# drop categorical variables
batting_df = batting_df.select_dtypes(exclude='object')


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
corrMatrix = batting_df.corr()
fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(corrMatrix, square=True)
plt.title('Correlation Matrix')

plt.show()

# multicollinearity detection
# 1. if correlations between independent variables are higher than 0.9 drop that variables
no_target = batting_df.iloc[:, batting_df.columns != 'RS']

corrMatrix = abs(no_target.corr())
upperTri = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))
vars_drop = [col for col in upperTri.columns if any(upperTri[col] > 0.95)]

df = batting_df.drop(vars_drop, axis=1)

# 2. drop variables that have lower correlations than 0.60 with 'RS'
corrMatrix = abs(df.corr())
cols = list(corrMatrix.columns)

for col in cols:
    if corrMatrix[col]['RS'] < 0.65:
        vars_drop = col
        df.drop(col, axis=1, inplace=True)

filtered_vars = list(df.columns)
print('Filtered Variables: {}'.format(filtered_vars))

df = batting_df[filtered_vars]

# new correlation matrix for filtered data features
fig, ax = plt.subplots(figsize=(10, 10))

corrMatrix = df.corr()
sns.heatmap(corrMatrix, square=True, annot=True, annot_kws={'size':10},
            xticklabels=corrMatrix.columns, yticklabels=corrMatrix.columns)
plt.title('Correlation Matrix')

plt.show()

# independent variables EDA
cols = list(df.drop('RS', axis=1).columns)
# histograms
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for col, ax in zip(cols, axes.flatten()[:8]):
    sns.histplot(df[col], kde=True, color='red', ax=ax)
    ax.set_title('Team {} Histogram'.format(col))

plt.show()

# Q-Q plots
fig, axes = plt.subplots(3, 3, figsize=(17, 17))

for col, ax in zip(cols, axes.flatten()[:8]):
    stats.probplot(df[col], plot=ax)
    ax.set_title('{} Q-Q Plot'.format(col))

plt.show()

# scatter plots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for col, ax in zip(cols, axes.flatten()[:8]):
    sns.regplot(x=col, y='RS', data=df, scatter_kws={'color': 'navy'},
                line_kws={'color': 'red'}, ax=ax)
    ax.set_title('Correlation between {} and RS'.format(col))
    ax.set_xlabel(col)
    ax.set_ylabel('RS')

plt.show()



### 3. Feature Scaling ###
# check independent data ranges
print(df.describe().to_string())

# StandardScaler
scaled_df = df.drop(['RS'], axis=1)
cols = list(scaled_df.columns)

std_scaler = StandardScaler()
scaled_data = std_scaler.fit_transform(scaled_df)
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



### 4. Multiple Linear Regression with feature selection ###
# Recursive Feature Elimination
df = pd.concat([df['RS'], scaled_df], axis=1)

x = df.loc[:, df.columns != 'RS']
y = df['RS']
cols = list(x.columns)

lm = LinearRegression()

rfe = RFE(lm, 2)
x_rfe = rfe.fit_transform(x, y)
lm.fit(x_rfe, y)

temp = pd.Series(rfe.support_,index = cols)
selected_vars = list(temp[temp==True].index)
print('Selected Features: {}'.format(selected_vars))

# check VIF
x = df[selected_vars]
x = sm.add_constant(x)
y = df['RS']

lm = sm.OLS(y, x)
result_rs = lm.fit()
print(result_rs.summary())

vif = pd.DataFrame()
vif['Feature'] = lm.exog_names
vif['VIF'] = [variance_inflation_factor(lm.exog, i) for i in range(lm.exog.shape[1])]
print(vif[vif['Feature'] != 'const'])


# split data into training and test data and build a multiple linear regression model
# multiple linear regression (x:'TB', 'OBP' / y:'RS')
x = df[selected_vars]
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
# univariate feature selection
x = batting_df.loc[:, batting_df.columns != 'RS']
y = batting_df['RS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

selector = SelectKBest(score_func=f_regression, k=1)
selected_x_train = selector.fit_transform(x_train, y_train)
selected_x_test = selector.transform(x_test)

all_cols = x.columns
selected_mask = selector.get_support()
selected_var = all_cols[selected_mask]

print('Selected Feature: {}'.format(selected_var.values))

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
x = df[['TB', 'OBP']]
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