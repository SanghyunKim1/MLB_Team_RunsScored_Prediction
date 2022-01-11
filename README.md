# MLB Team Runs Scored Prediction

## Table of Contents
1. [Motivation](#motivation)
2. [Metadata](#metadata)
3. [Data Cleaning](#datacleaning)
4. EDA (Exploratory Data Analysis)
5. Feature Scaling
7. Multiple Linear Regression with Feature Selection
8. Simple Linear Regression
9. Model Validation
10. Conclusion

## 🤔 Motivation <a name="motivation"></a>
Before we dive into our analysis, let's briefly talk about the nature of baseball.

Say you are an owner of a baseball team, then why do you think you are running your team spending billions of dollars per year? 
Making money, making fans enthusiastic, having fans in ballparks etc... But the ulitmate goal of running a baseball club, as with other sports teams, will be winning. As the owner of your team, your goal should be winning, and thus, you should try to buy wins not just players. This is where **sabermetrics** ([SABR](https://sabr.org) + metrics) has originated. Okay, interesting.

Now we understand that we should focus on wins, then HOW do we win? (i.e. what makes your team win?) As in other sports, a baseball team should score runs and prevent opponents from scoring to win a game at the same time.

Alright, we're almost there. Your goal is to win, and therefore, your team must outscore your opponents to do so.
That's the nature of baseball: **Runs Scored** and **Runs Allowed**. </br>
In this project, we'll focus on the first part: **Runs Scored**.


## 📂 Metadata <a name="metadata"></a>
| **Metadata** | **Information** |
| :-----------: | :-----------: |
| **Origin of Data** | [FanGraphs.com](https://www.fangraphs.com) |
| **Terms of Service** | [Terms of Service](https://www.fangraphs.com/about/terms-of-service) |
| **Data Structure** | 22 datasets (**CSV** files) each of which consists of 30 rows * 23 columns |

| **Data Feature** | **Description** |
| :-----------: | :-----------: |
| **Season** | Each year refers to the corresponding season |
| **TEAM** | Team names |
| **G** | Total number of games played by all batters for each team |
| **PA** | [Plate Appearance](http://m.mlb.com/glossary/standard-stats/plate-appearance) |
| **AB** | [At-bat](http://m.mlb.com/glossary/standard-stats/at-bat) |
| **R** | [Runs Scored](http://m.mlb.com/glossary/standard-stats/run) |
| **H** | [Hit](http://m.mlb.com/glossary/standard-stats/hit) |
| **AVG** | [Batting Average](http://m.mlb.com/glossary/standard-stats/batting-average) |
| **OBP** | [On-base Percentage](http://m.mlb.com/glossary/standard-stats/on-base-percentage) |
| **SLG** | [Slugging Percentage](http://m.mlb.com/glossary/standard-stats/slugging-percentage) |
| **OPS** | [On-base Plus Slugging](http://m.mlb.com/glossary/standard-stats/on-base-plus-slugging) |
| **ISO** | [Isolated Power](http://m.mlb.com/glossary/advanced-stats/isolated-power) |
| **HR** | [Home run](http://m.mlb.com/glossary/standard-stats/home-run) |
| **BB** | [Walk](http://m.mlb.com/glossary/standard-stats/walk) |
| **SO** | [Strikeout](http://m.mlb.com/glossary/standard-stats/strikeout) |
| **SB** | [Stolen Base](http://m.mlb.com/glossary/standard-stats/stolen-base) |
| **CS** | [Caught Stealing](http://m.mlb.com/glossary/standard-stats/caught-stealing) |
| **wOBA** | [Weighted On-base Average](https://www.mlb.com/glossary/advanced-stats/weighted-on-base-average) |
| **wRC+** | [Weighted Runs Created Plus](https://www.mlb.com/glossary/advanced-stats/weighted-runs-created-plus) |
| **WAR** | [Wins Above Replacement](https://library.fangraphs.com/misc/war/) |

| **Newly Created Data** | **Description** |
| :-----------: | :-----------: |
| **League** | AL: American League / NL: National League |
| **TB** | [TB](https://www.mlb.com/glossary/standard-stats/total-bases) |

## 🧹 Data Cleaning <a name="datacleaning"></a>
- Combined 22 different datasets (2000-2021 Season Team Batting Data).
- Renamed **'R'** data feature as **'RS'** for clarity.
- Created new data features: **League** and **TB**
- Confirmed that there are no missing duplicated data.
- Converted **object** data types into **category** data types to save memory usage.
- Reordered data features.

## 📊 EDA (Exploratory Data Analysis)


## Feature Selection


## Multiple Linear Regression


## Simple Linear Regression


## Model Validation



## Cross-era Comparison



## Conclusion


***5-2. Feature Selection: Filter Method***

<img src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/Filtered%20Correlation%20Matrix.png" width="500" height="500">

| ***Correlation*** | **RS** | **PA** | **TB** | **OBP** | **ISO** | **DRC+** | **DRAA** | **BWARP** |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| **RS** | 1.0 | 0.739 | 0.922 | 0.829 | 0.812 | 0.751 | 0.806 | 0.780 |

Initially, I had 24 independent variables. To avoid multicollinearity, I filtered some of them based on (a) correlation between each **independent** variable, and (b) correlation between those filtered features and the **dependent** variable, **RS**. As a result, I ended up with **7** independent varaibles as indicated in the correlation matrix above. 


***5-3. Filtered Independent Variables EDA***

<img src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/Histogram.png" width="800" height="800">

According to the histograms of each independent variable above, all the variables are normally distributed.

<img src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/Scatter%20plot.png" width="600" height="600">

Scatter plots also depict that there are reasonable linear trends between each independent variable and **RS** without notable outliers, and thus, it's safe to use the linear regression model.


### 6. Feature Scaling
Since the ranges of independent variables vary considerably, I scaled all the independent variables. As all the data attributes have normal distributions with few outliers, I used ***StandardScaler*** to scale them.

The result of feature scaling is the following:

<img src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/KDE%20plot.png" width="600" height="600">


### 7. Multiple Linear Regression with Feature Selection
With all the independent variables filtered above, I built a multiple linear regression model to check the degree of multicollinearity based on **VIF**.

<img width="207" alt="VIF" src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/VIF1.png">

According to the table above, there seems to be multicollinearity in the model because independent variables are still highly correlated one another.
Therefore, I used the wrapper method (**Recursive Feature Elimination**) to find the best two independent variables.

Through **RFE**, I got **TB** and **OBP** as independent variables and built a multiple linear regression.
The result of the model is:

<img width="601" alt="Multiple Linear Regression" src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/Multiple%20Linear%20Regression.png"> <img width="193" alt="VIF2" src="https://github.com/shk204105/MLB_Team_RunsScored_Prediction/blob/master/images/VIF2.png">


### 8. Simple Linear Regression
Apart from the multiple linear regression model, I also built a simple linear regression model. To find the sinlge best independent variable, I used the **SelectKBest** function. Based on F-statistics of each independent variable, **OPS** has been selected as the best independent variable.

Furthermore, I also splitted data into training(70%) and test(30%) datasets to avoid overfitting.

The result of the model is:

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***Intercept*** | -752.3837394309697 |
| ***Coefficient*** | 2009.36777208 |
| ***R-squared*** | 0.9079557000049954 |
| ***RMSE*** | 23.207166436311425 |


### 9. Model Validation
<img src="https://user-images.githubusercontent.com/67542497/105632704-f1848a80-5e97-11eb-8b69-f19913f1d3be.png" width="500" height="400">

To validate both multiple and simple linear regression models, I used the K-Fold Cross Validation method, where the number of folds is 10.

***9-1. Multiple Linear Regression model validtion***

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***Mean R-squared*** | 0.8584381525775473 |
| ***Mean RMSE*** | 24.92574073069298 |

***9-2. Simple Linear Regression model validtion***

| **Measurement** | **Score** | 
| :-----------: | :-----------: |
| ***Mean R-squared*** | 0.8610824571143239 |
| ***Mean RMSE*** | 24.37742789357559 |

Accoring to the results above, the simple linear regression model (x:**OPS** / y:**RS**) showed a slightly higher R-squared than the multiple linear regression model (x:**TB**, **OBP** / y:**RS**).
However, the differences in the R-squared between those two models are marginal, and as both models don't overfit data, it's safe to use either model to predict team **RS**.


### 10. Conclusion

Comparing those two models through 10-Fold Cross Validation, although the simple linear regression seems more accurate, the differences between these two models seem marginal.

One possible reason for such a result is because these two predictors (**OPS** vs **TB + OBP**) measure similar things in baseball. For those who are not familiar with baseball, let me briefly talk about what these three stats measure in baeball.


Frist, [**TB**](http://m.mlb.com/glossary/standard-stats/total-bases) measures total number of bases gained through hits. It assigns **1 total base** to a single, **2 total bases** to a double, **3 total bases** to a triple and **4 total bases** to a home run. Therefore, a team's TB shows how many **singles** as well as **extra hits (doubles + triples + home runs)**, which possibly advance runners on base, have been gained throughout the season by that team.

Second, [**OBP**](http://m.mlb.com/glossary/standard-stats/on-base-percentage) (On-Base Percentage) measures how *often* a batter reaches bases (e.g an **OBP** of 0.400 means that this batter has reached bases four times in 10 plate appearances). It includes *Hits (singles + extra hits)*, *Base-on-Balls* and *Hit-by-Pitches*. While **TB** measures total number of bases gained, **OBP** measures the efficiency of a batter in terms of the ability to reach bases.

Finally, [**OPS**](http://m.mlb.com/glossary/standard-stats/on-base-plus-sluggin) is the sum of **OBP** and [**SLG**](http://m.mlb.com/glossary/standard-stats/slugging-percentage). **SLG** here refers to *Slugging Percentage*. This **SLG** shows the total number of bases (**TB**) a hitter records *per at-bat*. As **SLG** doesn't include *Base-on-Balls* and *Hit-by-Pitches* (these two are measured by **OBP**), if we combine **OBP** and **SLG** together, we get a single statistic that measures similar things as **TB + OBP** does.

The nature of baseball again. As I mentioned at the beginning of this project, a team should outscore its opponents to win a game in baseball. To do so, that team has to score and it's indicated as **Runs Scored (RS)**, the dependent variable. Then how does a team score runs?

Simple. To score runs in baseball, a team's batters must reach bases (i.e. become runners on bases) and other batters must advance these runners on bases to drive runs. This is how a team scores in baseball.

And this is what either **OPS** or **TB + OBP** measure, (a) the ability to reach bases, and (b) the ability to advance runners on bases to drive runs.

Given this fact, there's no wonder that those two different models yield the similar level of accuracy. Both independet variables measure similar things. Therefore, we get the similar result. Although the simple linear regression model where the independent variable is **OPS** yields a marginally more accurate result, I believe we'd get similar results no matter which one we use to predict team **RS**.
