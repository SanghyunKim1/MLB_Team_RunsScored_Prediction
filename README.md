# MLB Team Runs Scored Prediction

## Content
1. Intro: The Nature of Baseball
2. Metadata
3. Data Cleaning
4. EDA (Exploratory Data Analysis)
5. Feature Scaling
6. Multiple Linear Regression with Feature Selection
7. Simple Linear Regression
8. Model Validation
9. Conclusion

### 1. Intro
Before we dive into our analysis, let's briefly talk about the nature of baseball.

Say you are an owner of a baseball team, then why do you think you are running your team spending billions of dollars per season? 
Making money, making fans enthusiastic, having fans in ballparks etc... But the ulitmate goal of running a baseball club, as with other sports teams, will be winning. As the owner of your team, your goal should be winning, and thus, you should try to buy wins not just players. This is where sabermetrics (SABR + metrics) has originated. Interesting, isn't it?

Okay, now we understand that we should focus on wins, then HOW do we win? (i.e. what makes that wins) As in other sports, a baseball team should score runs and prevent opponents from scoring to win a game.

Alright, we're almost there. Your goal is to win, and therefore, your team must outscore its opponents to do so.
That's the nature of baseball: Runs Scored and Runs Allowed. In this project, I'll pretty much focus on the first part of baseball, **Runs Scored**.

### 2. Metadata
| **Metadata** | **Information** |
| :-----------: | :-----------: |
| **Origin of Data** | [Baseball Prospectus](https://www.baseballprospectus.com) |
| **Terms of Use** | [Terms and Conditions](https://www.baseballprospectus.com/terms-and-conditions/) |
| **Data Structure** | 10 datasets each consisting of 301 rows * 28 columns |

| **Data Meaning** | **Explanation** |
| :-----------: | :-----------: |
| ***LG*** | AL: American League / NL: National League |
| ***YEAR*** | Each year refers to corresponding seasons |
| ***TM*** | All 30 Major League Baseball Teams |
| ***G*** | Number of games played in the corresponding season |
| ***PA*** | [Plate Appearance](http://m.mlb.com/glossary/standard-stats/plate-appearance) |
| ***AB*** | [At-bat](http://m.mlb.com/glossary/standard-stats/at-bat) |
| ***R*** | [Runs Scored](http://m.mlb.com/glossary/standard-stats/run) |
| ***TB*** | [Total Bases](http://m.mlb.com/glossary/standard-stats/total-bases) |
| ***H*** | [Hit](http://m.mlb.com/glossary/standard-stats/hit) |
| ***AVG*** | [Batting Average](http://m.mlb.com/glossary/standard-stats/batting-average) |
| ***OBP*** | [On-base Percentage](http://m.mlb.com/glossary/standard-stats/on-base-percentage) |
| ***SLG*** | [Slugging Percentage](http://m.mlb.com/glossary/standard-stats/slugging-percentage) |
| ***OPS*** | [On-base Plus Slugging](http://m.mlb.com/glossary/standard-stats/on-base-plus-slugging) |
| ***ISO*** | [Isolated Power](http://m.mlb.com/glossary/advanced-stats/isolated-power) |
| ***HR*** | [Home run](http://m.mlb.com/glossary/standard-stats/home-run) |
| ***HRr*** | [Home Run Rate](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=344) |
| ***BB*** | [Walk](http://m.mlb.com/glossary/standard-stats/walk) |
| ***BBr*** | [Walk Rate](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=283) |
| ***SO*** | [Strikeout](http://m.mlb.com/glossary/standard-stats/strikeout) |
| ***SOr*** | [Strikeout Rate](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=349) |
| ***SO/BB*** | [Strikeout-to-Walk ratio (equivalently K/BB)](http://m.mlb.com/glossary/advanced-stats/strikeout-to-walk-ratio) |
| ***SB*** | [Stolen Base](http://m.mlb.com/glossary/standard-stats/stolen-base) |
| ***CS*** | [Caught Stealing](http://m.mlb.com/glossary/standard-stats/caught-stealing) |
| ***SB%*** | [Stolen-base Percentage](http://m.mlb.com/glossary/standard-stats/stolen-base-percentage) |
| ***DRC+*** | [Deserved Runs Created for a batter](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=696) |
| ***DRAA*** | [Deserved Runs Above Average](https://legacy.baseballprospectus.com/glossary/index.php?search=DRC_RAA) |
| ***BWARP*** | [Batter Wins Above Replacement Player](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=591) |

### 3. Data Cleaning
- Combined 10 different datasets (2010-2019 Season Batting datasets).
- Dropped an unnecessary column made when combining datasets (Column: **'#'**).
- Renamed **'R'** data feature as **'RS'** for clarity.
- Eliminated commas in some data features and convert their data types from **integer** into **numeric** (**'PA'**, **'AB'**, **'H'**, **'SO'**).
- Confirmed that there are no missing values and duplicates.

### 4. EDA (Exploratory Data Analysis)
***4-1. RS EDA***
![RS Histogram:Probability Plot](https://user-images.githubusercontent.com/67542497/105629056-1cfd7a00-5e84-11eb-9166-ebbd49161ed1.png)
<img src="https://user-images.githubusercontent.com/67542497/105629055-1c64e380-5e84-11eb-94bb-60ff4948660d.png" width="500" height="500">

- **RS** Skewness: 0.35264844525853095
- **RS** Kurtosis: 0.061150254394042314

According to the histogram and probability plot above, **RS** seems to follow a normal distribution. The skewness of 0.35 and kurtosis of 0.06 also indicate that team **RS** data is normallly distributed. The boxplots above also shows that team **RS** has been normally distributed over the last 10 seasons with few outliers.


***4-2. Correlation EDA***

<img src="https://user-images.githubusercontent.com/67542497/105629053-1b33b680-5e84-11eb-8717-ee73189af8e3.png" width="500" height="500">

| ***Correlation*** | **RS** | **PA** | **TB** | **OBP** | **SLG** | **OPS** | **ISO** | **HR** | **HRr** | **DRC+** | **DRAA** | **BWARP** |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| **RS** | 1.0 | 0.739 | 0.922 | 0.829 | 0.920 | 0.950 | 0.812 | 0.745 | 0.711 | 0.751 | 0.806 | 0.780 |

Initially, we have too many independent variables (24 in total) to predict **RS**. Let's drop some of them based on their correlations with **RS** (cut-off point: 0.65).
As a result, 11 data features are selected indicated in the correlation heatmap above.


***4-3. Filtered Independent Variables EDA***

<img src="https://user-images.githubusercontent.com/67542497/105629048-17a02f80-5e84-11eb-977a-7d2a5dc43438.png" width="800" height="800">
According to the histograms of each independent variable above, all the variables are normally distributed.

<img src="https://user-images.githubusercontent.com/67542497/105629057-1cfd7a00-5e84-11eb-8ea9-ed824baf24aa.png" width="600" height="600">
Scatter plots also depict that there are reasonable linear trends between each independent variable and **RS** without notable outliers, and thus, it's safe to use the linear regression model.

### 5. Feature Scaling
Since the data ranges of independent variables considerably vary, we need to scale them. As all the data attributes have normal distributions with few outliers, we're going to use ***StandardScaler*** to scale our data.

The result of feature sacling is the following:

<img src="https://user-images.githubusercontent.com/67542497/105629054-1bcc4d00-5e84-11eb-934a-c7efe9d98fbc.png" width="600" height="600">


### 6. Multiple Linear Regression with Feature Selection


### 7. Simple Linear Regression


### 8. Model Validation


### 9. Conclusion
