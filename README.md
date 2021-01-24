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

Alright, we're almost there. Your goal is to win and therefore, you shoud outscore your opponents to do so. That's the nature of baseball: Runs Scored and Runs Allowed. In this project, I'll pretty much focus on the first part of baseball, **Runs Scored**.

### 2. Metadata
| **Metadata** | **Information** |
| :-----------: | :-----------: |
| **Origin of Data** | [Baseball Prospectus](https://www.baseballprospectus.com) |
| **Terms of Use** | [Terms and Conditions](https://www.baseballprospectus.com/terms-and-conditions/) |
| **Data Structure** | 301 rows * 28 columns |

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


### 4. EDA (Exploratory Data Analysis)


### 5. Feature Scaling


### 6. Multiple Linear Regression with Feature Selection


### 7. Simple Linear Regression


### 8. Model Validation


### 9. Conclusion
