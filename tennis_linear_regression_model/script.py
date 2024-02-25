import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

##Load/investigate data 
df = pd.read_csv('C:\\Users\\joshs\\Downloads\\tennis_ace_starting\\tennis_ace_starting\\tennis_stats.csv')
print(df.head())

BreakPointsOpportunities = df['BreakPointsOpportunities']
Winnings = df['Winnings']
plt.scatter(BreakPointsOpportunities, Winnings, alpha=0.4)


## perform single feature linear regressions here:

features = df[['FirstServeReturnPointsWon']]
outcome = df[['Winnings']]
features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)

model = LinearRegression()
model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)
prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.show()




## single feature linear regressions here:

DoubleFaults = df[['DoubleFaults']]
outcome = df[['Losses']]
features_train, features_test, outcome_train, outcome_test = train_test_split(DoubleFaults, outcome, train_size = 0.8)

model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)
prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.show()


## Multi feature linear regressions here:


features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)




model.fit(features_train,outcome_train)


model.score(features_test,outcome_test)
prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.show()













## perform multiple feature linear regressions here:
