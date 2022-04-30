import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import seaborn as sns

data = pd.read_csv('2019_NBA_Players.csv')
data_adv = pd.read_csv('2019_NBA_Players_Advanced.csv')
data['SA'] = data['FGA'] + 0.44 * data['FTA']
data['TS%'] = data['TS%'] * 100
temp = np.random.rand(len(data)) < 0.8
train = data[temp]
train_adv = data_adv[temp]
test = data[~temp]
test_adv = data_adv[~temp]
reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['PTS', 'AST', 'TS%']])
test_x = np.asanyarray(test[['PTS', 'AST', 'TS%']])
train_y = np.asanyarray(train_adv[['ORtg']])
test_y = np.asanyarray(test_adv[['ORtg']])
reg.fit(train_x, train_y)
predict_test_y = reg.predict(test_x)
intercept = reg.intercept_
coefficient = []
coefficient.append(float(reg.coef_[0][0]))
coefficient.append(float(reg.coef_[0][1]))
coefficient.append(float(reg.coef_[0][2]))
plt.scatter(data['PTS'], data_adv['ORtg'], edgecolors='blue')
plt.scatter(data['AST'], data_adv['ORtg'], edgecolors='red')
plt.scatter(data['TS%'], data_adv['ORtg'], edgecolors='green')
plt.plot(train_x, coefficient * train_x + intercept, 'r')
plt.xlabel("PTS, AST per 100 possesions, TS%")
plt.ylabel("ORtg")
plt.legend(["PTS, AST per 100 possesions, TS%", "PTS, AST per 100 possesions, TS% vs. ORtg Best Fit"], loc='upper left')
plt.show()

print("R^2 score = " + str(r2_score(predict_test_y, test_y)))
print('residual sum of squares: ' + str(np.mean((predict_test_y - test_y) ** 2)))
print('Variance Score: ' + str(reg.score(test_x, test_y)))

temp = np.random.rand(len(data)) < 0.8
train = data[temp]
test = data[~temp]
poly = PolynomialFeatures(degree=3)
train_x = np.asanyarray(train[['Age']])
test_x = np.asanyarray(test[['Age']])
train_y = np.asanyarray(train[['PTS']])
test_y = np.asanyarray(test[['PTS']])
train_x_poly = poly.fit_transform(train_x)
reg = linear_model.LinearRegression()
predict_train_y = reg.fit(train_x_poly, train_y)
intercept = float(reg.intercept_)
coefficient = reg.coef_[0]
plt.scatter(data['Age'], data['PTS'], edgecolors='blue')
x_increment = np.arange(19.0, 42.0, 1.0)
y_increment = intercept + coefficient[0] * x_increment + coefficient[1] * np.power(x_increment, 2) + coefficient[2] * np.power(x_increment, 3)
plt.plot(x_increment, y_increment, 'r')
plt.show()
print(coefficient)
print(intercept)

tot_x1 = data[['PTS', 'AST', 'TOV', 'TS%']]
tot_x2 = data_adv[['AST%', 'TOV%', 'USG%']]
tot_x = pd.concat([tot_x1, tot_x2], axis=1)
tot_x = (tot_x - tot_x.mean()) / tot_x.std()
tot_y = data_adv['ORtg']
tot_y = (tot_y - tot_y.mean()) / tot_y.std()
reg = linear_model.LinearRegression()
reg.fit(tot_x, tot_y)
print("R^2 score:")
print(reg.score(tot_x,tot_y)) #R squared value
predict_std = reg.predict(np.array([50, 10, 5, 0.6, 40, 15, 40]).reshape(1, -1))
Yhat = reg.predict(tot_x.to_numpy())
print(predict_std) #Number of std units away from average y variable
print(reg.coef_) #How much each variable affects the y variable prediction
sns.distplot(tot_y, hist=False, label="Actual")
sns.distplot(Yhat, hist=False, label="Predicted")
plt.title("Actual vs Predicted Offensive Rating")
plt.ylabel("Proportion of x variables")
plt.show() #Show actual vs predicted
print("Cross Val Score:")
print(cross_val_score(reg, tot_x, tot_y, cv=5).mean())

data_lbj = pd.read_csv('LeBron_James.csv')
polyfit = np.polyfit(data_lbj['Age'], data_lbj['PTS'], 4)
poly1d = np.poly1d(polyfit)
print(poly1d) #show equation
plt.plot(data_lbj['Age'], data_lbj['PTS'], '.') #Plot usg% vs ortg
plt.plot(data_lbj['Age'], poly1d(data_lbj['Age']), '-') #Plot equation for best fit usg% vs ortg
plt.xlabel("Age")
plt.ylabel("Points")
plt.show()