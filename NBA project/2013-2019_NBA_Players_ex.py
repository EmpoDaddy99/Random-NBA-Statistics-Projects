import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

data = pd.read_csv('2013-2019_NBA_Players.csv')
data_adv = pd.read_csv('2013-2019_NBA_Players_Adv.csv')
data2020 = pd.read_csv('2020_NBA_Players_(before_corona).csv')
data2020_adv = pd.read_csv('2020_NBA_Players_adv_(before_corona).csv')
x0 = data[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TS%', 'TOV', 'PF']]
x1 =  data_adv[['ORtg', 'DRtg']]
x2 = data['MP'] / data['G']
x = pd.concat([x0, x1, x2], axis=1)
#x = preprocessing.StandardScaler().fit(x).transform(x.astype(float)) # uncomment to switch to standard deviations instead of real numbers
y = data['All NBA']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
array0 = np.zeros((99))
array1 = np.zeros((99))
# Seems like 2 neighbors are the most accurate testing accuracy (96% accurate)
# and only goes downhill for both testing and training accuracy
for i in range(1, 100):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(x_train, y_train)
    train_accuracy = metrics.accuracy_score(y_train, neigh.predict(x_train))
    test_accuracy = metrics.accuracy_score(y_test, neigh.predict(x_test))
    array0[i-1] = train_accuracy
    array1[i-1] = test_accuracy
plt.plot(range(1, 100), array0)
plt.plot(range(1, 100), array1)
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel('K Neighbors')
plt.ylabel('Accuracy')
plt.show()

x0 = data2020[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TS%', 'TOV', 'PF']]
x1 = data2020_adv[['ORtg', 'DRtg']]
x2 = data2020['MP'] / data2020['G']
x2020 = pd.concat([x0, x1, x2], axis=1)
data2020 = pd.concat([data2020, x1], axis=1)
#x2020 = preprocessing.StandardScaler().fit(x2020).transform(x2020.astype(float)) # uncomment to switch to standard deviations instead of real numbers
neigh = KNeighborsClassifier(n_neighbors = 2)
neigh.fit(x, y)
data2020['All NBA'] = neigh.predict(x2020)
#print(neigh.predict(np.asanyarray([4, 4, 4, 4, 4, 4]).reshape(1, -1)))
logReg = LogisticRegression(C=1.0, solver='saga', max_iter=1000000)
logReg.fit(x, y)
data2020['All NBA Probability Logistic Regression'] = logReg.predict_proba(x2020)[:,1]
data2020['All NBA'] = logReg.predict(x2020)
data2020 = data2020.set_index('All NBA', drop=True)
data2020 = data2020.set_index('Player', drop=True)
#print(data2020.loc[data2020['All NBA Probability'] > 0.5])
print(data2020.nlargest(30, 'All NBA Probability Logistic Regression'))
clf = svm.SVC(kernel='poly', probability=True)
clf.fit(x, y)
data2020['All NBA Probability SVM'] = clf.predict_proba(x2020)[:,1]
print(data2020.nlargest(30, 'All NBA Probability SVM'))
print(logReg.coef_)

kmeans = KMeans(n_clusters= 4, max_iter=1000)
kmeans.fit(x, y)
cluster_centers = kmeans.cluster_centers_
print("Clusters:")
print(cluster_centers)
data2020["Type of Player"] = kmeans.predict(x2020)
kmeans_labels = kmeans.labels_
markertype = ['r.', 'b.', 'g.', 'y.']
legendlabel = ['','','','','','','','']
for i in range(4):
    df = data2020.loc[data2020['Type of Player'] == i]
    plt.plot(df['PTS'], df['ORtg'] - df['DRtg'], markertype[i])
    plt.plot(cluster_centers[i][0], cluster_centers[i][8] - cluster_centers[i][9], 'ko')
    if df['ORtg'].mean() > df['DRtg'].mean():
        if df['PTS'].mean() > 20:
            data2020.loc[data2020['Type of Player'] == i, ['Type of Player']] = 'Above Average Starter'
            legendlabel[2*i] = 'Above Average Starter'
            legendlabel[2*i+1] = 'Above Average Starter Center'
        else:
            data2020.loc[data2020['Type of Player'] == i, ['Type of Player']] = 'Above Average Role Player'
            legendlabel[2*i] = 'Above Average Role Player'
            legendlabel[2*i+1] = 'Above Average Role Player Center'
    else:
        if df['PTS'].mean() > 20:
            data2020.loc[data2020['Type of Player'] == i, ['Type of Player']] = 'Below Average Starter'
            legendlabel[2*i] = 'Below Average Starter'
            legendlabel[2*i+1] = 'Below Average Starter Center'
        else:
            data2020.loc[data2020['Type of Player'] == i, ['Type of Player']] = 'Below Average Role Player'
            legendlabel[2*i] = 'Below Average Role Player'
            legendlabel[2*i+1] = 'Below Average Role Player Center'
plt.xlabel('Points')
plt.ylabel('ORtg - DRtg')
plt.title('Cluster Graph uwu :3')
plt.legend(legendlabel, loc='best', fontsize='xx-small', markerscale=0.4)
plt.show()
print(data2020.groupby(['Type of Player']).mean())
print(data2020.groupby(['Type of Player']).count())

agglo = AgglomerativeClustering(n_clusters=2, linkage='ward')
agglo.fit(x2020)
agglo_labels = agglo.labels_
data2020['Type of Player Agglo'] = agglo_labels
data2020['ORtg - DRtg'] = data2020['ORtg'] - data2020['DRtg']
df0 = data2020.loc[data2020['Type of Player Agglo'] == 0]
df1 = data2020.loc[data2020['Type of Player Agglo'] == 1]
#plt.plot(data2020.loc[data2020['Type of Player Agglo'] == 0, ['PTS']], data2020.loc[data2020['Type of Player Agglo'] == 0, ['ORtg - DRtg']], 'r.')
#plt.plot(data2020.loc[data2020['Type of Player Agglo'] == 1, ['PTS']], data2020.loc[data2020['Type of Player Agglo'] == 1, ['ORtg - DRtg']], 'b.')
sns.regplot('PTS', 'ORtg - DRtg', df0, color='red', marker='.')
sns.regplot('PTS', 'ORtg - DRtg', df1, color='blue', marker='.')
plt.xlabel('Points')
plt.ylabel('ORtg - DRtg')
plt.title('Cluster Graph uwu :3')
plt.legend(['Group 0', 'Group 1'], loc='best', fontsize='xx-small', markerscale=0.4)
plt.show()
data2020.groupby(['Type of Player Agglo']).mean()[['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TS%', 'TOV', 'PF', 'ORtg', 'DRtg']].to_csv('csv.csv')
print(data2020.groupby(['Type of Player Agglo']).count())

db = DBSCAN(eps=9, min_samples=5)
db.fit(x2020)
db_labels = db.labels_
data2020['Type of Player DBSCAN'] = db_labels
markertype = np.array(['r.', 'b.', 'g.', 'c.', 'm.', 'y.', 'k.', 'r*', 'b*', 'g*', 'c*', 'm*', 'y*', 'k*', 'rx', 'bx', 'gx', 'cx', 'mx', 'yx', 'kx', 'r+', 'b+', 'g+', 'c+', 'm+', 'y+', 'k+'])
legendlabel = np.empty(db_labels.max())
for i in range(db_labels.max()):
    df = data2020.loc[data2020['Type of Player DBSCAN'] == i]
    plt.plot(df['PTS'], df['ORtg - DRtg'], markertype[i])
    legendlabel[i] = str(int(i))
plt.xlabel('Points')
plt.ylabel('ORtg - DRtg')
plt.title('Cluster Graph uwu :3')
plt.legend(legendlabel, loc='best', fontsize='xx-small', markerscale=0.4)
plt.show()
print(data2020.groupby(['Type of Player DBSCAN']).mean())
print(data2020.groupby(['Type of Player DBSCAN']).count())
data2020 = data2020.drop(['ORtg', 'DRtg', 'ORtg - DRtg'], axis=1)
