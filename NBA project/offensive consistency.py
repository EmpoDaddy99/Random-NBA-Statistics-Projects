import pandas as pd

data = pd.read_csv('Kobe_Bryant_2005-2006_Game_Log.csv')
data = data.dropna(subset=['G'])
data = data[['MP','FGA','FTA','ORB','AST','TOV','PTS']]
for i in data.index:
	if int(data['MP'][i][3]) >= 3:
		data['MP'][i] = int(data['MP'][i][:2]) + 1
	else:
		data['MP'][i] = int(data['MP'][i][:2])
data = data.astype(float)
data['TS%'] = data['PTS'] / (2 * (data['FGA'] + 0.44 * data['FTA']))
data['Off_Con'] = (36.0 / data['MP']) * (data['PTS'] * data['TS%'] + data['AST'] - data['TOV'] + data['ORB'])
Off_Con_Avg = data['Off_Con'].mean()
Off_Con = data['Off_Con'].std()
Per_Game = data[['MP','FGA','FTA','ORB','AST','TOV','PTS','TS%']].mean()
Off_Rat = (36.0 / Per_Game[0]) * (Per_Game[6] * Per_Game[7] + Per_Game[4] / Per_Game[5] + Per_Game[3])
print(data)
print('David\'s Offensive Rating: ' + str(Off_Rat))
print('Offensive Consistency: ' + str(Off_Con_Avg) + ' +/- ' + str(Off_Con))