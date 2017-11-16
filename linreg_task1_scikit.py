import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import sys

ch = [[100,500,1000,5000,10000,50000,100000,500000,1000000]]

#import CSV
filename = 'Sum+noise.csv'
d=pd.read_csv(filename,sep=';')
print(list(d))
#specify target and train coloumn
target_column = ['Noisy Target']
train_column = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']


for x in ch:
	for i in x:
		print(i)
		X = d[train_column].iloc[0:i]
		Y = d[target_column].iloc[0:i]
		
		X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
		
		clf = linear_model.LinearRegression()
		clf.fit(X_train,Y_train)
		y_pred = clf.predict(X_test)
		mean = np.mean(Y_test)
		mse = np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
		print(mse/mean)
		r2 = metrics.r2_score(Y_test,y_pred)
		print(r2)
sys.modules[__name__].__dict__.clear()
## for dataset Sum without noise

ch = [[100,500,1000,5000,10000,50000,100000,500000,1000000]]

#import CSV
filename = 'Sum-noise.csv'
d=pd.read_csv(filename,sep=';')

#specify target and train coloumn
target_column = ['Target']
train_column = ['Feature 1','Feature 2','Feature 3','Feature 4','Feature 6','Feature 7','Feature 8','Feature 9','Feature 10']


for x in ch:
	for i in x:
		print(i)
		X = d[train_column].iloc[0:i]
		Y = d[target_column].iloc[0:i]
		
		X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
		
		clf = linear_model.LinearRegression()
		clf.fit(X_train,Y_train)

		y_pred = clf.predict(X_test)
		mean = np.mean(Y_test)
		mse = np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
		print(mse/mean)
		r2 = metrics.r2_score(Y_test,y_pred)
		print(r2)


sys.modules[__name__].__dict__.clear()
## for dataset Skin_NonSkin
ch = [[100,500,1000,5000,10000,50000,100000]]

#import CSV
filename = 'Skin_NonSkin.txt'
d=pd.read_csv(filename,sep='\t')

#specify target and train coloumn
target_column = d.iloc[:,3] 
train_column = d.iloc[:,0:2]


for x in ch:
	for i in x:
		print(i)
		X = train_column.iloc[0:i]
		Y = target_column.iloc[0:i]
		
		X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
		
		clf = linear_model.LinearRegression()
		clf.fit(X_train,Y_train)

		y_pred = clf.predict(X_test)
		mean = np.mean(Y_test)
		mse = np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
		print(mse/mean)
		r2 = metrics.r2_score(Y_test,y_pred)
		print(r2)

sys.modules[__name__].__dict__.clear()
## for YearPredictionMSD

ch = [[100,500,1000,5000,10000,50000,100000,500000]]

#import CSV
filename = 'YearPredictionMSD.txt'
d=pd.read_csv(filename,sep=',')

#specify target and train coloumn
target_column = d.iloc[:,0] 
train_column = d.iloc[:,1:]

for x in ch:
	for i in x:
		print(i)
		X = train_column.iloc[0:i]
		Y = target_column.iloc[0:i]
		
		X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
		
		clf = linear_model.LinearRegression()
		clf.fit(X_train,Y_train)

		y_pred = clf.predict(X_test)
		mean = np.mean(Y_test)
		mse = np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
		print(mse/mean)
		r2 = metrics.r2_score(Y_test,y_pred)
		print(r2)
