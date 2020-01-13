# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_excel('Project_Data.xlsx')
#check=dataset.iloc[:, 13].values
x=dataset.iloc[:, 1:14].values
print(x)

#Getting rid of categorical data of region...

#Region
category_Hot_Encoded=pd.get_dummies(x[:, 4])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 4, 1)

#Gender
category_Hot_Encoded=pd.get_dummies(x[:, 1])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 1, 1)
x=np.delete(x, 15, 1)

#Goals
category_Hot_Encoded=pd.get_dummies(x[:, 4])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 4, 1)

#Seats
category_Hot_Encoded=pd.get_dummies(x[:, 7])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 7, 1)
x=pd.DataFrame(x)

#Hobbies
clist = ['Game','Musical','Gym','Others']
for i in range(0, 328):
    list1 = x[3][i].split(', ')
    list2 = [0, 0, 0, 0]
    
    for j in range(0, 4):
        if clist[j] in list1:
            list2[j] = 1 
    x[3][i] = list2
    
    
tags = x[3].apply(pd.Series)
tags = tags.rename(columns = lambda a : 'tag_' + str(a))

x = pd.concat([x[:], tags[:]], axis=1)
x=pd.DataFrame(x)
x.rename(columns = {'tag_0':'Game'}, inplace = True)
x.rename(columns = {'tag_1':'Musical'}, inplace = True)
x.rename(columns = {'tag_2':'Gym'}, inplace = True)
x.rename(columns = {'tag_3':'Others'}, inplace = True)


"""
#Making the data polynomial...
	from sklearn.preprocessing import PolynomialFeatures
	poly_reg=PolynomialFeatures(degree=i)
	x_poly=poly_reg.fit_transform(x)
"""



"""

#for i in range(5):
	
	#Making the data polynomial...
	from sklearn.preprocessing import PolynomialFeatures
	poly_reg=PolynomialFeatures(degree=4)
	x_poly=poly_reg.fit_transform(x)
	#Splitting the dataset...
	from sklearn.cross_validation import train_test_split
	x_train, x_test, y_train, y_test=train_test_split(x_poly, y, test_size=0.1, random_state=0)
	
	#applying feature scaling...
	from sklearn.preprocessing import StandardScaler
	sc=StandardScaler()
	x_train=sc.fit_transform(x_train)
	x_test=sc.transform(x_test)
	
	#fitting the linear regression model
	from sklearn.linear_model import LinearRegression
	regressor=LinearRegression()
	regressor.fit(x_train, y_train)
	#predicting the values
	y_pred=regressor.predict(x_test)
	
	loss=0
	err=0
	avg=0
	for j in range(77):
		print("Loss->",(y_test[j]-y_pred[j]))
		avg=avg+y_test[j]
		err=err+abs((y_test[j]-y_pred[j]))
		loss=loss+(y_test[j]-y_pred[j])*(y_test[j]-y_pred[j])
	cost=loss/77
	err=err/77
	avg=avg/77
	per_error=(err/avg)*100
	
	#plotting the result
	plt.scatter(check, y, color='red')
	plt.plot(check, regressor.predict(x), color='blue')
	plt.xlabel('AH--->')
	plt.ylabel('RH--->')
	plt.show()
	



"""
