# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#getting the dataset.....
dataset=pd.read_csv('project.csv')
x=dataset.iloc[:, 1:14].values
print(x)

#Gettign rid of the categorical data of regions...
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
x[:, 4]=labelencoder_X.fit_transform(x[:, 4])
print(x[:,4])

onehotencoderRegion=OneHotEncoder(categorical_features= [4])
x=onehotencoderRegion.fit_transform(x).toarray()

