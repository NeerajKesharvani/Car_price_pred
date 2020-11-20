# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:25:29 2020

@author: 2slim
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle

df = pd.read_csv ('car data.csv')

df.head()

print(df['Seller_Type'].unique())

df.isnull().sum()

df.describe()

df.columns

final_dataset = df[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner' ]]

final_dataset.head()

final_dataset['Current_year'] = 2020

final_dataset.head()

final_dataset['no_of_year'] = final_dataset['Current_year'] - final_dataset['Year']

final_dataset.head()

final_dataset.drop(['Year'], axis = 1, inplace =True)

final_dataset.head()

final_dataset.drop(['Current_year'], axis = 1, inplace =True)

final_dataset.head()

final_dataset = pd.get_dummies(final_dataset, drop_first= True)

final_dataset.head()

sns.pairplot(final_dataset)

corrmat=final_dataset.corr() 
top_corr_features=corrmat.index 
plt.figure(figsize=(20,20)) 
#plot heat map 
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

x = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]

x.head()

model = ExtraTreesRegressor()
model.fit(x,y)

print(model.feature_importances_)


feat_importances = pd.Series(model.feature_importances_, index= x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

x_train.shape

rf_rand = RandomForestRegressor()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num =12)]
print(n_estimators)

# No. of features to consider at every split
max_features = ['auto', 'sqrt']

# Max no. levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# Min no. of saamples required to split a node 
min_samples_split = [2, 5, 10, 15, 100]

# Min no. of saamples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf' : min_samples_leaf}

print(random_grid)

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                        scoring= 'neg_mean_squared_error', n_iter = 10, cv=5, verbose=2 , random_state= 42, n_jobs=1)

rf_random.fit(x_train, y_train)

# rf_random.best_params_
#
# rf_random.best_score_

predictions = rf_random.predict(x_test)

print(predictions)


sns.distplot(y_test-predictions)

plt.scatter(y_test, predictions)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



file = open('random_forest_regression_model.pkl', 'wb')
pickle.dump(rf_random, file)

















