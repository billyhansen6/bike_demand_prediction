"""
## Problem Description

The objective is to predict hourly bike rental demand. The target variable is "count", which is the total bikes
rented. The predictor variables are:

- datetime - hourly date + timestamp  
- season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
- holiday - whether the day is considered a holiday
- workingday - whether the day is neither a weekend nor holiday
- weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
- temp - temperature in Celsius
- atemp - "feels like" temperature in Celsius
- humidity - relative humidity
- windspeed - wind speed
- casual - number of non-registered user rentals initiated
- registered - number of registered user rentals initiated
- count - number of total rentals

The data can be downloaded here: https://www.kaggle.com/c/bike-sharing-demand/data

I borrowed a few ideas from this excellent kaggle kernal:
https://www.kaggle.com/viveksrinivasan/eda-ensemble-model-top-10-percentile """

 # Import relevant Libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from random import seed
from random import randrange
from sklearn.metrics import mean_squared_log_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from math import sqrt
import math
import warnings
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load Data
os.chdir(r'C:\Users\Billy Hansen\Desktop\Kaggle Practice\Bike Demand')
df = pd.read_csv('train.csv')
toast = pd.read_csv('test.csv')
# Make Copy of Test Data
test = toast.copy()

###############################################################################
# # EDA

# Check for missing Values
df.isnull().sum()


# Peak at data
df.head(10)


# Look at target variable
sb.boxplot('count', data=df);

###############################################################################
# How many outliers?

len(df) - len(df[(np.abs(stats.zscore(df['count'])) < 3)])

# Will keep outliers for now,
# and try running both ways. Let's look at how the categorical variables, and see their relationship to the target
# variable.

# Season Variable
sb.catplot('season', data=df, kind='count');


sb.boxplot(x = 'season', y='count', data=df);


# Holiday 
sb.catplot('holiday', data=df, kind='count');


sb.boxplot(x='holiday', y='count', data=df);


# Working Day
sb.catplot('workingday', data=df, kind='count');


sb.boxplot(x='workingday', y='count', data=df);


# Weather
sb.catplot('weather', data=df, kind='count');


sb.boxplot(x='weather', y='count', data=df);

# Summary Stats
df.describe()


# Let's look at a coorelation matrix for all of our numeric variables
corr = df[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots()
fig.set_size_inches(10, 9)
sb.heatmap(corr, mask=mask, vmax=.6, square=True, annot=True);

# ############################################################################## The number of rented bikes is
# positively correlated with temperature, and negatively coorelated with huminity, which makes sense intuitively.
# Atemp and temp are almost perfectly coorelated, so we should drop one of these variables to avoid
# multi-colinearity. I'll drop "feel like temperature", and keep actual temperature. I'll perform all data prep to
# both the train and test data sets.

df = df.drop(columns=['atemp'])
test = test.drop(columns=['atemp'])


# Change categorical variable types to objects
df['season'] = df['season'].astype('object')
df['holiday'] = df['holiday'].astype('object')
df['workingday'] = df['workingday'].astype('object')
df['weather'] = df['weather'].astype('object')
test['season'] = test['season'].astype('object')
test['holiday'] = test['holiday'].astype('object')
test['workingday'] = test['workingday'].astype('object')
test['weather'] = test['weather'].astype('object')

###############################################################################
# We can engineer date variable to potentially have some predictive value. We should create hour, weekday, and month values to use in our models.

df['datetime'] = pd.to_datetime(df['datetime'])
df['datetime'].describe()


df['Hour'] = df['datetime'].dt.hour
df['Day'] = df['datetime'].dt.weekday
df['Month'] = df['datetime'].dt.month
df['Year'] = df['datetime'].dt.year

###############################################################################
# Let's look at how these new variables are related to our target

hours = pd.DataFrame(df.groupby(['Hour'])['count'].mean())
days = pd.DataFrame(df.groupby(['Day'])['count'].mean())
months = pd.DataFrame(df.groupby(['Month'])['count'].mean())


# Mean rentals for each hour of the day



sb.boxplot(x = 'Hour', y = 'count', data = df);

###############################################################################
# Hours looks like it will be an important variable

# Mean rentals each month
sb.boxplot(x='Month', y='count', data=df);

###############################################################################
# The summer months seem to be the most popular.

# Mean Rentals each day of the week
test['datetime'] = pd.to_datetime(test['datetime'])
test['datetime'].describe()


test['Hour'] = test['datetime'].dt.hour
test['Day'] = test['datetime'].dt.weekday
test['Month'] = test['datetime'].dt.month
test['Year'] = test['datetime'].dt.year


sb.boxplot(x='Day', y='count', data=df);


# # Day doesnt look like a very good variable, so we'll drop it.
# df = df.drop(columns = ['Day'])
# test = test.drop(columns = ['Day'])


# Change Data Types
df['Hour'] = df['Hour'].astype('object')
df['Month'] = df['Month'].astype('object')
df['Day'] = df['Day'].astype('object')
df['Year'] = df['Year'].astype('object')
test['Hour'] = test['Hour'].astype('object')
test['Month'] = test['Month'].astype('object')
test['Day'] = test['Day'].astype('object')
test['Year'] = test['Year'].astype('object')


# Join data frames together

data = df.append(test)


data.head()

###############################################################################
# There are many values with 0 as the wind speed. I suspect that these are not really 0s, but rather incomplete data. We'll fill these 0s with a random forest model.

# Grab data with 0 as wind speed
zerowind = data[data['windspeed'] ==0]
wind = data[data['windspeed']!=0]

# Simple random forest
rf_wind = RandomForestRegressor()
# Select models for training
windy = wind[["season","weather","humidity","Day","temp","Hour", 'Month']]
# fit data
rf_wind.fit(windy, wind['windspeed'])

# Make predictions
zerowindy = zerowind[["season","weather","humidity","Day","temp","Hour", 'Month']]
wind_values = rf_wind.predict(zerowindy)
zerowind['windspeed'] = wind_values

# Append data back together
data = wind.append(zerowind)


# Restore Data Types
cat_feets = ["season","holiday","workingday","weather","Day","Month","Year","Hour"]
num_feets = ["temp","humidity","windspeed"]
drop_feets = ['casual',"count","datetime","registered"]

for var in cat_feets:
    data[var] = data[var].astype("category")


# Get dummy variables
data = pd.get_dummies(data, columns = ["season","holiday","workingday","weather","Day","Month","Year","Hour"])


data.head()


# Break data back into train and test
df = data[pd.notnull(data['count'])].sort_values(by= ['datetime'])
test = data[~pd.notnull(data['count'])].sort_values(by = ['datetime'])
y = df['count']
df  = df.drop(drop_feets,axis=1)
test  = test.drop(drop_feets,axis=1)
X = df.copy()

###############################################################################
# We'll create seperate data sets that are normalized for models that benefit from normalization.

sc = StandardScaler()
X_scaled = sc.fit_transform(X)
test_scaled = sc.transform(test)

###############################################################################
# Now let's split data into training and testing sets, for scaled and unscaled data.

X_train_sc, X_test_sc, y_train, y_test = train_test_split(X_scaled, y, test_size = .2, random_state = 37)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 37)


# Let's plot the target variable and look at the distribution

plt.hist(y, color = 'blue', edgecolor = 'black',
         bins = int(180/5));


# Now let's look at a plot when we transform the vector using the np.log1p function

plt.hist(np.log1p(y), color = 'green', edgecolor = 'black',
         bins = int(180/5));


###############################################################################
# The second distribtion looks like it might help the model be more skillful. So we'l transform the target vector before we train the model

###############################################################################
# The Kaggle competition is scored with Root Mean Squared Logrithmic Error. We'll build a function that uses this method to score our results.

def rmsle(y, y1):
    log1 = np.nan_to_num(np.array([np.log1p(v) for v in y]))
    log2 = np.nan_to_num(np.array([np.log1p(v) for v in y1]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

###############################################################################
# Now we'll transform our target variables

y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize linear regression model
lin_model = LinearRegression()

# Train the model
lin_model.fit(X = X_train_sc,y = y_train)

# Make predictions
preds = lin_model.predict(X=X_test_sc)
print ("RMSLE Value For Linear Regression: ",rmsle(np.exp(y_test),np.exp(preds)))

###############################################################################
# Let's try a neural net.

from keras.callbacks import EarlyStopping
# ANN
model = Sequential()
model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

es = EarlyStopping(monitor='loss', mode='min', patience = 3, restore_best_weights = True, verbose=1)
model.fit(X_train_sc, y_train, epochs=100, batch_size=5, callbacks = [es])

# Make predictions
preds = model.predict(X_test_sc)
print ("RMSLE Value For ANN: ",rmsle(np.exp(y_test),np.exp(preds)))

###############################################################################
# Now Ridge Regression with grid search for the optimal alpha

ridgey = Ridge()
params = {'max_iter': [3000], 'alpha': [0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}

# Make scorer using rmsle
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)

grid_ridge = GridSearchCV( ridgey,
                          params,
                          scoring = rmsle_scorer,
                          cv=5)

grid_ridge.fit(X_train_sc, y_train)


preds = grid_ridge.predict(X= X_test_sc)
print (grid_ridge.best_params_)
print ("RMSLE Value For Ridge Regression: ",rmsle(np.exp(y_test),np.exp(preds)))

###############################################################################
# Let's try random forest. We'll used unscaled data for tree based models

# Initialize
rf_mod = RandomForestRegressor(n_estimators = 300)
# Train
rf_mod.fit(X_train, y_train)
# Make Predictions
preds = rf_mod.predict(X_test)
print ("RMSLE Value For Random Forest: ",rmsle(np.exp(y_test),np.exp(preds)))

###############################################################################
# Wooooooo!!!! Let's try a gradient boosting model

from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01)


gbm.fit(X_train, y_train)
preds = gbm.predict(X_test)
print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(y_test),np.exp(preds)))

###############################################################################
# Ohhh shit!!! lol Let's try to use gridsearch to find an optimal alpha value.

gbm = GradientBoostingRegressor()
params = {'n_estimators': [4000], 'alpha': [0.1, .05, .15]}

# Make scorer using rmsle
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)

grid_gbm = GridSearchCV( gbm,
                          params,
                          scoring = rmsle_scorer,
                          cv=3)

grid_gbm.fit(X_train, y_train)

preds = grid_gbm.predict(X= X_test)
print (grid_gbm.best_params_)
print ("RMSLE Value For Gradient Bossing Regressor is: ",rmsle(np.exp(y_test),np.exp(preds)))


yy = np.log1p(y)

gbm = GradientBoostingRegressor(n_estimators = 4000, alpha = .05)
gbm.fit(X, yy)

predsTest = gbm.predict(X=test)

# Make submission, transforming predictions back to appropriate scale
submission = pd.DataFrame({
        "datetime": toast['datetime'],
        "count": [max(0, x) for x in np.exp(predsTest)]
    })
submission.to_csv('hoot4.csv', index=False)

'''This submission scored approx .39 on kaggle, which is in the top 6th percentile of all submissions.'''



