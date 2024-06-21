##********************************************************************************************
#********************************Python Library Imports***************************************
#*********************************************************************************************
import matplotlib
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use(backend='qt5agg', force=True)
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import skew
import math as mt
import os
##********************************************************************************************
# *************************************Read in CSV's******************************************
# ********************************************************************************************
# Changing the current working directory to where the main.py file is located
# this helps with any relative imports later on.
os.chdir("C:/Users/brian/OneDrive/Documents/Academics/Courses/NYC Bootcamp Machine Learning/\
Projects/Project-III (Ames Housing)/")
print(os.getcwd())
# read_csv starts looking in current folder, so you can navigate using ../
housing = pd.read_csv('data/Ames_HousePrice.csv', index_col=0)
print(housing.shape)
housing.head()
##********************************************************************************************
#*************************************Pre-processing******************************************
#*********************************************************************************************
housing.dtypes
# select only numeric columns and get the total number of numeric columns (38)
num_feat = housing.select_dtypes(np.number)
# print("numerical features = ",num_feat.shape[1])
# select only numeric columns and get the total number of categorical columns (43)
cat_feat = housing.select_dtypes(include = ['O'])
# print("categorical features = ",cat_feat.shape[1])
temp_feats = pd.array([num_feat.shape[1],cat_feat.shape[1]])
# unpacks the plot into a fig and ax container.
# Fig contains the plot which you can save as a picture
# ax contains the important stuff, like the axes, legend, type of plot, etc...
fig, ax = plt.subplots(figsize=(8,6))
# Create a barplot
bar_feats = ax.bar(['Num','Cat'], temp_feats)
# Adding labels and title
ax.set_title('Housing Features')
ax.set_xlabel('Feature Type')
ax.set_ylabel('# of Features')
ax.bar_label(bar_feats)
# Rotating x-axis labels
ax.tick_params(axis='x', labelrotation=45)
# Displaying the plot
plt.tight_layout() # Adjust layout to prevent clipping
plt.show(block=True)
##
print(num_feat.columns[num_feat.isna().any()]) #Print all colums that have NA values for h_num
##
miss_numfeats = num_feat.isna().sum().sort_values(ascending = False)*(100/2580)
miss_numfeats = miss_numfeats.round(1)

fig, ax = plt.subplots(figsize=(8,6))
bar_miss_numfeats = ax.bar(miss_numfeats.index[0:9], miss_numfeats[0:9]) # create a barplot
# Adding labels and title
ax.set_title('Top 10 num. feat. with missing values')
ax.set_xlabel('Num. Feature')
ax.set_ylabel('% Missing')
ax.bar_label(bar_miss_numfeats)
# Rotating x-axis labels
ax.tick_params(axis='x', labelrotation=45)
# Displaying the plot
plt.tight_layout() # Adjust layout to prevent clipping
plt.show(block=True)
##
miss_catfeats = cat_feat.isna().sum().sort_values(ascending = False)*(100/2580)
fig, ax = plt.subplots()
# create a barplot
bar_miss_catfeats = ax.bar(miss_catfeats.index[0:9], miss_catfeats[0:9])
# Adding labels and title
ax.set_title('Top 10 cat. feat. with missing values')
ax.set_xlabel('Num. Feature')
ax.set_ylabel('% Missing')
ax.bar_label(bar_miss_catfeats,fmt='%.1f')
# Rotating x-axis labels
ax.tick_params(axis='x', labelrotation=45)
# Displaying the plot
plt.tight_layout() # Adjust layout to prevent clipping
plt.show(block=True)
##
# What data types do we have?
print(housing.dtypes.unique())
##
# dummify, note that get_dummies automatically removes NaN values from categorical columns in final output
# however you still need to manually fill in NaN for non categorical features.
dfpp = pd.get_dummies(housing, columns=cat_feat.columns, drop_first=True)
# Filing NA's with the mean of each column.
dfpp = dfpp.fillna(dfpp.mean())
# Check if any missing values remain
dfpp.isna().mean()>0
print(dfpp.shape)
dfpp.head()
##-------------------------Checking that no data is missing---------------------------------------------
# We only need to check the numerical feats since the categorical ones are allready gauranteed
# to be taken of with the dummifiication.
miss_numfeats = dfpp.isna().sum().sort_values(ascending = False)*(100/2580)
fig, ax = plt.subplots()
# create a barplot
bar_num_feats = ax.bar(miss_numfeats.index[0:9], miss_numfeats[0:9])
# Adding labels and title
ax.set_title('Top 10 feat. with missing values')
ax.set_xlabel('Feature')
ax.set_ylabel('% Missing')
ax.bar_label(bar_num_feats)
# Rotating x-axis labels
ax.tick_params(axis='x', labelrotation=45)
# Displaying the plot
plt.tight_layout() # Adjust layout to prevent clipping
plt.show(block=True)
##
# Split into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(dfpp.drop(columns = 'SalePrice'), dfpp['SalePrice'], test_size=0.4, random_state=0)
##********************************************************************************************
#*************************************Gradient boosting**************************************
#********************************************************************************************
from sklearn.ensemble import GradientBoostingRegressor
GB = GradientBoostingRegressor
results = [] # Initializing array
# You sould only have five entries from MLR vanilla+Lasso+Ridge+Elasic+RF
# up to this point this ensures that, so that when you run this chunk
# multiple times you don't continously append values
# results = results[0:10] # commenting for the webapp dev, uncomment if you want to run entire program as was intended
# learning_rate = Learning rate shrinks the contribution of each tree
# by learning_rate. There is a trade-off between learning_rate and n_estimators.
# n_estimators: # trees in forest
# sub_sample: The fraction of samples to be used for fitting the individual base learners.
# If smaller than 1.0 this results in Stochastic Gradient Boosting.
# Subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to
# a reduction of variance and an increase in bias.
# np.linspace(0.05,1,11)
gbGrid = {'random_state': [0], 'learning_rate': np.linspace(0.05,1,11),
          'n_estimators': range(100,300,25), 'subsample': np.linspace(0.05,1,11)}
# CV = Kfold cross validation. This says how many subgroups to divide up your test data
gbCV = GridSearchCV(GB(), gbGrid, cv = 2, return_train_score = True, n_jobs = 4)
GB_Train = gbCV.fit(x_train,y_train)
results = np.append(results, GB_Train.best_score_)
GB_Test = GB(**gbCV.best_params_).fit(x_test,y_test)
results = np.append(results,GB_Test.score(x_test,y_test))
# ------------------------------------------Saving results------------------------------------------------------------
# Reshaping the array from [n,] to [n/2 ; 2].
results_reshape = results.reshape(int(results.shape[0]/2),2)
resultsDF = pd.DataFrame(data = results_reshape, index=['GB'], columns = ['Train','Test'])
# resultsDF = pd.DataFrame(data = results_reshape, index=['GB'], columns = ['Train','Test'])
# speak('DONE')
## ----------------------Printing Results---------------------------------
print('\nBest hyper-parms:\n',gbCV.best_params_)
resultsDF
#### ----------------------Saving the Model---------------------------------
# with open('GB_Train_Full.pkl','wb') as f:
#     pickle.dump(GB_Train, f)
#### ----------------------Loading the Full Model---------------------------------
# i.e. the model trained on all the features
with open('GB_Test_Full.pkl', 'rb') as f:
    gb_test_full = pickle.load(f)
with open('GB_Train_Full.pkl', 'rb') as f:
    gb_train_full = pickle.load(f)
##
LR = LinearRegression
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color = 'red', linewidth = '3')

predicted = gb_full.predict(x_test)
predicted = predicted.reshape(1032,1)
# print(predicted.shape)
actual = y_test.values.reshape(1032,1)
# print(actual.shape)
toyLR = LR().fit(predicted,actual)
# print("score: ",toyLR.score(predicted,actual))
# print("intercept:\n",toyLR.intercept_)
# print("slope:\n",toyLR.coef_)
# print(predicted.shape)
# y_test.shape
# ----------------Plotting-------------------------
fig, ax = plt.subplots()
gb_vs_actual = ax.scatter(predicted, y_test)
# Adding labels and title
ax.set_xlabel('GB Prediction', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
ax.set_ylabel('Actual', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
ax.set_title('GB vs Actual',{'fontsize': 15, 'fontweight': 'bold'})
# Rotating xtick labels by 90deg
plt.xticks(rotation=90)
# ax.grid(color='black', linestyle='-', linewidth=0.1)
abline(toyLR.coef_.reshape(1,),toyLR.intercept_)
# Displaying the plot
plt.tight_layout() # Adjust layout to prevent clipping
plt.show(block=True)
##----------------------Gradient boosting Feature Importance---------------------------------
# gbCV.best_estimator_.feature_importances_
# create a zipped tuple of training column names and feature importance score.
GB_feat_imp = pd.Series(gb_train_full.best_estimator_.feature_importances_, x_train.columns).sort_values(ascending = False)
# print(feature_importance)
fig, ax = plt.subplots()
bar_GB_FeatImp = ax.bar(GB_feat_imp[0:10].keys(), GB_feat_imp[0:10])
# Adding labels and title
ax.set_title('Top 10 Features (GB model)',{'fontsize': 15, 'fontweight': 'bold'})
ax.set_xlabel('Features', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
ax.set_ylabel('GB Feat. Imp.', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
ax.bar_label(bar_GB_FeatImp, fmt='%.2f')
# Rotating x-axis labels
ax.tick_params(axis='x', rotation=45)
# Displaying the plot
plt.tight_layout() # Adjust layout to prevent clipping
plt.show(block=True)
##----------------------GB Model Top 6 Features---------------------------------
select_feats = dfpp[GB_feat_imp.index.values[0:6]]
results2 = []
x_train2, x_test2, y_train2, y_test2 = train_test_split(select_feats, housing['SalePrice'], test_size=0.4, random_state=0)
#     print(select_feats.columns)
GB_Train2 = gbCV.fit(x_train2,y_train2)
results2 = np.append(results2, GB_Train2.best_score_)
GB_Test2 = GB(**GB_Train2.best_params_).fit(x_test2,y_test2)
results2 = np.append(results2,GB_Test2.score(x_test2,y_test2))
# ------------------------------------------Saving results------------------------------------------------------------
# Reshaping the array from [n,] to [n/2 ; 2].
results_reshape2 = results2.reshape(int(results2.shape[0]/2),2)
resultsDF2 = pd.DataFrame(data = results_reshape2, columns = ['Train','Test'])
# speak('DONE')
resultsDF2
#### ----------------------Saving the 6 feature Model---------------------------------
with open('GB_Train_6feats.pkl','wb') as f:
    pickle.dump(GB_Train2, f)
with open('GB_Test_6feats.pkl', 'wb') as f:
    pickle.dump(GB_Test2, f)
# NOTE: The top 6 features are:
# ['OverallQual' 'GrLivArea' 'TotalBsmtSF' 'BsmtFinSF1' 'GarageArea'
#      'YearBuilt']