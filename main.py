# Random Tips:
# Display the docstring of a function: Shift+Tab
# Commenting: Ctrl + /
# type() checks variable type
# axis = 0 "rows" 1 "columns

##********************************************************************************************
#********************************Python Library Imports***************************************
#*********************************************************************************************
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from IPython.display import display, HTML
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import math as mt
import os
##********************************************************************************************
#********************************Local Library Imports***************************************
#*********************************************************************************************
from Ames_Housing.Libraries import speak


##********************************************************************************************
#***************************************Global Parms******************************************
#*********************************************************************************************

# sound_file = '../data/Beep.mp3'

##********************************************************************************************
#***************************************Global Funcs******************************************
#*********************************************************************************************


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

#********************************************************************************************
#*************************************Pre-processing******************************************
#*********************************************************************************************
##
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
fig, ax = plt.subplots()
p1 = ax.bar(['Num','Cat'], temp_feats) # create a barplot
ax.set_xlabel('Feature Type')
ax.set_ylabel('# of Features')
ax.bar_label(p1, temp_feats)
# NOTE: Pycharm doesn't like plt.show() which tries to open a figure window in "interactive" mode
# instea you need to use the below code:
plt.show(block=True)
# Displays the entire table. .head() will omit columns for very long tables. Uses IPython.display library
# display(HTML(h_num.to_html()))
##
print(num_feat.columns[num_feat.isna().any()]) #Print all colums that have NA values for h_num
##
miss_numfeats = num_feat.isna().sum().sort_values(ascending = False)*(100/2580)
miss_numfeats = miss_numfeats.round(1)
fig, ax = plt.subplots()
p2 = ax.bar(miss_numfeats.index[0:9], miss_numfeats[0:9]) # create a barplot
ax.set_title('Top 10 num. feat. with missing values')
ax.set_xlabel('Num. Feature')
ax.bar_label(p2, miss_numfeats)
# ax.get_xticks(): gets tick positions of current plot
# ax.get_xticklabels(): gets tick labels of current plot
ax.tick_params(axis = 'x', rotation=90)
# ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90, ha='right')
ax.set_ylabel('% Missing')
plt.show(block=True)
##
miss_catfeats = cat_feat.isna().sum().sort_values(ascending = False)*(100/2580)
fig, ax = plt.subplots()
p2 = ax.bar(miss_catfeats.index[0:9], miss_catfeats[0:9]) # create a barplot
ax.set_title('Top 10 cat. feat. with missing values')
ax.set_xlabel('Num. Feature')
# ax.get_xticks(): gets tick positions of current plot
# ax.get_xticklabels(): gets tick labels of current plot
ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90, ha='right')
ax.set_ylabel('% Missing')
plt.show(block=True)

##
# What data types do we have?
print(housing.dtypes.unique())
##
# dummify, note that get_dummies automatically removes NaN values from categorical columns in final output
# however you still need to manually fill in NaN for non categorical features.
dfpp = pd.get_dummies(housing, columns = cat_feat.columns, drop_first = True)
# Filing NA's with the mean of each column.
dfpp = dfpp.fillna(dfpp.mean())
# Check if any missing values remain
dfpp.isna().mean()>0
print(dfpp.shape)
dfpp.head()
##-------------------------Checking that no data is missing---------------------------------------------
miss_numfeats = dfpp.isna().sum().sort_values(ascending = False)*(100/2580)
fig, ax = plt.subplots()
p2 = ax.bar(miss_numfeats.index[0:9], miss_numfeats[0:9]) # create a barplot
ax.set_title('Top 10 feat. with missing values')
ax.set_xlabel('Feature')
# ax.get_xticks(): gets tick positions of current plot
# ax.get_xticklabels(): gets tick labels of current plot
ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90, ha='right')
ax.set_ylabel('% Missing')
plt.show(block=True)
##
# Split into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(dfpp.drop(columns = 'SalePrice'), dfpp['SalePrice'], test_size=0.4, random_state=0)
#********************************************************************************************
#*************************************Feature Importance*************************************
#********************************************************************************************
##---------------------------------------------------------------------------------------
# NOTE: This is a good excersice in creating numpy structred arrays but not the most optimal way to
# create and store arrays for plotting. Just use it as an example.

# f_values, p_values = f_regression(x_train, y_train)
# linear_FI =[]
# # Making p_values a series with indices names by the columns of years df, then sorting by p-value mag.
# # pd.Series(p_values, index=years.columns).sort_values()
# # p_values
# # create a zipped tuple of training column names and feature importance score.
# linear_FI = list(zip(x_train.columns,p_values))
# # define numpy data type for each element in tuple.
# # S20: string of size 20
# # float: self explanatory
# dtype = [('feature', 'S20'), ('importance', 'float')]
# # Create a numpy structured array: https://numpy.org/doc/stable/user/basics.rec.html
# linear_FI = np.array(linear_FI, dtype=dtype)
# # [::-1] reverse order of array
# linearsort_FI = np.sort(linear_FI, order='importance')
# # unpack the zipped array into column names and scores
# name, score = zip(*linearsort_FI)
# # create a DF from unpacked names and scores
# linearFI_DF = pd.DataFrame({'name':name,'score':score})[:15]
# fig, ax = plt.subplots()
# plotFI = ax.bar(linearFI_DF.name[0:9], linearFI_DF.score[0:9]) # create a barplot
# ax.set_title('Top 10 Featues by fregression')
# ax.set_xlabel('Feature')
# # ax.get_xticks(): gets tick positions of current plot
# # ax.get_xticklabels(): gets tick labels of current plot
# ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=90, ha='right')
# ax.set_ylabel('Log(p_value)')
# ax.set_yscale("log")
##
# Unpacking f and p values
f_values, p_values = f_regression(x_train, y_train)
# Converting p values into a pandas series (sorted)
p_values = pd.Series(p_values, index=x_train.columns).sort_values()
# print(p_values)
# creating a plot object with only the top 10 features
plt.bar(p_values[0:11].keys(), abs(np.log10(p_values[0:11])))
# Rotating xtick labels by 90deg
plt.xticks(rotation=90)
plt.xlabel('Features', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.ylabel('log(p_value)', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.title('Top 10 feat. by Linear regression', fontdict = {'fontsize': 15})
plt.show(block=True)
## mutual info tries to find "p-values" for non-linear dependancies amongst data sets.
mi = mutual_info_regression(x_train, y_train)
mi = pd.Series(mi, index = x_train.columns).sort_values(ascending = False)
# creating a plot object with only the top 10 features
plt.bar(mi[0:10].keys(), mi[0:10])
# Rotating xtick labels by 90deg
plt.xticks(rotation=90)
plt.xlabel('Features', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.ylabel('MI', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.title('Top 10 feat. by Non-Linear regression', fontdict = {'fontsize': 15})
plt.show(block=True)
#********************************************************************************************
#*************************************Normalizing SKewed Data********************************
#********************************************************************************************
# Still needs work since data with negative skewness is not easily normalized by using the
# np.log1p(). Leaving it as a future work.
##
# num_feat[['OverallCond','YearBuilt','YearRemodAdd','GarageCars']].hist()
# np.log1p(num_feat[['OverallCond','YearBuilt','YearRemodAdd','GarageCars']]).hist()
# skewed = num_feat.apply(lambda x: skew(x.dropna()))
# skewed.sort_values()
#********************************************************************************************
#*************************************MLR Vanilla********************************************
#********************************************************************************************
##
from sklearn.linear_model import LinearRegression
LR = LinearRegression
results = [] # Initializing array
# Note that MLR doesn't really have any hyper-parameters you can tune.
# so we're only tunning the normalization option, which is technically
# a parameter of MLR, not a hyper-parm.
mlrGrid = {'normalize': [False, True]}
mlrCV = GridSearchCV(LR(), mlrGrid, cv = 2, return_train_score = True, n_jobs = 4, scoring = 'r2')
MLR_Train = mlrCV.fit(x_train,y_train)
# This returns the mean r^2 of the CV splits, not the highest
results = np.append(results,MLR_Train.best_score_)
# "**" this oper. unpacks dictionaries into labels and values. "best_params_" is a dictionary of
# params (labesl) and their values (values)
MLR_Test = LR(**mlrCV.best_params_).fit(x_test,y_test)
results = np.append(results,MLR_Test.score(x_test,y_test))
# -----------------------------Saving results------------------------------------------------------------
# Reshaping the array from [n,] to [n/2 ; 2].
results_reshape = results.reshape(int(results.shape[0]/2),2)
resultsDF = pd.DataFrame(data = results_reshape, index=['MLR'], columns = ['Train','Test'])
# speak('DONE')
##
print('\nBest hyper-parms:\n',MLR_Train.best_params_)
resultsDF
#********************************************************************************************
#*************************************MLR Lasso**********************************************
#********************************************************************************************
##
from sklearn.linear_model import Lasso
LS=Lasso
# You sould only have two entries from MLR vanilla up to this point
# this ensures that, so that when you run this chunk multiple times
# you don't continously append values
results = results[0:2]
# Note that Lasso only has alpha as hyper-parameters you can tune in addition
# to normalization
lassoGrid = {'random_state': [0],'alpha': range(0,100,1), 'normalize': [False, True], 'tol': [1e-1]}
lassoCV = GridSearchCV(LS(), lassoGrid, cv = 2, return_train_score = True, n_jobs = 4)
Lasso_Train = lassoCV.fit(x_train,y_train)
results = np.append(results, Lasso_Train.best_score_)
Lasso_Test = LS(**lassoCV.best_params_).fit(x_test,y_test)
results = np.append(results,Lasso_Test.score(x_test,y_test))
# -----------------------------Saving results------------------------------------------------------------
# Reshaping the array from [n,] to [n/2 ; 2].
results_reshape = results.reshape(int(results.shape[0]/2),2)
resultsDF = pd.DataFrame(data = results_reshape, index=['MLR','Lasso'], columns = ['Train','Test'])
# speak('DONE')
##
print('\nBest hyper-parms:\n',lassoCV.best_params_)
resultsDF
#********************************************************************************************
#*************************************MLR Ridge**********************************************
#********************************************************************************************
##
from sklearn.linear_model import Ridge
RG = Ridge
# You sould only have four entries from MLR vanilla+Lasso up to this point
# this ensures that, so that when you run this chunk multiple times
# you don't continously append values
results = results[0:4]
# Note that Lasso only has alpha as hyper-parameters you can tune in addition
# to normalization
ridgeGrid = {'random_state': [0],'alpha': range(0,100,1), 'normalize': [False, True], 'tol': [1e-1]}
ridgeCV = GridSearchCV(RG(), ridgeGrid, cv = 2, return_train_score = True, n_jobs = 4)
Ridge_Train = ridgeCV.fit(x_train,y_train)
results = np.append(results, Ridge_Train.best_score_)
Ridge_Test = RG(**ridgeCV.best_params_).fit(x_test,y_test)
results = np.append(results,Ridge_Test.score(x_test,y_test))
# -----------------------------Saving results------------------------------------------------------------
# Reshaping the array from [n,] to [n/2 ; 2].
results_reshape = results.reshape(int(results.shape[0]/2),2)
resultsDF = pd.DataFrame(data = results_reshape, index=['MLR','Lasso','Ridge'], columns = ['Train','Test'])
# speak('DONE')
##
print('\nBest hyper-parms:\n',ridgeCV.best_params_)
resultsDF
#********************************************************************************************
#*************************************MLR Elastic Net****************************************
#********************************************************************************************
##
from sklearn.linear_model import ElasticNet
EN = ElasticNet
# You sould only have three entries from MLR vanilla+Lasso+Ridge up to this point
# this ensures that, so that when you run this chunk multiple times
# you don't continously append values
results = results[0:6]
elasticGrid = {'random_state': [0], 'alpha': range(1,100,10), 'l1_ratio': np.linspace(0,1,11), 'normalize': [False, True]}
elasticCV = GridSearchCV(EN(), elasticGrid, cv = 2, return_train_score = True, n_jobs = 4)
Elastic_Train = elasticCV.fit(x_train,y_train)
results = np.append(results, Elastic_Train.best_score_)
Elastic_Test = EN(**elasticCV.best_params_).fit(x_test,y_test)
results = np.append(results,Elastic_Test.score(x_test,y_test))
# -----------------------------Saving results------------------------------------------------------------
# Reshaping the array from [n,] to [n/2 ; 2].
results_reshape = results.reshape(int(results.shape[0]/2),2)
resultsDF = pd.DataFrame(data = results_reshape, index=['MLR','Lasso','Ridge','Elastic'], columns = ['Train','Test'])
# speak('DONE')
##-------------------Printing Results--------------------------
print('\nBest hyper-parms:\n',Elastic_Train.best_params_)
resultsDF
##---------------Plotting ElasticNet Feature Importance-----------------------------------
# Since you actually can't get the coef. of the best prediction from gridsearchCV, then
# we instead use the test prediction with the best params from gridsearchCV
# NOTE: **elasticCV.best_params_ unpacks the dict. into "label = value" pairs so that it
# can be used as input to the ElasticNet function.
elastic_FI = pd.Series(abs(Elastic_Test.coef_), x_train.columns ).sort_values(ascending = False)
plt.bar(elastic_FI[1:10].keys(), elastic_FI[1:10])
# Rotating xtick labels by 90deg
plt.xticks(rotation=90)
plt.xlabel('Features', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.ylabel('Elast Net Coef.', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.show(block=True)
# The values here are the coeff. for the linear elastic model, these don't neccesarily
# predict significance, You can hava a large coeff. for bad normalization. Also, these
# coefficients are how the target varies with the rest held constant, therefore not really
# a significance amongst all inputs. P_values for these models would be better, but they
# are not available yet. Do not infer feature importance from the below histogram.
# speak("DONE")
#********************************************************************************************
#*************************************Random Forrest****************************************
#********************************************************************************************
##
from sklearn import ensemble
RF = ensemble.RandomForestRegressor
# You sould only have four entries from MLR vanilla+Lasso+Ridge+Elasic
# up to this point this ensures that, so that when you run this chunk
# multiple times you don't continously append values
results = results[0:8]
# max_features = Each tree gets the full set of features, but at each node,
#                only a random subset of features is considered.
#                https://sebastianraschka.com/faq/docs/random-forest-feature-subsets.html
#                Sci-kit reccomends m=p for regression problems & m=sqrt(p) for classification
#                You should always tune m regardless of what scikit reccomends (see below link)
#                https://stats.stackexchange.com/questions/324370/references-on-number-of-features-to-use-in-random-forest-regression
# max_samples: ratio of how many features each tree uses [0->1]
# n_estimators: # trees in forest
rfGrid = {'random_state': [0], 'n_estimators': range(10,150,10), 'max_features': range(1,100,10), 'max_samples': np.linspace(0,1,11)} #, 'max_depth':range(3,20,2)}
# CV = Kfold cross validation. This says how many subgroups to divide up your test data
rfCV = GridSearchCV(RF(), rfGrid, cv = 2, return_train_score = True, n_jobs = 4)
RF_Train = rfCV.fit(x_train,y_train)
results = np.append(results, RF_Train.best_score_)
RF_Test = RF(**RF_Train.best_params_).fit(x_test,y_test)
results = np.append(results,RF_Test.score(x_test,y_test))
# ------------------------------------------Saving results------------------------------------------------------------
# Reshaping the array from [n,] to [n/2 ; 2].
results_reshape = results.reshape(int(results.shape[0]/2),2)
resultsDF = pd.DataFrame(data = results_reshape, index=['MLR','Lasso','Ridge','Elastic','RF'], columns = ['Train','Test'])
# speak('DONE')
##--------------------------------Printing Results-------------------------------
print('\nBest hyper-parms:\n',RF_Train.best_params_)
resultsDF
##--------------------Feature importance of RF-----------------------------------
# create a zipped tuple of training column names and feature importance score.
feature_importance = pd.Series(RF_Test.feature_importances_, x_train.columns).sort_values(ascending = False)
# print(feature_importance)
# creating a plot object with only the top 10 features
plt.bar(feature_importance[0:10].keys(), feature_importance[0:10])
# Rotating xtick labels by 90deg
plt.xticks(rotation=90)
plt.xlabel('Features', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.ylabel('RF Feat. Imp.', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.title('Top 10 Features (RF model)',{'fontsize': 15, 'fontweight': 'bold'})
plt.show(block=True)
#********************************************************************************************
#*************************************Gradient boosting**************************************
#********************************************************************************************
##
from sklearn.ensemble import GradientBoostingRegressor
GB = GradientBoostingRegressor
# You sould only have five entries from MLR vanilla+Lasso+Ridge+Elasic+RF
# up to this point this ensures that, so that when you run this chunk
# multiple times you don't continously append values
results = results[0:10]
# learning_rate = Learning rate shrinks the contribution of each tree
# by learning_rate. There is a trade-off between learning_rate and n_estimators.
# n_estimators: # trees in forest
# sub_sample: The fraction of samples to be used for fitting the individual base learners.
# If smaller than 1.0 this results in Stochastic Gradient Boosting.
# Subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to
# a reduction of variance and an increase in bias.
gbGrid = {'random_state': [0], 'learning_rate': np.linspace(0,1,11), 'n_estimators': range(100,300,25), 'subsample': np.linspace(0,1,11)}
# CV = Kfold cross validation. This says how many subgroups to divide up your test data
gbCV = GridSearchCV(GB(), gbGrid, cv = 2, return_train_score = True, n_jobs = 4)
GB_Train = gbCV.fit(x_train,y_train)
results = np.append(results, GB_Train.best_score_)
GB_Test = GB(**gbCV.best_params_).fit(x_test,y_test)
results = np.append(results,GB_Test.score(x_test,y_test))
# ------------------------------------------Saving results------------------------------------------------------------
# Reshaping the array from [n,] to [n/2 ; 2].
results_reshape = results.reshape(int(results.shape[0]/2),2)
resultsDF = pd.DataFrame(data = results_reshape, index=['MLR','Lasso','Ridge','Elastic','RF','GB'], columns = ['Train','Test'])
# resultsDF = pd.DataFrame(data = results_reshape, index=['GB'], columns = ['Train','Test'])
# speak('DONE')
## ----------------------Printing Results---------------------------------
print('\nBest hyper-parms:\n',gbCV.best_params_)
resultsDF
# gbCV
##
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color = 'red', linewidth = '3')

predicted = GB_Test.predict(x_test)
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
plt.scatter(predicted, y_test)
# Rotating xtick labels by 90deg
plt.xticks(rotation=90)
plt.xlabel('GB PRediction', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.ylabel('Actual', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.title('GB vs Actual',{'fontsize': 15, 'fontweight': 'bold'})
plt.grid(color='black', linestyle='-', linewidth=0.1)
abline(toyLR.coef_.reshape(1,),toyLR.intercept_)
plt.show(block=True)
##----------------------Gradient boosting Feature Importance---------------------------------
# gbCV.best_estimator_.feature_importances_
# create a zipped tuple of training column names and feature importance score.
GB_feat_imp = pd.Series(GB_Train.best_estimator_.feature_importances_, x_train.columns).sort_values(ascending = False)
# print(feature_importance)
# creating a plot object with only the top 10 features
plt.bar(feature_importance[0:10].keys(), feature_importance[0:10])
# Rotating xtick labels by 90deg
plt.xticks(rotation=90)
plt.xlabel('Features', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.ylabel('GB Feat. Imp.', fontweight= None, color = 'black', fontsize='17', horizontalalignment='center')
plt.title('Top 10 Features (GB model)',{'fontsize': 15, 'fontweight': 'bold'})
plt.show(block=True)
#********************************************************************************************
#*************************************Feature Selection**************************************
#********************************************************************************************
##
GB_feat_imp.index.values[0:15]
##
GB_feat_imp = pd.Series(GB_Test.feature_importances_, x_train.columns).sort_values(ascending = False)
GB_feat_imp.index.values[0:15]
select = []
results2 = []
type(select)
for l in GB_feat_imp.index.values[0:15]:
    select.append(l)
    select_feats = dfpp[select] # Selecting feats from dummified dataset
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
##
resultsDF2.iloc[:,1]
# unpacks the plot into a fig and ax container.
# Fig continas the plot which you can save as a picture
# ax contains the important stuff, like the axes, legend, type of plot, etc...
fig2, ax2 = plt.subplots()
p1 = ax2.plot(resultsDF2.index, resultsDF2.iloc[:,0], label = 'Train') # create a simple scatter
p2 = ax2.plot(resultsDF2.index, resultsDF2.iloc[:,1], label = 'Test') # create a simple scatter
p3 = ax2.plot(np.array(ax2.get_xlim()), [0.9887,0.9887], label = 'Target') # plot horizontal line
ax2.set_xlabel('# of Features')
ax2.set_ylabel('R^2')
ax2.set_title('GB w/ incres. # of features')
ax2.legend() # Displays the legend
plt.show(block=True)