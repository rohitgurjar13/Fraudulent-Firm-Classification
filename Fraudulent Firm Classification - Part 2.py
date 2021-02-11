#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils


# Importing the data

# In[2]:


audit=pd.read_csv('audit_risk.csv')
trial=pd.read_csv('trial.csv')


# In[3]:


audit.shape


# In[4]:


audit.head()


# In[5]:


trial.shape


# In[6]:


trial.head()


# In[7]:


audit.info()


# In[8]:


trial.info()


# Finding the common columns in Two given datasets

# In[9]:


audit_columns = set(audit.columns)
trial_columns = set(trial.columns)

audit_columns.intersection(trial_columns)


# In[10]:


trial=trial.drop_duplicates(keep='first')
audit=audit.drop_duplicates(keep='first')


# Merging the Audit_Risk and Trials on common columns using Inner Join

# In[11]:


result = pd.merge(audit,trial, how = 'inner', on = ['History', 'LOCATION_ID', 'Money_Value','PARA_A','PARA_B','Score','Sector_score','TOTAL','numbers'], sort = False)


# In[12]:


result.shape


# In[13]:


l=['LOHARU','NUH','SAFIDON']

for item in l:
    result=result[result.LOCATION_ID != item]


# Converting the LOCATION_ID column with replaced string values to a float column
# 

# In[14]:


result['LOCATION_ID'] = result['LOCATION_ID'].astype(float)


# In[15]:


result.isnull().sum()


# Imputing the missing value in the Money_Value column
# 

# In[16]:


result['Money_Value'] = result['Money_Value'].replace('', np.nan)
result['Money_Value'] = result['Money_Value'].replace(np.nan,result.Money_Value.mean())


# In[17]:


result.isnull().sum()


# In[18]:


result.shape


# Correlation Matrix

# In[19]:


result.corr().style.format("{:.2}")


# Statistical Summary

# In[20]:


result.describe().T


# Examining the merged data and the correlation matrix, it is clearly evident that there are several columns with similar data having different scales. In such case, there is a need to eliminate one of the two existing similar columns.
# Following is the list of similar columns:
# 1."Score_A" and "SCORE_A"
# 2."Score_B" and "SCORE_B"
# 3."Score_MV" and "MONEY_Marks"
# 4."District_Loss" and "District"

# In[21]:


result=result.drop(['SCORE_A','SCORE_B','MONEY_Marks','District'], axis = 1)


# From the Statistical Summary, is can be seen that the "Detection_Risk" column has same value for all the observations. Thus, this column should be eliminated.

# In[22]:


result=result.drop(['Detection_Risk'], axis = 1)


# Removing the outliers from Audit_Risk column 

# In[23]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.025)
    q3 = df_in[col_name].quantile(0.95)
    fence_low  = q1
    fence_high = q3
    df_in = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_in
result=remove_outlier(result, 'Audit_Risk')


# In[24]:


result.info()


# In[25]:


Y=result['Audit_Risk']
X=result.drop(['Audit_Risk','Risk_x','Risk_y'], axis = 1)
X.shape


# Scaling is an essential step before fitting the models as most estimators are developed with an assumption 
# that the features involved vary on a comparable scale. Since it is evident from the above distribution plots that the 
# distribution of data is not Gaussian, MinMax Scaler is a better fit for our data. After MinMax scaling, we have smaller 
# standard deviations in our data and we therefore end up with suppressed effect of outliers.

# In[26]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train_org, X_test_org, y_train, y_test = train_test_split(X, Y, random_state = 0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# ## ENSEMBLE MODELS (REGRESSION)

# #### 1st BAGGING MODEL (KNN REGRESSOR)

# In[27]:


#Finding the best model for KNN Regressor
param_grid = {'n_neighbors': np.arange(1, 10)}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
grid_searchknn = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, return_train_score=True)

grid_searchknn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(grid_searchknn.score(X_test, y_test)))

print("Best parameters: {}".format(grid_searchknn.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_searchknn.best_score_))


# In[28]:


#base model (Using the best model of KNN Regressor)
Best_KNNR = KNeighborsRegressor(n_neighbors = grid_searchknn.best_params_['n_neighbors'])


# In[29]:


#Grid Search for finding the best number of estimators
from sklearn.ensemble import BaggingRegressor
bag_knn = BaggingRegressor(Best_KNNR, bootstrap = True,max_samples=100, oob_score=True,random_state=0)
param_grid = {'n_estimators': [50,100,200]}
grid_bag_knn = GridSearchCV(bag_knn, param_grid, cv = 5,return_train_score=True)
grid_bag_knn.fit(X_train, y_train)


# In[30]:


print("Best parameters: {}".format(grid_bag_knn.best_params_))
print("Best cross-validation accuracy: {:.2f}".format(grid_bag_knn.best_score_))
print("Train Set Score: {}".format(grid_bag_knn.score(X_train, y_train)))
print("Test Set Score: {}".format(grid_bag_knn.score(X_test,y_test)))


# Building the bagging regressor model with best parameters

# In[31]:


bag_reg = BaggingRegressor(Best_KNNR, n_estimators = 200, max_samples = 100, bootstrap = True, oob_score = True, random_state=0)

#train the model
bag_reg.fit(X_train, y_train)

#model attributes
print("Out of bag score: ",bag_reg.oob_score_)
print("Train score: ", bag_reg.score(X_train,y_train))
print("Test score: ", bag_reg.score(X_test,y_test))


# #### 2nd BAGGING MODEL (LINEAR REGRESSOR)

# In[32]:


#Grid Search for finding the best number of estimators
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

#base model
LINR = LinearRegression()

bag_lin = BaggingRegressor(LINR, bootstrap = True, max_samples=100, oob_score=True,random_state=0)
param_grid = {'n_estimators': [50,100,200]}
grid_bag_lin = GridSearchCV(bag_lin, param_grid, cv = 5, return_train_score=True)
grid_bag_lin.fit(X_train, y_train)


# In[33]:


print("Best parameters: {}".format(grid_bag_lin.best_params_))
print("Best cross-validation accuracy: {:.2f}".format(grid_bag_lin.best_score_))
print("Train Set Score: {}".format(grid_bag_lin.score(X_train, y_train)))
print("Test Set Score: {}".format(grid_bag_lin.score(X_test,y_test)))


# Building the bagging regressor model with best parameters

# In[34]:


bag_reg1 = BaggingRegressor(LINR, n_estimators = 200, max_samples = 100, bootstrap = True, oob_score = True, random_state=0)

#train the model
bag_reg1.fit(X_train, y_train)

#model attributes
print("Out of Bag score: ", bag_reg1.oob_score_)
print("Train score: ", bag_reg1.score(X_train,y_train))
print("Test score: ", bag_reg1.score(X_test,y_test))


# #### 1st PASTING MODEL (DECISION TREE REGRESSOR)

# In[35]:


#Finding the best model for Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

param_grid = {'max_depth':[1, 2, 3, 4, 5, 6]}
dtree = DecisionTreeRegressor()

grid_tree = GridSearchCV(dtree, param_grid, cv = 5, return_train_score=True, n_jobs = -1)
grid_tree.fit(X_train, y_train)

print("Test set score: {:.2f}".format(grid_tree.score(X_test, y_test)))

print("Best parameters: {}".format(grid_tree.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_tree.best_score_))


# In[36]:


#Base model (Using the best model of Decision Tree Regressor)
Best_DTR = DecisionTreeRegressor(max_depth = grid_tree.best_params_['max_depth'])


# In[37]:


#Grid Search for finding the best number of estimators
from sklearn.ensemble import BaggingRegressor
bag_dtr = BaggingRegressor(Best_DTR, bootstrap = True,max_samples=100,random_state=0)
param_grid = {'n_estimators': [50,100,200]}
grid_bag_dtr = GridSearchCV(bag_dtr, param_grid, cv = 5,return_train_score=True)
grid_bag_dtr.fit(X_train, y_train)


# In[38]:


print("Best parameters: {}".format(grid_bag_dtr.best_params_))
print("Best cross-validation accuracy: {:.2f}".format(grid_bag_dtr.best_score_))
print("Train Set Score: {}".format(grid_bag_dtr.score(X_train, y_train)))
print("Test Set Score: {}".format(grid_bag_dtr.score(X_test,y_test)))


# Building the pasting regressor model with best parameters

# In[39]:


pas_reg = BaggingRegressor(Best_DTR, n_estimators = 100, max_samples = 100, bootstrap = False, random_state=0)

#train the model
pas_reg.fit(X_train, y_train)

#model attributes
print("Train score: ", pas_reg.score(X_train,y_train))
print("Test score: ", pas_reg.score(X_test,y_test))


# #### 2nd PASTING MODEL (RIDGE)

# In[40]:


#Finding the best model for Ridge Regressor

param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
grid_searchrid = GridSearchCV(Ridge(), param_grid, cv=5, return_train_score=True)

grid_searchrid.fit(X_train, y_train)

print("Test set score: {:.2f}".format(grid_searchrid.score(X_test, y_test)))

print("Best parameters: {}".format(grid_searchrid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_searchrid.best_score_))


# In[41]:


#Base model (Using the best model of Ridge Regressor)
Best_RID = Ridge(alpha = grid_searchrid.best_params_['alpha'])


# In[42]:


#Grid Search for finding the best number of estimators
from sklearn.ensemble import BaggingRegressor
bag_rid = BaggingRegressor(Best_RID, bootstrap = True,max_samples=100,random_state=0)
param_grid = {'n_estimators': [50,100,200]}
grid_bag_rid = GridSearchCV(bag_rid, param_grid, cv = 5,return_train_score=True)
grid_bag_rid.fit(X_train, y_train)


# In[43]:


print("Best parameters: {}".format(grid_bag_rid.best_params_))
print("Best cross-validation accuracy: {:.2f}".format(grid_bag_rid.best_score_))
print("Train Set Score: {}".format(grid_bag_rid.score(X_train, y_train)))
print("Test Set Score: {}".format(grid_bag_rid.score(X_test,y_test)))


# Building the pasting regressor model with best parameters

# In[44]:


pas_reg1 = BaggingRegressor(Best_RID, n_estimators = 200, max_samples = 100, bootstrap = False, random_state=0)

#train the model
pas_reg1.fit(X_train, y_train)

#model attributes
print("Train score: ", pas_reg1.score(X_train,y_train))
print("Test score: ", pas_reg1.score(X_test,y_test))


# #### 1st ADABOOST BOOSTING MODEL (DECISION TREE REGRESSOR)

# In[45]:


#Finding the best model for Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

param_grid = {'max_depth':[1, 2, 3, 4, 5, 6]}
dtree = DecisionTreeRegressor()

grid_tree = GridSearchCV(dtree, param_grid, cv = 5, return_train_score=True, n_jobs = -1)
grid_tree.fit(X_train, y_train)

print("Test set score: {:.2f}".format(grid_tree.score(X_test, y_test)))

print("Best parameters: {}".format(grid_tree.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_tree.best_score_))


# In[46]:


#Base model (Using the best model of Decision Tree Regressor)
Best_DTR = DecisionTreeRegressor(max_depth = grid_tree.best_params_['max_depth'])


# In[47]:


from sklearn.ensemble import AdaBoostRegressor

adbst_dtr = AdaBoostRegressor(Best_DTR,random_state=0)
param_grid = {'learning_rate': [0.01,0.03,0.1,0.3,1.0],
              'n_estimators': [50,100,200]}
grid_adbst_dtr = GridSearchCV(adbst_dtr, param_grid, cv = 5,return_train_score=True)
grid_adbst_dtr.fit(X_train, y_train)


# In[48]:


print("Best parameters: {}".format(grid_adbst_dtr.best_params_))
print("Best cross-validation accuracy: {:.2f}".format(grid_adbst_dtr.best_score_))
print("Train Set Score: {}".format(grid_adbst_dtr.score(X_train, y_train)))
print("Test Set Score: {}".format(grid_adbst_dtr.score(X_test,y_test)))


# Building the adaboost regressor model with best parameters

# In[49]:


ada_reg = AdaBoostRegressor(Best_DTR, n_estimators=100, learning_rate=0.1, random_state=0)

#train the model
ada_reg.fit(X_train, y_train)

#model attributes
print("Train score: ", ada_reg.score(X_train,y_train))
print("Test score: ", ada_reg.score(X_test,y_test))


# #### 2nd ADABOOST BOOSTING MODEL (LASSO)

# In[50]:


#Finding the best model for Lasso Regressor

param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
grid_searchlas = GridSearchCV(Lasso(), param_grid, cv=5, return_train_score=True)

grid_searchlas.fit(X_train, y_train)

print("Test set score: {:.2f}".format(grid_searchlas.score(X_test, y_test)))

print("Best parameters: {}".format(grid_searchlas.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_searchlas.best_score_))


# In[51]:


#Base model (Using the best model of Lasso Regressor)
Best_LAS = Lasso(alpha = grid_searchlas.best_params_['alpha'])


# In[52]:


from sklearn.ensemble import AdaBoostRegressor

adbst_las = AdaBoostRegressor(Best_DTR,random_state=0)
param_grid = {'learning_rate': [0.01,0.03,0.1,0.3,1.0],
              'n_estimators': [50,100,200]}
grid_adbst_las = GridSearchCV(adbst_las, param_grid, cv = 5,return_train_score=True)
grid_adbst_las.fit(X_train, y_train)


# In[53]:


print("Best parameters: {}".format(grid_adbst_las.best_params_))
print("Best cross-validation accuracy: {:.2f}".format(grid_adbst_las.best_score_))
print("Train Set Score: {}".format(grid_adbst_las.score(X_train, y_train)))
print("Test Set Score: {}".format(grid_adbst_las.score(X_test,y_test)))


# Building the adaboost regressor model with best parameters

# In[54]:


ada_reg1 = AdaBoostRegressor(Best_LAS, n_estimators=100, learning_rate=0.1, random_state=0)

#train the model
ada_reg1.fit(X_train, y_train)

#model attributes
print("Train score: ", ada_reg1.score(X_train,y_train))
print("Test score: ", ada_reg1.score(X_test,y_test))


# #### GRADIENT BOOSTING REGRESSOR MODEL

# In[55]:


from  sklearn.ensemble import GradientBoostingRegressor

gbrt_lreg = GradientBoostingRegressor(random_state=0)
param_grid = {'max_depth':[1,2,5],
              'learning_rate' : [0.01,0.03,0.1,0.3,1.0],
              'n_estimators': [50,100,200]}
grid_gbrt_lreg = GridSearchCV(gbrt_lreg, param_grid, cv = 5,return_train_score=True)
grid_gbrt_lreg.fit(X_train, y_train)


# In[56]:


print("Best cross-validation accuracy: {:.2f}".format(grid_gbrt_lreg.best_score_))
print("Best parameters: {}".format(grid_gbrt_lreg.best_params_))
print("Train Set Score: {}".format(grid_gbrt_lreg.score(X_train, y_train)))
print("Test Set Score: {}".format(grid_gbrt_lreg.score(X_test,y_test)))


# ## Principal Component Analysis

# The main idea of principal component analysis (PCA) here is to reduce the dimensionality of a data set consisting of many variables correlated with each other, either heavily or lightly, while retaining 95% of the variation present in the dataset.

# In[57]:


from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("{} PCA components are selected to explain atleast 95% of the total variation in data".format(pca.n_components_))


# In[58]:


print("Variation explained by each PCA component is: \n {}".format(pca.explained_variance_ratio_))
print("Total data variation explained is: \n {}".format(pca.explained_variance_ratio_.sum()))


# #### LINEAR REGRESSION

# In[59]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lreg = LinearRegression()
lreg.fit(X_train_pca, y_train)
y_pred = lreg.predict(X_test_pca)
print("R^2: {}".format(lreg.score(X_test_pca, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
print("Train Score: {}".format(lreg.score(X_train_pca, y_train)))
print("Test Score: {}".format(lreg.score(X_test_pca, y_test)))


# In[60]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

X_train_rm = X_train_pca[:,4].reshape(-1,1)
lreg.fit(X_train_rm, y_train)
y_predict = lreg.predict(X_train_rm)

plt.plot(X_train_rm, y_predict, c = 'r')
plt.scatter(X_train_rm,y_train)
plt.xlabel('RM')


# Linear regressor using Cross-Validation

# In[61]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from sklearn.metrics import mean_squared_error
linreg = LinearRegression()
linreg.fit(X_train_pca, y_train)
kfold = KFold(n_splits=5)
scoring={'neg_mean_squared_error':'neg_mean_squared_error','r2':'r2'}

cv_scores_linreg = cross_validate(linreg, X, Y, cv=kfold, scoring = scoring, return_train_score=True)

#print("Mean 5-Fold CV Score: {}".format(np.mean(cv_scores_linreg)))
print("Train Score: {}".format(cv_scores_linreg['train_r2']))
print("Mean 5-Fold CV Train Score: {}".format(np.mean(cv_scores_linreg['train_r2'])))
print("Test Score: {}".format(cv_scores_linreg['test_r2']))
print("Mean 5-Fold CV Test Score: {}".format(np.mean(cv_scores_linreg['test_r2'])))


# #### LASSO MODEL

# In[62]:


from  sklearn.linear_model import Lasso

x_range = [0.01, 0.1, 1, 10, 100]
train_score_list = []
test_score_list = []

for alpha in x_range: 
    lasso = Lasso(alpha)
    lasso.fit(X_train_pca,y_train)
    train_score_list.append(lasso.score(X_train_pca,y_train))
    test_score_list.append(lasso.score(X_test_pca, y_test))


# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(x_range, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_range, test_score_list, c = 'b', label = 'Test Score')
plt.xscale('log')
plt.legend(loc = 3)
plt.xlabel(r'$\alpha$')
print("Train Score List:{}".format(train_score_list))
print("Test Score List:{}".format(test_score_list))


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')

x_range1 = np.linspace(0.001, 1, 1000).reshape(-1,1)
x_range2 = np.linspace(1, 1000, 1000).reshape(-1,1)

x_range = np.append(x_range1, x_range2)
coeff = []

for alpha in x_range: 
    lasso = Lasso(alpha)
    lasso.fit(X_train_pca,y_train)
    coeff.append(lasso.coef_ )
    
coeff = np.array(coeff)

for i in range(0,9):
    plt.plot(x_range, coeff[:,i], label = i)

plt.axhline(y=0, xmin=0.001, xmax=9999, linewidth=1, c ='gray')
plt.xlabel(r'$\alpha$')
plt.xscale('log')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
          ncol=5, fancybox=True, shadow=True)
plt.show()


# GridSearch with Cross-Validation (Finding the Best parameter)

# In[65]:


param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
grid_search = GridSearchCV(Lasso(), param_grid, cv=5, return_train_score=True)

grid_search.fit(X_train_pca, y_train)

print("Test set score: {:.2f}".format(grid_search.score(X_test_pca, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# #### RIDGE

# In[66]:


from  sklearn.linear_model import Ridge

x_range = [0.01, 0.1, 1, 10, 100]
train_score_list = []
test_score_list = []

for alpha in x_range: 
    ridge = Ridge(alpha)
    ridge.fit(X_train_pca,y_train)
    train_score_list.append(ridge.score(X_train_pca,y_train))
    test_score_list.append(ridge.score(X_test_pca, y_test))


# In[67]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(x_range, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_range, test_score_list, c = 'b', label = 'Test Score')
plt.xscale('log')
plt.legend(loc = 3)
plt.xlabel(r'$\alpha$')
print("Train Score List:{}".format(train_score_list))
print("Test Score List:{}".format(test_score_list))


# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

x_range1 = np.linspace(0.001, 1, 100).reshape(-1,1)
x_range2 = np.linspace(1, 10000, 10000).reshape(-1,1)

x_range = np.append(x_range1, x_range2)
coeff = []

for alpha in x_range: 
    ridge = Ridge(alpha)
    ridge.fit(X_train_pca,y_train)
    coeff.append(ridge.coef_ )
    
coeff = np.array(coeff)

for i in range(0,9):
    plt.plot(x_range, coeff[:,i], label = 'feature {:d}'.format(i))

plt.axhline(y=0, xmin=0.001, xmax=9999, linewidth=1, c ='gray')
plt.xlabel(r'$\alpha$')
plt.xscale('log')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
          ncol=3, fancybox=True, shadow=True)
plt.show()


# GridSearch with Cross-Validation (Finding the Best parameter)

# In[69]:


param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
kfold = KFold(5)
grid_search = GridSearchCV(Ridge(), param_grid, cv=kfold, return_train_score=True)

grid_search.fit(X_train_pca, y_train)

print("Test set score: {:.2f}".format(grid_search.score(X_test_pca, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# #### KNN REGRESSOR

# In[70]:


from sklearn.neighbors import KNeighborsRegressor

get_ipython().run_line_magic('matplotlib', 'inline')
train_score_list = []
test_score_list = []

for k in range(1,10):
    knn_reg = KNeighborsRegressor(k)
    knn_reg.fit(X_train_pca, y_train)
    train_score_list.append(knn_reg.score(X_train_pca, y_train))
    test_score_list.append(knn_reg.score(X_test_pca, y_test))

x_axis = range(1,10)
plt.plot(x_axis, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_axis, test_score_list, c = 'b', label = 'Test Score')
plt.legend()
plt.xlabel('k')
plt.ylabel('MSE')
print("Train Score List:{}".format(train_score_list))
print("Test Score List:{}".format(test_score_list))


# GridSearch with Cross-Validation (Finding the Best parameter)

# In[71]:


param_grid = {'n_neighbors': np.arange(1, 10)}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, return_train_score=True)

grid_search.fit(X_train_pca, y_train)

print("Test set score: {:.2f}".format(grid_search.score(X_test_pca, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# #### POLYNOMIAL REGRESSOR

# In[72]:


from  sklearn.preprocessing  import PolynomialFeatures

train_score_list = []
test_score_list = []

for n in range(1,3):
    poly = PolynomialFeatures(n)
    X_train_poly = poly.fit_transform(X_train_pca)
    X_test_poly = poly.transform(X_test_pca)
    lreg.fit(X_train_poly, y_train)
    train_score_list.append(lreg.score(X_train_poly, y_train))
    test_score_list.append(lreg.score(X_test_poly, y_test))
    


# In[73]:


get_ipython().run_line_magic('matplotlib', 'inline')

x_axis = range(1,3)
plt.plot(x_axis, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_axis, test_score_list, c = 'b', label = 'Test Score')
plt.xlabel('degree')
plt.ylabel('accuracy')
plt.legend()
print("Train Score List:{}".format(train_score_list))
print("Test Score List:{}".format(test_score_list))


# In[74]:


poly = PolynomialFeatures(n)

X_train_1 = X_train_pca[:,3].reshape(-1,1)
X_train_poly = poly.fit_transform(X_train_1)
lreg.fit(X_train_poly, y_train)

x_axis = np.linspace(0,1,100).reshape(-1,1)
x_poly = poly.transform(x_axis)
y_predict = lreg.predict(x_poly)

plt.scatter(X_train_1,y_train)
plt.plot(x_axis, y_predict, c = 'r')


# Naive GridSearch (Finding the Best parameter)

# In[75]:


from  sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LinearRegression

best_score=0
degrees = np.arange(1, 5)
rmses = []
min_rmse, min_deg = 1e10, 0

for deg in degrees:
    poly = PolynomialFeatures(deg, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_pca)
    X_test_poly = poly.transform(X_test_pca)
    lreg = LinearRegression()
    lreg.fit(X_train_poly, y_train)
    poly_predict = lreg.predict(X_test_poly)
    score=lreg.score(X_test_poly, y_test)
    poly_mse = mean_squared_error(y_test, poly_predict)
    poly_rmse = np.sqrt(poly_mse)
    rmses.append(poly_rmse)
     # Cross-validation of degree
    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        min_deg = deg
        
# Plot and present results
print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(degrees, rmses)
ax.set_yscale('log')
ax.set_xlabel('Degree')
ax.set_ylabel('RMSE')


# GridSearch with Cross-Validation (Finding the Best parameter)

# In[76]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


# In[77]:


from sklearn.model_selection import GridSearchCV

param_grid = {'polynomialfeatures__degree': np.arange(1,3),
              'linearregression__fit_intercept': [True, False],
              'linearregression__normalize': [True, False]}

grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)


# In[78]:


grid.fit(X_train_pca,y_train)


# In[79]:


print("Test set score: {:.2f}".format(grid.score(X_test_pca, y_test)))


# In[80]:


print("Best parameters: {}".format(grid.best_params_))
print("Best cross-validation score: {:.2f}".format(grid.best_score_))


# #### Linear SVR

# In[81]:


from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score
LSVR = LinearSVR()
from sklearn.model_selection import KFold
kfold = KFold(5)

scores = cross_val_score(LSVR, X, Y, cv = kfold)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))


# GridSearch with Cross-Validation (Finding the Best parameter)

# In[82]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
grid_search = GridSearchCV(LinearSVR(), param_grid, cv=5, return_train_score=True)

grid_search.fit(X_train_pca, y_train)

print("Test set score: {:.2f}".format(grid_search.score(X_test_pca, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# #### SVR with Kernel rbf

# In[83]:


from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
SVR_rbf = SVR(kernel='rbf', gamma=0.1, C=100)
from sklearn.model_selection import KFold
kfold = KFold(5)

scores = cross_val_score(SVR_rbf, X, Y, cv = kfold)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))


# GridSearch with Cross-Validation (Finding the Best parameter)

# In[84]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, return_train_score=True)

grid_search.fit(X_train_pca, y_train)

print("Test set score: {:.2f}".format(grid_search.score(X_test_pca, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# #### SVR with Kernel linear

# In[85]:


from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
SVR_linear = SVR(kernel='linear',C=1)
from sklearn.model_selection import KFold
kfold = KFold(5)

scores = cross_val_score(SVR_linear, X, Y, cv = kfold)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))


# GridSearch with Cross-Validation (Finding the Best parameter)

# In[86]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
grid_search = GridSearchCV(SVR(kernel='linear'), param_grid, cv=5, return_train_score=True)

grid_search.fit(X_train_pca, y_train)

print("Test set score: {:.2f}".format(grid_search.score(X_test_pca, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# #### SVR with Kernel poly

# GridSearch with Cross-Validation (Finding the Best parameter)

# In[87]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'degree': [1, 2, 3, 4]}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
grid_search = GridSearchCV(SVR(kernel='poly',gamma='auto'), param_grid, cv=5, return_train_score=True)

grid_search.fit(X_train_pca, y_train)

print("Test set score: {:.2f}".format(grid_search.score(X_test_pca, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# ## DEEP LEARNING MODELS (REGRESSION)

# In[88]:


import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# #### SIMPLE PERCEPTRON

# In[89]:


n_col = X_train.shape[1]

#step 1: build model
model_1 = Sequential()
#input layer
model_1.add(Dense(10, input_dim = n_col, activation = 'relu'))
#hidden layers
#NO hidden layers are added in this model
#output layer(No activation function)
model_1.add(Dense(1))

#step 2: make computational graph - compile
model_1.compile(loss= 'mse' , optimizer = 'sgd', metrics = ['mse'] )

#step 3: train the model - fit
model_1.fit(X_train, y_train, epochs = 50, batch_size = 100)


# In[90]:


model_1.evaluate(X_train, y_train)


# In[91]:


model_1.evaluate(X_test, y_test)


# In[92]:


from sklearn.metrics import r2_score
print("Train score: ", r2_score(y_train, model_1.predict(X_train)))


# In[93]:


print("Test score: ", r2_score(y_test, model_1.predict(X_test)))


# #### MULTI LAYER PERCEPTRON

# In[94]:


#step 1: build model
model_2 = Sequential()
#input layer
model_2.add(Dense(20, input_dim = n_col, activation = 'relu'))
#hidden layers
model_2.add(Dense(20, activation = 'relu'))
model_2.add(Dense(15, activation = 'relu'))
#output layer(No activation function)
model_2.add(Dense(1))

#step 2: make computational graph - compile
model_2.compile(loss= 'mse' , optimizer = 'sgd', metrics = ['mse'] )

#step 3: train the model - fit
model_2.fit(X_train, y_train, epochs = 50, batch_size = 100)


# In[95]:


model_2.evaluate(X_train, y_train)


# In[96]:


model_2.evaluate(X_test, y_test)


# In[97]:


from sklearn.metrics import r2_score
print("Train score: ", r2_score(y_train, model_2.predict(X_train)))


# In[98]:


print("Test score: ", r2_score(y_test, model_2.predict(X_test)))


# # CLASSIFICATION

# Selecting the target column for classification

# In[99]:


Z=result['Risk_x']
X=result.drop(['Audit_Risk','Risk_x', 'Risk_y'], axis = 1)
X.shape


# SCALING THE DATA USING MINMAX SCALER

# In[100]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train_org, X_test_org, z_train, z_test = train_test_split(X, Z, random_state = 0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# #### 1st BAGGING MODEL (KNN CLASSIFIER)

# In[101]:


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors': np.arange(1, 10)}
print("Parameter grid:\n{}".format(param_grid))

grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring = 'recall')

grid_search_knn.fit(X_train, z_train)

print("Test set score: {:.2f}".format(grid_search_knn.score(X_test, z_test)))

print("Best parameters: {}".format(grid_search_knn.best_params_))
print("Best cross-validation score (recall): {:.2f}".format(grid_search_knn.best_score_))


# In[102]:


#base model (Using the best model of KNN Classifier)
Best_KNNC = KNeighborsClassifier(n_neighbors = grid_search_knn.best_params_['n_neighbors'])


# In[103]:


#Grid Search for finding the best number of estimators
from sklearn.ensemble import BaggingClassifier
bag_knnc = BaggingClassifier(Best_KNNC, bootstrap = True,max_samples=100, oob_score=True,random_state=0)
param_grid = {'n_estimators': [50,100,200]}
grid_bag_knnc = GridSearchCV(bag_knnc, param_grid, cv = 5, scoring = 'recall', return_train_score=True)
grid_bag_knnc.fit(X_train, z_train)


# In[104]:


print("Best parameters: {}".format(grid_bag_knnc.best_params_))
print("Best cross-validation (recall): {:.2f}".format(grid_bag_knnc.best_score_))
print("Train Set Score: {}".format(grid_bag_knnc.score(X_train, z_train)))
print("Test Set Score: {}".format(grid_bag_knnc.score(X_test,z_test)))


# In[105]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

bag_class = BaggingClassifier(Best_KNNC, n_estimators = 50, max_samples = 100, bootstrap = True, oob_score = True, random_state=0)

#train the model
bag_class.fit(X_train, z_train)

# Predict
best_bag_pred = bag_class.predict(X_test)

print("Accuracy: {:.3f}".format(accuracy_score(z_test, best_bag_pred)))
print("f1 score: {:.3f}".format(f1_score(z_test, best_bag_pred)))
print("Precision:",metrics.precision_score(z_test, best_bag_pred))
print("Recall:",metrics.recall_score(z_test, best_bag_pred))
print("Confusion matrix:\n{}".format(confusion_matrix(z_test, best_bag_pred)))


# #### 2nd BAGGING MODEL (LOGISTIC REGRESSION)

# In[106]:


param_grid={"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], "penalty":["l1","l2"]}
print("Parameter grid:\n{}".format(param_grid))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression 
grid_search_log = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring = 'recall')

grid_search_log.fit(X_train, z_train)

print("Test set score: {:.2f}".format(grid_search_log.score(X_test, z_test)))

print("Best parameters: {}".format(grid_search_log.best_params_))
print("Best cross-validation score (recall): {:.2f}".format(grid_search_log.best_score_))


# In[107]:


#base model (Using the best model of Logistic Regression Classifier)
Best_log = LogisticRegression( C = grid_search_log.best_params_['C'], penalty = grid_search_log.best_params_['penalty'] )


# In[108]:


#Grid Search for finding the best number of estimators
from sklearn.ensemble import BaggingClassifier
bag_log = BaggingClassifier(Best_log, bootstrap = True,max_samples=100, oob_score=True,random_state=0)
param_grid = {'n_estimators': [50,100,200]}
grid_bag_log = GridSearchCV(bag_log, param_grid, cv = 5, scoring = 'recall',return_train_score=True)
grid_bag_log.fit(X_train, z_train)


# In[109]:


print("Best parameters: {}".format(grid_bag_log.best_params_))
print("Best cross-validation (recall): {:.2f}".format(grid_bag_log.best_score_))
print("Train Set Score: {}".format(grid_bag_log.score(X_train, z_train)))
print("Test Set Score: {}".format(grid_bag_log.score(X_test,z_test)))


# In[110]:


from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

Best_bag_log = BaggingClassifier(Best_log, n_estimators=100, max_samples=100, bootstrap=True, random_state=0)

Best_bag_log.fit(X_train, z_train)
Best_bag_pred1 = Best_bag_log.predict(X_test)
print("Accuracy: {:.3f}".format(accuracy_score(z_test, Best_bag_pred1)))
print("f1 score: {:.3f}".format(f1_score(z_test,Best_bag_pred1)))
print("Precision:",metrics.precision_score(z_test, Best_bag_pred1))
print("Recall:",metrics.recall_score(z_test, Best_bag_pred1))
print("Confusion matrix:\n{}".format(confusion_matrix(z_test, Best_bag_pred1)))


# #### 1st PASTING MODEL (DECISION TREE CLASSIFIER)

# In[111]:


from sklearn.tree import DecisionTreeClassifier

param_grid = {'max_depth':[1, 2, 3, 4, 5, 6]}


grid_search_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, scoring = 'recall', return_train_score=True, n_jobs = -1)
grid_search_tree.fit(X_train, z_train)

print("Test set score: {:.2f}".format(grid_search_tree.score(X_test, z_test)))

print("Best parameters: {}".format(grid_search_tree.best_params_))
print("Best cross-validation score (recall): {:.2f}".format(grid_search_tree.best_score_))


# In[112]:


#base model (Using the best model of Decision Tree Classifier)
Best_tree = DecisionTreeClassifier( max_depth= grid_search_tree.best_params_['max_depth'])


# In[113]:


#Grid Search for finding the best number of estimators
from sklearn.ensemble import BaggingClassifier
bag_tree = BaggingClassifier(Best_tree, bootstrap = False, max_samples=100,random_state=0)
param_grid = {'n_estimators': [50,100,200]}
grid_bag_tree = GridSearchCV(bag_tree, param_grid, cv = 5, scoring = 'recall', return_train_score=True)
grid_bag_tree.fit(X_train, z_train)


# In[114]:


print("Best parameters: {}".format(grid_bag_tree.best_params_))
print("Best cross-validation (recall): {:.2f}".format(grid_bag_tree.best_score_))
print("Train Set Score: {}".format(grid_bag_tree.score(X_train, z_train)))
print("Test Set Score: {}".format(grid_bag_tree.score(X_test,z_test)))


# In[115]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

Best_bag_tree = BaggingClassifier(Best_tree, n_estimators = 50, max_samples = 100, bootstrap = False, random_state=0)

#train the model
Best_bag_tree.fit(X_train, z_train)

Best_bag_pred2 = Best_bag_tree.predict(X_test)
print("Accuracy: {:.3f}".format(accuracy_score(z_test, Best_bag_pred2)))
print("f1 score: {:.3f}".format(f1_score(z_test, Best_bag_pred2)))
print("Precision:",metrics.precision_score(z_test, Best_bag_pred2))
print("Recall:",metrics.recall_score(z_test, Best_bag_pred2))
print("Confusion matrix:\n{}".format(confusion_matrix(z_test, Best_bag_pred2)))


# #### 2nd PASTING MODEL (SVC [KERNEL POLY])

# In[116]:


from sklearn.svm import SVC

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
               'degree':[1,2,3,4,5,6,7,8]}

grid_search_poly = GridSearchCV(SVC(kernel = 'poly'), param_grid, cv = 5, scoring = 'recall', return_train_score=True, n_jobs = -1)
grid_search_poly.fit(X_train, z_train)

print("Test set score: {:.2f}".format(grid_search_poly.score(X_test, z_test)))

print("Best parameters: {}".format(grid_search_poly.best_params_))
print("Best cross-validation score (recall): {:.2f}".format(grid_search_poly.best_score_))


# In[117]:


#base model (Using the best model of Kernel SVC (Poly) Classifier)
Best_poly = SVC(kernel='poly', C = grid_search_poly.best_params_['C'], gamma = grid_search_poly.best_params_['gamma'],
                degree = grid_search_poly.best_params_['degree'], probability = True)


# In[118]:


#Grid Search for finding the best number of estimators
from sklearn.ensemble import BaggingClassifier
bag_poly = BaggingClassifier(Best_poly, bootstrap = False, max_samples=100,random_state=0)
param_grid = {'n_estimators': [50,100,200]}
grid_bag_poly = GridSearchCV(bag_poly, param_grid, cv = 5, scoring ='recall',return_train_score=True)
grid_bag_poly.fit(X_train, z_train)


# In[119]:


print("Best parameters: {}".format(grid_bag_poly.best_params_))
print("Best cross-validation accuracy: {:.2f}".format(grid_bag_poly.best_score_))
print("Train Set Score: {}".format(grid_bag_poly.score(X_train, z_train)))
print("Test Set Score: {}".format(grid_bag_poly.score(X_test,z_test)))


# In[120]:


from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

Best_bag_poly = BaggingClassifier(Best_poly, n_estimators = 50, max_samples = 100, bootstrap = True, oob_score = True, random_state=0)

#train the model
Best_bag_poly.fit(X_train, z_train)

Best_bag_pred3 = Best_bag_poly.predict(X_test)

print("Accuracy: {:.3f}".format(accuracy_score(z_test, Best_bag_pred3)))
print("f1 score: {:.3f}".format(f1_score(z_test, Best_bag_pred3)))
print("Precision:",metrics.precision_score(z_test, Best_bag_pred3))
print("Recall:",metrics.recall_score(z_test, Best_bag_pred3))
print("Confusion matrix:\n{}".format(confusion_matrix(z_test, Best_bag_pred3)))


# #### 1st ADABOOST BOOSTING MODEL (DECISION TREE CLASSIFIER)

# In[121]:


from sklearn.tree import DecisionTreeClassifier

param_grid = {'max_depth':[1, 2, 3, 4, 5, 6]}


grid_search_TREE = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, scoring = 'recall', return_train_score=True, n_jobs = -1)
grid_search_TREE.fit(X_train, z_train)

print("Test set score: {:.2f}".format(grid_search_TREE.score(X_test, z_test)))

print("Best parameters: {}".format(grid_search_TREE.best_params_))
print("Best cross-validation score (recall): {:.2f}".format(grid_search_TREE.best_score_))


# In[122]:


#base model (Using the best model of Decision Tree Classifier)
Best_TREE = DecisionTreeClassifier( max_depth= grid_search_TREE.best_params_['max_depth'])


# In[123]:


from sklearn.ensemble import AdaBoostClassifier

adbst_TREE = AdaBoostClassifier(Best_TREE,random_state=0)
param_grid = {'learning_rate': [0.01,0.03,0.1,0.3,1.0],
              'n_estimators': [50,100,200]}
grid_adbst_TREE = GridSearchCV(adbst_TREE, param_grid, cv = 5, scoring = 'recall', return_train_score=True)
grid_adbst_TREE.fit(X_train, z_train)


# In[124]:


print("Best parameters: {}".format(grid_adbst_TREE.best_params_))
print("Best cross-validation (recall): {:.2f}".format(grid_adbst_TREE.best_score_))
print("Train Set Score: {}".format(grid_adbst_TREE.score(X_train, z_train)))
print("Test Set Score: {}".format(grid_adbst_TREE.score(X_test,z_test)))


# Building the AdaBoost Classifier Model with Best Parameters

# In[125]:


from sklearn.ensemble import AdaBoostClassifier
Best_adbst_TREE = AdaBoostClassifier(Best_TREE, n_estimators=50, algorithm="SAMME.R"
                             , learning_rate=0.01, random_state=0)
Best_adbst_TREE.fit(X_train, z_train)
Best_adbst_pred = Best_adbst_TREE.predict(X_test)
print("Accuracy: {:.3f}".format(accuracy_score(z_test,Best_adbst_pred)))
print("f1 score: {:.3f}".format(f1_score(z_test, Best_adbst_pred)))
print("Precision:",metrics.precision_score(z_test, Best_adbst_pred))
print("Recall:",metrics.recall_score(z_test, Best_adbst_pred))
print("Confusion matrix:\n{}".format(confusion_matrix(z_test, Best_adbst_pred)))


# #### 2nd ADABOOST BOOSTING MODEL (SVC [KERNEL LINEAR])

# In[126]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}


grid_search_lin = GridSearchCV(SVC(kernel='linear'), param_grid, cv = 5, scoring = 'recall', return_train_score=True, n_jobs = -1)
grid_search_lin.fit(X_train, z_train)

print("Test set score: {:.2f}".format(grid_search_lin.score(X_test, z_test)))

print("Best parameters: {}".format(grid_search_lin.best_params_))
print("Best cross-validation score (recall): {:.2f}".format(grid_search_lin.best_score_))


# In[127]:


#base model (Using the best model of Kernel SVC(Linear) Classifier)
Best_SVClin = SVC(kernel = 'linear', C = grid_search_lin.best_params_['C'], 
                                gamma = grid_search_lin.best_params_['gamma'], probability = True)


# In[128]:


from sklearn.ensemble import AdaBoostClassifier

adbst_lin = AdaBoostClassifier(Best_SVClin,random_state=0)
param_grid = {'learning_rate': [0.01,0.03,0.1,0.3,1.0],
              'n_estimators': [50,100,200]}
grid_adbst_lin = GridSearchCV(adbst_lin, param_grid, cv = 5, scoring = 'recall', return_train_score=True, n_jobs = -1)
grid_adbst_lin.fit(X_train, z_train)


# In[129]:


print("Best parameters: {}".format(grid_adbst_lin.best_params_))
print("Best cross-validation (recall): {:.2f}".format(grid_adbst_lin.best_score_))
print("Train Set Score: {}".format(grid_adbst_lin.score(X_train, z_train)))
print("Test Set Score: {}".format(grid_adbst_lin.score(X_test,z_test)))


# Building the AdaBoost Classifier Model with Best Parameters

# In[130]:


from sklearn.ensemble import AdaBoostClassifier
Best_adbst_lin = AdaBoostClassifier(Best_SVClin, n_estimators=200, algorithm="SAMME.R"
                             , learning_rate=0.01, random_state=0)
Best_adbst_lin.fit(X_train, z_train)
Best_adbst_pred1 = Best_adbst_lin.predict(X_test)
print("Accuracy: {:.3f}".format(accuracy_score(z_test,Best_adbst_pred1)))
print("f1 score: {:.3f}".format(f1_score(z_test, Best_adbst_pred1)))
print("Precision:",metrics.precision_score(z_test, Best_adbst_pred1))
print("Recall:",metrics.recall_score(z_test, Best_adbst_pred1))
print("Confusion matrix:\n{}".format(confusion_matrix(z_test, Best_adbst_pred1)))


# #### GRADIENT BOOSTING CLASSIFIER MODEL

# In[131]:


from  sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0)
param_grid = {'max_depth':[1,2,5],
              'learning_rate' : [0.01,0.03,0.1,0.3,1.0],
              'n_estimators': [50,100,200]}
grid_gbc = GridSearchCV(gbc, param_grid, cv = 5, scoring = 'recall', return_train_score=True, n_jobs = -1)
grid_gbc.fit(X_train, z_train)


# In[132]:


print("Best parameters: {}".format(grid_gbc.best_params_))
print("Best cross-validation (recall): {:.2f}".format(grid_gbc.best_score_))
print("Train Set Score: {}".format(grid_gbc.score(X_train, z_train)))
print("Test Set Score: {}".format(grid_gbc.score(X_test,z_test)))


# In[133]:


gbc_pred = grid_gbc.predict(X_test)

print("Accuracy: {:.3f}".format(accuracy_score(z_test, gbc_pred)))
print("f1 score: {:.3f}".format(f1_score(z_test, gbc_pred)))
print("Precision:",metrics.precision_score(z_test, gbc_pred))
print("Recall:",metrics.recall_score(z_test, gbc_pred))
print("Confusion matrix:\n{}".format(confusion_matrix(z_test, gbc_pred)))


# #### HARD VOTING CLASSIFIER

# In[134]:


from sklearn.ensemble import VotingClassifier

hard_voting = VotingClassifier(estimators=[('knn',Best_KNNC),('lm', Best_log),('svc',Best_SVClin)],voting='hard')
hard_voting.fit(X_train, z_train)


# In[135]:


for clf in (Best_KNNC, Best_log, Best_SVClin, hard_voting):
    clf.fit(X_train, z_train)
    z_test_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(z_test, z_test_pred))


# #### SOFT VOTING CLASSIFIER

# In[136]:


from sklearn.ensemble import VotingClassifier

soft_voting = VotingClassifier(estimators=[('knn',Best_KNNC),('dt', Best_TREE),('svc',Best_poly)],voting='soft')
soft_voting.fit(X_train, z_train)


# In[137]:


for clf in (Best_KNNC, Best_TREE, Best_poly, soft_voting):
    clf.fit(X_train, z_train)
    z_test_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(z_test, z_test_pred))


# ### PRINCIPAL COMPONENT ANALYSIS

# In[138]:


from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("{} PCA components are selected to explain atleast 95% of the total variation in data".format(pca.n_components_))


# In[139]:


print("Variation explained by each PCA component is: \n {}".format(pca.explained_variance_ratio_))
print("Total data variation explained is: \n {}".format(pca.explained_variance_ratio_.sum()))


# In[140]:


# Stratified k-fold
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10)


# #### LOGISTIC REGRESSION

# In[141]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
norm = LogisticRegression(penalty = 'l2', C = 1)
norm.fit(X_train_pca, z_train)
pred1 = norm.predict(X_test_pca)

print("Accuracy: {:.3f}".format(accuracy_score(z_test, pred1)))
print("f1 score: {:.3f}".format(f1_score(z_test, pred1)))
print("Precision:",metrics.precision_score(z_test, pred1))
print("Recall:",metrics.recall_score(z_test, pred1))
print("Confusion matrix:\n{}".format(confusion_matrix(z_test, pred1)))


# In[142]:


from sklearn.linear_model import LogisticRegression

c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_score_l1 = []
train_score_l2 = []
test_score_l1 = []
test_score_l2 = []

for c in c_range:
    log_l1 = LogisticRegression(penalty = 'l1', C = c)
    log_l2 = LogisticRegression(penalty = 'l2', C = c)
    log_l1.fit(X_train_pca, z_train)
    log_l2.fit(X_train_pca, z_train)
    train_score_l1.append(log_l1.score(X_train_pca, z_train))
    train_score_l2.append(log_l2.score(X_train_pca, z_train))
    test_score_l1.append(log_l1.score(X_test_pca, z_test))
    test_score_l2.append(log_l2.score(X_test_pca, z_test))


# In[143]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(c_range, train_score_l1, label = 'Train score, penalty = l1')
plt.plot(c_range, test_score_l1, label = 'Test score, penalty = l1')
plt.plot(c_range, train_score_l2, label = 'Train score, penalty = l2')
plt.plot(c_range, test_score_l2, label = 'Test score, penalty = l2')
plt.legend()
plt.xlabel('Regularization parameter: C')
plt.ylabel('Accuracy')
plt.xscale('log')


# In[144]:


get_ipython().run_line_magic('matplotlib', 'inline')
import mglearn
X_b = X_train_pca[10:50, [1,3]]
z_b = z_train[10:50]

lreg = LogisticRegression()
lreg.fit(X_b, z_b) 

mglearn.plots.plot_2d_separator(lreg, X_b, fill=True, eps=0.5, alpha=.4)
mglearn.discrete_scatter(X_b[:, 0], X_b[:, 1], z_b)


# In[145]:


# Grid search with CV for best parameters

from sklearn.model_selection import GridSearchCV

grid={"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], "penalty":["l1","l2"]}# l1 lasso l2 ridge
LR=LogisticRegression()
LR_cv=GridSearchCV(LR,grid,cv=kfold, scoring = 'recall')
LR_cv.fit(X_train_pca,z_train)

print("tuned hpyerparameters :(best parameters) ",LR_cv.best_params_)
print("Cross-validation score :",LR_cv.best_score_)
print("Test set score: {:.2f}".format(LR_cv.score(X_test_pca, z_test)))


# In[146]:


# Evaluating metrics of the model using best parameters
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
Best_model = LogisticRegression(penalty = 'l1', C = 100)
Best_model.fit(X_train_pca, z_train)
pred1 = Best_model.predict(X_test_pca)

print("Accuracy: {:.3f}".format(accuracy_score(z_test, pred1)))
print("f1 score: {:.3f}".format(f1_score(z_test, pred1)))
print("Precision:",metrics.precision_score(z_test, pred1))
print("Recall:",metrics.recall_score(z_test, pred1))
print("Confusion matrix:\n{}".format(confusion_matrix(z_test, pred1)))


# In[147]:


# Precision Recall curve for logistic regression
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(z_test, pred1)

close_zero = np.argmin(np.abs(thresholds))

plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c='k', mew=2)

plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")


# #### DECISION TREE CLASSIFIER

# In[148]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train_pca, z_train)
pred_tree = tree.predict(X_test_pca)

print("\nDecision tree:")
print("Accuracy: {:.3f}".format(accuracy_score(z_test, pred_tree)))
print("f1 score tree: {:.3f}".format(f1_score(z_test, pred_tree)))
print("Precision:",metrics.precision_score(z_test, pred_tree))
print("Recall:",metrics.recall_score(z_test, pred_tree))
print("Confusion matrix:\n",confusion_matrix(z_test, pred_tree))


# In[149]:


# Grid search with CV for best parameters
parameters={'max_depth': range(1,20,2)}
tree=DecisionTreeClassifier()
grid_search=GridSearchCV(tree,parameters,cv = kfold, scoring = 'recall')
grid_search.fit(X_train_pca,z_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[150]:


# Evaluating metrics of the model using best parameters
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
Best_Model1 = DecisionTreeClassifier(max_depth=3).fit(X_train_pca, z_train)
pred_tree = Best_Model1.predict(X_test_pca)

print("\nDecision tree:")
print("Accuracy: {:.3f}".format(accuracy_score(z_test, pred_tree)))
print("f1 score tree: {:.3f}".format(f1_score(z_test, pred_tree)))
print("Precision:",metrics.precision_score(z_test, pred_tree))
print("Recall:",metrics.recall_score(z_test, pred_tree))
print("Confusion matrix:\n",confusion_matrix(z_test, pred_tree))


# In[151]:


# Precision Recall curve for Decision Tree
get_ipython().run_line_magic('matplotlib', 'notebook')

precision, recall, thresholds = precision_recall_curve(z_test, pred_tree)

close_zero = np.argmin(np.abs(thresholds))

plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c='k', mew=2)

plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")


# #### KNN CLASSIFIER

# In[152]:


from sklearn.neighbors import KNeighborsClassifier

train_score_array = []
test_score_array = []

for k in range(1,20):
    knn = KNeighborsClassifier(k)
    knn.fit(X_train_pca, z_train)
    train_score_array.append(knn.score(X_train_pca, z_train))
    test_score_array.append(knn.score(X_test_pca, z_test))


# In[153]:


x_axis = range(1,20)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(x_axis, train_score_array, label = 'Train Score', c = 'g')
plt.plot(x_axis, test_score_array, label = 'Test Score', c='b')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()


# In[154]:


from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train_pca, z_train)

#Predict the response for test dataset
pred_knn = knn.predict(X_test_pca)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(z_test, pred_knn))
print("f1 score: {:.3f}".format(f1_score(z_test, pred_knn)))
print("Precision:",metrics.precision_score(z_test, pred_knn))
print("Recall:",metrics.recall_score(z_test, pred_knn))
print(confusion_matrix(z_test, pred_knn))


# In[155]:


# Grid Search with CV for best parameters
param_grid = {'n_neighbors': np.arange(1, 25)}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=kfold, scoring='recall')
grid_search.fit(X_train_pca, z_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[156]:


# Evaluating metrics of the model using best parameters
from sklearn.neighbors import KNeighborsClassifier
Best_model_knn= KNeighborsClassifier(1)
Best_model_knn.fit(X_train_pca, z_train)
pred_knn = Best_model_knn.predict(X_test_pca)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(z_test, pred_knn))
print("f1 score: {:.3f}".format(f1_score(z_test, pred_knn)))
print("Precision:",metrics.precision_score(z_test, pred_knn))
print("Recall:",metrics.recall_score(z_test, pred_knn))
print(confusion_matrix(z_test, pred_knn))


# #### LINEAR SVC

# In[157]:


from sklearn.svm import LinearSVC

linear_svm = LinearSVC(C=0.1).fit(X_train_pca, z_train)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)
pred_linear_svc = linear_svm.predict(X_test_pca)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(z_test, pred_linear_svc))
print("f1 score: {:.3f}".format(f1_score(z_test, pred_linear_svc)))
print("Precision:",metrics.precision_score(z_test, pred_linear_svc))
print("Recall:",metrics.recall_score(z_test, pred_linear_svc))
print(confusion_matrix(z_test, pred_linear_svc))


# In[158]:


# Grid Search with CV for best parameters
param_grid = {'C':[0.01,0.1,10,100]}
linearSVC = GridSearchCV(linear_svm,param_grid,cv=kfold,scoring='recall')
linearSVC.fit(X_train_pca,z_train)
print("Best parameters: {}".format(linearSVC.best_params_))
print("Best cross-validation score: {:.2f}".format(linearSVC.best_score_))


# In[159]:


# Evaluating metrics of the model using best parameters

from sklearn.svm import LinearSVC
best_linear_svc = LinearSVC(C=10)
best_linear_svc.fit(X_train_pca, z_train)
best_pred_linear_svc = best_linear_svc.predict(X_test_pca)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(z_test, best_pred_linear_svc))
print("f1 score: {:.3f}".format(f1_score(z_test, best_pred_linear_svc)))
print("Precision:",metrics.precision_score(z_test, best_pred_linear_svc))
print("Recall:",metrics.recall_score(z_test, best_pred_linear_svc))
print(confusion_matrix(z_test, best_pred_linear_svc))


# In[160]:


# Precision Recall curve for Linear SVC
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(z_test, best_pred_linear_svc)

close_zero = np.argmin(np.abs(thresholds))

plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c='k', mew=2)

plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")


# #### KERNELIZED SVC (LINEAR)

# In[161]:


from sklearn.svm import SVC
svc1 = SVC(kernel = 'linear', C = 0.1, gamma = 10)
svc1.fit(X_train_pca, z_train)
print("Accuracy: {:.3f}".format(accuracy_score(z_test, svc1.predict(X_test_pca))))
print("f1_score of svc1:{:.3f}".format(f1_score(z_test, svc1.predict(X_test_pca))))
print("Precision:",metrics.precision_score(z_test, svc1.predict(X_test_pca)))
print("Recall:",metrics.recall_score(z_test, svc1.predict(X_test_pca)))
print(confusion_matrix(z_test, svc1.predict(X_test_pca)))


# In[162]:


# Grid Search CV for best parameters
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(svc1, param_grid, cv=kfold,scoring='recall')
grid_search.fit(X_train_pca, z_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[163]:


# Evaluating metrics of the model using best parameters

from sklearn.svm import SVC
best_svc1 = SVC(kernel = 'linear', C = 10, gamma = 0.001)
best_svc1.fit(X_train_pca, z_train)
print("Accuracy: {:.3f}".format(accuracy_score(z_test, best_svc1.predict(X_test_pca))))
print("f1_score of svc1:{:.3f}".format(f1_score(z_test, best_svc1.predict(X_test_pca))))
print("Precision:",metrics.precision_score(z_test, best_svc1.predict(X_test_pca)))
print("Recall:",metrics.recall_score(z_test,best_svc1.predict(X_test_pca)))
print(confusion_matrix(z_test, best_svc1.predict(X_test_pca)))


# In[164]:


# Precision Recall curve for SVC (Linear)
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(z_test, best_svc1.decision_function(X_test_pca))

close_zero = np.argmin(np.abs(thresholds))

plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c='k', mew=2)

plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")


# #### KERNELIZED SVC (GAUSSIAN = RBF)

# In[165]:


from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
svc = SVC(kernel = 'rbf', C = 1, gamma = 10)
svc.fit(X_train_pca, z_train)
print("Accuracy: {:.3f}".format(accuracy_score(z_test, svc.predict(X_test_pca))))
print("f1_score of svc:{:.3f}".format(f1_score(z_test, svc.predict(X_test_pca))))
print("Precision:",metrics.precision_score(z_test, svc.predict(X_test_pca)))
print("Recall:",metrics.recall_score(z_test, svc.predict(X_test_pca)))
print(confusion_matrix(z_test, svc.predict(X_test_pca)))


# In[166]:


# Grid Search CV for best parameters
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(svc, param_grid, cv=kfold, scoring='recall')
grid_search.fit(X_train_pca, z_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[167]:


# Evaluating metrics of the model using best parameters

from sklearn.svm import SVC
best_svc = SVC(kernel = 'rbf', C = 1, gamma = 100)
best_svc.fit(X_train_pca, z_train)
print("Accuracy: {:.3f}".format(accuracy_score(z_test, best_svc.predict(X_test_pca))))
print("f1_score of svc:{:.3f}".format(f1_score(z_test, best_svc.predict(X_test_pca))))
print("Precision:",metrics.precision_score(z_test, best_svc.predict(X_test_pca)))
print("Recall:",metrics.recall_score(z_test, best_svc.predict(X_test_pca)))
print(confusion_matrix(z_test, best_svc.predict(X_test_pca)))


# In[168]:


# Precision Recall curve for SVC (Gaussian = rbf)
get_ipython().run_line_magic('matplotlib', 'notebook')

precision, recall, thresholds = precision_recall_curve(z_test, best_svc.decision_function(X_test_pca))

close_zero = np.argmin(np.abs(thresholds))

plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c='k', mew=2)

plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")


# In[169]:


get_ipython().run_line_magic('matplotlib', 'notebook')

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(z_test, best_svc.decision_function(X_test_pca))

plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)
plt.show()


# #### KERNELIZED SVC (POLY)

# In[170]:


from sklearn.svm import SVC  
svc_poly = SVC(kernel='poly',C = 1, gamma = 10, degree=6)  
svc_poly.fit(X_train_pca, z_train)  
pred_poly = svc_poly.predict(X_test_pca) 
print("Accuracy: {:.3f}".format(accuracy_score(z_test, pred_poly)))
print("f1_score of svc:{:.3f}".format(f1_score(z_test,pred_poly)))
print("Precision:",metrics.precision_score(z_test,pred_poly))
print("Recall:",metrics.recall_score(z_test,pred_poly))
print(confusion_matrix(z_test, pred_poly))


# In[171]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
               'degree':[1,2,3,4,5,6,7,8]}
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(svc_poly, param_grid, cv=kfold, scoring='recall')
grid_search.fit(X_train_pca, z_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[172]:


from sklearn.svm import SVC  
best_svc_poly = SVC(kernel='poly',C = 100, gamma = 100, degree=1)  
best_svc_poly.fit(X_train_pca, z_train)  
pred_best_svc_poly = best_svc_poly.predict(X_test_pca) 
print("Accuracy: {:.3f}".format(accuracy_score(z_test, pred_best_svc_poly)))
print("f1_score of svc:{:.3f}".format(f1_score(z_test,pred_poly)))
print("Precision:",metrics.precision_score(z_test,pred_poly))
print("Recall:",metrics.recall_score(z_test,pred_poly))
print("Confusion_matrix:\n",confusion_matrix(z_test, pred_poly))


# In[173]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(z_test, pred_best_svc_poly)

close_zero = np.argmin(np.abs(thresholds))

plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label="threshold zero", fillstyle="none", c='k', mew=2)

plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")


# ## DEEP LEARNING MODELS (CLASSIFICATION)

# In[174]:


import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# #### SIMPLE PERCEPTRON

# In[175]:


n_cols = X_train.shape[1]

#step 1: build model
model1 = Sequential()
#input layer
model1.add(Dense(10, input_dim = n_cols, activation = 'relu'))
#hidden layers
#NO hidden layers are added in this model
#output layer
model1.add(Dense(1, activation = 'sigmoid'))

#step 2: make computational graph - compile
model1.compile(loss= 'binary_crossentropy' , optimizer = 'adam',metrics = ['accuracy'] )

#step 3: train the model - fit
model1.fit(X_train, z_train, epochs = 50, batch_size = 20)


# In[176]:


model1.evaluate(X_train, z_train)


# In[177]:


model1.evaluate(X_test, z_test)


# In[178]:


z_pred1 = model1.predict_classes(X_test)


# In[179]:


print("Confision Matrix: \n", confusion_matrix(z_test, z_pred1) )
print("Accuracy: {:.3f}".format(accuracy_score(z_test, z_pred1)))
print("f1_score of svc:{:.3f}".format(f1_score(z_test, z_pred1)))
print("Precision:",metrics.precision_score(z_test, z_pred1))
print("Recall:",metrics.recall_score(z_test, z_pred1))


# In order to avoid the problem of underfitting, we add hidden layers to the simple perceptron model (i.e. build a Multilayer Perceptron model) which helps to create customized decision boundaries by decreasing gap between the train and the test score.

# #### MULTILAYER PERCEPTRON

# In[180]:


#step 1: build model
model2 = Sequential()
#input layer
model2.add(Dense(32, input_dim = n_cols, activation = 'relu'))
#hidden layers
model2.add(Dense(16, activation = 'relu'))
model2.add(Dense(8, activation = 'relu'))
#output layer
model2.add(Dense(1, activation = 'sigmoid'))

#step 2: compile the model
model2.compile(loss= 'binary_crossentropy' , optimizer = 'adam',metrics = ['accuracy'] )

#step 3: train the model
model2.fit(X_train, z_train, epochs = 50, batch_size = 20)

#step 4: evaluate


# In[181]:


model2.evaluate(X_train, z_train)


# In[182]:


model2.evaluate(X_test, z_test)


# In[183]:


z_pred2 = model2.predict_classes(X_test)


# In[184]:


print("Confision Matrix: \n", confusion_matrix(z_test, z_pred2))
print("Accuracy: {:.3f}".format(accuracy_score(z_test, z_pred2)))
print("f1_score of svc:{:.3f}".format(f1_score(z_test, z_pred2)))
print("Precision:",metrics.precision_score(z_test, z_pred2))
print("Recall:",metrics.recall_score(z_test, z_pred2))

