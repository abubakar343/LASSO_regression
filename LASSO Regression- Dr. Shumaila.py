#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston


# In[2]:


#data
boston = load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)

boston_df['Price']=boston.target
boston_df.head()


# In[3]:


boston_df.shape


# In[4]:


boston_df.describe()


# In[5]:


#Exploration
plt.figure(figsize = (10, 10))
sns.heatmap(boston_df.corr(), annot = True)


# In[6]:


#There are cases of multicolinearity, we will drop a few columns
boston_df.drop(columns = ["INDUS", "NOX"], inplace = True)


# In[7]:


boston_df.head()


# In[8]:


#pairplot
sns.pairplot(boston_df)


# In[9]:


#Variables should be normally distributed and linear. However, the relationship between LSTAT and Price is nonlinear. Hence, we log it.
boston_df.LSTAT = np.log(boston_df.LSTAT)


# In[10]:


#pairplot again
sns.pairplot(boston_df)


# In[11]:


#train test split
boston_df.shape


# In[12]:


#preview
features = boston_df.columns[0:11]
target = boston_df.columns[-1]

#X and y values
X = boston_df[features].values
y = boston_df[target].values

#splot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

print("The dimension of X_train is {}".format(X_train.shape))
print("The dimension of X_test is {}".format(X_test.shape))
#Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[13]:


#Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[15]:


#Model
#Lasso
#print("The train score for lr model is {}".format(train_score_lr))
#print("The test score for lr model is {}".format(test_score_lr))

#Building a ri
print("\nLasso Model............................................\n")
lasso = Lasso(alpha = 10)
lasso.fit(X_train,y_train)
train_score_ls =lasso.score(X_train,y_train)
test_score_ls =lasso.score(X_test,y_test)

print("The train score for ls model is {}".format(train_score_ls))
print("The test score for ls model is {}".format(test_score_ls))


# In[16]:


pd.Series(lasso.coef_, features).sort_values(ascending = True).plot(kind = "bar")
plt.title("Lasso Regression Coefficients at alpha = 10")
plt.show()


# In[17]:


#Using the linear CV model
from sklearn.linear_model import LassoCV

#Lasso Cross validation
lasso_cv = LassoCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10], random_state=0).fit(X_train, y_train)

#score
print("The train score for lasso model is: {}".format(lasso_cv.score(X_train, y_train)))
print("The train score for lasso model is: {}".format(lasso_cv.score(X_test, y_test)))


# In[ ]:




