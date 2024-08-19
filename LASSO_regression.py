#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

# Loading the Boston housing dataset and converting it into a DataFrame
boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Adding the target variable 'Price' to the DataFrame
boston_df['Price'] = boston.target

# Displaying the first few rows of the DataFrame
boston_df.head()

# Checking the shape (dimensions) of the DataFrame
boston_df.shape

# Describing the dataset to get summary statistics
boston_df.describe()

# Exploratory Data Analysis: Plotting a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(boston_df.corr(), annot=True)

# Identifying and dropping columns with multicollinearity (highly correlated variables)
boston_df.drop(columns=["INDUS", "NOX"], inplace=True)

# Displaying the first few rows after dropping columns
boston_df.head()

# Plotting pairwise relationships in the dataset
sns.pairplot(boston_df)

# The relationship between LSTAT and Price is nonlinear, so we take the logarithm of LSTAT to make it more linear
boston_df.LSTAT = np.log(boston_df.LSTAT)

# Plotting the pairplot again after transforming LSTAT
sns.pairplot(boston_df)

# Preparing for model training by splitting the dataset into features and target
# The features include all columns except the last one, which is the target
features = boston_df.columns[0:11]
target = boston_df.columns[-1]

# Extracting the feature values (X) and target values (y)
X = boston_df[features].values
y = boston_df[target].values

# Splitting the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Displaying the dimensions of the training and testing sets
print("The dimension of X_train is {}".format(X_train.shape))
print("The dimension of X_test is {}".format(X_test.shape))

# Scaling the features to standardize them (mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building and evaluating a Lasso regression model
print("\nLasso Model............................................\n")
lasso = Lasso(alpha=10)  # Setting the regularization strength (alpha)
lasso.fit(X_train, y_train)  # Fitting the model to the training data

# Calculating and printing the R^2 scores for the training and testing sets
train_score_ls = lasso.score(X_train, y_train)
test_score_ls = lasso.score(X_test, y_test)
print("The train score for ls model is {}".format(train_score_ls))
print("The test score for ls model is {}".format(test_score_ls))

# Visualizing the coefficients learned by the Lasso model
pd.Series(lasso.coef_, features).sort_values(ascending=True).plot(kind="bar")
plt.title("Lasso Regression Coefficients at alpha = 10")
plt.show()

# Using LassoCV for cross-validated Lasso regression to find the best alpha
from sklearn.linear_model import LassoCV

# Performing cross-validated Lasso regression with a range of alpha values
lasso_cv = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10], random_state=0).fit(X_train, y_train)

# Calculating and printing the R^2 scores for the training and testing sets using the cross-validated Lasso model
print("The train score for lasso model is: {}".format(lasso_cv.score(X_train, y_train)))
print("The train score for lasso model is: {}".format(lasso_cv.score(X_test, y_test)))
