# LASSO Regression

## Overview

This project analyzes the Boston Housing dataset using various regression models, focusing on Lasso regression. The dataset is first explored and preprocessed, including handling multicollinearity and transforming variables. The project demonstrates the use of Lasso regression and cross-validated Lasso to predict housing prices.

## Features

- **Data Exploration**: Visualize correlations and pairwise relationships in the Boston Housing dataset.
- **Data Preprocessing**: Handle multicollinearity and transform non-linear relationships.
- **Modeling**: Implement and evaluate Lasso regression for predicting housing prices.
- **Cross-Validation**: Use LassoCV to perform cross-validated Lasso regression and identify the best regularization parameter (alpha).

## Requirements

- Python 3.x
- Required Python packages:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - statsmodels

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/lasso_regression.git
   cd lasso_regression
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the necessary data files are available, such as the Boston Housing dataset.

## Usage

1. **Data Exploration**: Start by exploring the dataset, visualizing the correlation matrix, and identifying multicollinearity.

2. **Preprocessing**: Drop highly correlated features and apply transformations to variables with non-linear relationships.

3. **Modeling**: Train a Lasso regression model on the preprocessed data and evaluate its performance using R² scores.

4. **Cross-Validation**: Use LassoCV to find the optimal alpha for the Lasso regression model through cross-validation.

To run the script, use the following command:

```bash
python lasso_regression.py
```

print("Test score:", test_score)
```

## License

This project is licensed under the MIT License.
