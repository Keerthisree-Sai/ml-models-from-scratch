# Multiple Linear Regression using Ordinary Least Squares (OLS)

## Overview

This repository contains a Python implementation of Multiple Linear Regression using the Ordinary Least Squares (OLS) method. The MultipleLinearRegressionwithOLS class provides functionality to fit a linear model to a dataset and make predictions based on the trained model.

## How OLS Works

The Ordinary Least Squares (OLS) method estimates the coefficients (𝛽) of the multiple linear regression model by minimizing the sum of squared residuals between the observed and predicted values. The core formula for calculating the coefficients is:

$\[
\hat{\beta} = (X^T X)^{-1} X^T y
\]$

Where:
- \( X \) is the matrix of input features (with each row representing a data point and each column representing a feature).
- \( y \) is the vector of target values (the dependent variable).
- $\( \hat{\beta} \)$ is the vector of estimated coefficients (weights) of the regression model.

The formula computes the best-fitting line by finding the coefficient values that minimize the squared differences between the observed target values and the predicted values. The intercept (bias) term is included in the matrix \( X \) as a column of ones to account for the constant term in the linear model.

## Features

- **Train a Model:** Fit a multiple linear regression model to the training dataset.

- **Predict Outcomes:** Generate predictions for a given test dataset.

- **Custom Implementation:** The class implements OLS from scratch using NumPy, without relying on high-level libraries like scikit-learn.

## Class Details

### `MultipleLinearRegressionwithOLS`

This class encapsulates the OLS method for multiple linear regression.

#### Attributes:

- `coef_`: Stores the coefficients (weights) of the regression model.

- `intercept_`: Stores the intercept (bias) of the regression model.

#### Methods:

1. **`fit(X_train, y_train)`:**
  
    - Fits the model to the training data.
    
    - **Parameters:**
    
      - `X_train`: A 2D NumPy array of shape `(n_samples, n_features)` representing the input features.
        
      - `y_train`: A 1D NumPy array of shape `(n_samples,)` representing the target values.
  
    - Updates the `coef_` and `intercept_` attributes.

2. **predict(X_test):**

    - Generates predictions for the given test data.
      
    - **Parameters:**
      
      - X_test: A 2D NumPy array of shape (n_samples, n_features) representing the input features.
      
    - **Returns:**
      
      - A 1D NumPy array of predicted values.

## How to Use

**1. Clone the Repository**

  To get started, clone the repository to your local machine:

    git clone https://github.com/Keerthisree-Sai/ml-models-from-scratch.git

**2. Import the Class**

  Import the MultipleLinearRegressionwithOLS class into your Python script:
  
    from multiple_lr_ols import MultipleLinearRegressionwithOLS

**3. Train the Model**

  Use the fit() method to train the model on your training data (X_train and y_train):

    model = MultipleLinearRegressionwithOLS()
    
    model.fit(X_train, y_train)

**4. Make Predictions**

  Once the model is trained, you can use the predict() method to make predictions on new data (X_test):

    predictions = model.predict(X_test)

## Limitations

  - Assumes no multicollinearity in the input data. High multicollinearity can lead to unreliable coefficient estimates.

  - Assumes that the input data is preprocessed (e.g., normalized or standardized) if necessary. Proper preprocessing helps improve the model's stability and performance.

  - Does not include regularization techniques like Ridge or Lasso regression, which can help prevent overfitting.
