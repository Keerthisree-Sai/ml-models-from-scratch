**Simple Linear Regression using OLS (Ordinary Least Squares)**

This repository contains a simple implementation of Linear Regression using the Ordinary Least Squares (OLS) method, which is a basic algorithm for fitting a straight line to a set of data points.


**Overview**

The SimpleLRwithOLS class implements the core functionality of linear regression using the OLS method. It can be used to train a model on a given dataset and make predictions based on the learned parameters (slope m and intercept b).


**Key Features:**

Fit the model: The fit() method trains the model by calculating the slope (m) and intercept (b) using the OLS formula.

Predict: The predict() method makes predictions based on the learned linear regression model.


**What is Ordinary Least Squares (OLS)?**

Ordinary Least Squares (OLS) is a method used in linear regression to estimate the parameters of a linear relationship between the input features and the output variable. The goal of OLS is to find the line (or hyperplane, in higher dimensions) that minimizes the sum of the squared differences between the observed values and the predicted values.

  **The Linear Regression Equation**
  
  In simple linear regression, the relationship between the independent variable 𝑥 and the dependent variable 𝑦 is modeled by the equation: **𝑦 = 𝑚𝑥 + 𝑏**


**OLS Method**

OLS works by minimizing the sum of squared residuals (the difference between the observed values and the predicted values). The formula for OLS is derived by solving for the parameters m and b that minimize the following:

Residual Sum of Squares (RSS) = ∑(y − ( mx + b))<sup>2</sup>


**Deriving the OLS Formulas**
To find the best-fitting line, OLS minimizes the RSS. The formulas for the slope m and the intercept b are derived as follows:

The slope m and the intercept b are calculated using the formulas: 

m = (∑(x − x_mean)(y − y_mean))/ ∑(x − x_mean)<sup>2</sup>

b = y_mean − m*x_mean

**How OLS Works:**
OLS finds the values of m and b that minimize the residual sum of squares, thus ensuring the best-fitting line for the given data points.


**How to Use**

**1. Clone the repository**
   
   To get started, clone the repository to your local machine:
      
      git clone https://github.com/Keerthisree-Sai/ml-models-from-scratch.git
   
**2. Import the class**

  Import the SimpleLRwithOLS class in your Python script:
      
      from simple_lr_ols import SimpleLRwithOLS

**3. Train the model**

   Use the fit() method to train the model on your training data (X_train and y_train):
      
      model = SimpleLRwithOLS()
      
      model.fit(X_train, y_train)

**4. Make predictions**

   Once the model is trained, you can use the predict() method to make predictions on new data (X_test):
      
      predictions = model.predict(X_test)
