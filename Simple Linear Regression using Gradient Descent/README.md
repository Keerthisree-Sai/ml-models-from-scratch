## Simple Linear Regression using Gradient Descent
This project demonstrates the implementation of Simple Linear Regression using two approaches:

1. **Ordinary Least Squares (OLS)** via scikit-learn's `LinearRegression`.

2. **Custom Gradient Descent Regressor** implemented from scratch in Python.

The code generates synthetic regression data, trains the models, evaluates their performance, and visualizes the results.

## Gradient Descent:
**Note:** The explanation for Ordinary Least Squares (OLS) is provided in the "Simple Linear Regression using OLS" section.

**Gradient Descent** is an optimization algorithm used to minimize the cost function in machine learning models. It iteratively adjusts the model parameters (slope and intercept) by calculating the gradient (or derivative) of the cost function with respect to each parameter. The parameters are then updated in the direction of the steepest descent (opposite of the gradient) to find the minimum. The **learning rate** determines the size of the steps taken during each iteration, while the **number of epochs** controls how many times the model parameters are updated. Gradient Descent is especially useful for large datasets or more complex models, where direct methods like OLS can be computationally expensive.

## Features

- Synthetic Data Generation: Uses scikit-learn's make_regression to generate a noisy dataset.
 
- Model Comparison:
  - OLS-based Linear Regression using scikit-learn.
    
  - Custom Gradient Descent-based Linear Regression.

- Visualization: Plots the data points and regression lines for both models.

- Performance Evaluation: Compares models using the $R^2$ score.

## Requirements

The following Python libraries are required to run the code:

  - `numpy`
  - `matplotlib`
  - `scikit-learn`

Install them using: `pip install numpy matplotlib scikit-learn`

## Code Walkthrough

### 1. Data Generation
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=500, n_features=1, noise=20)
  Generates a dataset with 500 samples, 1 feature, and a specified noise level.
### 2. Data Splitting
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
  Splits the dataset into training and testing sets (80%-20%).
### 3. OLS Linear Regression
  - **Training:**
    
          from sklearn.linear_model import LinearRegression
          lr = LinearRegression()
          lr.fit(X_train, y_train)
  - **Prediction:**
          y_pred = lr.predict(X_test)
  - **Evaluation:**
          from sklearn.metrics import r2_score
          print("R2 score of linear regression model:", r2_score(y_test, y_pred))
### 4. Custom Gradient Descent Regressor
  - **Implementation:** A custom class `GradientDescentRegressor` is defined with methods for training `fit` and prediction `predict`.
    
        class GradientDescentRegressor:
        def __init__(self, learning_rate, epochs):
            self.m = 0
            self.b = 0
            self.alpha = learning_rate
            self.epochs = epochs
    
        def fit(self, X, y):
            n = len(y)
            for i in range(self.epochs):
                slope_m = -(2/n) * np.sum((y - self.m * X.ravel() - self.b) * X.ravel())
                slope_b = -(2/n) * np.sum(y - self.m * X.ravel() - self.b)
                self.m = self.m - (self.alpha * slope_m)
                self.b = self.b - (self.alpha * slope_b)
    
        def predict(self, X):
            return self.m * X + self.b
  - **Training:**
    
        gdregressor = GradientDescentRegressor(learning_rate=0.001, epochs=150)
        gdregressor.fit(X_train, y_train)
  - **Prediction:**
    
        y_pred_gdregressor = gdregressor.predict(X_test)
  - **Evaluation:**

        print("R2 score of Gradient Descent Regression Model:", r2_score(y_test, y_pred_gdregressor))

### 5. Visualization:
Two separate plots are generated to compare the performance of the OLS and Gradient Descent models.

- **OLS Regression Plot:** The first plot shows the data points and the regression line generated using the OLS method (from scikit-learn). The data points are shown in blue, and the regression line is shown in red.

- **Gradient Descent Regression Plot:** The second plot shows the data points and the regression line generated using the custom Gradient Descent implementation. The data points are shown in blue, and the regression line is shown in red.

By creating these two plots separately, we can compare how each model fits the data individually.

        #Plotting the regression line for the OLS model
        plt.scatter(X, y, color='blue', alpha=0.5, label="Data Points")
        plt.plot(X, y_prediction, color='red', label="OLS Regression Line")
        plt.title("Linear Regression from sklearn")
        plt.legend()
        plt.show()
  
        #Plotting the regression line for the Gradient Descent model
        plt.scatter(X, y, color='blue', alpha=0.5, label="Data Points")
        plt.plot(X, y_prediction_gdregressor, color='red', label="Gradient Descent Regression Line")
        plt.title("Gradient Descent Regressor")
        plt.legend()
        plt.show()

## Results:
The $R^2$ scores of both models are printed, showing their performance. The OLS method is expected to perform slightly better due to its direct optimization approach, but its performance may also be influenced by the dataset size. With a smaller dataset, OLS can fit the data more accurately. The Gradient Descent model's performance depends on the chosen learning rate and epochs, and it may not perform as well on smaller datasets without proper tuning.
