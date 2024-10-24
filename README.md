
# Predicting GDP Using Machine Learning Models

This project demonstrates how to predict GDP using various economic variables like inflation, interest rates, unemployment rate, and others with **four different models**:
1. **Random Forest Regressor**
2. **XGBoost Regressor**
3. **LSTM (Long Short-Term Memory)**
4. **Fully Connected Neural Network (Dense Neural Network)**

The code uses **mock data** to simulate economic indicators and shows how to train each model. It also compares the performance of the models using the Root Mean Squared Error (RMSE) metric.

---

## Table of Contents

1. [Introduction to the Models](#introduction-to-the-models)
    - Random Forest Regressor
    - XGBoost Regressor
    - LSTM
    - Fully Connected Neural Network
2. [Code Structure](#code-structure)
3. [How to Use the Code](#how-to-use-the-code)
4. [Model Explanations](#model-explanations)
    - Random Forest
    - XGBoost
    - LSTM
    - Fully Connected Neural Network
5. [Parameter Explanations](#parameter-explanations)
6. [Performance Evaluation](#performance-evaluation)
7. [Things to Keep in Mind](#things-to-keep-in-mind)
8. [References](#references)

---

## Introduction to the Models

### 1. **Random Forest Regressor**
Random Forest is an **ensemble learning method** that operates by constructing multiple decision trees during training and outputting the average of the trees' predictions for regression tasks. It reduces the risk of **overfitting** and is effective for both small and large datasets.

- **Usage**: Commonly used for structured/tabular data.
- **Strengths**: Handles missing data, reduces overfitting, and is less sensitive to outliers.
- **Parameters**:
  - `n_estimators`: The number of trees in the forest.
  - `max_depth`: The maximum depth of each tree.
  - `random_state`: Ensures reproducibility.

### 2. **XGBoost Regressor**
XGBoost (Extreme Gradient Boosting) is a **boosting algorithm** that focuses on sequentially correcting the errors made by previous models. It builds trees iteratively, where each tree tries to minimize the loss of the previous model's predictions.

- **Usage**: Suitable for large datasets, structured data, and competitions (e.g., Kaggle).
- **Strengths**: High speed, performance, and flexibility. Includes regularization to prevent overfitting.
- **Parameters**:
  - `n_estimators`: Number of boosting rounds.
  - `learning_rate`: Controls the contribution of each tree.
  - `max_depth`: Maximum depth of trees.

### 3. **LSTM (Long Short-Term Memory)**
LSTM is a type of **Recurrent Neural Network (RNN)**, well-suited for **time-series prediction**. It remembers patterns over long sequences, which is essential for forecasting problems.

- **Usage**: Used for sequential data (e.g., time series, speech recognition).
- **Strengths**: Able to capture long-term dependencies in data.
- **Parameters**:
  - `units`: The number of memory units (or cells) in the LSTM layer.
  - `input_shape`: Shape of the input data (timesteps, features).
  - `activation`: Activation function used in the LSTM cells.

### 4. **Fully Connected Neural Network**
A Fully Connected Neural Network (also known as a **Dense Neural Network**) consists of layers where each neuron is connected to every neuron in the previous and next layer. It's a basic structure for **deep learning models** and is useful when there's no inherent structure in the input data.

- **Usage**: Suitable for complex non-linear relationships and high-dimensional data.
- **Strengths**: Flexible in handling a variety of problems by adjusting the network architecture.
- **Parameters**:
  - `units`: The number of neurons in each layer.
  - `activation`: Activation functions control the output of each neuron (`relu`, `sigmoid`, etc.).

---

## Code Structure

The provided Python script consists of:
1. **Data Preparation**: Mock data generation for economic indicators like inflation, interest rates, unemployment rates, etc.
2. **Model Training**: Training four different models (Random Forest, XGBoost, LSTM, and a Fully Connected Neural Network) using the economic data.
3. **Model Evaluation**: Each model is evaluated using RMSE and performance is visualized with plots.
4. **Plotting**: Visualization of actual vs predicted GDP for each model.

---

## How to Use the Code

1. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib xgboost keras
   ```
2. Run the script:
   ```bash
   python gdp_prediction.py
   ```
3. The script will generate predictions using each model and print out the RMSE (Root Mean Squared Error) for each model, along with charts showing the predicted vs actual GDP.

---

## Model Explanations

### Random Forest
Random Forest creates multiple decision trees from randomly selected subsets of the training data. It takes the average of all tree predictions to make a final prediction, reducing the chances of overfitting and increasing robustness.

### XGBoost
XGBoost works by training models sequentially, each one correcting the mistakes of the previous one. It employs gradient descent to minimize the error in predictions.

### LSTM
LSTM networks are designed to remember patterns over long sequences of data. They are useful when predicting time series, as they are capable of learning long-term dependencies between time steps in data.

### Fully Connected Neural Network
A fully connected neural network, or Dense Neural Network, is the basic building block of deep learning. Each neuron in one layer is connected to every neuron in the next layer. It’s powerful for capturing complex non-linear patterns in data.

---

## Parameter Explanations

Each model has its own set of parameters that can be tuned:

### Random Forest Parameters:
- **n_estimators**: The number of trees in the forest (default: 100).
- **max_depth**: The maximum depth of a tree (default: None, meaning nodes are expanded until all leaves are pure).
- **random_state**: Controls the randomness of the estimator.

### XGBoost Parameters:
- **n_estimators**: The number of boosting rounds.
- **learning_rate**: Shrinks the contribution of each tree by the learning rate.
- **max_depth**: Controls the maximum depth of each tree.

### LSTM Parameters:
- **units**: The number of memory cells in the LSTM layer.
- **input_shape**: Defines the shape of the input (timesteps, features).
- **activation**: Determines the activation function applied to each LSTM cell.

### Neural Network Parameters:
- **units**: The number of neurons in a dense layer.
- **activation**: The activation function used in each layer (`relu` for hidden layers, no activation for output).

---

## Performance Evaluation

Each model's performance is measured using the **Root Mean Squared Error (RMSE)**. RMSE provides a measure of how far the predictions are from the actual values.

- **Random Forest RMSE**: Gives an understanding of how well the ensemble method of decision trees predicts GDP.
- **XGBoost RMSE**: Often outperforms other models on structured data due to its boosting nature.
- **LSTM RMSE**: Shows the network's ability to learn long-term dependencies.
- **Neural Network RMSE**: Displays how well the network can capture complex, non-linear relationships.

---

## Things to Keep in Mind

1. **Data Scaling**: When using neural networks (like LSTM or Fully Connected Networks), it's important to scale your data (e.g., between 0 and 1) for better performance.
2. **Overfitting**: Some models (like neural networks or XGBoost) are prone to overfitting if not regularized properly.
3. **Hyperparameter Tuning**: Each model's performance can be significantly improved by tuning its parameters (like the number of trees in Random Forest or the learning rate in XGBoost).
4. **Sequential Data**: For LSTM, it's important that the data have a temporal or sequential nature (e.g., time-series data).

---

## References

1. [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
2. [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
3. [Keras Documentation](https://keras.io/)
4. [Random Forests](https://en.wikipedia.org/wiki/Random_forest)
5. [LSTM Networks](https://en.wikipedia.org/wiki/Long_short-term_memory)
