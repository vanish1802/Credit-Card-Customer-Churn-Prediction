# **Credit Card Customer Churn Prediction**

This project focuses on predicting customer churn for a credit card company using a neural network model built with TensorFlow and Keras.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Acknowledgments](#acknowledgments)

---

## **Introduction**
Customer churn prediction is crucial for businesses to retain customers and reduce revenue loss. This project uses a dataset containing customer information to predict whether a customer will churn (exit) or not.

---

## **Dataset**
The dataset used in this project is **[Credit Card Customer Churn Prediction](https://www.kaggle.com/rjmanoj/credit-card-customer-churn-prediction)**. It contains the following features:
- **Demographic Information**: Geography, Gender, Age.
- **Account Details**: Credit Score, Balance, Number of Products, etc.
- **Target Variable**: `Exited` (1 = Churned, 0 = Retained).

---

## **Installation**
To run this project locally:
1. Clone the repository:
   ```bash
   git clone 
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script.

---

## **Project Workflow**
1. **Data Preprocessing**:
   - Load the dataset.
   - Handle categorical variables using one-hot encoding (e.g., `Geography` and `Gender`).
   - Scale numerical features using `StandardScaler`.

2. **Model Building**:
   - A Multilayer Perceptron (MLP) model is built using TensorFlow/Keras.
   - The architecture includes:
     - Input Layer: Features from the dataset.
     - Hidden Layers: Fully connected layers with ReLU activation.
     - Output Layer: A single neuron with sigmoid activation for binary classification.

3. **Model Training**:
   - The model is trained using the Adam optimizer and binary cross-entropy loss for 100 epochs.

4. **Evaluation**:
   - Evaluate the model on the test set using accuracy as the metric.
   - Plot training and validation loss over epochs.

5. **Prediction**:
   - Use the trained model to predict churn probabilities and classify customers as churned or retained.

---

## **Results**
- The model achieved an accuracy of **85.95%** on the test set.
- Loss trends during training and validation are visualized to ensure proper convergence.

---

## **Technologies Used**
- Python 3
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## **Acknowledgments**
Special thanks to Kaggle for providing the dataset:  
[Credit Card Customer Churn Prediction Dataset](https://www.kaggle.com/rjmanoj/credit-card-customer-churn-prediction).

---
