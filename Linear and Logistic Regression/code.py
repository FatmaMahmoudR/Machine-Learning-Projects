import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# a) Load the "loan_old.csv" dataset
data = pd.read_csv('loan_old.csv')

# i) Check for missing values
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# ii) Check the type of each feature
print("\nData Types:\n")
for column in data.columns:
    if data[column].dtype == 'int64' or data[column].dtype == 'float64':
        print(f"{column: <20}: numerical")
    else:
        print(f"{column: <20}: categorical")
        

# iii) Check whether numerical features have the same scale
numerical_features = ['Income', 'Coapplicant_Income', 'Loan_Tenor', 'Credit_History', 'Max_Loan_Amount']

# Calculate the ranges
ranges = data[numerical_features].max() - data[numerical_features].min()

# Calculate the standard deviations
std_devs = data[numerical_features].std()

for feature in numerical_features:
    print(f"Range of {feature}: {ranges[feature]}")
    print(f"Standard deviation of {feature}: {std_devs[feature]}")
    print()

# histograms for each feature
data[numerical_features].hist(bins=10, figsize=(12, 6))
plt.show()

# iv) Visualize a pairplot between numerical columns
sns.pairplot(data, diag_kind='kda')

plt.show

# c) Data Preprocessing

# i) Remove records containing missing values
data = data.dropna()

# ii) Separate features and targets
features = data.drop(columns=['Max_Loan_Amount', 'Loan_Status'])
target = data[['Max_Loan_Amount', 'Loan_Status']]


print("\n----------- features -------------\n")
print(features)
print("\n----------- target ---------------\n")
print(target)

# iii) Shuffle and split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

#print(x_train.count())
#print(y_train)
#print(x_test.count())
#print(y_test)


# iv) Encode categorical features
label_encoder = LabelEncoder()
categorical_columns = features.select_dtypes(include='object').columns

# Exclude 'Loan_ID' from encoding
categorical_columns = categorical_columns.drop('Loan_ID', errors='ignore')

for column in categorical_columns:
    x_train[column] = label_encoder.fit_transform(x_train[column])
    x_test[column] = label_encoder.transform(x_test[column])


print("\n------------ Features (encoded) ------------- \n")
print("\nx_train :")
print(x_train.head())
print("\n--------------------------------------------- \n")
print("\nx_test :")
print(x_test.head())


# v) Encode categorical targets
encoder = LabelEncoder()

y_train['Loan_Status'] = encoder.fit_transform(y_train['Loan_Status'])
y_test['Loan_Status'] = encoder.fit_transform(y_test['Loan_Status'])

print(y_train.head())
print(y_test.head())

# vi) Scale numerical features to the range [0, 1]
scaler = MinMaxScaler()
numerical_features = features.select_dtypes(include=['float64', 'int64']).columns

for i in numerical_features:
    x_train[i] = scaler.fit_transform(x_train[[i]])
    x_test[i] = scaler.transform(x_test[[i]])

print("\n----------  Numerical features (Scaled)  -----------\n")
print("x_train :")
print(x_train.head())
print("\n----------------------------------------------------\n")
print("x_test :")
print(x_test.head())

print("\n****************************************************\n")
print("\n****************************************************\n")
#Laod_ID is string so i drop it
x_train = x_train.drop(columns=['Loan_ID'])
x_test = x_test.drop(columns=['Loan_ID'])

# d)Fit a linear regression model to the data to predict the loan amount.
linear_model = LinearRegression()
linear_model.fit(x_train, y_train['Max_Loan_Amount'])
#predict the loan amount.
y_pred = linear_model.predict(x_test)
print("Predicted Loan Amounts:")
print(y_pred)

# e) Evaluate the linear regression model using sklearn's R2 score.
r2 = r2_score(y_test['Max_Loan_Amount'], y_pred)
print("R2 Score: ")
print(r2)
print("\n****************************************************\n")



##(f) Fit a logistic regression model to the data to predict the loan status.
# Implement logistic regression from scratch using gradient descent.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta = theta - learning_rate * gradient
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history
# Convert target variable to binary (0 or 1)
y_train_binary = y_train['Loan_Status']
# Add a bias term to the features
x_train_logistic = np.c_[np.ones((len(x_train), 1)), x_train]
theta_initial = np.zeros(x_train_logistic.shape[1])
# Set hyperparameters
learning_rate = 0.01
iterations = 1000
theta_final, cost_history = gradient_descent(x_train_logistic, y_train_binary, theta_initial, learning_rate, iterations)
x_test_logistic = np.c_[np.ones((len(x_test), 1)), x_test]
probabilities = sigmoid(np.dot(x_test_logistic, theta_final))
# Convert probabilities to binary predictions (0 or 1)
predictions_logistic = [1 if p >= 0.5 else 0 for p in probabilities]

def predict_logistic(X, weights, bias):
    logits = np.dot(X, weights) + bias
    probabilities = sigmoid(logits)
    return np.round(probabilities)

def calculate_accuracy_logistic(predictions, targets):
    correct_predictions = np.sum(predictions == targets)
    total_instances = len(targets)
    accuracy = correct_predictions / total_instances
    return accuracy

# Perform gradient descent
theta_final_logistic, cost_history_logistic = gradient_descent(x_train_logistic, y_train_binary, theta_initial, learning_rate, iterations)

# Add a bias term to the test features for logistic regression
x_test_logistic = np.c_[np.ones((len(x_test), 1)), x_test]
probabilities_logistic = sigmoid(np.dot(x_test_logistic, theta_final_logistic))
predictions_logistic = predict_logistic(x_test_logistic, theta_final_logistic, 0)
accuracy_logistic = calculate_accuracy_logistic(predictions_logistic, y_test['Loan_Status'])
print(f"\nAccuracy of Logistic Regression (from scratch): {accuracy_logistic*100}%")


#read new data
data2 = pd.read_csv('loan_new.csv')

# i) Check for missing values for new data set
missing_values = data2.isnull().sum()
print(missing_values[missing_values > 0])

# ii) Check the type of each feature
print("\nData Types of new data:\n")
for column in data2.columns:
    if data2[column].dtype == 'int64' or data2[column].dtype == 'float64':
        print(f"{column: <20}: numerical")
    else:
        print(f"{column: <20}: categorical")

# iii) Check whether numerical features have the same scale
numerical_features = ['Income', 'Coapplicant_Income', 'Loan_Tenor', 'Credit_History']

# Calculate the ranges for new data
ranges = data2[numerical_features].max() - data2[numerical_features].min()

# Calculate the standard deviations for new data
std_devs = data2[numerical_features].std()
print("New data Ranges & standerd deviations")
for feature in numerical_features:
    print(f"Range of {feature}: {ranges[feature]}")
    print(f"Standard deviation of {feature}: {std_devs[feature]}")
    print()

# histograms for each feature
data2[numerical_features].hist(bins=10, figsize=(12, 6))
plt.show()

# iv) Visualize a pairplot between numerical columns
sns.pairplot(data2, diag_kind='kda')

plt.show()


# c) Data Preprocessing

# i) Remove records containing missing values
data2 = data2.dropna()

# iv) Encode categorical features in new data
label_encoder = LabelEncoder()
categorical_columns = data2.select_dtypes(include='object').columns

categorical_columns = categorical_columns.drop('Loan_ID', errors='ignore')


for column in categorical_columns:
    data2[column] = label_encoder.fit_transform(data2[column])


print("\n------------ data2 (encoded) ------------- \n")
print(data2.head())
print("\n--------------------------------------------- \n")


#scale numrical new data
scaler = MinMaxScaler()
numerical_features = features.select_dtypes(include=['float64', 'int64']).columns

for i in numerical_features:
    data2[i] = scaler.fit_transform(data2[[i]])

print("\n----------  Numerical features (Scaled)  -----------\n")
print("data2 :")
print(data2.head())
print("\n****************************************************\n")

#drop string
data2 = data2.drop(columns=['Loan_ID'])

#linear regression prediction on new data to pridict Max loan amount
y_pred2 = linear_model.predict(data2)

print("Predicted Max loan amount in new data")
print (y_pred2)
##(f) Fit a logistic regression model to the new data to predict the loan status.
def sigmoid2(z):
    return 1 / (1 + np.exp(-z))
def compute_cost2(X, y, theta):
    m = len(y)
    h = sigmoid2(np.dot(X, theta))
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent2(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        h = sigmoid2(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta = theta - learning_rate * gradient
        cost_history[i] = compute_cost2(X, y, theta)

    return theta, cost_history
# Convert target variable to binary (0 or 1)
y_train_binary = y_train['Loan_Status']
# Add a bias term to the features
x_train_logistic = np.c_[np.ones((len(x_train), 1)), x_train]
theta_initial = np.zeros(x_train_logistic.shape[1])
# Set hyperparameters
learning_rate = 0.01
iterations = 1000
theta_final, cost_history = gradient_descent2(x_train_logistic, y_train_binary, theta_initial, learning_rate, iterations)
x_test_logistic = np.c_[np.ones((len(data2), 1)), data2]
probabilities = sigmoid(np.dot(x_test_logistic, theta_final))
# Convert probabilities to binary predictions (0 or 1)
predictions_logistic = [1 if p >= 0.5 else 0 for p in probabilities]

print (predictions_logistic)