# %%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

data =pd.read_csv("drug.csv")
data.head()
data.info()
print("Missing Values:\n", data.isnull().sum())
freq_value_BP = data['BP'].mode()[0]
data['BP'] = data['BP'].fillna(freq_value_BP)

freq_value_Cholesterol = data['Cholesterol'].mode()[0]
data['Cholesterol'] = data['Cholesterol'].fillna(freq_value_Cholesterol)

mean_Na_to_K = data['Na_to_K'].mean()
data['Na_to_K'] = data['Na_to_K'].fillna(mean_Na_to_K)


print("After Handling Missing Values:\n", data.isnull().sum())
print(data)
encoder = LabelEncoder()
data['Sex'] = encoder.fit_transform(data['Sex'])
data['BP']=encoder.fit_transform(data['BP'])
data['Cholesterol'] = encoder.fit_transform(data['Cholesterol'])
data['Drug']=encoder.fit_transform(data['Drug'])
print("Updated Dataset:\n", data.head())
############First experiment##################

best_accuracy = 0.0
for i in range(5):
    X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]  
    y = data['Drug']
    # Split the data into features (X) and target variable (y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Experiment {i + 1}:")
    print(f"Size of Training Set: {len(X_train)}")
    print(f"Size of Testing Set: {len(X_test)}")
    print(f"Accuracy: {accuracy}")
    print(f"Tree Size: {model.tree_.node_count}\n")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
print(f"Best Model Accuracy: {best_accuracy}")
print("Best Model Tree Size:", best_model.tree_.node_count)
print("Best Model:")
print(best_model)
plt.figure(figsize=(15, 10))
plot_tree(best_model, feature_names=X_train.columns, class_names=[str(i) for i in best_model.classes_], filled=True, rounded=True)
plt.show()





# %%
# Second experiment: Training and Testing with a Range of Train-Test Split Ratios

train_sizes = range(30, 71, 10)  # Train set sizes from 30% to 70%
results = {'Train_Size': [], 'Mean_Accuracy': [], 'Max_Accuracy': [], 'Min_Accuracy': [], 'Mean_Tree_Size': [],
           'Max_Tree_Size': [], 'Min_Tree_Size': []}

for train_size in train_sizes:
    accuracies = []
    tree_sizes = []
    for i in range(5):
        X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
        y = data['Drug']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_size) / 100, random_state=i)
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        tree_sizes.append(model.tree_.node_count)
    
    results['Train_Size'].append(train_size)
    results['Mean_Accuracy'].append(sum(accuracies) / len(accuracies))
    results['Max_Accuracy'].append(max(accuracies))
    results['Min_Accuracy'].append(min(accuracies))
    results['Mean_Tree_Size'].append(sum(tree_sizes) / len(tree_sizes))
    results['Max_Tree_Size'].append(max(tree_sizes))
    results['Min_Tree_Size'].append(min(tree_sizes))

# Storing statistics in a report
report = pd.DataFrame(results)
print("Experiment Report:")
print(report)

# Creating plots
plt.figure(figsize=(10, 6))

# Plotting accuracy against training set size
plt.subplot(1, 2, 1)
plt.plot(results['Train_Size'], results['Mean_Accuracy'], label='Mean Accuracy')
plt.plot(results['Train_Size'], results['Max_Accuracy'], label='Max Accuracy')
plt.plot(results['Train_Size'], results['Min_Accuracy'], label='Min Accuracy')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Set Size')
plt.legend()

# Plotting number of nodes in the final tree against training set size
plt.subplot(1, 2, 2)
plt.plot(results['Train_Size'], results['Mean_Tree_Size'], label='Mean Tree Size')
plt.plot(results['Train_Size'], results['Max_Tree_Size'], label='Max Tree Size')
plt.plot(results['Train_Size'], results['Min_Tree_Size'], label='Min Tree Size')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Number of Nodes in Tree')
plt.title('Tree Size vs Training Set Size')
plt.legend()

plt.tight_layout()
plt.show()



# %%
