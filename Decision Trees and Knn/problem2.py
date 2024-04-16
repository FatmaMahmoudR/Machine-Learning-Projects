# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
# a) Load the "diabetes.csv" dataset
data = pd.read_csv('diabetes.csv')

# %%
#Data preprocessing
# Function to normalize data using Min-Max Scaling
def min_max_scaling(data):
    min_vals = data.min()
    max_vals = data.max()
    return (data - min_vals) / (max_vals - min_vals)

# %%
#Euclidean distance between two data points
def euclidean_distance(instance1, instance2):
    return np.sqrt(np.sum((instance1 - instance2)**2))

# %%
# Function to perform KNN classification
def knn_classifier(train_data, test_instance, k):
    distances = np.sqrt(np.sum((train_data.iloc[:, :-1].values - test_instance[:-1].values)**2, axis=1))
    
    #indices of k-nearest neighbors
    nearest_neighbors_indices = np.argpartition(distances, k)[:k]
    
    #classes of the k-nearest neighbors
    nearest_classes = train_data.iloc[nearest_neighbors_indices]['Outcome'].values
    
    #the most frequent class
    unique_classes, counts = np.unique(nearest_classes, return_counts=True)
    predicted_class = unique_classes[np.argmax(counts)]
    
    return predicted_class

def knn_classifier_v(train_data, test_instance, k):
    distances = np.sqrt(np.sum((train_data.iloc[:, :-1].values - test_instance[:-1].values)**2, axis=1))
    # Get the k-nearest neighbors directly without sorting
    nearest_neighbors_indices = np.argpartition(distances, k)[:k]
    nearest_distances = distances[nearest_neighbors_indices]
    
    # Apply distance weights to classes
    nearest_classes = train_data.iloc[nearest_neighbors_indices]['Outcome'].values
    weighted_votes = 1 / (nearest_distances + 1e-10)  # Add a small value to avoid division by zero
    class_votes = np.bincount(nearest_classes, weights=weighted_votes)
    
    # Break ties by selecting the class with the maximum weighted votes
    predicted_class = np.argmax(class_votes)
    
    return predicted_class



# %%
train_size = int(0.7 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# %%
train_data.iloc[:, :-1] = min_max_scaling(train_data.iloc[:, :-1])
test_data.iloc[:, :-1] = min_max_scaling(test_data.iloc[:, :-1])


# %%
accuracies = []
k_values = [2, 3, 4]

for k in k_values:
    correct_predictions = 0

    for index, test_instance in test_data.iterrows():
        predicted_class = knn_classifier_v(train_data, test_instance, k)
        actual_class = test_instance['Outcome']

        if predicted_class == actual_class:
            correct_predictions += 1

    total_instances = len(test_data)
    accuracy = (correct_predictions / total_instances) * 100
    
    accuracies.append(accuracy)

    print(f"k value: {k}")
    print(f"Number of correctly classified instances: {correct_predictions}")
    print(f"Total number of instances: {total_instances}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 40)


# %%
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average Accuracy across all iterations: {average_accuracy:.2f}%")


# %%



