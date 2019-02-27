# Name: Chris Maltais
# Student Number: 10155183
# Header: This file is used to both train the ANN BP Model, as well as test the model

import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.special import softmax

def initialize_weights(input_values, num_hidden_layer_nodes, output_values):
    # + 1's account extra weight that becomes bias
    layer_1_weights = np.random.rand(input_values.shape[1], num_hidden_layer_nodes + 1)
    layer_2_weights = np.random.rand(num_hidden_layer_nodes + 1, output_values.shape[1])
    return layer_1_weights, layer_2_weights

def standardize_inputs(x_train, x_validate, x_test):
    # Standard Scalar is used to standardize data
    sc = StandardScaler()
    # Train the scaler on the training data
    sc.fit(x_train)
    # Apply scaler to feature training data
    x_train_std = sc.transform(x_train)
    # Apply scaler to feature training data
    x_test_std = sc.transform(x_test)
    # Apply scaler to feature training data
    x_validate_std = sc.transform(x_validate)
    return x_train_std, x_validate_std, x_test_std

def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

def feed_forward(x_inputs, layer_1_weights, layer_2_weights):
    layer1_output = sigmoid(np.dot(x_inputs, layer_1_weights))
    output = sigmoid(np.dot(layer1_output, layer_2_weights)) #, axis=1)
    return layer1_output, output

def backprop(x_inputs, expected_output, layer1_actual_output, layer2_actual_output, layer1_weights, layer2_weights, d_weights1_old, d_weights2_old, learning_rate):
    # Define Mu for Momentum
    mu = 0.5
    
    # Find the change in weights with sum of squares loss function
    # From chapter 7 lecture slide 25:
    delta_o = 2 * (expected_output - layer2_actual_output) * sigmoid_derivative(layer2_actual_output)
    delta_h = np.dot(delta_o, layer2_weights.T) * sigmoid_derivative(layer1_actual_output)

    # Change in Layer 2 weights = c * (Layer 1 outputs) * delta_o + mu * (old change in Layer 2 weights)
    d_weights2 = (learning_rate) * np.dot(layer1_actual_output.T, delta_o) + mu * d_weights2_old

    # Change in Layer 1 weights = c * X * delta_h + mu + (old change in Layer 1 weights)
    d_weights1 = (learning_rate) * np.dot(x_inputs.T, delta_h) + mu * d_weights1_old

    layer1_weights += d_weights1 
    layer2_weights += d_weights2 

    return layer1_weights, layer2_weights, d_weights1, d_weights2

# Set max element in array equal to 1 and all other elements 0
# Range goes from 0-5
def find_max(array_of_vectors):
    # Copy structure of array of vectors but set everything to 0
    results = np.zeros_like(array_of_vectors)
    # Sets the max value in each vector to 1
    results[np.arange(len(array_of_vectors)), array_of_vectors.argmax(1)] = 1
    return results

def generate_performance_stats(confusion_matrix, classification_report, layer1_weights, layer2_weights, training_accuracy, testing_accuracy, num_epochs_elapsed, terminating_condition, predicted_test_output_category, actual_test_output_category):
    filename = 'results/performanceStats.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Convert results to strings to write to file...
    target_test_string = " ".join(str(x) for x in actual_test_output_category)
    predicted_test_target_string = " ".join(str(x) for x in predicted_test_output_category)
    # Write to file
    with open(filename, 'w+') as f:
        wr = csv.writer(f)
        f.write('Performance Statistics:\n\n')
        f.write('Training Set Accuracy: %.2f \n' % (training_accuracy * 100))
        f.write('Testing Set Accuracy: %.2f \n\n' % (testing_accuracy * 100))
        f.write('Final weights:\n')
        f.write('Layer One:\n')
        wr.writerows(layer1_weights)
        f.write('\nLayer Two:\n')
        wr.writerows(layer2_weights)
        f.write('\n\nNumber of Epochs to train the model:\n')
        f.write(str(num_epochs_elapsed))
        f.write('\n\n')
        f.write('Termination Criteria: ')
        f.write(terminating_condition)
        f.write('\n\n')
        f.write('Testing Targets:\n')
        f.write(target_test_string)
        f.write('\n')
        f.write('Predicted Testing Targets:\n')
        f.write(predicted_test_target_string)
        f.write('\n\nConfusion Matrix:\n')
        f.write(np.array2string(confusion_matrix))
        f.write('\n\n')
        f.write('Classification Report:\n')
        f.write(classification_report)
        f.write('\n')

if __name__ == "__main__":
    # Import Dataset
    dataset = pd.read_csv('data/GlassData.csv')
    
    # Extract feature values (X) from dataset
    dataset_X_values = dataset.drop(['ID', 'Glass_Type'], axis=1)

    # Add bias to dataset
    dataset_X_values['Bias'] = 1

    # Extract label values (Y) from dataset
    dataset_Y_values = dataset['Glass_Type']

    # Split data into 85% training and 15% testing using stratified sampling (to ensure statistically equivalent datasets)
    x_train, x_test, y_train, y_test = train_test_split(dataset_X_values, dataset_Y_values, test_size=0.15, stratify=dataset_Y_values)

    # Split training data into training and validation. 
    # 17.76% of 85% yields a 70/15/15 train/val/test split
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.1776, stratify=y_train)

    # Convert categorical variable into indicator variables
    y_train_encoded = pd.get_dummies(y_train, prefix='Glass_Type')
    y_test_encoded = pd.get_dummies(y_test, prefix='Glass_Type')
    y_validate_encoded = pd.get_dummies(y_validate, prefix='Glass_Type')

    # Standardize inputs
    x_train_std, x_validate_std, x_test_std = standardize_inputs(x_train, x_validate, x_test)

    # Define number of nodes in the hidden layer
    # Good estimate is mean(Input nodes, Output nodes)
    num_hidden_layer_nodes = 8

    # Create initial weights and save into variables
    weights1, weights2 = initialize_weights(x_test, num_hidden_layer_nodes, y_test_encoded)
    
    layer1_initial_weights = weights1
    layer2_initial_weights = weights2

    layer1_weights = weights1
    layer2_weights = weights2

    # Define learning rate
    learning_rate = 0.001

    # Set initial d_weights
    d_weights1 = 0
    d_weights2 = 0
   
    # Define number of epochs
    epochs = 5000
    num_epochs_elapsed = 0

    # Define terminating condition
    terminating_condition = '5000 Epochs'

    # Iterate through epochs
    for x in range(0, epochs):
        # Feedforward (No correction)
        layer1_output, output = feed_forward(x_train_std, layer1_weights, layer2_weights)

        # Get output in proper form
        encoded_output = find_max(output)

        # Calculate accuracy of model against training data
        score = accuracy_score(encoded_output, y_train_encoded)

        # Control accuracy output to be a reasonable amount
        if (num_epochs_elapsed % 50 == 0):
            print('Score is: ', score)

        # Define terminating condition as "high enough" training accuracy
        if (score > 0.75):
            terminating_condition = '> 75% Training Accuracy'
            break

        # Recalculate weights using backprop
        layer1_weights, layer2_weights, d_weights1, d_weights2 = backprop(x_train_std, y_train_encoded, layer1_output, output, layer1_weights, layer2_weights, d_weights1, d_weights2, learning_rate)
        
        # Increase epoch count
        num_epochs_elapsed  = num_epochs_elapsed + 1

    ### Test model ###
    # Feedforward (No correction)
    layer1_output, output = feed_forward(x_test_std, layer1_weights, layer2_weights)

    # Get output in proper form
    encoded_output = find_max(output)
    test_score = accuracy_score(encoded_output, y_test_encoded)
    print('Score is for testing is: ', test_score)

    # List of predicted test categorical glass outputs obtained from model
    predicted_test_output_category = [np.argmax(x) + 1 for x in encoded_output]

    # Re-label data over 3 to be n + 1 (sloppy post-processing to account for no label "4")
    for i in range(len(predicted_test_output_category)):
        if (predicted_test_output_category[i] > 3):
            predicted_test_output_category[i] += 1
    
    print(predicted_test_output_category)

    # List of actual test categorical glass outputs obtained from data
    actual_test_output_category = y_test.tolist()

    # Calculate confusion matrix and classification report for recall and precision
    # Confusion Matrix
    glass_types = [1, 2, 3, 5, 6, 7]
    confusion_matrix_results = confusion_matrix(actual_test_output_category, predicted_test_output_category, labels=glass_types)

    # Classification report
    classification_report_results = classification_report(actual_test_output_category, predicted_test_output_category)

    generate_performance_stats(
        confusion_matrix_results, 
        classification_report_results, 
        layer1_weights, 
        layer2_weights, 
        score, 
        test_score, 
        num_epochs_elapsed, 
        terminating_condition,
        predicted_test_output_category,
        actual_test_output_category
    )