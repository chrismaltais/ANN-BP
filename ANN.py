import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

def backprop(x_inputs, expected_output, layer1_actual_output, layer2_actual_output, layer1_weights, layer2_weights, learning_rate):
    # Find the change in weights with sum of squares loss function
    # From chapter 7 lecture slide 25:
    error = (expected_output - layer2_actual_output)
    #print('Layer 2 outputs is: ', layer2_actual_output)
    #print('Actual output is: ', expected_output)
    #print('Error is: ', error)
    delta_o = 2*(expected_output - layer2_actual_output) * sigmoid_derivative(layer2_actual_output)
    delta_h = np.dot(delta_o, layer2_weights.T) * sigmoid_derivative(layer1_actual_output)

    d_weights2 = np.dot(layer1_actual_output.T, delta_o)
    d_weights1 = np.dot(x_inputs.T, delta_h)
    print('Dweights1 before learning rate: ', d_weights1)
    print('Dweights2 before learning rate: ', d_weights2)

    
    d_weights2 = np.multiply(-learning_rate, d_weights2)
    d_weights1 = np.multiply(-learning_rate, d_weights1)

    #print('Delta weights2: ', d_weights2)

    # Update weights w derivative of sum of squares loss function
    layer1_weights += d_weights1
    layer2_weights += d_weights2

    print('Layer 1 weights: ', layer1_weights)
    print('Layer 2 weights: ', layer2_weights)

    return layer1_weights, layer2_weights

# Set max element in array equal to 1 and all other elements 0
def find_max(array_of_vectors):
    # Copy structure of array of vectors but set everything to 0
    results = np.zeros_like(array_of_vectors)
    # Sets the max value in each vector to 1
    results[np.arange(len(array_of_vectors)), array_of_vectors.argmax(1)] = 1
    return results

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

    # Create initial weights
    layer_1_initial_weights, layer_2_initial_weights = initialize_weights(x_test, num_hidden_layer_nodes, y_test_encoded)

    layer1_weights = layer_1_initial_weights
    layer2_weights = layer_2_initial_weights

    # Define learning rate
    learning_rate = 0.1
   
    for x in range(0, 1000):
        # Feedforward (No correction)
        layer1_output, output = feed_forward(x_test_std, layer1_weights, layer2_weights)

        # Get output in proper form
        encoded_output = find_max(output)
        score = accuracy_score(encoded_output, y_test_encoded)
        print('Score is: ', score)

        # Recalculate weights
        layer1_weights, layer2_weights = backprop(x_test_std, y_test_encoded, layer1_output, output, layer1_weights, layer2_weights, learning_rate)

    

# Where d is the predicted output of the entire network:
# delta_layer2_weights = (d - output_of_network) * sigmoid_derivative(output_of_network)
# delta_layer1_weights = (d - output_of_hidden_layer) * sigmoid_derivative(output_of_hidden_layer)
# delta_layer1_weights = delta_layer2_weights * weights_from_hidden_to_output * sigmoid(output of hidden lager)