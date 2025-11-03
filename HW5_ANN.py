import numpy as np # for matrix multiplication
import random

data = [
    ([0, 0, 0, 0], [0]),
    ([0, 0, 0, 1], [1]),
    ([0, 0, 1, 0], [0]),
    ([0, 0, 1, 1], [1]),
    ([0, 1, 0, 0], [0]),
    ([0, 1, 0, 1], [1]),
    ([0, 1, 1, 0], [0]),
    ([0, 1, 1, 1], [1]),
    ([1, 0, 0, 0], [1]),
    ([1, 0, 0, 1], [1]),
    ([1, 0, 1, 0], [1]),
    ([1, 0, 1, 1], [1]),
    ([1, 1, 0, 0], [0]),
    ([1, 1, 0, 1], [0]),
    ([1, 1, 1, 0], [0]),
    ([1, 1, 1, 1], [1])
]

# class definition adapted from:
# https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/#

class ANN():
  # network layer dimensions 
  INPUT_DIMENSION = 4
  HIDDEN_LAYER_DIMENSION = 8
  OUTPUT_DIMENSION = 1
  
  # network properties
  LEARNING_RATE = 0.1
  
  def __init__(self, input_size=INPUT_DIMENSION, hidden_size=HIDDEN_LAYER_DIMENSION, output_size=OUTPUT_DIMENSION):
    # define the network's layer dimensions
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    
    # initialize random weights as a matrix with proper dimensions
    self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
    self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
    
    # initialize biases for hidden layer and output layer
    self.bias_hidden = np.zeros((1, self.hidden_size))
    self.bias_output = np.zeros((1, self.output_size))
    
  
  """
  Sigmoid activation function. Takes one input and applies the sigmoid function to it.
  """
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  """
  Sigmoid Derivative
  """
  def sigmoid_derivative(self, x):
    return x * (1 - x)
  
  def feedforward(self, X):
    self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
    self.hidden_output = self.sigmoid(self.hidden_activation)
    
    self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
    self.predicted_output = self.sigmoid(self.output_activation)
    
    return self.predicted_output
  
  def backward(self, x, y, learning_rate):
    x = np.array([x]) # convert X to a 2D array from matrix multiplication purposes
    output_error = y - self.predicted_output
    output_delta = output_error * self.sigmoid_derivative(self.predicted_output)
    # print(output_delta, output_delta.shape)

    hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
    hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
    # print(hidden_delta, hidden_delta.shape)

    self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
    self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    # print(X)
    # print(X.shape)
    # print(X.T.shape, hidden_delta.shape)
    self.weights_input_hidden += np.dot(x.T, hidden_delta) * learning_rate
    self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
  
  def train(self, X, Y, epochs, learning_rate):
    average_loss = 0
    for epoch in range(epochs+1):
      
      losses = []
      for x, y in zip(X, Y):
        y_hat = self.feedforward(x)
        # perform backpropagation for every epoch
        losses.append(abs(y - y_hat))
        self.backward(x, y, learning_rate)
      
      # calculate the average loss of the epoch
      avg_loss = sum(losses) / len(losses)
      if epoch % 100 == 0:
        # print the average loss % every 100 epochs (should see a decrease in error)
        print("Epoch:\t", epoch, "Error:\t", avg_loss)
      
      # if the average loss is less than 0.05, stop training to prevent overfitting
      if avg_loss < 0.05:
        break
      
  
def main():
  nn = ANN()
  
  # randomly select 10 data points to train on
  train_set = random.sample(data, 10)
  
  # separate the training data into inputs and outputs
  X = np.array([inputs[0] for inputs in train_set])
  Y = np.array([output[1] for output in train_set])
  
  # train the nueral network
  nn.train(X, Y, epochs=50000, learning_rate=0.1)
  print("Training Complete.")
  
  
if __name__ == "__main__":
  main()