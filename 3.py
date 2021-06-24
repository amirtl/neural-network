import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
import sys

def max_min(matrix):
    min_x = sys.maxsize
    max_x = -sys.maxsize
    min_y = sys.maxsize
    max_y = -sys.maxsize
    for i in range(len(matrix)):
        for j in range(2):
            min_x = min(min_x, matrix[i][0])
            max_x = max(max_x,matrix[i][0])
            min_y = min(min_y,matrix[i][1])
            max_y = max(max_y,matrix[i][1])
    
    return min_x,max_x,min_y,max_y


def make_matrix(x1, x2, y1, y2, rate):
    a = []
    i = y1
    while i <= y2:
        j = x1
        while j <= x2:
            a.append([j,i])
            j+= rate        
        i += rate
    
    return np.array(a)


def decision_boundry(x1, x2, y1, y2, rate, y):
    j = x1
    k = y1
    x = []
    z = []
    x3 = []
    z3 = []
    for i in range(len(y)):
        if y[i] == 1:
            x.append(j)
            z.append(k)
        else:
            x3.append(j)
            z3.append(k)
        j += rate
        if j > x2:
            j = x1
            k += rate
    plt.plot(x,z,'ro')
    plt.plot(x3,z3,'o')
    plt.ylabel('blue : 0')
    plt.xlabel('red : 1')
    plt.show()


def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    
    return 1./num_examples * data_loss


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    return np.argmax(probs, axis=1)


#This function learns parameters for the neural network and returns the model.
#- nn_hdim: Number of nodes in the hidden layer
#- num_passes: Number of passes through the training data for gradient descent
#- print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):# Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    # This is what we return at the end
    model = {}
    # Gradient descent. For each batch...
    for i in range(0, num_passes):
        j = 0
        X, y = shuffle(X, y, random_state=0)
        while j < len(X):
            X1 = X[j:j+num_batch]
            y1 = y[j:j+num_batch]
            j += num_batch
            # Forward propagation
            z1 = X1.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            # Backpropagation
            delta3 = probs
            delta3[range(num_batch), y1] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X1.T, delta2)
            db1 = np.sum(delta2, axis=0)
            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1
            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2
            # Assign new parameters to the model
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" %(i, calculate_loss(model)))

    return model



# Generate a dataset and plot it
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
num_examples = 200 # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
num_batch = 50 # minibatch Gradient descent parameters
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength
model = build_model(X, y, 3, 20000, True)
a = max_min(X)
b = make_matrix(a[0],a[1],a[2],a[3], 0.01)
l = predict(model, b)
decision_boundry(a[0],a[1],a[2],a[3], 0.01, l)
