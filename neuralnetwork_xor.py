#two-layer neural network for classification

import numpy as np  # matrix multiplication, important for neural network
import time     #time how long training will be

#variables
n_hidden = 10  #number of hidden neurons, 10 1&0s comparison
n_in = 10 
#outputs
n_out = 10
#sample data
n_sample = 300 #300 samples generated

#hyperparameters
learning_rate = 0.01 #how fast we want it to learn 2d knot
momentum = 0.9 #lower loss function called cross entropy 

#non deterministic seeding
np.random.seed(0) #generate random data, make sure everytime to generate the same random numbers

#activation function - is the sigmoid function, run on every neuron
#sigmoid function turns numbers into probabilities 
#neural network, each weight is a probability, probability is updated
#every time input data hits neuron and layers, turn that number into a probability
#usually one activation function
#two this time, for XOR specifically tangent function is helpful, makes loss better
#activation function each for two layers

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
    return 1 - np.tanh(x)**2

#training function
#x is input data 
#t is transpose for matrix multiplication
#V layer #1
#W layer #2
#bv bias, help make more accurate predication, one bias for each layer in our network
#bw bias

#input data, transpose, layer 1, layer 2, biases
def train(x, t, V, W, bv, w) :
    #forward propogation -- matrix multiply + biases
    #simple perceptron, top 1%, feed-forward neural network (deep learning)
    #been around since the 50s
    
    #A - taking dot product of the input data x, put into first layer v, compute matrix multiplication and account for the bias, Activation function on data
    
    ## forward propagation --matrix multiply + bias
    A = np.dot(x, V) + bv  #
    Z = np.tanh(A)
    
    #B take computation and dot product of first for the second and second bias
    B = np.dot(Z, W) + bw 
    Y = sigmoid(B)
    
    #Feed-Forward Neural Network is not recurrent but has backward propagation, goes forward and backwards, update weights one way and backward
    
    ##backward propagation
    Ew = Y - t  #two deltas and transpose = matrix of weights flipped, and we flip because we're going backwards, use to calculate backward propagation
    Ev = tanh_prime(A) + np.dot(W, Ew) #Ultimately we want this Ev value, to calculate our loss, compare predicated with actual to minimize our loss
    
    ##predict our loss - both deltas using Z value, x-input
    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)
    
    #cross entropy 
    #t is a transpose
    #using cross entropy because classification, usually use cross entropy, better results
    #math, you don't need math to get good, but you should, series of abstractions, but helps 
    loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1-Y))
    
    return loss, (dV, dW, Ev, Ew)

#prediction step - means we're going to perform matrix multiplication to predict value, end result
def predict(x, V, W, bv, bw):
    
    #A & B are our final values using var we already use
    #return is our prediction, which is a 1 or 0, 0.5 or greater is going to return a 1, else 0
    #probability

    A = np.dot(x, V) + bv  
    B = np.dot(np.tanh(A), W) + bw
    return (sigmoid(B) > 0.5).astype(int)

#create layers
#size is the number on input layers, input values ten, 10 in and 10 out
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

#create biases
bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params= [V, W, bv, bw]

#generate our data
#sample of 300 samples is used here
#transpose 
X = np.random.binomial(1, 0.5, (n_sample, n_in))
T = X ^ 1

#TRAINING TIME
#100 epoch
for epoch in range(100):
    err = []
    upd = [0]*len(params)

    t0 = time.clock()
    #for each data point, update our weights of our network
    #neural network is to find XOR of 1 and 0s
    
    #shape of x input data 
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i], *params)

        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]

        err.append( loss )

    print "Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 )

# Try to predict something

x = np.random.binomial(1, 0.5, n_in)
print "XOR prediction:"
print x
print predict(x, *params)

