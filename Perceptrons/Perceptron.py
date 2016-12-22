"""
    Title: Multilayer Perceptron Example
    Author: Jones Agwata
    Student ID: 200899592

"""
import numpy as np
import math
import matplotlib.pyplot as plt

def train_perceptron(w,x,eta):
    """
        Function to train perceptron layer
        Params: Weight w, Dataset x, learning rate eta
        returns Weight w
    """
    for a,b,c,d in x:
        points = np.array([a,b,c])
        target = d
        result = np.sum(points.T*w)
        if result > 0:
            y = 1
        else:
            y = 0

        w  -= eta * np.dot(points,(y-target))
        return w


def evaluate_perceptron(w,inputs):
    """
        Function to train perceptron layer
        Params: Weight w, Dataset inputs
    """
    errors = 0
    res = []
    target = inputs[:,3]
    points = inputs[:,0:3]

    result = np.dot(points,w)


    for i in range(len(result)):
        if result[i] > 0:
            res.append(1)
        else:
            res.append(0)
    res = np.array(res)

    dev = abs(res - target)
    error = 0.5*np.sum(dev)

    return error
def g(x):
    return 1/(1+(math.e**(-x)))
def train_mlp(w_in,w_out,b,D,eta,h):
    '''
        Function to train multilayer perceptron that
        accepts two weight matrices, a bias input, a matrix of points
        a learning rate and number of hidden neurons.
    '''
    #Specifies input values and target values
    inp = D[:,0:2]
    c = []
    tar= D[:,3]
    #creates new array for mlp output
    for i in tar:
        if i == 0:
            c.append([1,0,0,0])
        if i == 1:
            c.append([0,1,0,0])
        if i == 2:
            c.append([0,0,1,0])
        if i == 3:
            c.append([0,0,0,1])
    c = np.array(c)

    #parses bias for the various layers
    b_in = b[0:h]
    b_out = b[h:h+4]

    #Forward propagation,
    #hidden layer  input and output after
    #computing activation
    hi_in = np.dot(inp,w_in) + b_in
    hi_out = g(hi_in)

    #ouptut layer input and output after
    ##computing activation
    out_in = np.dot(hi_out,w_out) + b_out
    out_out = g(out_in)
    #computes difference between output and target
    dev = out_out-c
    #Begin back propagation
    #computes activation of output to hidden layer
    g_outin = dev * out_out * (1.0 - out_out)
    #computes new activation of hidden layer
    g_hi_out = np.dot(g_outin, w_out.T)
    g_hi_in = g_hi_out * (1 - hi_out**2)
    #updates new bias values
    b_in -= eta * np.sum(g_hi_in, axis=0)
    b_out -= eta * np.sum(g_outin, axis=0)

    #Update new weights
    w_in -= eta * np.dot(inp.T, g_hi_in)

    w_out -= eta * np.dot(hi_out.T, g_outin)

    b = np.concatenate((b_in,b_out))

    return w_in,w_out,b
def evaluate_mlp(w_in,w_out,b, D, h):
    """
        Function to evaluate accuracy of mlp using weights generated
        by training the perceptron
    """
    #Specifies input values and target values
    inp = D[:,0:2]
    c = []
    tar= D[:,3]
    #creates new array for mlp output
    for i in tar:
        if i == 0:
            c.append([1,0,0,0])
        if i == 1:
            c.append([0,1,0,0])
        if i == 2:
            c.append([0,0,1,0])
        if i == 3:
            c.append([0,0,0,1])
    c = np.array(c)

    #parses bias for the various layers
    b_in = b[0:h]
    b_out = b[h:h+4]

    #Forward propagation,
    #hidden layer  input and output after
    #computing activation
    hi_in = np.dot(inp,w_in) + b_in
    hi_out = g(hi_in)

    #ouptut layer input and output after
    ##computing activation
    out_in = np.dot(hi_out,w_out) + b_out
    out_out = g(out_in)

    #computes difference between output and target
    dev = out_out-c
    #calculates overall error
    error = 0.5*np.sum(dev**2)


    return error
