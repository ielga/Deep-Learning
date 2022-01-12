#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        yhat  = self.predict(x_i)
        if yhat != y_i:
             self.W[y_i] +=  x_i 
             self.W[yhat] -= x_i
      


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        yhat = self.predict(x_i) 
        
        self.W = self.W - learning_rate * ( 2*(yhat - y_i)  * x_i)

        # Q3.1b
        #raise NotImplementedError


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size=1):
        # Initialize an MLP with a single hidden layer.
        
        self.W1 = np.random.normal(0.1, 0.01, size=(n_features, 200)) # weight for hidden layer
        self.B1 = np.zeros((1, 200)) # bias for hidden layer
        self.W2 = np.random.normal(0.1, 0.01, size=(n_classes, 1)) #weight for output layer
        self.B2 = np.zeros(( 1, 1)) #bias for output layer
        print(self.W1.shape)
        print(self.W2.shape)
        print(self.B1.shape)
        print(self.B2.shape)

        #self.W = [W1, W2]
        #self.B = [B1, B2]
        #self.num_layers = len(self.W)
        #self.hiddenLayer = hidden_size

    def sigmoid (self, x):
        return 1/(1 + np.exp(-x))

    # derivative of Sigmoid Function
    def derivatives_sigmoid(self, x):
        return x * (1 - x)

    def relu (self, x, der=False):
    
        if (der == True): # the derivative of the ReLU is the Heaviside Theta
            f = np.heaviside(x, 1)
        else :
            f = np.maximum(x, 0)
    
        return f
    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        houtput = np.dot(X, self.W1 ) + self.B1
        output1 = self.relu(houtput, False)
        Ooutput = np.dot(output1, self.W2) + self.B2
        output2 = self.sigmoid(Ooutput)
        return output2
   
    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X) 
        
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):

        print("fazer forward")
        print(y.shape)
        hinput= np.dot(X,self.W1) + self.B1
        hiddenlayer_activations = self.sigmoid(hinput)
        oinput=np.dot(hiddenlayer_activations,self.W2) + self.B2
        output = self.sigmoid(oinput)
        print(output)
        print("forward feito")

        #Backpropagation
        probs = np.exp(output) / np.sum(np.exp(output))
        error = -y.dot(np.log(probs))
        
        #error = y-output
        print("calculei erro")
        grad_output_layer = self.derivatives_sigmoid(output)
        grad_hidden_layer = self.derivatives_sigmoid(hiddenlayer_activations)
        print("passei1")
        d_output = error * grad_output_layer
        print("passei2")
        error_hidden_layer = d_output.dot(self.W2.T)
        print("passei3")
        d_hiddenlayer = error_hidden_layer * grad_hidden_layer

        print("vou atualizar")
        self.W2 += hiddenlayer_activations.T.dot(d_output) *learning_rate
        self.B2 += np.sum(d_output, axis=0,keepdims=True) *learning_rate
        self.W1 += X.T.dot(d_hiddenlayer) *learning_rate
        self.B1 += np.sum(d_hiddenlayer, axis=0,keepdims=True) *learning_rate
        print("update done")

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()
    plt.savefig("epochs.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
