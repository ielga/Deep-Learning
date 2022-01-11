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
        # Q3.1a
        #raise NotImplementedError


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
        units = [64, 50, 10]

        # Initialize all weights and biases randomly.
        W1 = .1 * np.random.randn(units[1], units[0])
        b1 = .1 * np.random.randn(units[1])
        W2 = .1 * np.random.randn(units[2], units[1])
        b2 = .1 * np.random.randn(units[2])
        self.W = [W1, W2]
        self.B = [b1, b2]
        self.hiddenLayer = hidden_size
        print(self.W)
        print(self.B)
       
        #raise NotImplementedError
   
    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        predicted_y = []
        g = np.tanh #put a relu function
        nm_layers = len(self.W)
        hiddens = []
        for x in X:
        #compute forward
            print(x)
            for i in range(nm_layers):
                h = x if i == 0 else hiddens[i-1]
                print(self.W[i])
                print(h)
                z = self.W[i].dot(h) + self.B[i]
                if i < nm_layers-1:  # Assume the output layer has no activation.
                    hiddens.append(np.maximum(z, 0))
            output = z
        #finished forward
            yhat =  np.zeros_like(output)
            yhat[np.argmax(output)] = 1
            predicted_y.append(yhat)
        predicted_y = np.array(predicted_y)
        return predicted_y
        #raise NotImplementedError


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
        total_loss = 0
        hiddens = []
        grad_weights = []
        grad_biases = []
        nm_layers = len(self.W)
        g = np.tanh
        for x, y in zip(X, y):
            #forward
            for i in range(nm_layers):
                h = X if i == 0 else hiddens[i-1]
                z = self.W[i] * (h) + self.B[i]
                if i < nm_layers-1:  # Assume the output layer has no activation.
                    hiddens.append(g(z))
            output = z
            #compute loss using cross_entropy
            probs = np.exp(output) / np.sum(np.exp(output))
            loss = -y.dot(np.log(probs))
            total_loss += loss

            #compute backward propagation
            z = output
            grad_z = probs - y  # Grad of loss wrt last z.

            for i in range(nm_layers-1, -1, -1):
                # Gradient of hidden parameters.
                h = x if i == 0 else hiddens[i-1]
                grad_weights.append(grad_z[:, None].dot(h[:, None].T))
                grad_biases.append(grad_z)

                # Gradient of hidden layer below.
                grad_h = self.W[i].T.dot(grad_z)

                # Gradient of hidden layer below before activation.
                assert(g == np.tanh)
                grad_z = grad_h * (1-h**2)   # Grad of loss wrt z3.

                grad_weights.reverse()
                grad_biases.reverse()

            #updating parameters
            for i in range(nm_layers):
                self.W[i] -= learning_rate*grad_weights[i]
                self.B[i] -= learning_rate*grad_biases[i]
                


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
