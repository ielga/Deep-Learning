import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils

def distance(analytic_solution, model_params):
    return np.linalg.norm(analytic_solution - model_params)


def solve_analytically(X, y):
    """
    X (n_points x n_features)
    y (vector of size n_points)


    Q2.1: given X and y, compute the exact, analytic linear regression solution.
    This function should return a vector of size n_features (the same size as
    the weight vector in the LinearRegression class).

   his problem can be solved by adding a small value λ to the diagonal  of X⊤X before inverting, i.e., 
    w^=(X⊤X+λI)−1X⊤y
    where I is the identity matrix.
    You can use λ=1×10−4 for this exercise. 
    """
    
    
    i = (1*10)**(-4)
    w = np.linalg.inv(X.transpose().dot(X) 
                    + i * np.identity( 349 )
                    )
    w = w.dot(X.transpose())
    w = w.dot(y)
    return w
    
    #raise NotImplementedError


class _RegressionModel:
    """
    Base class that allows evaluation code to be shared between the
    LinearRegression and NeuralRegression classes. You should not need to alter
    this class!
    """
    def train_epoch(self, X, y, **kwargs):
        """
        Iterate over (x, y) pairs, compute the weight update for each of them.
        Keyword arguments are passed to update_weight
        """
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def evaluate(self, X, y):
        """
        return the mean squared error between the model's predictions for X
        and he ground truth y values
        """
        yhat = self.predict(X)
        error = yhat - y
        squared_error = np.dot(error, error)
        mean_squared_error = squared_error / y.shape[0]
        return np.sqrt(mean_squared_error)


class LinearRegression(_RegressionModel):
    def __init__(self, n_features, **kwargs):
        self.w = np.zeros((n_features))
        self.b = 0

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        Q2.2a

        x_i, y_i: a single training example

        This function makes an update to the model weights (in other words,
        self.w).
        """
        y_pred = self.predict(x_i)
        loss = y_i - y_pred
        dW = - ( 2 * x_i * loss )
        db = - 2 * np.sum( loss )

        self.w = self.w - learning_rate * dW
        self.b = self.b - learning_rate * db

        #raise NotImplementedError

    def predict(self, X):
        return np.dot(X, self.w) + self.b


class NeuralRegression(_RegressionModel):
    """
    Q2.2b
    """
    def __init__(self, n_features, hidden):
        """
        In this _init_, you should define the weights of the neural
        regression model (for example, there will probably be one weight
        matrix per layer of the model).
        """
        self.b1 = np.zeros((hidden))
        self.b2 = 0
        self.w1 = np.random.normal(0.1, 0.1**2, (hidden, n_features)) 
        self.w2 = np.random.normal(0.1, 0.1**2, (hidden)) 
    
    def relu(self, x):
        return np.maximum(0.0, x)

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i, y_i: a single training example

        This function makes an update to the model weights
        """
        z1 = self.w1.dot(x_i) + self.b1 # hidden layer pre-activation
        h1 = self.relu(z1)              # hidden layer activation
        z2 = self.w2.dot(h1) + self.b2  # output layer pre-activation
        y_pred = z2                     # output layer activation
    
        # calculate gradients
        d_z2 = y_pred - y_i

        d_w2 = d_z2 * h1.T
        d_b2 = d_z2
        d_h1 = (self.w2.T).dot(d_z2)

        d_z1 = np.multiply(d_h1, np.sign(z1))

        # d_w1 = d_z1.dot(x_i.transpose())
        d_w1 = d_z1.reshape(-1, 1).dot(x_i.reshape(-1, 1).T)
        d_b1 = d_z1
     
        # update parameters
        self.w1 = self.w1 - learning_rate * d_w1
        self.b1 = self.b1 - learning_rate * d_b1
        self.w2 = self.w2 - learning_rate * d_w2
        self.b2 = self.b2 - learning_rate * d_b2

    def predict(self, X):
        """
        X: a (n_points x n_feats) matrix.

        This function runs the forward pass of the model, returning yhat, a
        vector of size n_points that contains the predicted values for X.

        This function will be called by evaluate(), which is called at the end
        of each epoch in the training loop. It should not be used by
        update_weight because it returns only the final output of the network,
        not any of the intermediate values needed to do backpropagation.
        """
        y_pred = []
        for x in X:
            z1 = self.w1.dot(x) + self.b1   # hidden layer pre-activation
            h1 = self.relu(z1)              # hidden layer activation
            z2 = self.w2.dot(h1) + self.b2  # output layer pre-activation
            o = z2                          # output layer activation

            y_pred.append(o)

        return np.array(y_pred)


def plot(epochs, train_loss, test_loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, epochs[-1] + 1, step=10))
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, test_loss, label='test')
    plt.legend()
    plt.show()
    plt.savefig('epochs.png')
    plt.close()


def plot_dist_from_analytic(epochs, dist):
    plt.xlabel('Epoch')
    plt.ylabel('Dist')
    plt.xticks(np.arange(0, epochs[-1] + 1, step=10))
    plt.plot(epochs, dist, label='dist')
    plt.legend()
    plt.show()
    plt.savefig('analytic.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['linear_regression', 'nn'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=150, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=150)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != 'nn'
    data = utils.load_regression_data(bias=add_bias)
    train_X, train_y = data["train"]
    test_X, test_y = data["test"]

    n_points, n_feats = train_X.shape

    # Linear regression has an exact, analytic solution. Implement it in
    # the solve_analytically function defined above.
    if opt.model == "linear_regression":
        analytic_solution = solve_analytically(train_X, train_y)
    else:
        analytic_solution = None

    # initialize the model
    if opt.model == "linear_regression":
        model = LinearRegression(n_feats)
    else:
        model = NeuralRegression(n_feats, opt.hidden_size)

    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_losses = []
    test_losses = []
    dist_opt = []
    for epoch in epochs:
        print('Epoch %i... ' % epoch)
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(train_X, train_y, learning_rate=opt.learning_rate)

        # Evaluate on the train and test data.
        train_losses.append(model.evaluate(train_X, train_y))
        test_losses.append(model.evaluate(test_X, test_y))

        if analytic_solution is not None:
            model_params = model.w
            dist_opt.append(distance(analytic_solution, model_params))

        print('Loss (train): %.3f | Loss (test): %.3f' % (train_losses[-1], test_losses[-1]))

    plot(epochs, train_losses, test_losses)
    if analytic_solution is not None:
        plot_dist_from_analytic(epochs, dist_opt)


if __name__ == '__main__':
    main()