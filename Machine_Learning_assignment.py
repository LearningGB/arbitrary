import numpy as np
import matplotlib.pyplot as plt



def MSE(y_pred, y):
    """
     y_pred :ndarray, shape (n_samples,)

    y : ndarray, shape (n_samples,)

    Returns
    ----------
    mse : numpy.float
      Mean squared error
    """
    pass
    return mse
class ScratchLinearRegression(): ## neg object##

    def __init__(self, num_iter, lr, no_bias, verbose):
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    def fit(self, X, y, X_val=None, y_val=None):

        if self.no_bias=False
             

        for epoch in range(self.iter):

            y_pred=self._linear_hypothesis(X)
            self.loss[epoch]=np.mean((y-y_pred)**2)


        if self.verbose:

            print()
        pass

    def predict(self, X):

        pass
        return

    def _linear_hypothesis(self, X):
        """
        線形のHypothetical functionTo計算する

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)


        Returns
        -------
          ndarray, shape (n_samples, 1)


        """
        y_pred=np.dot(X, self.parameters)
        pass
        return

    def _gradient_descent(self, X, error):

        pass