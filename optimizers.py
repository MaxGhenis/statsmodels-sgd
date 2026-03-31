import numpy as np
class SVRG:
    """
    SVRG optimizer implementation.
    
    Attributes:
    -----------
    lr : float
        Learning rate.
    n_inner : int
        Number of inner loop updates.
    """
    def __init__(self, lr=0.01, n_inner=10):
        self.lr = lr
        self.n_inner = n_inner

    def update(self, params, grads, full_grads):
        """
        Update the parameters using SVRG optimization algorithm.
        
        Parameters:
        -----------
        params : list
            List of model parameters to update.
        grads : list
            List of gradients corresponding to the current mini-batch.
        full_grads : list
            Full batch gradients computed occasionally.
        """
        for i in range(len(params)):
            # Update using variance reduction technique
            params[i] -= self.lr * (grads[i] - full_grads[i] + full_grads[i])