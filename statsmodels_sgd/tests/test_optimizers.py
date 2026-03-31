import unittest
import numpy as np 
from optimizers import SVRG

class TestSVRGOptimizer(unittest.TestCase):
    
    def setUp(self):
        """
        Set up the parameters for SVRG optimizer tests.
        """
        self.lr = 0.01
        self.n_inner = 10
        self.svrg = SVRG(lr=self.lr, n_inner=self.n_inner)
        self.params = [np.array([1.0, 2.0])]
        self.initial_params = self.params[0].copy()  # Store initial params for later comparison
    
    def test_single_update(self):
        """
        Test a single update of the SVRG optimizer.
        """
        grads = [np.array([0.5, 0.5])]
        full_grads = [np.array([0.3, 0.3])]

        # Update
        self.svrg.update(self.params, grads, full_grads)

        # Expected output after one update (hand-calculated)
        expected_params = self.initial_params - self.lr * (grads[0] - full_grads[0] + full_grads[0])
        
        # Assertions
        self.assertTrue(np.allclose(self.params[0], expected_params), 
                        msg="Single update failed: Expected parameters do not match.")
    
    def test_multiple_updates(self):
        """
        Test multiple updates of the SVRG optimizer.
        """
        updates = [
            (np.array([0.5, 0.5]), np.array([0.3, 0.3])),
            (np.array([0.1, 0.1]), np.array([0.1, 0.1])),
            (np.array([0.4, 0.4]), np.array([0.2, 0.2]))
        ]
        
        for grads, full_grads in updates:
            self.svrg.update(self.params, [grads], [full_grads])
        
        # Calculate expected params after multiple updates
        expected_params = self.initial_params.copy()
        for grads, full_grads in updates:
            expected_params -= self.lr * (grads - full_grads + full_grads)

        # Assertions
        self.assertTrue(np.allclose(self.params[0], expected_params), 
                        msg="Multiple updates failed: Expected parameters do not match after multiple updates.")

    def test_zero_gradients(self):
        """
        Test the SVRG optimizer behavior with zero gradients.
        """
        grads = [np.zeros(2)]
        full_grads = [np.zeros(2)]
        
        # Update with zero gradients
        self.svrg.update(self.params, grads, full_grads)

        # Expected parameters should remain unchanged
        self.assertTrue(np.array_equal(self.params[0], self.initial_params),
                        msg="Update with zero gradients should not change parameters.")

if __name__ == '__main__':
    unittest.main()