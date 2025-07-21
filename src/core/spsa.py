"""
Simultaneous Perturbation Stochastic Approximation (SPSA) implementation.
"""
import numpy as np

class SPSA:
    """
    Implementation of the Simultaneous Perturbation Stochastic Approximation algorithm.
    """
    def __init__(self, learning_rate: float, perturbation_coefficient: float):
        self.learning_rate = learning_rate
        self.perturbation_coefficient = perturbation_coefficient

    def optimize(self, objective_function, initial_params: np.ndarray, iterations: int) -> np.ndarray:
        """
        Optimize the given objective function using SPSA.

        Args:
            objective_function: The function to be optimized.
            initial_params: Initial parameters for the optimization.
            iterations: Number of iterations to perform.

        Returns:
            np.ndarray: Optimized parameters.
        """
        params = np.copy(initial_params)

        for k in range(1, iterations + 1):
            ak = self.learning_rate / (k ** 0.101)
            ck = self.perturbation_coefficient / (k ** 0.101)

            delta = 2 * np.random.randint(2, size=params.shape) - 1

            params_plus = params + ck * delta
            params_minus = params - ck * delta

            y_plus = objective_function(params_plus)
            y_minus = objective_function(params_minus)

            grad = (y_plus - y_minus) / (2.0 * ck * delta)

            params = params - ak * grad

        return params

    def _perturb(self, params: np.ndarray) -> np.ndarray:
        """
        Generate a perturbation for SPSA.
        """
        return 2 * np.random.randint(2, size=params.shape) - 1
