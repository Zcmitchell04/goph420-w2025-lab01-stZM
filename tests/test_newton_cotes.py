import unittest
import numpy as np
from src.goph420_lab01.integration import integrate_newton

class TestIntegrateNewtonCotes(unittest.TestCase):

    def test_trapezoid_linear(self):
        # Test for a linear function with Trapezoidal Rule
        x = np.array([0, 1])
        f = np.array([0, 1])  # Linear function f(x) = x
        result = integrate_newton(x, f, 'trap')
        self.assertAlmostEqual(result, 0.5, places=5)  # Integral of x from 0 to 1 is 0.5

    def test_trapezoid_linear_even_points(self):
        # Test for linear function with an even number of data points
        x = np.linspace(0, 1, 4)  # Even number of points (4)
        f = x  # Linear function f(x) = x
        result = integrate_newton(x, f, 'trap')
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_simpson_quadratic(self):
        # Test for quadratic function with Simpson's Rule
        x = np.array([0, 0.5, 1])
        f = np.array([0, 0.25, 1])  # Quadratic function f(x) = x^2
        result = integrate_newton(x, f, 'simp')
        self.assertAlmostEqual(result, 1/3, places=5)  # Integral of x^2 from 0 to 1 is 1/3

    def test_simpson_quadratic_odd_points(self):
        # Test for quadratic function with an odd number of data points
        x = np.linspace(0, 1, 5)  # Odd number of points (5)
        f = x**2  # Quadratic function f(x) = x^2
        result = integrate_newton(x, f, 'simp')
        self.assertAlmostEqual(result, 1/3, places=5)

    def test_invalid_alg(self):
        # Test for invalid algorithm string
        x = np.array([0, 1])
        f = np.array([0, 1])
        with self.assertRaises(ValueError):
            integrate_newton(x, f, 'invalid')

    def test_incompatible_shapes(self):
        # Test for incompatible x and f arrays
        x = np.array([0, 1])
        f = np.array([0])
        with self.assertRaises(ValueError):
            integrate_newton(x, f, 'trap')

    def test_sorting_of_data(self):
        # Test for sorting functionality
        x = np.array([1, 0])
        f = np.array([1, 0])
        result = integrate_newton(x, f, 'trap')
        self.assertAlmostEqual(result, 0.5, places=5)

if __name__ == "__main__":
    unittest.main()