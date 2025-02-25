import unittest
import numpy as np
from src.goph420_lab01.integration import integrate_gauss

class TestIntegrateGaussLegendre(unittest.TestCase):

    def test_gauss_1_point(self):
        """ Tests a case for 1 - point Gauss - Legendre Quadrature

    Parameters:
    ---------
    self

    Returns:
    -------
    x - squared
    """

        def f(x):
            return x**2
        result = integrate_gauss(f, (0, 1), 1)
        self.assertAlmostEqual(result, 1/3, places=5)

    def test_gauss_2_points(self):
        """ Testing a simple case for 2-point Gauss-Legendre Quadrature.

    Parameters:
    ---------
    self

    Returns:
    -------
    x - squared
        """

        def f(x):
            return x**2
        result = integrate_gauss(f, (0, 1), 2)
        self.assertAlmostEqual(result, 1/3, places=5)

    def test_gauss_3_points(self):
        """ Testing a simple case for 3-point Gauss-Legendre Quadrature.

            Parameters:
            ---------
            self

            Returns:
            -------
            x - squared
            """

        def f(x):
            return x**2
        result = integrate_gauss(f, (0, 1), 3)
        self.assertAlmostEqual(result, 1/3, places=5)

    def test_gauss_5_points(self):
        """ Testing a simple case for 5-point Gauss-Legendre Quadrature.

        Parameters:
        ---------
        self

        Returns:
        -------
        x - squared
        """

        def f(x):
            return x**2
        result = integrate_gauss(f, (0, 1), 5)
        self.assertAlmostEqual(result, 1/3, places=5)

    def test_polynomial_integration(self):
        """ Testing for polynomials of various orders up to n = 9.

            Parameters:
            ---------
            self

            Returns:
            -------
            poly(x)
            """

        for n in range(10):  # from x^0 to x^9
            poly = np.poly1d([0] * (9 - n) + [1])  # Generates polynomial x^n
            def f(x):
                return poly(x)

            for npts in [1, 2, 3, 4, 5]:  # Test with different number of points
                result = integrate_gauss(f, (0, 1), npts)
                exact_result = 1 / (n + 1)  # Integral of x^n from 0 to 1
                self.assertAlmostEqual(result, exact_result, places=5)

    def test_non_polynomial_function(self):
        """ Testing for non-polynomial functions using Gauss-Legendre Quadrature.

            Parameters:
            ---------
            self

            Returns:
            -------
            np.sin(x)
            """
        # Test for non-polynomial functions using Gauss-Legendre Quadrature
        def f(x):
            return np.sin(x)

        result = integrate_gauss(f, (0, np.pi), 5)
        exact_result = 2  # Integral of sin(x) from 0 to pi is 2
        self.assertAlmostEqual(result, exact_result, places=5)


if __name__ == "__main__":
    unittest.main()