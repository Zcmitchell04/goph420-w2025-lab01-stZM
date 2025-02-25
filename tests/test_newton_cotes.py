
import numpy as np
from src.goph420_lab01.integration import integrate_newton


def test_trap_():
    """Test Trapezoid rule on linear data with an even number of data points."""

    # Define a linear function f(x) = 2x + 1
    a, b = 2, 1  # y = 2x + 1
    x = np.linspace(0, 1, 6)  # Generate 6 points evenly spaced between 0 and 1
    f = a * x + b  # f(x) = 2x + 1

    # Expected integral for f(x) = 2x + 1 from 0 to 1 is (x^2 + x) evaluated from 0 to 1
    expected_result = (1 ** 2 + 1) - (0 ** 2 + 0)  # Exact result = 2

    # Perform the integration using Trapezoid rule
    result = integrate_newton(x, f, "trap")

    # Check if the result is very close to the exact result
    print("Testing trapezoid rule...")
    print(f"the expected result is {expected_result}")
    print(f"The calculated result is {result}")

    if expected_result == result:
        print("TEST PASSED")

    else:
        print("TEST FAILED")

def test_simp():
    """Test Simpson's rule on quadratic data with an even number of data points."""
    # Define a quadratic function f(x) = x^2 + 2x + 1
    a, b, c = 1, 2, 1  # y = x^2 + 2x + 1
    x = np.linspace(0, 2, 6)  # Generate 6 points evenly spaced between 0 and 2
    f = a * x ** 2 + b * x + c  # f(x) = x^2 + 2x + 1

    # Exact integral for f(x) = x^2 + 2x + 1 from 0 to 2 is (x^3/3 + x^2 + x) evaluated from 0 to 2
    expected_result = (2 ** 3 / 3 + 2 ** 2 + 2) - (0 ** 3 / 3 + 0 ** 2 + 0)  # Exact result = 14/3 ≈ 4.6667

    # Perform the integration using Simpson's rule
    result = integrate_newton(x, f, "simp")

    # Check if the result is very close to the exact result
    print("Testing Simpson's 1/3 rule...")
    print(f"the expected result is {expected_result}")
    print(f"The calculated result is {result}")

    if expected_result == result:
        print("TEST PASSED")

    else:
        print("TEST FAILED")


if __name__ == '__main__':
    test_trap_()
    test_simp()
