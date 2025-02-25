import numpy as np
import matplotlib.pyplot as plt

def integrate_newton(x,f,alg):
    """performs numerical integration of discrete data using
    Newton-Cotes rules.

    Parameters:
    ---------
       x: array_like
       f: array_like
        * x and f have the same shape (len(x) == len(f))
       alg: (optional) string flag that decides between Trapezoid rule and Simpson's rule
       default value "trap", alt value "simp"
                *raise ValueError if str contains anything else


    Returns:
    -------
    float:
        """

    alg_clean = alg.lower().strip()

    x = np.array(x)
    f = np.array(f)

    # make sure the dimensions are correct
    if x.ndim != 1 or f.ndim != 1 or (x.shape[0] != f.shape[0]):
        raise ValueError ("x and f need to be 1D with the same length")

    # sorting x and f if not already
    if not np.all(np.diff(x) > 0):
        sort_indices = np.argsort(x)
        x = x[sort_indices]
        f = f[sort_indices]

        dx = x[1] - x[0]

    # implementation of the trapezoid rule
    if alg_clean == "trap":
        I = 0.0
        for i in range (len(x) - 1):
            I += 0.5 * (f[i] + f[i+1]) * (x[i+1] - x[i])
        return I

    # implementation of Simpson's Rule

    elif alg_clean == "simp":
        # implement Simpson's 1/3 rule for pairs of intervals
        # implement Simpson's 3/8 rule for odd intervals
        # the last three intervals use the combined approach

        N = len(x) - 1
        I = 0.0
        i = 0

    while i < N:
        if  (N - i) >= 2 and ((N - i) % 2 == 0):

            I += (x[i + 2] - x[i]) * (f[i] + 4*f[i + 1] + f[i + 2]) / 6.0
            i += 2
        elif (N - i) == 1: #interval is odd
            I += 0.5 * (f[i] + f[i + 1])  *  (x[i+1] - x[i])
            i += 1


        # for if exactly 2 intervals remain
        else:
            if (N - i) == 3:
                h = (x[i+3] - x[i]) / 3.0
                I += 3.0 * h * (f[i] + 3.0 * f[i+1] + 3.0 * f[i + 2] + f[i + 3]) / 8.0
                i += 3

            else:
                I += 0.5 * (f[i] + f[i+1]) * (x[i + 1] - x[i])
                i += 1
        return I

    else:
        raise ValueError ("Invalid 'alg' parameter. Use 'trap' or 'simp'.")




def integrate_gauss(f, lims, npts):
    """Numerical integration of a function using gauss-legendre quadrature.

    Parameters:
    ----------
    f: ref. callable object (implements __call__() method)
    lims: and object with a length of 2, contains lower and upper boundaries of the function
        *(x=a, x=b)
    npts: (optional) an integer that gives number of integration points to use. default is 3, possible values are 1,2,3,4,5.
    Returns:
    ---------
    float: provides integral estimate

    Errors:
    ------
    TypeError: f is not callable, uses callable()
    ValueError: lims does not have len(2)
    ValueError: if lims[0] or lims[1] are not convertible to float
    ValueError: if npts is not in range [1,2,3,4,5]

    """
    # Check that f is callable
    if not callable(f):
        raise TypeError("f must be a callable object (like a function).")

    # Check lims
    if len(lims) != 2:
        raise ValueError("lims must have length 2.")
    a = float(lims[0])
    b = float(lims[1])

    # Check npts
    if npts not in [1, 2, 3, 4, 5]:
        raise ValueError("npts must be in [1, 2, 3, 4, 5].")


    gauss_data = {
        1: {
            'x': np.array([0.0]),
            'w': np.array([2.0])
        },
        2: {
            'x': np.array([-0.5773502692, 0.5773502692]),
            'w': np.array([1.0, 1.0])
        },
        3: {
            'x': np.array([-0.7745966692, 0.0, 0.7745966692]),
            'w': np.array([ 0.5555555556, 0.8888888889, 0.5555555556])
        },
        4: {
            'x': np.array([-0.8611363116, -0.3399810436,
                            0.3399810436,   0.8611363116]),
            'w': np.array([ 0.3478548451,  0.6521451549,
                            0.6521451549,  0.3478548451])
        },
        5: {
            'x': np.array([-0.9061798459, -0.5384693101, 0.0,
                            0.5384693101,  0.9061798459]),
            'w': np.array([ 0.2369268850,  0.4786286705, 0.5688888889,
                            0.4786286705,  0.2369268850])
        }
    }

    x_std = gauss_data[npts]['x']   # standard points in [-1,1]
    w_std = gauss_data[npts]['w']   # standard weights

    # Perform linear transformation
    midpoint = 0.5*(a + b)
    half_range = 0.5*(b - a)

    # accumulate integral
    I = 0.0
    for i in range(npts):
        xk = midpoint + half_range*x_std[i]
        wk = half_range*w_std[i]
        I += wk * f(xk)

    return I
