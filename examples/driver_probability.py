import numpy as np
import matplotlib.pyplot as plt
from src.goph420_lab01.integration import integrate_gauss


def stdnorm_probability(z):
    """
    Computes the probability density function (Equation #17).

   parameters:
   --------
   z

   Returns:
    -------
    (1 / np.sqrt(2 * np.pi)) * np.exp((-1 / 2) * z ** 2) : Equation 17
    """
    return (1 / np.sqrt(2 * np.pi)) * np.exp((-1 / 2) * z ** 2)


def prob_seismic_mag4(mag, mean, std, npts):
    """
    Computes the probability of a seismic event with a magnitude greater than 4.0

    Parameters:
    ---------
    mag :
    mean:
    std:
    npts:


    Returns:
    -------
    integrate_gauss: the result of parameters calculated using the function itegrate_gauss
    """

    # equation 18
    z = (mag - mean) / std


    lims = [z, 10]

    return integrate_gauss(stdnorm_probability, lims, npts)


def prob_trueval_dist(L1, L2, Lmean, std, npts):
    """
    Determines the probability that the true value is between 10.25-10.35m

    Parameters:
    ---------
    L1:
    L2:
    Lmean:
    std:
    npts:

    Returns:
    -------
    integrate_gauss : result of parameters calculated with the integrate_gauss function.
    """

    lims = [((L1 - Lmean) / std), ((L2 - Lmean) / std)]

    return integrate_gauss(stdnorm_probability, lims, npts)


def h_refinement(npts_list, interval_list):
    """
    Calculate h-refinement by performing Gauss integration over sub-intervals and determining relative error

    Parameters:
    ---------
    npts_list:
    interval_list:

    Returns:
    -------
    interval values:
    error:
    """

    lims = [4, 10]

    exact_gauss = integrate_gauss(stdnorm_probability, lims, 5)
    error = {npts: [] for npts in npts_list}  # initializing a dictionary to add diff intervals to each list
    interval_values = []

    for intervals in interval_list:
        # subinterval value
        interval_num = (10 - 4)/intervals
        interval_values.append(interval_num)

        for npts in npts_list:
            # sub-intervals
            result_gauss = sum(
                integrate_gauss(stdnorm_probability, [4 + i * interval_num, 4 + (i + 1) * interval_num], npts)
                for i in range(intervals))

            # relative error
            rel_error = abs((result_gauss - exact_gauss) / exact_gauss)
            error[npts].append(rel_error)

    return interval_values, error


def main():
    """ Takes data and previous functions and creates required plots

        Parameters:
        ---------
        none

        Returns:
        -------
        none
        """
    mean = 1.5
    std = 0.5
    mag = 4
    Lmean = 10.28
    L1 = 10.25
    L2 = 10.35
    npts_list = [1, 2, 3, 4, 5]

    prob_gt_4 = [prob_seismic_mag4(mag, mean, std, npts) for npts in npts_list]
    print(f'Probability that an earthquake greater than magnitude 4.0 will occur:', [float(val) for val in prob_gt_4])

    prob_true_value = [prob_trueval_dist(L1, L2, mean, std, npts) for npts in npts_list]
    print(f'The probably of earthquake distance between 10.25 - 10.35m is:', [float(val) for val in prob_true_value])

    # now we plot the convergence of probability estimates with increasing integration points.
    seis_prob = []
    dist_prob = []

    for npts in npts_list:
        seis_prob.append(integrate_gauss(stdnorm_probability, [4, 10], npts))
        dist_prob.append(
            integrate_gauss(stdnorm_probability, [(L1 - Lmean) / std, (L2 - Lmean) / std], npts))

    print("Seismic Probabilities:", [float(val) for val in seis_prob])
    print("Distance Probabilities:", [float(val) for val in dist_prob])

    plt.figure(figsize=(10, 4))

    # seismic Probability Plot
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.plot(npts_list, seis_prob, "-r", label="Seismic Convergence")
    plt.xlabel("Seismic Points")
    plt.ylabel("Probability")
    plt.title("Seismic Convergence Using Probability Density Function")
    plt.legend()

    # distance Probability Plot
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.plot(npts_list, dist_prob, "-b", label="Distance Convergence")
    plt.xlabel("Distance Points")
    plt.ylabel("Probability")
    plt.title("Distance Convergence Using Probability Density Function")
    plt.legend()

    plt.tight_layout()
    plt.savefig("C:/Users/zcmit/git/Goph420projects/goph420-w2025-lab01-stZM/figures/convergence_plots.png")
    plt.show()

    # h-refinement plot
    intervals_list = [1, 2, 4, 8, 16, 32]
    interval_values, error = h_refinement([2, 3, 4], intervals_list)

    print(f'Interval values: {interval_values}')
    print(f'Relative error:', [float(val) for val in dist_prob])

    plt.figure(figsize=(6, 5))

    for npts in error:
        plt.loglog(interval_values, error[npts], label=f"{npts} Integration Points", marker="o")
    plt.grid()
    plt.xlabel("Interval")
    plt.ylabel("Relative Error")
    plt.title("Relative Error with respect to interval value")
    plt.legend()
    plt.savefig("C:/Users/zcmit/git/Goph420projects/goph420-w2025-lab01-stZM/figures/h_refinement_plot.png")
    plt.show()


if __name__ == "__main__":
    main()