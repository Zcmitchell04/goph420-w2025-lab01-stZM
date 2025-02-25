import numpy as np
import matplotlib.pyplot as plt
from src.goph420_lab01.integration import integrate_newton

def main():
    """
    plots raw data including squared velocity, finds max velocity, integrates using both Trapezoid and Simpson's rules,
    plots convergence on loglog plots,

    Parameters:
    ---------
    none

    returns:
    -------
    none
    """
    data = np.loadtxt("C:/Users/zcmit/git/Goph420projects/goph420-w2025-lab01-stZM/data/s_wave_data.txt")
    t_data = data[:, 0]
    v_data = data[:, 1]
    v2_data = v_data**2

    # part 1

    # find the maximum absolute velocity and the boundary when v > 0.005 vmax which will be T

    # find the max abs velocity (strongest seismic wave)
    boundary = 0.005 * np.max(np.abs(v_data))  # and multiply by 0.005 to find the significant velocity points
    T_boundary = np.where(np.abs(v_data) > boundary)[0]  # find velocity about the boundary
    T = t_data[T_boundary[-1]]  # find the last time when velocity is above the boundary

    print(f'Estimated duration when velocity is significant: {T} seconds')

    # integrate v^2 over T using trap and simp

    integral_trap = integrate_newton(t_data, v2_data, "trap")
    integral_simp = integrate_newton(t_data, v2_data, "simp")

    #plotting the raw data
    print(f'Average squared velocity using trapezoid rule: {integral_trap} mm²/s²')
    print(f"Average squared velocity using Simpson's 1/3 rule: {integral_simp} mm²/s²")

    plt.figure(figsize=(11, 5))

    plt.subplot(1, 2, 1)
    plt.grid()
    plt.plot(t_data, v_data, "-b", label="Seismic wave data")
    plt.axvline(float(T), color='r', linestyle='--', label=f"Significant velocity ({T:.2f} s)")
    plt.scatter(T, np.interp(T, t_data, v_data), color='r', zorder=3)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mm²/s²)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.grid()
    plt.plot(t_data, v2_data, "-g", label="Squared Seismic wave data")
    plt.axvline(float(T), color='r', linestyle='--', label=f"Significant velocity ({T:.2f} s)")
    plt.scatter(T, np.interp(T, t_data, v_data), color='r', zorder=3)
    plt.xlabel("Time (s)")
    plt.ylabel("Squared velocity (mm²/s²)")
    plt.legend()

    plt.suptitle("Velocity as a Function of Time", fontsize=14)
    plt.savefig("C:/Users/zcmit/git/Goph420projects/goph420-w2025-lab01-stZM/figures/seismic_raw_wave.png")
    plt.show()

    # plot the convergence by plotting delta t against approx rel error in log-log space

    samp_int = [0.02, 0.04, 0.08, 0.16]  # sampling intervals
    trap_error = []
    simp_error = []

    # reference integral
    I_ref_simp = integrate_newton(t_data, v2_data, "simp")

    for i in samp_int:
        values_keep = np.arange(0, len(t_data), int(i/0.01))  # determine the values to keep based on the sampling interval

        # we need to downsample the og t and v2 data with the kept values
        t_samp = t_data[values_keep]
        v2_samp = v2_data[values_keep]

        # we need to keep points odd for simpson
        if len(t_samp) % 2 == 0:
            t_samp = t_samp[:-1]  # remove the last point
            v2_samp = v2_samp[:-1]  # remove the last velocity to match

        I_down_trap = integrate_newton(t_samp, v2_samp, "trap")
        I_down_simp = integrate_newton(t_samp, v2_samp, "simp")

        app_rel_e_trap = abs((I_down_trap - I_ref_simp)/I_ref_simp)
        trap_error.append(float(app_rel_e_trap))

        app_rel_e_simp = abs((I_down_simp - I_ref_simp)/I_ref_simp)
        simp_error.append(float(app_rel_e_simp))

    print(f"Trapezoid error: {trap_error}")
    print(f"Simpson's 1/3 error: {simp_error}")

    #plots

    plt.figure(figsize=(8, 6))

    plt.grid()
    plt.loglog(samp_int, trap_error, "-.r", label="Convergence using Trapezoid rule")
    plt.loglog(samp_int, simp_error, "-.g", label="Convergence using Simpson's 1/3 rule")
    plt.xlabel("Sampling Interval (s)")
    plt.ylabel("Error")
    plt.legend()

    plt.suptitle("Convergence with Newton-Cotes Rules", fontsize=14)
    plt.savefig("C:/Users/zcmit/git/Goph420projects/goph420-w2025-lab01-stZM/figures/seismic_convergence_ds.png")
    plt.show()


if __name__ == "__main__":
    main()