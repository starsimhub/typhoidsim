"""
Numerical and mathy utilities like math function that can be used
for parametrisation of otherwise constant parameters. .
"""

import numpy as np
import sciris as sc

# Specify all externally visible things this file defines
__all__ = ['sigmoid', 'gompertz', 'gompertz_dfun', 'double_sigmoid_exp', 'double_sigmoid_tanh',
           'asym_trapezoidal', 'box_exponential']


def sigmoid(x, max_x, slope):
    """
    Compute a sigmoid-like function. The range of x is between [0, max_x].

    This function is used in the age-based exposure mechanism in typhoid.

    Arguments:
        x (array-like): The array of x values at which the function is evaluated
        max_x (float): The asymptote of the function as x approaches infinity.
        slope (float): The slope parameter that controls how 'fast' we reach
        100% exposure (or 100% susceptible). If slope=0, this function is
        a linear function, if slope > 0, higher susceptibility is achieved
        faster. If slope < 0, higher susceptibility is reached slower.

    Returns:
      (ndarray): An array of y values corresponding to the sigmoid-like function
       evaluated at the input x values -- which are expected to represent age.
    """
    return 1.0 - (max_x - x) / (x * slope + max_x)


def double_sigmoid_exp(x, l1, l2, l3, x_12, x_23, s12, s23):
    """
    Calculate a stair-like function with 3 steps using two exponential
    sigmoid functions.

    This function simulates a stair-like characteristic with 3 steps
    (l1, l2, l3). The transition from l1 to l2 occurs around the
    point x_12, and from l2 to l3 around point x_23.

    The steepness between any two levels is governed independently
    by s12 and s23 respectively.

    Arguments:
        x (float or numpy.array): Input data.
        l1, l2, l3 (float): y-levels for each step.
        x_12, x_23 (float): x-points that indicate halfway between two levels.
        s12, s23 (float): Steepness of the curve between l1 and l2, and l2 and l3,
        respectively. The larger s_ij is, the closer to a step-like
        stair function we get.

    Returns:
        float or numpy.array: the function evaluated at x
    """

    y = l1 + (l2 / (1.0 + np.exp(-s12 * (x - x_12))) +
              l3 / (1.0 + np.exp(-s23 * (x - x_23))))

    return y


def double_sigmoid_tanh(x, l1, l2, l3, x_12, x_23, s=1.0):
    """
    This function simulates a stair-like function with 3 steps
    (l1, l2, l3), using a single specified steepness value (s) for
    the hyperbolic tangent.

    l1> l2 > l3 or l1 < l2 < l3

    Arguments:
        x (float or numpy.array): Input data.
        l1, l2, l3 (float): y-levels for each step.
        x_12, x_23 (float): x-points that indicate halfway between two levels.
        s (float): Steepness of the curve between two levels. The larger s is,
            the closer to a step-like stair function we get.

    Returns:
        float or numpy.array: y-values.

    Example:
        >>> import numpy as np
        >>> import typhoidsim.utils_math as tyum
        >>> x = np.arange(1_000_000, 60_000_000, 1024)
        >>> tyum.double_sigmoid_tanh(x, 2.235, 2.002, 1.5487, 5_050_000, 55_000_000, 1)
    """

    y = l1 + ((l2 - l1) * ((np.tanh(s * (x-x_12)) + 1)/2) +
              (l3 - l2) * ((np.tanh(s * (x-x_23)) + 1)/2))
    return y


def gompertz(x, a, b, c):
    """
    Compute the Gompertz function for a given set of parameters.
    This function is used for describing mortality and ageing-like processes.

    See:
    https://en.wikipedia.org/wiki/Gompertz_function

    The Gompertz function is defined as:
    f(x) = a * exp(-b * exp(-c * x))

    Arguments
        x (array-like): The array of x values at which the function is evaluated
        a (float): The asymptote of the function as x approaches infinity.
        b (float): Displacement along the x-axis
        c (float): The growth rate

    Returns:
        (ndarray): An array of y values corresponding to the gomeprtz function
        evaluated at the input x values.
    """
    return a*np.exp(-b*np.exp(-c*x))


def gompertz_dfun(x, a, b, c):
    """
    Compute the derivative of the Gompertz function with respect to x for a
    given set of parameters.

    This function is used in ecology for modelling ageing processes.

    The derivative of the Gompertz function is defined as:
    f'(x) = a * b * c * exp(-(b / exp(c * x)) - c * x)

    Arguments
        x (array-like): The array of x values at which the function is evaluated
        a (float): The asymptote of the Gompertz function as x approaches infinity.
        b (float): Displacement along the x-axis
        c (float): The growth rate

    Returns:
       (ndarray): An array of derivative values corresponding to the input x values.
    """
    return a*b*c*np.exp(-(b/np.exp(c*x)) - c*x)


def asym_trapezoidal(x, period=365.0, peak_start_doy=45.0, ramp_up_dur=15.0,
                     ramp_dw_dur=25.0, cutoff_dur=0.0, amp=1.0):
    """
    Specify an asymmetric trapezoidal 'wave' profile like the one used
    in Gauld et al 2018 and Kraay et al 2024.
    Args:
        x (array-like): The array of x values at which the function is evaluated, usually time.
        period (float): The period, in days, over which the seasonal repeats.
        peak_start_doy (float): The day of the year at which the environmental exposure reaches its peak.
        ramp_up_dur (float): Duration, in days, of the period over which the environmental exposure route increases seasonally.
        ramp_dw_dur (float): Duration, in days, of the period over which the environmental exposure route descreases seasonally.
        cutoff_dur (float): Duration, in days, in which environmental exposure halts during the low season
        amp (float): usually a numvber between 0 and 1 with the modulating scaling factor of exposure to the environment
    Returns:
         (ndarray): An array of values corresponding to the input asym_trapezoidal(x) values.
    """

    shift_days = ramp_up_dur - peak_start_doy
    half_day = 0.5

    # Adjust in case the "start of the ramp up period" starts before what would usually be the first of january
    ramp_up_start_doy = ((peak_start_doy - ramp_up_dur) + shift_days) % period
    peak_start_doy = (peak_start_doy + shift_days) % period
    peak_dur = (period - cutoff_dur) - (ramp_dw_dur + ramp_up_dur)
    peak_end_doy = ((peak_start_doy + peak_dur) + shift_days) % period
    ramp_dw_end_doy = ((peak_end_doy + ramp_dw_dur) + shift_days) % period

    time_mod = (x + shift_days) % period

    # Define slope up as vectorized operations instead of direct operations
    slope_up = np.where(
        ((time_mod >= ramp_up_start_doy) & (time_mod < peak_start_doy)),
        ((time_mod - ramp_up_start_doy) + half_day) * (amp / ramp_up_dur),
        0
    )

    # Define slope down as vectorized operations instead of direct operations
    slope_dw = np.where(
        ((time_mod >= peak_end_doy) & (time_mod <= ramp_dw_end_doy)),
        amp - (((time_mod - peak_end_doy) - half_day) * (amp / ramp_dw_dur)),
        0
    )

    trapezoidal_pulse = np.where(
        (time_mod < ramp_up_start_doy),
        0,
        np.where(
            ((time_mod >= peak_start_doy-1) & (time_mod <= peak_end_doy)),
            amp,
            slope_up + slope_dw
        )
    )
    return trapezoidal_pulse


def box_exponential(x, start, box_duration, decay_time_constant):
    """
    Generate a time signal which is 0 up to a 'start' time; 1 from
    'start' until 'start + box_duration', and from 'start + box_duration'
    decays as an exponential function.

    From EMDO: `currentEffect *= (1 - dt/decayTimeConstant)`.

    Args:
        z (numpy.ndarray): The array of points at which we calculate the signal, usually time.
        start (float): The start time of the box_duration in the signal.
        box_duration (float): The duration for which the signal remains at 1.
        decay_time_constant (float): The time constant for the exponential decay.

    Returns:
        numpy.ndarray: The resulting signal as an array.

    Example:
        >>> time = np.linspace(0, 10, 1000)
        >>> efficacy_modulation = box_exponential(time, 2, 5, 3)
    """
    # Definition by parts
    conditions = [
        x < start,
        (x >= start) & (x < start + box_duration),
        x >= start + box_duration
    ]

    functions = [
        0.0,
        1.0,
        # decay part, equivalent to emod's effect *= (1 - dt/decayTimeConstant)
        lambda x: np.exp(-((x - (start + box_duration)) / decay_time_constant))
    ]
    val = np.piecewise(x, conditions, functions)
    val = np.where(val < 0, 0.0, val)
    return val
