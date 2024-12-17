
"""
Numerical and mathematical utility functions for parameterizing model parameters.
"""

import numpy as np
import sciris as sc

# Specify all externally visible functions this file defines
__all__ = ['sigmoid', 'gompertz', 'gompertz_dfun', 'double_sigmoid_exp', 'double_sigmoid_tanh',
           'asym_trapezoidal', 'box_exponential']

def sigmoid(x, max_x, slope):
    """
    Compute a sigmoid-like function, bounded between 0 and 1.

    This function is used in the age-based exposure mechanism in typhoid models.

    Args:
        x (array-like): The input values.
        max_x (float): The asymptote of the function.
        slope (float): Controls the steepness of the curve.

    Returns:
        ndarray: The evaluated sigmoid-like function.
    """
    vals = 1.0 - (max_x - x) / (x * slope + max_x)
    vals = np.clip(vals, 0, 1)
    vals[x >= max_x] = 1.0
    return vals

def double_sigmoid_exp(x, l1, l2, l3, x_12, x_23, s12, s23):
    """
    Compute a stair-like function with 3 steps using two exponential sigmoids.

    Args:
        x (array-like): Input data.
        l1, l2, l3 (float): Levels for each step.
        x_12, x_23 (float): Transition points between levels.
        s12, s23 (float): Steepness between levels.

    Returns:
        array-like: The evaluated function.
    """
    y = l1 + (l2 / (1.0 + np.exp(-s12 * (x - x_12))) +
              l3 / (1.0 + np.exp(-s23 * (x - x_23))))
    return y

def double_sigmoid_tanh(x, l1, l2, l3, x_12, x_23, s=1.0):
    """
    Simulate a stair-like function with 3 steps using tanh.

    Args:
        x (array-like): Input data.
        l1, l2, l3 (float): Levels for each step.
        x_12, x_23 (float): Transition points between levels.
        s (float): Steepness parameter.

    Returns:
        array-like: The evaluated function.
    """
    y = l1 + ((l2 - l1) * ((np.tanh(s * (x - x_12)) + 1) / 2) +
              (l3 - l2) * ((np.tanh(s * (x - x_23)) + 1) / 2))
    return y

def gompertz(x, a, b, c):
    """
    Compute the Gompertz function.

    Args:
        x (array-like): Input values.
        a (float): Asymptote.
        b (float): Displacement.
        c (float): Growth rate.

    Returns:
        array-like: The evaluated Gompertz function.
    """
    return a * np.exp(-b * np.exp(-c * x))

def gompertz_dfun(x, a, b, c):
    """
    Compute the derivative of the Gompertz function.

    Args:
        x (array-like): Input values.
        a, b, c (float): Gompertz parameters.

    Returns:
        array-like: The derivative values.
    """
    return a * b * c * np.exp(-(b / np.exp(c * x)) - c * x)

def asym_trapezoidal(x, period=365.0, peak_start_doy=45.0, ramp_up_dur=15.0,
                     ramp_dw_dur=25.0, cutoff_dur=0.0, max_amp=1.0):
    """
    Generate an asymmetric trapezoidal wave profile.

    Args:
        x (array-like): Input values, typically time in days.
        period (float): Period of the seasonal pattern.
        peak_start_doy (float): Day of the year for peak start.
        ramp_up_dur, ramp_dw_dur, cutoff_dur (float): Durations for ramp up, down, and cutoff.
        max_amp (float): Maximum amplitude.

    Returns:
        array-like: The trapezoidal pulse values.
    """
    shift_days = ramp_up_dur - peak_start_doy
    half_day = 0.5

    ramp_up_start_doy = ((peak_start_doy - ramp_up_dur) + shift_days) % period
    peak_start_doy = (peak_start_doy + shift_days) % period
    peak_dur = (period - cutoff_dur) - (ramp_dw_dur + ramp_up_dur)
    peak_end_doy = (peak_start_doy + peak_dur) % period
    ramp_dw_end_doy = (peak_end_doy + ramp_dw_dur) % period

    time_mod = (x + shift_days) % period

    slope_up = np.where(
        ((time_mod >= ramp_up_start_doy) & (time_mod < peak_start_doy)),
        ((time_mod - ramp_up_start_doy) + half_day) * (max_amp / ramp_up_dur),
        0
    )

    slope_dw = np.where(
        ((time_mod >= peak_end_doy) & (time_mod <= ramp_dw_end_doy)),
        max_amp - ((time_mod - peak_end_doy) * (max_amp / ramp_dw_dur)),
        0
    )

    trapezoidal_pulse = np.where(
        (time_mod < ramp_up_start_doy),
        0,
        np.where(
            ((time_mod >= peak_start_doy - 1) & (time_mod <= peak_end_doy)),
            max_amp,
            slope_up + slope_dw
        )
    )
    return trapezoidal_pulse

def box_exponential(x, start, box_duration, decay_time_constant):
    """
    Generate a box-exponential time signal.

    Args:
        x (array-like): Input points, typically time.
        start (float): Start time.
        box_duration (float): Duration of the 'box' period.
        decay_time_constant (float): Time constant for the exponential decay.

    Returns:
        array-like: The resulting signal.
    """
    x_offset = x - start
    vals = np.zeros(len(x))
    vals = np.where((x_offset >= 0.0) & (x_offset < box_duration), 1.0, vals)
    vals = np.where(x_offset >= box_duration, np.exp(-((x - (start + box_duration)) / decay_time_constant)), vals)
    vals = np.where(vals < 0, 0.0, vals)
    return vals
