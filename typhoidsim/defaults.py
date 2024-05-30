"""
Define some default values, constants for use throughout Typhoidsim
"""

import numpy as np
import sciris as sc


# Datetime
months_per_year = 12   # Months per year
days_per_year   = 365  # Not quite, because of leap years ...
days_per_week   = 7    #

# Numeric
eps             = np.finfo(np.float64).resolution  # To avoid divisions by-zero

# Plotting defaults
default_plot_granularity = 512