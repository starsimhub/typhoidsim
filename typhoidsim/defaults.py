"""
Define some default values, constants for use throughout Typhoidsim
"""

import numpy as np
import sciris as sc


# Datetime
months_per_year = 12.0   # Months per year
days_per_year   = 365.0  # Not quite, because of leap years ...
days_per_week   = 7.0    #
day2year        = 1.0 / days_per_year   # A factor to transfor quantities expressed in days, to quantities expressed in years

# Numeric
tinynum         = np.finfo(np.float64).resolution  # To avoid divisions by-zero

# Plotting defaults
default_plot_granularity = 512  # How many points between x-min and x-max we will plot.

# Age
min_age = 0
max_age = 120

# Average blood volume per kilo of body weight
average_bv_bw = 75  # ml/kg
