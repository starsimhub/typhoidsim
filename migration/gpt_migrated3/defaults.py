
"""
Define some default values, constants for use throughout Typhoidsim
"""

import numpy as np
import starsim as ss

# Time constants
months_per_year = ss.time.time_units['year'] / ss.time.time_units['month']  # Convert year to months, assuming time_units are consistent in starsim
days_per_year = ss.time.time_units['year']  # Use the year unit from starsim for consistency
days_per_week = ss.time.time_units['week']  # Use the week unit from starsim for consistency

# Conversion factor from days to years
day2year = ss.time_ratio(unit1='day', unit2='year')  # Use starsim functionality for time conversion

# Numeric constant
tinynum = np.finfo(ss.dtypes.float).resolution  # Use starsim's default float type for consistency

# Plotting defaults
default_plot_granularity = 512  # How many points between x-min and x-max we will plot.

# Age
min_age = 0
max_age = 120

# Average blood volume per kilo of body weight
average_bv_bw = 75  # ml/kg

sorry_mssg = "Sorry 🙈!"

# Enumeration to track origin of infections
class TransmissionRoute(ss.Base):  # Inheriting from ss.Base for consistency with starsim's object model
    ENVIRONMENT = 0
    CONTACT = 1
