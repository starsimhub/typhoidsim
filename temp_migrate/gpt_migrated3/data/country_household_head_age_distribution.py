
"""
Distribution of age of the head of household
https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/undesa_pd_2022_hh-size-composition.xlsx
"""

import starsim as ss

# Define a function to create AgeGroup distributions for each region
def create_age_distributions(data):
    age_distributions = {}
    for region, age_data in data.items():
        age_groups = []
        for age_range, proportion in age_data.items():
            low, high = map(int, age_range.replace('+', '-100').split('-'))
            age_groups.append((ss.AgeGroup(low=low, high=high), proportion))
        age_distributions[region] = age_groups
    return age_distributions

# Example data
data = {
    'Chile': {
        '0-15': 0.0,     # Assumed
        '16-19': 0.39,   # Chile: Data Source: IPUMS: Ref. year: 2017
        '20-64': 71.82,
        '65-84': 19.7,
        '85+': 0.0       # Assumed
    },
    'Pakistan': {
        '0-15': 0.0,     # Assumed
        '16-19': 0.52,   # Pakistan: Data Source: DHS; Ref. year: 2013
        '20-64': 86.62,
        '65-84': 12.87,
        '85+': 0.0       # Assumed
    }
}

# Generate age distributions
age_distributions = create_age_distributions(data)

# Print the results for verification
for region, distributions in age_distributions.items():
    print(f"\n{region} Age Distribution:")
    for age_group, proportion in distributions:
        print(f"  {age_group}: {proportion}%")
