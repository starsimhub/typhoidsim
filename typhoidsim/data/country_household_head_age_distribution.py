"""
Distribution of age of the head of household
https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/undesa_pd_2022_hh-size-composition.xlsx
"""


data = {
     'Chile':     {'0-15' :  0.0,     # Assumed
                   '16-19':  0.39,    # Chile: Data Soruce:IPUMS: Ref. year: 2017
                   '20-64':  71.82,
                   '65-84':  19.7,
                   '85+'  :  0.0},    # Assumed
     'Pakistan':  {'0-15' :  0.0,     # Assumed
                   '16-19':  0.52,    # Pakistan: Data Source: DHS; Ref. year: 2013
                   '20-64': 86.62,
                   '65-84': 12.87,
                   '85+'  : 0.0, },   # Assumed
}
