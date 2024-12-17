"""
Script to use Starsim-AI to migrate Typhoidsim
"""

import starsim as ss
import starsim_ai as ssai

# The files to migrate
files = [
    'environment.py',
    'demographics.py',
    'interventions.py',
    'monitors.py',
    'networks.py',
    'patterns.py',
    'utils.py',
    'typhoid.py',
    'calibration.py',
]

M = ssai.Migrate(
    source_dir = '../typhoidsim', # folder with the code to migrate
    dest_dir = './gpt_migrated5', # folder to output migrated code into
    files = files, # the specific files to migrate
    library = ss, # can also be the path to the starsim folder, which must be the cloned repo (not from pypi)
    v_from = 'v1.0.3', # can be any valid git tag or hash
    v_to = 'v2.2.0', # ditto
    model = 'gpt-4o', # see ssai.Models for list of allowed models
    parallel = True,
    die = False,
)
M.run()