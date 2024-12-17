"""
Script to use Starsim-AI to migrate Typhoidsim from Starsim v1.0.3 to v2.2.0

Run with e.g.:

    python -i migrate_v1-v2.py
"""

import starsim as ss
import starsim_ai as ssai

# The files to migrate
files = [
    'typhoid.py', # Put this first, most likely to fail
    'environment.py',
    'demographics.py',
    'interventions.py',
    'monitors.py',
    'networks.py',
    'patterns.py',
    'utils.py',
]

exclude = [
    "docs/*",
    'tests/devtests/*',
    'tests/archive/*',
    "setup.py",
    'starsim/arrays.py',
    'starsim/calibration.py',
    'starsim/demographics.py',
    'starsim/distributions.py',
    'starsim/__init__.py',
    'starsim/interventions.py',
    'starsim/loop.py',
    'starsim/networks.py',
    'starsim/parameters.py',
    'starsim/people.py',
    'starsim/products.py',
    'starsim/results.py',
    'starsim/run.py',
    'starsim/samples.py',
    'starsim/settings.py',
    'starsim/sim.py',
    'starsim/utils.py',
    'starsim/version.py',
]

M = ssai.Migrate(
    source_dir = '../typhoidsim', # folder with the code to migrate
    dest_dir = './migrated_new_tests', # folder to output migrated code into
    files = files, # the specific files to migrate
    library = ss, # can also be the path to the starsim folder, which must be the cloned repo (not from pypi)
    v_from = 'v1.0.3', # can be any valid git tag or hash
    v_to = 'v2.2.0', # ditto
    exclude = exclude,
    model = 'gpt-4o', # see ssai.Models for list of allowed models
    parallel = True,
    die = False,
)
M.run()