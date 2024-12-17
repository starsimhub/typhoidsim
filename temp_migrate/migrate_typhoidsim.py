"""
Script to use Starsim-AI to migrate Typhoidsim
"""

import starsim as ss
import starsim_ai as ssai

M = ssai.Migrate(
    source = '/home/cliffk/idm/typhoidsim/typhoidsim', # folder with the code to migrate
    dest = './migrated', # folder to output migrated code into
    library = ss, # can also be the path to the starsim folder, which must be the cloned repo (not from pypi)
    v_from = 'v1.0.3', # can be any valid git tag or hash
    v_to = 'v2.2.0', # ditto
    model = 'gpt-4o', # see ssai.Models for list of allowed models
)
M.run()