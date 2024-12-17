
""" Import all Starsim modules """

# Start imports: version and settings
from .version import __version__, __versiondate__, __license__
from .settings import dtypes, options

# Optionally print the license
if options.license:
    print(__license__)

# Finish imports
from .calibration import *
from .defaults import *
from .demographics import *
from .environment import *
from .interventions import *
from .ingest import *
from .monitors import *
from .networks import *
from .patterns import *
from .typhoid import *
from .utils import *
from .utils_eligibility import *
from .utils_math import *
from .visualizers import *
