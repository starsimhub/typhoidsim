
"""
Define options for Typhoidsim
All options should be set using set() or directly, e.g.::
    ty.options(verbose=False)
"""

import numpy as np
import sciris as sc

__all__ = ['dtypes', 'options']

# Default datatypes, compatible with Starsim
class dtypes:
    bool = bool
    int = np.int64
    float = np.float64
    str = "<U8"
    result_float = np.float64

class Options(sc.objdict):
    """
    Set options for Typhoid -- from Starsim options. This class non-overlapping
    options with ss.options.

    Use ``ty.options.set('defaults')`` to reset all values to default, or
    ``ty.options.set(dpi='default')`` to reset one parameter to default.
    See ``ty.options.help(detailed=True)`` for more information.

    """

    def __init__(self):
        super().__init__()
        optdesc, opts = self.get_orig_options()  # Get the options
        self.update(opts)  # Update this object with them
        self.setattribute('optdesc', optdesc)  # Set the description as an attribute, not a dict entry
        self.setattribute('orig_options', sc.dcp(opts))  # Copy the default options
        return

    @staticmethod
    def get_orig_options():
        """
        Set the default options for Starsim -- not to be called by the user, use
        ``ss.options.set('defaults')`` instead.
        """

        # Options acts like a class, but is actually an objdict for simplicity
        optdesc = sc.objdict()  # Help for the options
        options = sc.objdict()  # The options

        # Data directory
        data_home = sc.thisdir(__file__, 'data')  # Use Sciris utility to get directory
        optdesc.data_home = 'Set default value of data directory with useful input datasets. Use an absolute path.'
        options.data_home = sc.parse_env('TYPHOIDSIM_DATA', data_home, 'str')

        return optdesc, options

    def __call__(self, *args, **kwargs):
        """Allow ``ty.options(dpi=150)`` instead of ``ty.options.set(dpi=150)`` """
        return self.set(*args, **kwargs)

    def __repr__(self):
        """ Brief representation """
        output = sc.objectid(self)
        output += 'Typhoid options (see also ss.options.disp()):\n'
        output += sc.pp(self.to_dict(), output=True)
        return output

    def disp(self):
        """ Detailed representation """
        output = 'Typhoid options (see also ss.options.help()):\n'
        keylen = 10  # Maximum key length  -- "interactive"
        for k, v in self.items():
            keystr = sc.colorize(f'  {k:>{keylen}s}: ', fg='cyan', output=True)
            reprstr = sc.pp(v, output=True)
            reprstr = sc.indent(n=keylen + 4, text=reprstr, width=None)
            output += f'{keystr}{reprstr}'
        print(output)
        return

    def set(self, key=None, value=None, **kwargs):
        """
        Actually change the style. See ``ss.options.help()`` for more information.

        Args:
            key    (str):    the parameter to modify, or 'defaults' to reset everything to default
            value  (varies): the value to specify; use None or 'default' to reset to default
            kwargs (dict):   if supplied, set multiple key-value pairs

        **Example**::
            ty.options.set(dpi=50) # Equivalent to ty.options(dpi=50)
        """

        # Reset to defaults
        if key in ['default', 'defaults']:
            kwargs = self.orig_options  # Reset everything to default

        # Handle other keys
        elif key is not None:
            kwargs = sc.mergedicts(kwargs, {key: value})

        # Reset options
        for key, value in kwargs.items():
            if key not in self:
                keylist = self.orig_options.keys()
                keys = '\n'.join(keylist)
                errormsg = (f"Option {key} not recognized; options are defaults "
                            f"or:\n{keys}\n\nSee help(ty.options.set) for more information.")
                raise sc.KeyNotFoundError(errormsg)
            else:
                if value in [None, 'default']:
                    value = self.orig_options[key]
                self[key] = value

                # Handle special cases
                if key == 'precision':
                    self.set_precision()

        return

    def context(self, **kwargs):
        """
        Alias to set(), for use in a "with" block.

        **Examples**::

            # Silence all output
            with ty.options.context(verbose=0):
                ss.Sim().run()

            # Convert warnings to errors
            with ty.options.context(warnings='error'):
                ss.Sim(location='not a location').init()

            # Use with_style(), not context(), for plotting options
            with ss.options.with_style(dpi=50):
                ss.Sim().run().plot()
        """

        # Store current settings
        on_entry = {k: self[k] for k in kwargs.keys()}
        self.setattribute('on_entry', on_entry)

        # Make changes
        self.set(**kwargs)
        return self

    def get_default(self, key):
        """ Helper function to get the original default options """
        return self.orig_options[key]

    def changed(self, key):
        """ Check if current setting has been changed from default """
        if key in self.orig_options:
            return self[key] != self.orig_options[key]
        else:
            return None

    def set_precision(self):
        if self.precision == 32:
            dtypes.int = np.int32
            dtypes.float = np.float32
        elif self.precision == 64:
            dtypes.int = np.int64
            dtypes.float = np.float64
        else:
            errormsg = f'Precision {self.precision} not recognized; must be 32 or 64'
            raise ValueError(errormsg)
        return

    def to_dict(self):
        """ Convert to dictionary """
        return {k: v for k, v in self.items()}

# Create the options on module load
options = Options()
