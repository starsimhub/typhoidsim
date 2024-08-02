"""
Classes that supporte introducing 'pre-defined'
spatiotemporal profiles as parameters. These patterns
can be based on a callable or on data.

"""

import numpy as np
import numexpr as ne
import matplotlib.pyplot as pl

import sciris as sc
import starsim as ss

import typhoidsim.defaults as tyd

__all__ = ["Pattern", "StateVariable"]


class Pattern:
    """
    Base class for evaluating a function as a function of an independent
    variable like age or time, or 'agent-space'. Based on starsim's Distribution class
    can be expanded in functionality.

    Args:
        equation (str): required; a simple string the defines the pattern
            as a function of var
        pattern_name (str): the name for this class of pattern (e.g. "temporal")
        name (str): the name for this particular pattern (e.g. "seasonal_modulation_exposure")
        sim (Sim): usually determined on initialization; the sim to use as input to callable parameters
        module (Module): usually determined on initialization; the module to use as input to callable parameters
        kwargs (dict): parameters of the patter

    **Examples**::
        pattern = typ.Pattern("a + var * b",
                              pars={'a':2, 'b': 0})
        res = pattern(var)
    """

    def __init__(
        self,
        equation,
        pattern_name: str = None,
        name=None,
        sim=None,
        module=None,
        debug=False,
        **kwargs,
    ):
        self.equation = equation
        self.pattern_name = pattern_name
        self.name = name
        self.pars = sc.dictobj(kwargs)  # The user-defined kwargs
        self.module = module
        self.sim = sim
        self.debug = debug
        self._pars = None  # Validated and transformed (if necessary) parameters

        # History and random state
        self.trace = None  # The path of this object within the parent
        self.ready = True
        self.initialized = False
        self.initialize()
        return

    def __repr__(self):
        """Custom display to show state of object"""
        tracestr = "<no trace>" if self.trace is None else f"{self.trace}"
        classname = self.__class__.__name__
        pttrnstr = ""
        if classname == "Pattern":
            if self.equation is not None:
                pttrnstr = f"y={self.equation}, "
        string = f"typ.{classname}({tracestr}, {pttrnstr}pars={dict(self.pars)})"
        return string

    def __call__(self, var=None):
        """Alias to self.evaluate(var)"""
        pattern = None
        if var is not None:
            pattern = self.evaluate(var)
        return pattern

    def disp(self):
        """Return full display of object"""
        return sc.pr(self)

    def show_state(self, output=False):
        """Show the state of the object"""
        keys = [
            "pars",
            "equation",
            "ready",
        ]
        data = {key: getattr(self, key) for key in keys}
        s = f"{self}"
        for key, val in data.items():
            s += f"\n{key:9s} = {val}"
        if output:
            return data
        else:
            print(s)
            return

    def validate_equation(self):
        # TODO: perform some basic checks that the equation is well formed?
        # See: https://github.com/pyparsing/pyparsing/blob/master/examples/fourFn.py
        # Numexpr 2.8.5 added stricter checks on which operators can be used
        # See: https://github.com/pydata/numexpr/issues/442
        pass

    def set(self, *args, equation=None, **kwargs):
        """
        Set (change) the equation of this pattern, or one or more of its
        parameters
        """
        if equation:
            self.equation = equation
            self.validate_equation()
        if args:
            if kwargs:
                errormsg = f"You can supply args or kwargs, but not both (args={len(args)}, kwargs={len(kwargs)})"
                raise ValueError(errormsg)
            else:  # Convert from args to kwargs
                parkeys = list(self.pars.keys())
                for i, arg in enumerate(args):
                    kwargs[parkeys[i]] = arg
        if kwargs:
            self.pars.update(kwargs)
            # self.process_pars(call=False)
        return

    def initialize(self, trace=None, module=None, sim=None, force=False):
        """Calculate the starting seed and create the RNG"""

        if (
            self.initialized is True and not force
        ):  # Don't warn if we have a partially initialized class
            msg = (
                f"Pattern {self} is already initialized, use force=True if intentional"
            )
            ss.warn(msg)

        # Handle the sim, module, and slots
        self.link_sim(sim)
        self.link_module(module)
        # self.process_pars(call=False)
        self.ready = True
        self.initialized = True
        return self

    def link_sim(self, sim=None, overwrite=False):
        """Shortcut for linking the sim, only overwriting an existing one if overwrite=True"""
        if (not self.sim or overwrite) and sim is not None:
            self.sim = sim
        return

    def link_module(self, module=None, overwrite=False):
        """Shortcut for linking the module"""
        if (not self.module or overwrite) and module is not None:
            self.module = module
        return

    def evaluate(self, var):
        """
        Generate a discretised representation of the equation for the domain
        represented by ``var``.

        The argument ``var`` can represent time, or age. It can be be a
        single number, a numpy.ndarray or pandas series???
        """
        self._pars = sc.cp(self.pars['pars'])
        self._pars['var'] = var
        return ne.evaluate(self.equation, local_dict=self._pars)

    def generate_data(self, vmin=0, vmax=128, step=None):
        """
        NOTE: The variable name of the actual independent variable
        in the equation expression should be named var
        """
        if step is None:
            step = float(vmax - vmin) / tyd.default_plot_granularity

        var = np.arange(vmin, vmax + step, step)
        var = var[np.newaxis, :]

        y = self.evaluate(var)
        return var.flat, y.flat

    def plot(self, data_kw=None, fig_kw=None, var_name=None):
        """Plot the pattern"""
        pl.figure(**sc.mergedicts(fig_kw))
        x, y = self.generate_data(**sc.mergedicts(data_kw))
        pl.plot(x, y)
        pl.title(str(self))
        pl.xlabel(var_name or "var")
        pl.ylabel(f"{self.equation}")
        return


class StateVariable(np.ndarray):
    """
    This class is identical to Results, but named to something more generic,
    so it can be used in different parts of the code and still make sense
    from a code-readability perspective. The concept of Results is very
    does not necessarily align with the internal-state of the model/simulation.
    A results array may have outputs that are a transformed version of the
    internal 'state variables' and as such ideally we don't want them to be used
    directly. A Results arrary *can* also map one-to-one to one of the internal
    state variables, but this is not always guaranteed.

    This is a structure that holds an internal state of our system (ie, module)
    prior to any transformations for output.

    A pattern can also be used to pass a timeseries that will be used
    to perform amplitude-modulation of a module parameter.
    """

    def __new__(cls, module=None, name=None, shape=None, dtype=None, scale=None):
        arr = np.zeros(shape=shape, dtype=dtype).view(cls)
        arr.name = name
        arr.module = module
        arr.scale = scale
        return arr

    def __repr__(self):
        modulestr = f"{self.module}." if (self.module is not None) else ""
        cls_name = self.__class__.__name__
        arrstr = super().__repr__().removeprefix(cls_name)
        out = f"{cls_name}({modulestr}{self.name}):\narray{arrstr}"
        return out

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", None)
        self.module = getattr(obj, "module", None)
        self.scale = getattr(obj, "scale", None)
        return

    def __array_wrap__(self, obj, **kwargs):
        if obj.shape == ():
            return obj[()]
        else:
            return super().__array_wrap__(obj, **kwargs)

    def to_df(self):
        return sc.dataframe({self.name: self})


class StateVariables(ss.ndict):

    def __init__(self, module, strict=True, *args, **kwargs):
        super().__init__(type=StateVariable, strict=strict)
        if hasattr(module, 'name'):
            module = module.name
        self.setattribute('_module', module)
        return

    def append(self, arg, key=None):
        if isinstance(arg, (list, tuple)):
            state_variable = StateVariable(self._module, *arg)
        elif isinstance(arg, dict):
            state_variable = StateVariable(self._module, **arg)
        else:
            state_variable = arg
        if state_variable.module != self._module:
            state_variable.module = self._module

        super().append(state_variable, key=key)
        return

    def to_df(self):
        pass

    def __repr__(self, *args, **kwargs):
        return super().__repr__(*args, **kwargs)

    def plot(self):
        pass


## ----------------- Vignettes of  Data Classes --------------------------- ####
# class Equation(DataClass):
#     equation = Field(
#         field_type=str,
#         label="Equation as a string",
#         doc=""" The equation in a format that can be interpreted by numexpr""",
#     )
#
#     pars = Field(
#         field_type=dict,
#         label="Parameters for this equation passed in a dictionary.",
#         default=lambda: {},
#         doc="""Should be a list of the parameters and their meaning, Traits
#                 should be able to take defaults and sensible ranges from any
#                 traited information that was provided.""",
#     )
#
#     def evaluate(self, var):
#         """
#         Generate a discrete representation of the equation for the domain
#         represented by ``var`` (or x).
#
#         The argument ``var`` can represent time, or age. It can be be a
#         single number, a numpy.ndarray or pandas series???
#         """
#         return ne.evaluate(self.equation, global_dict=self.pars)
#
#
# class Linear(Equation):
#     """
#     A linear equation. The idea is that would be to use it as ss.Distributions
#     are used. IE,
#
#     class MyModule:
#     ...
#     self.pars = {
#     my_parameter=ss.Linear()
#     }
#
#      my_parameter.evaluate(ti) or my_parameter(ti)
#      my_parameter.evaluate(age) or my_parameter(age)
#
#     """
#
#     equation = Field(
#         label="Linear Equation",
#         default="a * var + b",
#         doc=""":math:`result = a * x + b`""",
#     )
#
#     pars = Field(
#         field_type=dict,
#         label="Parameters of a linear function",
#         default=lambda: {"a": 1.0, "b": 0.0},
#     )
