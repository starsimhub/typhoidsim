"""
Define the calibration class - migrated from Starsim 2.1.1 and 2.2.0
"""
import os
import datetime

import numpy as np
import pandas as pd
import sciris as sc
import optuna as op
import matplotlib.pyplot as plt
import starsim as ss
from scipy.special import gammaln


__all__ = ['Calibration220', 'CalibComponent220', 'compute_gof']


def compute_gof(expected, predicted, normalize=True, use_frac=False, use_squared=False,
                as_scalar='none', eps=1e-9, skestimator=None, estimator=None, **kwargs):
    """
    Calculate the goodness of fit. By default use normalized absolute error, but
    highly customizable. For example, mean squared error is equivalent to
    setting normalize=False, use_squared=True, as_scalar='mean'.

    Args:
        expected    (arr):   array of reference/expected data points
        predicted   (arr):   corresponding array of predicted (model) points
        normalize   (bool):  whether to divide the values by the largest value in either series
        use_frac    (bool):  convert to fractional mismatches rather than absolute
        use_squared (bool):  square the mismatches
        as_scalar   (str):   return as a scalar instead of a time series: choices are sum, mean, median
        eps         (float): to avoid divide-by-zero
        skestimator (str):   if provided, use this scikit-learn estimator instead
        estimator   (func):  if provided, use this custom estimator instead
        kwargs      (dict):  passed to the scikit-learn or custom estimator

    Returns:
        gofs (arr): array of goodness-of-fit values, or a single value if as_scalar is True

    **Examples**::

        x1 = np.cumsum(np.random.random(100))
        x2 = np.cumsum(np.random.random(100))

        e1 = compute_gof(x1, x2) # Default, normalized absolute error
        e2 = compute_gof(x1, x2, normalize=False, use_frac=False) # Fractional error
        e3 = compute_gof(x1, x2, normalize=False, use_squared=True, as_scalar='mean') # Mean squared error
        e4 = compute_gof(x1, x2, skestimator='mean_squared_error') # Scikit-learn's MSE method
        e5 = compute_gof(x1, x2, as_scalar='median') # Normalized median absolute error -- highly robust
    """

    # Handle inputs
    expected  = np.array(sc.dcp(expected), dtype=float)
    predicted = np.array(sc.dcp(predicted), dtype=float)

    # Scikit-learn estimator is supplied: use that
    if skestimator is not None: # pragma: no cover
        try:
            import sklearn.metrics as sm
            sklearn_gof = getattr(sm, skestimator) # Shortcut to e.g. sklearn.metrics.max_error
        except ImportError as E:
            errormsg = f'You must have scikit-learn >=0.22.2 installed: {str(E)}'
            raise ImportError(errormsg) from E
        except AttributeError as E:
            errormsg = f'Estimator {skestimator} is not available; see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for options'
            raise AttributeError(errormsg) from E
        gof = sklearn_gof(expected, predicted, **kwargs)
        return gof

    # Custom estimator is supplied: use that
    if estimator is not None: # pragma: no cover
        try:
            gof = estimator(expected, predicted, **kwargs)
        except Exception as E:
            errormsg = f'Custom estimator "{estimator}" must be a callable function that accepts `expected` and `predicted` arrays, plus optional kwargs'
            raise RuntimeError(errormsg) from E
        return gof

    # Default case: calculate it manually
    else:
        # Key step -- calculate the mismatch!
        gofs = abs(np.array(expected) - np.array(predicted))

        if normalize and not use_frac:
            expected_max = abs(expected).max()
            if expected > 0:
                gofs /= expected_max

        if use_frac:
            if (expected < 0).any() or (predicted < 0).any():
                print('Warning: Calculating fractional errors for non-positive quantities is ill-advised!')
            else:
                maxvals = np.maximum(expected, predicted) + eps
                gofs /= maxvals

        if use_squared:
            gofs **= 2

        if as_scalar == 'sum':
            gofs = np.sum(gofs)
        elif as_scalar == 'mean':
            gofs = np.mean(gofs)
        elif as_scalar == 'median':
            gofs = np.median(gofs)
        return gofs


def validate_sim_data(data=None, die=None):
    """
    Validate data intended to be compared to the sim outputs, e.g. for calibration

    Args:
        data (df/dict): a dataframe (or dict) of data, with a column "time" plus data columns of the form "module.result", e.g. "hiv.new_infections"
        die (bool): whether to raise an exception if the data cannot be converted (default: die if data is not None but cannot be converted)

    """
    success = False
    if data is not None:
        # Try loading the data
        try:
            data = sc.dataframe(data) # Convert it to a dataframe
            timecols = ['t', 'timevec', 'tvec', 'time', 'day', 'date', 'year'] # If a time column is supplied, use it as the index
            found = False
            for timecol in timecols:
                if timecol in data.cols:
                    if found:
                        errormsg = f'Multiple time columns found: please ensure only one of {timecols} is present; you supplied {data.cols}.'
                        raise ValueError(errormsg)
                    data.set_index(timecol, inplace=True)
                    found = True
            success = True

        # Data loading failed
        except Exception as E:
            errormsg = f'Failed to add data "{data}": expecting a dataframe-compatible object. Error:\n{E}'
            if not die:
                print(errormsg)
            else:
                raise ValueError(errormsg)

    # Validation
    if not success and die == True:
        errormsg = 'Data "{data}" could not be converted and die == True'
        raise ValueError(errormsg)

    return data


def return_fig(fig, **kwargs):
    """ Do postprocessing on the figure: by default, don't return if in Jupyter, but show instead """
    is_jupyter = False
    if is_jupyter:
        print(fig)
        plt.show()
        return None
    else:
        return fig


class Calibration220(sc.prettyobj):
    """
    A class to handle calibration of Starsim simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org).

    Args:
        sim          (Sim)  : the base simulation to calibrate
        calib_pars   (dict) : a dictionary of the parameters to calibrate of the format dict(key1=dict(low=1, high=2, guess=1.5, **kwargs), key2=...), where kwargs can include "suggest_type" to choose the suggest method of the trial (e.g. suggest_float) and args passed to the trial suggest function like "log" and "step"
        n_workers    (int)  : the number of parallel workers (if None, will use all available CPUs)
        total_trials (int)  : the total number of trials to run, each worker will run approximately n_trials = total_trial / n_workers
        reseed       (bool) : whether to generate new random seeds for each trial
        build_fn  (callable): function that takes a sim object and calib_pars dictionary and returns a modified sim
        build_kw  (dict): a dictionary of options that are passed to build_fn to aid in modifying the base simulation. The API is self.build_fn(sim, calib_pars=calib_pars, **self.build_kw), where sim is a copy of the base simulation to be modified with calib_pars
        components (list): CalibComponents independently assess pseudo-likelihood as part of evaluating the quality of input parameters
        eval_fn  (callable): Function mapping a sim to a float (e.g. negative log likelihood) to be maximized. If None, the default will use CalibComponents.
        eval_kwargs  (dict): Additional keyword arguments to pass to the eval_fn
        label        (str)  : a label for this calibration object
        study_name   (str)  : name of the optuna study
        db_name      (str)  : the name of the database file (default: 'starsim_calibration.db')
        keep_db      (bool) : whether to keep the database after calibration (default: false)
        storage      (str)  : the location of the database (default: sqlite)
        sampler (BaseSampler): the sampler used by optuna, like optuna.samplers.TPESampler
        die          (bool) : whether to stop if an exception is encountered (default: false)
        debug        (bool) : if True, do not run in parallel
        verbose      (bool) : whether to print details of the calibration

    Returns:
        A Calibration object
    """

    def __init__(self, sim, calib_pars, n_workers=None, total_trials=None,
                 reseed=True,
                 build_fn=None, build_kw=None, eval_fn=None, eval_kwargs=None,
                 components=None,
                 label=None, study_name=None, db_name=None, keep_db=None,
                 storage=None,
                 sampler=None, die=False, debug=False, verbose=True):

        # Handle run arguments
        if total_trials is None: total_trials = 100
        if n_workers is None: n_workers = sc.cpu_count()
        if study_name is None: study_name = 'starsim_calibration'
        if db_name is None: db_name = f'{study_name}.db'
        if keep_db is None: keep_db = False
        if storage is None: storage = f'sqlite:///{db_name}'

        self.build_fn = build_fn or self.translate_pars
        self.build_kw = build_kw or dict()
        self.eval_fn = eval_fn or self._eval_fit
        self.eval_kwargs = eval_kwargs or dict()
        self.components = components

        n_trials = int(np.ceil(total_trials / n_workers))
        kw = dict(n_trials=n_trials, n_workers=int(n_workers), debug=debug,
                  study_name=study_name,
                  db_name=db_name, keep_db=keep_db, storage=storage,
                  sampler=sampler)
        self.run_args = sc.objdict(kw)

        # Handle other inputs
        self.label = label
        self.sim = sim
        self.calib_pars = calib_pars
        self.reseed = reseed
        self.die = die
        self.verbose = verbose
        self.calibrated = False
        self.before_msim = None
        self.after_msim = None

        # Temporarily store a filename for storing intermediate results
        self.tmp_filename = 'tmp_calibration_%06i.obj'
        self.json = None
        return

    def run_sim(self, calib_pars=None, label=None):
        """ Create and run a simulation """
        sim = sc.dcp(self.sim)
        if label: sim.label = label

        if 'rand_seed' in calib_pars:
            sim.pars['rand_seed'] = calib_pars.pop('rand_seed')

        sim = self.build_fn(sim, calib_pars=calib_pars, **self.build_kw)

        # Run the sim
        try:
            sim.run()
            return sim

        except Exception as E:
            if self.die:
                raise E
            else:
                print(f'Encountered error running sim!\nParameters:\n{calib_pars}\nTraceback:\n{sc.traceback()}')
                output = None
                return output

    @staticmethod
    def translate_pars(sim=None, calib_pars=None):
        """ Take the nested dict of calibration pars and modify the sim """

        if 'rand_seed' in calib_pars:
            sim.pars['rand_seed'] = calib_pars.pop('rand_seed')

        for parname, spec in calib_pars.items():
            if 'path' not in spec:
                raise ValueError(
                    f'Cannot map {parname} because "path" is missing from the parameter configuration.')

            p = spec['path']

            # TODO: Allow longer paths
            if len(p) != 3:
                raise ValueError(
                    f'Cannot map {parname} because "path" must be a tuple of length 3.')

            modtype = p[0]
            dkey = p[1]
            dparkey = p[2]
            dparval = spec['value']
            targetpar = sim[modtype][dkey].pars[dparkey]

            if sc.isnumber(targetpar):
                sim[modtype][dkey].pars[dparkey] = dparval
            elif isinstance(targetpar, ss.Dist):
                sim[modtype][dkey].pars[dparkey].set(dparval)
            else:
                errormsg = 'Type not implemented'
                raise ValueError(errormsg)

        return sim

    @staticmethod
    def _sample_from_trial(pardict=None, trial=None):
        """
        Take in an optuna trial and sample from pars, after extracting them from the structure they're provided in
        """
        pars = sc.dcp(pardict)
        for parname, spec in pars.items():

            if 'value' in spec:
                # Already have a value, likely running initial or final values as part of checking the fit
                continue

            if 'suggest_type' in spec:
                suggest_type = spec.pop('suggest_type')
                sampler_fn = getattr(trial, suggest_type)
            else:
                sampler_fn = trial.suggest_float

            path = spec.pop('path', None)  # remove path for the sampler
            guess = spec.pop('guess', None)  # remove guess for the sampler
            spec['value'] = sampler_fn(name=parname, **spec)  # suggest values!
            spec['path'] = path
            spec['guess'] = guess

        return pars

    def _eval_fit(self, sim, **kwargs):
        """ Evaluate the fit by evaluating the negative log likelihood """
        nll = 0  # Negative log likelihood
        for component in sc.tolist(self.components):
            nll += component(sim)
        return nll

    def run_trial(self, trial):
        """ Define the objective for Optuna """
        if self.calib_pars is not None:
            pars = self._sample_from_trial(self.calib_pars, trial)
        else:
            pars = None

        if self.reseed:
            pars['rand_seed'] = trial.suggest_int('rand_seed', 0, 1_000_000)  # Choose a rand_seed

        sim = self.run_sim(pars)

        # Compute fit
        fit = self.eval_fn(sim, **self.eval_kwargs)
        return fit

    def worker(self):
        """ Run a single worker """
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = op.load_study(storage=self.run_args.storage,
                              study_name=self.run_args.study_name,
                              sampler=self.run_args.sampler)
        output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials,
                                callbacks=None)
        return output

    def run_workers(self):
        """ Run multiple workers in parallel """
        if self.run_args.n_workers > 1 and not self.run_args.debug:  # Normal use case: run in parallel
            output = sc.parallelize(self.worker,
                                    iterarg=self.run_args.n_workers)
        else:  # Special case: just run one
            output = [self.worker()]
        return output

    def remove_db(self):
        """ Remove the database file if keep_db is false and the path exists """
        try:
            if 'sqlite' in self.run_args.storage:
                # Delete the file from disk
                if os.path.exists(self.run_args.db_name):
                    os.remove(self.run_args.db_name)
                if self.verbose: print(
                    f'Removed existing calibration file {self.run_args.db_name}')
            else:
                # Delete the study from the database e.g., mysql
                op.delete_study(study_name=self.run_args.study_name,
                                storage=self.run_args.storage)
                if self.verbose: print(
                    f'Deleted study {self.run_args.study_name} in {self.run_args.storage}')
        except Exception as E:
            if self.verbose:
                print('Could not delete study, skipping...')
                print(str(E))
        return

    def make_study(self):
        """ Make a study, deleting one if it already exists """
        if not self.run_args.keep_db:
            self.remove_db()
        if self.verbose: print(self.run_args.storage)
        output = op.create_study(storage=self.run_args.storage,
                                 study_name=self.run_args.study_name)
        return output

    def calibrate(self, calib_pars=None, load=False, tidyup=True, **kwargs):
        """
        Perform calibration.

        Args:
            calib_pars (dict): if supplied, overwrite stored calib_pars
            load (bool): whether to load existing trials from the database (if rerunning the same calibration)
            tidyup (bool): whether to delete temporary files from trial runs
            verbose (bool): whether to print output from each trial
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        """
        # Load and validate calibration parameters
        if calib_pars is not None:
            self.calib_pars = calib_pars
        self.run_args.update(kwargs)  # Update optuna settings

        # Run the optimization
        t0 = sc.tic()
        self.make_study()
        self.run_workers()
        study = op.load_study(storage=self.run_args.storage,
                              study_name=self.run_args.study_name,
                              sampler=self.run_args.sampler)
        self.best_pars = sc.objdict(study.best_params)
        self.elapsed = sc.toc(t0, output=True)

        self.sim_results = []
        if load:
            if self.verbose: print('Loading saved results...')
            for trial in study.trials:
                n = trial.number
                try:
                    filename = self.tmp_filename % trial.number
                    results = sc.load(filename)
                    self.sim_results.append(results)
                    if tidyup:
                        try:
                            os.remove(filename)
                            if self.verbose: print(
                                f'    Removed temporary file {filename}')
                        except Exception as E:
                            errormsg = f'Could not remove {filename}: {str(E)}'
                            if self.verbose: print(errormsg)
                    if self.verbose: print(f'  Loaded trial {n}')
                except Exception as E:
                    errormsg = f'Warning, could not load trial {n}: {str(E)}'
                    if self.verbose: print(errormsg)

        # Compare the results
        self.parse_study(study)

        if self.verbose: print('Best pars:', self.best_pars)

        # Tidy up
        self.calibrated = True
        if not self.run_args.keep_db:
            self.remove_db()

        return self

    def check_fit(self, n_runs=5):
        """ Run before and after simulations to validate the fit """

        if self.verbose: print('\nChecking fit...')

        before_pars = sc.dcp(self.calib_pars)
        for spec in before_pars.values():
            spec['value'] = spec['guess']  # Use guess values

        after_pars = sc.dcp(self.calib_pars)
        for parname, spec in after_pars.items():
            spec['value'] = self.best_pars[parname]

        before_sim = self.build_fn(self.sim, calib_pars=before_pars,
                                   **self.build_kw)
        before_sim.label = 'Before calibration'
        self.before_msim = ss.MultiSim(before_sim)
        self.before_msim.run(n_runs=n_runs)
        self.before_fits = np.array(
            [self.eval_fn(sim, **self.eval_kwargs) for sim in
             self.before_msim.sims])

        after_sim = self.build_fn(self.sim, calib_pars=after_pars,
                                  **self.build_kw)
        after_sim.label = 'Before calibration'
        self.after_msim = ss.MultiSim(after_sim)
        self.after_msim.run(n_runs=n_runs)
        self.after_fits = np.array(
            [self.eval_fn(sim, **self.eval_kwargs) for sim in
             self.after_msim.sims])

        print(f'Fit with original pars: {self.before_fits}')
        print(f'Fit with best-fit pars: {self.after_fits}')
        if self.after_fits.mean() <= self.before_fits.mean():
            print('✓ Calibration improved fit')
        else:
            print(
                '✗ Calibration did not improve fit, but this sometimes happens stochastically and is not necessarily an error')

        return self.before_fits, self.after_fits

    def parse_study(self, study):
        """Parse the study into a data frame -- called automatically """
        best = study.best_params
        self.best_pars = best

        if self.verbose: print('Making results structure...')
        results = []
        n_trials = len(study.trials)
        failed_trials = []
        for trial in study.trials:
            data = {'index': trial.number, 'mismatch': trial.value}
            for key, val in trial.params.items():
                data[key] = val
            if data['mismatch'] is None:
                failed_trials.append(data['index'])
            else:
                results.append(data)
        if self.verbose: print(
            f'Processed {n_trials} trials; {len(failed_trials)} failed')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i, r in enumerate(results):
            for key in keys:
                if key not in r:
                    warnmsg = f'Key {key} is missing from trial {i}, replacing with default'
                    print(warnmsg)
                    r[key] = best[key]
                data[key].append(r[key])
        self.study_data = data
        self.df = sc.dataframe.from_dict(data)
        self.df = self.df.sort_values(by=['mismatch'])  # Sort
        return

    def to_json(self, filename=None, indent=2, **kwargs):
        """ Convert the results to JSON """
        order = np.argsort(self.df['mismatch'])
        json = []
        for o in order:
            row = self.df.iloc[o, :].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'),
                           pars={})
            for key, val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        self.json = json
        if filename:
            return sc.savejson(filename, json, indent=indent, **kwargs)
        else:
            return json

    def plot_sims(self, **kwargs):
        """
        Plot sims, before and after calibration.

        Args:
            kwargs (dict): passed to MultiSim.plot()
        """
        if self.before_msim is None:
            self.check_fit()

        # Turn off jupyter mode so we can receive the figure handles
        jup = ss.options.jupyter if 'jupyter' in ss.options else sc.isjupyter()
        ss.options.jupyter = False

        self.before_msim.reduce()
        fig_before = self.before_msim.plot()
        fig_before.suptitle('Before calibration')

        self.after_msim.reduce()
        fig_after = self.after_msim.plot(fig=fig_before)
        fig_after.suptitle('After calibration')

        ss.options.jupyter = jup

        return fig_before, fig_after

    def plot_trend(self, best_thresh=None, fig_kw=None):
        """
        Plot the trend in best mismatch over time.

        Args:
            best_thresh (int): Define the threshold for the "best" fits, relative to the lowest mismatch value (if None, show all)
            fig_kw (dict): passed to plt.figure()
        """
        df = self.df.sort_values(
            'index')  # Make a copy of the dataframe, sorted by trial number
        mismatch = sc.dcp(df['mismatch'].values)
        best_mismatch = np.zeros(len(mismatch))
        for i in range(len(mismatch)):
            best_mismatch[i] = mismatch[:i + 1].min()
        smoothed_mismatch = sc.smooth(mismatch)
        fig = plt.figure(**sc.mergedicts(fig_kw))

        ax1 = plt.subplot(2, 1, 1)
        plt.plot(mismatch, alpha=0.2, label='Original')
        plt.plot(smoothed_mismatch, lw=3, label='Smoothed')
        plt.plot(best_mismatch, lw=3, label='Best')

        ax2 = plt.subplot(2, 1, 2)
        max_mismatch = mismatch.min() * best_thresh if best_thresh is not None else np.inf
        inds = sc.findinds(mismatch <= max_mismatch)
        plt.plot(best_mismatch, lw=3, label='Best')
        plt.scatter(inds, mismatch[inds], c=mismatch[inds], label='Trials')
        for ax in [ax1, ax2]:
            plt.sca(ax)
            plt.grid(True)
            plt.legend()
            sc.setylim()
            sc.setxlim()
            plt.xlabel('Trial number')
            plt.ylabel('Mismatch')
        sc.figlayout()
        return fig


class CalibComponent220(sc.prettyobj):
    """
    A class to compare a single channel of observed data with output from a
    simulation. The Calibration class can use several CalibComponent objects to
    form an overall understanding of how will a given simulation reflects
    observed data.

    Args:
        name (str) : the name of this component. Importantly, if
            extract_fn is None, the code will attempt to use the name, like
            "hiv.prevalence" to automatically extract data from the sim object.
        expected (pd.Dataframe) : pandas dataframe containing calibration data.
            The index should be the time in either floating point years or datetime.
        extract_fn (callable) : a function to extract predicted/actual data in the same
            format and with the same columns as `expected`.
        conform (str | callable): specify how to handle timepoints that don't
            align exactly between the expected and the actual/predicted/simulated
            data so they conform to common time grid. Whether the data represents
            a 'prevalent' or an 'incident' quantity impacts how this alignment is performed.

            If 'prevalent', it means data in expected & actual dataframes represent
            the current state of the system, like the number of currently infected
            individuals. In this case, the data in 'actual'  will be interpolated
            to match the timepoints in 'expected', allowing for pointwise comparisons
            between the expected and actual data.

            If 'incident', it means data reflects new instances over a period of time,
            like the number of new vaccinated. In this case, the data in 'actual'
            will be transformed into a cumulative incidence form and then interpolated, ensuring they can be compared accurately with the progressive tally represented by incident data.

        nll_fn (str | callable)
    """

    def __init__(self, name, expected, extract_fn, conform, nll_fn, weight=1):
        self.name = name
        self.expected = expected
        self.extract_fn = extract_fn
        self.weight = weight
        self.nll = np.nan
        self.avail_conforms = {"incident":  linear_accum,
                               "prevalent": linear_interp}

        self.avail_nll_fns = {"beta": nll_beta,
                              "gamma": nll_gamma}

        self.conform = self._validate_conform(conform)
        self.nll_fn  = self._validate_nll_fn(nll_fn)
        return

    def _validate_conform(self, conform):
        if not isinstance(conform, str) and not callable(conform):
            raise Exception(f"The conform argument must be a string or a callable function, not {type(conform)}.")
        elif isinstance(conform, str):
            conform_ = self.avail_conforms.get(conform)
            if conform_ is None:
                avail = self.avail_conforms.keys()
                raise ValueError(f"The conform argument must be one of {avail}, not {conform}.")
        else:
            conform_ = conform
        return conform_

    def _validate_nll_fn(self, nll_fn):
        if not isinstance(nll_fn, str) and not callable(nll_fn):
            msg = f"The nll_fn (negative log-likelihood function) argument must be a string or a callable function, not {type(nll_fn)}."
            raise Exception(msg)
        elif isinstance(nll_fn, str):
            nll_fn_ = self.avail_nll_fns.get(nll_fn)
            avail = self.avail_nll_fns.keys()
            if nll_fn_ is None:
                raise ValueError(f"The nll_fn (negative log-likelihood function) argument must be one of {avail}, not {nll_fn}")
        else:
            nll_fn_ = nll_fn
        return nll_fn_

    def eval(self, sim):
        """ Compute and return the negative log likelihood """
        predicted = self.extract_fn(sim)                     # Extract simulated data
        predicted = self.conform(self.expected, predicted)   # Conform
        self.nll = self.nll_fn(self.expected, predicted)     # Negative log likelihood
        return self.weight * np.sum(self.nll)

    def __call__(self, sim):
        return self.eval(sim)

    def __repr__(self):
        return f"Calibration component with name {self.name}"

    def plot(self):
        raise NotImplementedError


def nll_beta(expected, actual):
    """
    For the beta-binomial negative log-likelihood, we begin with a Beta(1,1) prior
    and subsequently observe actual['x'] successes (positives) in actual['n'] trials (total observations).
    The result is a Beta(actual['x']+1, actual['n']-actual['x']+1) posterior.
    We then compare this to the real data, which has expected['x'] successes (positives) in expected['n'] trials (total observations).
    To do so, we use a beta-binomial likelihood:
    p(x|n, x, a, b) = (n choose x) B(x+a, n-x+b) / B(a, b)
    where
      x=expected['x']
      n=expected['n']
      a=actual['x']+1
      b=actual['n']-actual['x']+1
    and B is the beta function, B(x, y) = Gamma(x)Gamma(y)/Gamma(x+y)

    We compute the log of p(x|n, x, a, b), noting that gammaln is the log of the gamma function

    Args:
        expected (pd.Dataframe): dataframe containing reference data (usually empirical data).
             The index should be the time in either floating point years or datetime.
        actual (pd.Dataframe): dataframe containing the 'current' or 'actual' data we have (usually simulated) data
             The index should be the time in either floating point years or datetime.
    Returns
        -logL (float): negative likelihood
    """
    e_n, e_x = expected['n'], expected['x']
    a_n, a_x = actual['n'], actual['x']
    logL = gammaln(e_n + 1) - gammaln(e_x + 1) - gammaln(e_n - e_x + 1)
    logL += gammaln(e_x + a_x + 1) + gammaln(
        e_n - e_x + a_n - a_x + 1) - gammaln(e_n + a_n + 2)
    logL += gammaln(a_n + 2) - gammaln(a_x + 1) - gammaln(a_n - a_x + 1)
    return -logL


def nll_gamma(expected, actual):
    """
    Also called negative binomial, but parameterized differently. The gamma-poisson
    likelihood is a Poisson likelihood with a gamma-distributed rate parameter

    Args:
        expected (pd.Dataframe): dataframe containing reference data (usually empirical data).
             The index should be the time in either floating point years or datetime.
        actual (pd.Dataframe): dataframe containing the 'current' or 'actual' data we have (usually simulated) data
             The index should be the time in either floating point years or datetime.

    Returns
        -logL (float): negative likelihood
    """
    e_n, e_x = expected['n'], expected['x']
    a_n, a_x = actual['n'], actual['x']
    logL = gammaln(e_x + a_x + 1) - gammaln(e_x + 1) - gammaln(e_x + 1)
    logL += (e_x + 1) * np.log(e_n)
    logL += (a_x + 1) * np.log(a_n)
    logL -= (e_x + a_x + 1) * np.log(e_n + a_n)
    return -logL


def linear_interp(expected, actual):
    """
    Use for prevalent data like prevalence

    Args:
        expected (pd.Dataframe): dataframe containing reference data (usually from empirical sources).
             The index should be the time in either floating point 'calendar years' or datetime.
        actual (pd.Dataframe): dataframe containing the 'current' or 'actual' data we have (usually from simulated sources) data
             The index should be the time in either floating point 'calendar years' or datetime.
    Returns:
        conformed (pd.Dataframe): dataframe containing the actual or current data
            that have been interpolated to match a common timeframe with the data in `expected`.
            The interpolation ensures that the two datasets (expected and actual)
            can be compared directly (one-to-one) or used together in further
            analysis, because they are now aligned to the same time grid.
    """
    conformed = pd.DataFrame(index=expected.index)
    common_time_grid = expected.index
    for col in actual:
        conformed[col] = np.interp(x=common_time_grid, xp=actual.index, fp=actual[col])
    return conformed


def linear_accum(expected, actual):
    """
    Interpolate in the accumulation, then difference.
    Use for incident data like incidence or new_deaths

    Args:
        expected (pd.Dataframe): dataframe containing reference data (usually from empirical sources).
             The index should be the time in either floating point 'calendar years' or datetime.
        actual (pd.Dataframe): dataframe containing the 'current' or 'actual' data we have (usually from simulated sources) data
             The index should be the time in either floating point 'calendar years' or datetime.

    Returns:
        conformed (pd.Dataframe): dataframe containing the actual or current data
            that have been interpolated to match a common timeframe with the data in `expected`.
            The interpolation ensures that the two datasets (expected and actual)
            can be compared directly (one-to-one) or used together in further
            analysis, because they are now aligned to the same time grid.

    """
    conformed = pd.DataFrame(index=expected.index)
    common_time_grid = expected.index
    t_step = np.diff(common_time_grid )
    assert np.all(t_step == t_step[0])  # Check we have regularly sampled data

    # Make common time grid
    cum_time_grid = np.append(common_time_grid, common_time_grid[-1] + t_step)  # Add one more because later we'll diff

    if isinstance(actual.index, pd.DatetimeIndex):
        actual_time_grid = np.array([sc.datetoyear(t) for t in actual.index if isinstance(t, datetime.date)])
    else:
        actual_time_grid = actual.index

    for col in actual:
        sdi = np.interp(x=cum_time_grid, xp=actual_time_grid, fp=actual[col].cumsum())
        conformed[col] = pd.Series(np.diff(sdi), index=common_time_grid)
    return conformed


def dirichlet_single(raw_data, sim_data):
    # from emod-based calibration
    # TODO: document and refactor
    num_cat_bins = len(raw_data)
    raw_nobs = sum(raw_data)
    sim_nobs = sum(sim_data)
    ll = 0.
    ll += gammaln(raw_nobs + 1)
    ll += gammaln(sim_nobs + num_cat_bins)
    ll -= gammaln(raw_nobs + sim_nobs + num_cat_bins)
    for catbin in range(num_cat_bins):
        ll += gammaln(raw_data[catbin] + sim_data[catbin] + 1)
        ll -= gammaln(sim_data[catbin] + 1)
        ll -= gammaln(raw_data[catbin] + 1)
    ll /= num_cat_bins
    return ll


def beta_binomial(raw_nobs, sim_nobs, raw_data, sim_data, return_mean=True):
    # from emod-based calibration
    # TODO: document and refactor
    num_bins = len(raw_data)
    ll = 0.
    for this_bin in range(num_bins):
        ll += gammaln(raw_nobs[this_bin] + 1)
        ll += gammaln(sim_nobs[this_bin] + 2)
        ll -= gammaln(raw_nobs[this_bin] + sim_nobs[this_bin] + 2)
        ll += gammaln(raw_data[this_bin] + sim_data[this_bin] + 1)
        ll += gammaln(raw_nobs[this_bin] - raw_data[this_bin] + sim_nobs[this_bin] - sim_data[this_bin] + 1)
        ll -= gammaln(raw_data[this_bin] + 1)
        ll -= gammaln(raw_nobs[this_bin] - raw_data[this_bin] + 1)
        ll -= gammaln(sim_data[this_bin] + 1)
        ll -= gammaln(sim_nobs[this_bin] - sim_data[this_bin] + 1)
    if num_bins != 0 and return_mean:
        ll /= num_bins
    return ll