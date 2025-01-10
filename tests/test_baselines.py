"""
Test that the current version of Typhoidsim exactly matches
the baseline results.
"""

import sciris as sc
import starsim as ss
import typhoidsim as ty

baseline_filename = sc.thisdir(__file__, 'baseline.json')
benchmark_filename = sc.thisdir(__file__, 'benchmark.json')
parameters_filename = sc.thisdir(ss.__file__, 'regression',
                                 f'pars_v{ss.__version__}.json')
sc.options(interactive=False)  # Assume not running interactively

# Define the parameters
pars = sc.objdict(
    n_agents=10e3,             # Number of agents
    start=2000,                # Starting year
    dur=2,                # Number of years to simulate
    dt=1.0/ty.days_per_year,   # Timestep of 1 day, expressed in years
    verbose=0,                 # Don't print details of the run
    rand_seed=2,               # Set a non-default seed
)


def make_sim(run=False):
    """
    Define a default simulation for testing the baseline. If run directly (not
    via pytest), also plot the sim by default.
    """
    diseases = [ty.Typhoid()]
    networks = [ss.RandomNet({'n_contacts': 1})]
    demographics = [ty.EnvironmentalPool()]

    sim = ss.Sim(pars=pars, networks=networks, demographics=demographics, diseases=diseases)

    # Optionally run and plot
    if run:
        sim.run()
        sim.plot()

    return sim


def save_baseline():
    """
    Refresh the baseline results. This function is not called during standard testing,
    but instead is called by the update_baseline script.
    """
    sc.heading('Updating baseline values...')

    # Make and run sim
    sim = make_sim()
    sim.run()

    # Export results
    sim.to_json(filename=baseline_filename, keys='summary')

    print('Done.')
    return


def test_baseline():
    """ Compare the current default sim against the saved baseline """

    # Load existing baseline
    baseline = sc.loadjson(baseline_filename)
    old = baseline['summary']

    # Calculate new baseline
    new = make_sim()
    new.run()

    # Compute the comparison
    ss.diff_sims(old, new, die=True)

    return new


def test_benchmark(do_save=False, repeats=1, verbose=True):
    """ Compare benchmark performance """

    if verbose: print('Running benchmark...')
    try:
        previous = sc.loadjson(benchmark_filename)
    except FileNotFoundError:
        previous = None

    t_inits = []
    t_runs = []

    def normalize_performance(reference=0.1):
        """
        Calculate a performance ratio based on the execution time of array
        multiplication compared to a benchmarked time on an Intel
        Core i9-9900K CPU @ 3.60GHz. This indicates the CPU performance
        relative to the reference CPU.

        Returns:
        float: Ratio of reference time to actual time.
        Greater than 1 implies better performance than the reference CPU,
        less than 1 implies lower performance.
        """
        local_baseline_performance = ty.test_cpu_performance()
        ratio = reference / local_baseline_performance
        return ratio

    # Test CPU performance before the run
    r1 = normalize_performance()

    # Do the actual benchmarking
    for r in range(repeats):
        print(f'Repeat {r}')

        # Time initialization
        t0 = sc.tic()
        sim = make_sim()
        sim.init()
        t_init = sc.toc(t0, output=True)

        # Time running
        t0 = sc.tic()
        sim.run()
        t_run = sc.toc(t0, output=True)

        # Store results
        t_inits.append(t_init)
        t_runs.append(t_run)

    # Test CPU performance after the run
    r2 = normalize_performance()
    ratio = (r1 + r2) / 2
    t_init = ratio * min(t_inits)
    t_run = ratio * min(t_runs)

    # Construct json
    n_decimals = 3
    json = {'time': {
        'initialize': round(t_init, n_decimals),
        'run': round(t_run, n_decimals),
    },
        'parameters': {
            'n_agents': sim.pars.n_agents,
            'dur': sim.pars.dur,
            'dt': sim.pars.dt,
        },
        'cpu_performance': ratio,
    }

    if verbose:
        if previous:
            print('Previous benchmark:')
            sc.pp(previous)

        print('\nNew benchmark:')
        sc.pp(json)
    else:
        brief = sc.dcp(json['time'])
        brief['cpu_performance'] = json['cpu_performance']
        sc.pp(brief)

    if do_save:
        sc.savejson(filename=benchmark_filename, obj=json, indent=2)

    if verbose:
        print('Done.')

    return json


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    json = test_benchmark()  # Run this first so benchmarking is available even if results are different
    new = test_baseline()
    sim = make_sim(run=do_plot)

    T.toc()
