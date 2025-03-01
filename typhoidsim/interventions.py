"""
Define Typhoid-specific treatments and diagnostics (interventions). It includes
the intervention (eg, campaign that finds eligible people) and also products.
"""
import numpy as np
import sciris as sc
import starsim as ss
import functools

from .patterns import Pattern
from .defaults import days_per_year
from . import utils_math as tyum
from .products import typhoid_test
from . import immunity as tyi


# Interventions
# Diagnostics
__all__  = ['routine_acute_screening']
# Treatments applied to people
__all__ += ['acute_treatment', 'infection_clearence', 'vaccination_with_waning']
# Interventions applied to the environment or environmental transmission
__all__ += ['shedding_reduction', 'environmental_cleanup', 'environmental_exposure_reduction',
            'environmental_seasonality', 'environmental_trapezoidal_modulation']
# Interventions that are not treatments but change some of the agents properties
__all__ += ['behavioral_change']


# -- Treatments
class acute_treatment(ss.Intervention):
    """
    Treat acute (symptomatic) subjects.

    This is expected to use the "infectiousness_redux" product, which
    results in an effective reduction or blocking in shedding, by reducing
    an agent's infectiousness level.

    For instance, Infectiousness is constant throughout the acute duration and is
    determined TAI (Typhoid Acute Infectiousness). This is mediated by the
    product's treatment multiplier when an individual seeks treatment.
    For an agent who sought and received treatment,
    from treatment day to the end of the stage, its new infectiousness level is
    I(Acute) = TAI * product treatment multiplier

    The basic mechanics of this treatment are:
    - 0. Find candidate agents:
       - a. new candidates: acute agents who are untreated and would
            seek treatment today
       - b. previous candidates: acute agents under treatment continue to
            be under treatment until they are no longer acute
    - 1. Of the ones that are eligible to 'seek' treatment today, decide who
         does and who doesnt receive treatment
    - 2. Treat new candidates
    - 3. Count how many people receive treatment today
    """

    def __init__(self, product=None, prob=1.0, eligibility=None, **kwargs):
        super().__init__(**kwargs) # CK: TODO: move to update_pars()?
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.coverage_dist = ss.bernoulli(p=self.prob)
        self.treated = ss.State('treated')
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_treated', dtype=int, label='Number treated')
        )
        return

    def step(self):
        new_candidates = self.get_eligible()
        receive_uids = self.coverage_dist.filter(new_candidates)
        newly_treated = len(receive_uids)
        if newly_treated:
            self.product.administer(self.sim.people, receive_uids)
            self.treated[receive_uids] = True
            # How may accepted and started treatment today
        self.results['n_treated'][self.ti] = newly_treated
        return

    def get_eligible(self):
        """
        Get candidates for treatment on this timestep. This includes new patients
        and old patients.
        """
        # TODO: use self.eligibility
        # Only agents experience the acute stage of infection
        acute_uids = self.sim.people.typhoid.acute.uids
        # Those who would seek treatment today
        seeks_treatment = (self.sim.people.typhoid.ti_seek_trtmnt == self.sim.ti).uids
        new_candidates = acute_uids.intersect(seeks_treatment)
        return new_candidates


class infection_clearence(ss.Intervention):
    """
    Clears an infected individual's Typhoid infection. It will clear Typhoid
    infections of all types (prepatent, acute, subclinical, chronic)

    This is expected to use the "infectiousness_clearence" product, which
    results in an effective reduction or blocking in shedding, by reducing
    an agent's infectiousness level at every timestep.

    The basic mechanics of this treatment are:
    - 0. Find candidate agents:
       - a. new candidates: infected agents
       - b. previous candidates: people under treatment continue to
            be under treatment until they are no longer infected.
    - 2. Treat new candidates AND patients under treatment
    - 3. Count how many people receive treatment today and how many NEW started treatment today
    """

    def __init__(self, product=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.eligibility = eligibility
        self._parse_product(product)
        self.treated = ss.State('treated')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.define_results(
            ss.Result('n_treated', dtype=int, label='Number treated'),
            ss.Result('n_started_tr', dtype=int, label='Started treatment')
        )
        self.initialized = True
        return

    def step(self):
        sim = self.sim
        new_patients, under_treatment = self.get_eligible()
        newly_treated = len(new_patients)
        all_treated = newly_treated + len(under_treatment)

        if len(new_patients):
            self.product.administer(new_patients)
            self.treated[new_patients] = True

        if len(under_treatment):
            self.product.administer(under_treatment)

        self.results['n_started_tr'][self.ti] = newly_treated
        self.results['n_treated'][self.ti] = all_treated

        # Check if infectiousness was cleared in this timestep
        treated = ss.uids.cat(new_patients, under_treatment)
        cleared_uids = treated.intersect((sim.people.typhoid.infectiousness < 0).uids)
        if len(cleared_uids):
            # Reset infectiousness
            sim.people.typhoid.infectiousness[cleared_uids] = 0.0

            # Reset infected states
            for state in [
                "prepatent",
                "acute",
                "subclinical",
                "chronic",
            ]:
               sim.people.typhoid.statesdict[state][cleared_uids] = False

            # Reset time of death if this patient was supposed die
            for state in [
                "ti_prepatent",
                "ti_acute",
                "ti_seek_trtment",
                "ti_subclinical",
                "ti_chronic",
                "ti_dead",
            ]:
                sim.people.typhoid.statesdict[state][cleared_uids] = np.nan

            # Set recovered state and when this agent becomes susceptible
            sim.people.typhoid.statesdict["recovered"][cleared_uids] = True
            sim.people.typhoid.statesdict["ti_recovered"][cleared_uids] = sim.ti # CK: TODO: should this be sim.ti or self.ti?
            sim.people.typhoid.statesdict["ti_susceptible"][cleared_uids] = sim.ti + 1
            # Count again
            sim.people.typhoid.update_results()

        return

    def get_eligible(self):
        """
        Get candidates for treatment on this timestep. This includes new patients
        and old patients.
        """
        # TODO: use self.eligibility
        # Only agents experience the acute stage of infection
        sim = self.sim
        infected_uids = (sim.people.typhoid.infected).uids

        # Those who are under treatment
        under_treatment = (sim.people.infection_clearence.treated).uids
        untreated = (~sim.people.infection_clearence.treated).uids

        # and still are in the acute stage
        old_patients = infected_uids.intersect(under_treatment)
        new_patients = infected_uids.intersect(untreated)
        return new_patients, old_patients


# -- Diagnostics
# PSL: this copy is here because the starsim version assumes every product has a hierarchy
# which then prevents us from using BaseTest, BaseScreeening and routine_screeening clases
class BaseTest(ss.Intervention):
    """
    Base class for screening and triage.

    Args:
         product        (Product)       : the diagnostic to use
         prob           (float/arr)     : annual probability of eligible people receiving the diagnostic
         eligibility    (inds/callable) : indices OR callable that returns inds
         kwargs         (dict)          : passed to Intervention()
    """

    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.screened = ss.BoolArr('screened')
        self.screens = ss.FloatArr('screens', default=0)
        self.ti_screened = ss.FloatArr('ti_screened')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        try:
            self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        except AttributeError:
            # PSL: Not every product will have a hierarchy -- also I don't know what a product hierarchy is
            self.outcomes = ss.uids()
        return

    def deliver(self):
        """
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        """
        sim = self.sim
        ti = sc.findinds(self.timepoints, sim.ti)[0]
        prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
        eligible_uids = self.check_eligibility()  # Check eligibility
        self.coverage_dist.set(p=prob)
        accept_uids = self.coverage_dist.filter(eligible_uids)
        if len(accept_uids):
            self.outcomes = self.product.administer(accept_uids)  # Actually administer the diagnostic
        return accept_uids

    def check_eligibility(self):
        raise NotImplementedError


class base_screening(BaseTest):
    """
    By default, the eligible people to be screened are people in the acute stage.

    Args:
         eligibility_kwargs (dict)  : keyword arguments passed to eligibilty() if eligibility is a function/callable with the signature eligibility(sim, **eligibility_kwargs)
         kwargs         (dict)      : passed to Intervention()
    """
    # BaseTest kwargs: product=None, prob=None, eligibility=None,
    def __init__(self, eligibility_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.eligibility_kwargs = sc.mergedicts(eligibility_kwargs)  # Converts None to {}
        self.coverage_dist = ss.bernoulli(p=self.prob)
        self.ti_positive = ss.FloatArr('ti_positive')  # store when a person tested positive
        self.validate_eligibility()
        self.outcomes = ss.uids()
        return

    def validate_eligibility(self):
        import functools
        if self.eligibility is None and self.eligibility_kwargs is None:
            self.eligibility = self.default_eligibility

        if callable(self.eligibility):
            self.eligibility = functools.partial(self.eligibility, **self.eligibility_kwargs)
        return

    @staticmethod
    def default_eligibility(self, sim, **kwargs):
        # By default only test acute and people can be tested more than once
        acute_uids = (sim.people.typhoid.acute).uids
        return acute_uids

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_positive', dtype=int, label='New Positive'),
            ss.Result('new_screened', dtype=int, label='New Screeened'),
            ss.Result('positivity', dtype=float, scale=False, label='Positivity')
        )
        return

    def deliver(self):
        """
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        """
        sim = self.sim
        ti = sc.findinds(self.timepoints, sim.ti)[0]
        prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep

        # SELECT ELIGIBLE PEOPLE
        eligible_uids = self.check_eligibility()  # Check eligibility
        self.coverage_dist.set(p=prob)
        tested_uids = self.coverage_dist.filter(eligible_uids)

        # ADMINISTER PRODUCT
        if len(tested_uids):
            # Outcomes has the uids of eligible people, who were selected for a test, and tested positive
            self.outcomes = self.product.administer(sim.people, tested_uids)  # Actually administer the diagnostic, and get the uids of those who tested positive
        return tested_uids

    def step(self):
        """ Where everything happens at each time step"""
        sim = self.sim
        accept_uids = ss.uids()
        self.outcomes = ss.uids()

        if sim.ti in self.timepoints:
            accept_uids = self.deliver()
            self.screened[accept_uids] = True
            self.screens[accept_uids] += 1
            self.ti_screened[accept_uids] = sim.ti
            self.ti_positive[self.outcomes] = sim.ti
            self.results['new_screened'][sim.ti] = len(accept_uids)
            self.results['new_positive'][sim.ti] = len(self.outcomes)
            self.results['positivity'][sim.ti] = sc.safedivide(self.results['new_positive'][sim.ti], self.results['new_screened'][sim.ti])
        return accept_uids

    def check_eligibility(self):
        if callable(self.eligibility):
            return self.eligibility(self.sim)
        else:  # Assume self.eligibility is an array of uids
            return self.eligibility


class routine_acute_screening(base_screening, ss.RoutineDelivery):
    """
    Routine screening - an instance of base screening combined with routine delivery.
    See base classes for a description of input arguments.

    **Examples**::
        screen1 = ty.routine_screening(product=my_prod, prob=0.02) # Screen 2% of the eligible (acute) population every year
        screen2 = ty.routine_screening(product=my_prod, prob=0.02, start_year=2020) # Screen 2% every year starting in 2020
        screen3 = ty.routine_screening(product=my_prod, prob=np.linspace(0.005,0.025,5), years=np.arange(2020,2025)) # Scale up screening over 5 years starting in 2020
    """
    pass


# -- Environmental interventions
class WASH(ss.Intervention):
    """
    An environmental intervention that targets one of three different
    factors in transmission. Interventions can impact:
     - shedding into the contagion population (affects transmission parameter people -> environment)
     - CFU dose (applied to dose received by each person, or the current CFU level of the environment)
     - and frequency of exposures (lambda in the poisson distribution that produces number of exposures).

    Assumes the intervention is applied over an interval of (continuous) time.
    """

    def __init__(self, start_year=None, dur=None, efficacy=None, efficacy_kwargs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start_year
        self.dur   = dur
        self.efficacy_pattern = efficacy  # (temporal) pattern of efficacy of this intervention
        self.efficacy = None,             # current value of efficacy
        self.end = None
        self.time = None
        self.tidx = 0  # time index relative to the start of the simulation
        self.target_baseline = None  # baseline value of the factor this intervention targets at the start of the simulation
        self.target_attr_path = None
        self.target_attr = None
        return

    def init_pre(self, sim):
        # starsim base time units are in years, but the base unit of typhpoid is days # CK: TODO: fix
        if self.start is None:
            self.start = sim.pars['start']
        if self.dur is None:
            self.dur = sim.pars['stop'] - sim.pars['start']
        super().init_pre(sim)
        return

    def init_results(self):
        super().init_results()
        # This is the "time" vector or variable that will be evaluated.
        # time is the compact support to evaluate the pattern over.
        # time = 0, represents time relative to the start of the temporal pattern.
        # so a sin() pattern would always return a value of 0.0 on its start
        self.time = sc.inclusiverange(0, self.dur, self.t.dt)  # CK: TODO: replace with self.t
        self.define_results(
            ss.Result('efficacy', dtype=float, scale=False),
            ss.Result('effective_value', dtype=float, scale=False, label='Effective Value')
        )

        if self.efficacy_pattern is None:
            raise ValueError('No efficacy value or pattern specified')
        if sc.isnumber(self.efficacy_pattern):
            self.efficacy_pattern = Pattern("efficacy", pars={'efficacy': self.efficacy_pattern})
        return

    def _get_target_baseline(self):
        attr = self.sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        val = getattr(attr, target_attr)
        return val

    def _get_target_val_arr(self, idx):
        """Get target value of an attribute that is an interable"""
        attr = self.sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        val = getattr(attr, target_attr[idx])
        return val

    def _set_target_val_par(self, val):
        attr = self.sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        setattr(attr, target_attr, val)
        return

    def _set_target_val_arr(self, idx, val):
        attr = self.sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        target_attr[idx] = val
        return

    def step(self):
        sim = self.sim
        self.results['effective_value'][self.sim.ti] = self.target_baseline
        if sim.t.now('year')  >= self.start and len(self.time):
            self.efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            self._set_target_val_par((1.0 - self.efficacy) * self.target_baseline)
            self.results['efficacy'][self.tidx] = self.efficacy
            self.results['effective_value'][self.tidx] = (1.0 - self.efficacy) * self.target_baseline
            self.tidx += 1
        return

    def update_results(self):
        super().update_results()
        return


class shedding_reduction(WASH):
    """
    Simulates sanitation interventions such as latrines and sewage disposal.
    Efficacy for this intervention is a multiplier on the daily shedding amounts.
    """
    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['demographics', 'environmentalpool', 'pars', 'transmission', 'shedding_rate']
        self.target_baseline = self._get_target_baseline()
        return


class environmental_exposure_reduction(WASH):
    """
    Results in a reduction in frequency of exposure due to interventions such
    as dietary changes, crop irrigation, health inspections of food vendors.

    Efficacy for this intervention is a multiplier on # exposures. This means
    that if we want to reduce the n_exposure rate by half throughout the
    simulation, we have to apply the reduction at a single point in time, because
    this intervention modifies the model/module parameteter.
    """
    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['diseases', 'typhoid', 'pars', 'transmission', 'exposure2env_rate', 'lam']
        self.target_baseline = super()._get_target_baseline(sim)
        return


class environmental_trapezoidal_modulation(WASH):
    """
    Results in a reduction of the relative exposure to the environment
    due crop irrigation, health inspections of food vendors.
    """
    def __init__(self, efficacy_kwargs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.efficacy_pars = efficacy_kwargs
        return

    def validate_efficacy_pars(self):
        total_duration = (self.efficacy_pars['ramp_up_dur'] +
                          self.efficacy_pars['ramp_dw_dur'] +
                          self.efficacy_pars['cutoff_dur'])

        if total_duration> self.efficacy_pars['period']:
            raise ValueError(f"the duration of the pattern is longer than the period")
        if total_duration == self.efficacy_pars['period']:
            raise ValueError(f"the duration of the pattern is exactly the period, will get a triangular waveform")
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.validate_efficacy_pars()
        self.efficacy_pattern = functools.partial(self.efficacy_pattern, **self.efficacy_pars)
        self.target_attr_path = ['demographics', 'environmentalpool', 'pars', 'transmission', 'rel_trans']
        self.target_baseline = self._get_target_baseline()
        return

    def step(self):
        self.efficacy = 0.0
        self.results['effective_value'][self.sim.ti] = self.target_baseline
        sim_year = self.sim.t.now('year')
        if sim_year >= self.start and len(self.time):
            time_days = sim_year * days_per_year # CK: TODO: rewrite
            self.efficacy = self.efficacy_pattern(time_days)
        self._set_target_val_par(self.efficacy * self.target_baseline)
        return

    def update_results(self):
        super().update_results()
        self.results['efficacy'][self.ti] = self.efficacy
        self.results['effective_value'][self.ti] = self.efficacy * self.target_baseline
        return


class environmental_cleanup(WASH):
    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['demographics', 'environmentalpool', 'sv', 'cfu_conc']
        return

    def step(self):
        if self.t.now('year') >= self.start and len(self.time):
            efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            val = self._get_target_val_arr(self.ti-1)
            r_val = (1.0 - efficacy) * val
            self._set_target_val_arr(self.ti-1, r_val)
            self.results['efficacy'][self.ti] = efficacy
        return


class behavioral_change(WASH):
    """
    Simulates reduction in exposure amount that may be due to behavioral
    changes (washing vegetables, handwashing). Efficacy for this intervention
    is a multiplier on the dose.

    In starsim behavioural changes will be represented by a reduction in each
    agent's relative susceptibility and transmissibility.

    """
    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['diseases', 'typhoid', 'rel_sus']
        return

    def step(self):
        if  (self.sim.t.now('year') >= self.start) and (self.sim.t.now('year') < (self.start + self.sim.t.dt/2)):
            efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            val = self._get_target_baseline()
            self._set_target_val_par((1.0 - efficacy)*val)
            self.results['efficacy'][self.ti] = efficacy
        return


# Environmental modulation
class environmental_seasonality(ss.Intervention):
    """
    Use the mechanism of interventions to increase the number of CFUs in the environment.
    """
    def __init__(self, start_year=None, dur=None, seasonal_pattern=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start_year
        self.dur = dur
        self.pattern = seasonal_pattern  # pattern of seasonal cfu
        self.end_day = None
        self.t_relative = None # CK: TODO: refactor
        return

    def init_pre(self, sim):  # CK: TODO: refactor
        if self.start is None:
            self.start = sim.pars['start']
        if self.dur is None:
            self.dur = sim.pars['stop'] - sim.pars['start']

        # This is the "time" variable relative to the specified start year
        self.t_relative = sc.inclusiverange(0, self.dur, sim.t.dt)
        super().init_pre(sim)  # PSL: TODO: refactor? starsim interventions init_pre() call self.init_results() (?), so some attributes used by that method are not defined
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('seasonal_cfu', shape=len(self.t_relative), dtype=float, label='Seasonal CFUs') # additional cfu in the environment due to seasonality
        )
        return

    def step(self):
        sim = self.sim
        if sim.t.now('year') >= self.start and len(self.t_relative):   # CK: TODO: refactor
            seasonal_cfu = self.pattern(self.t_relative[0])
            self.t_relative = self.t_relative[1:]
            val = (sim.demographics['environmentalpool'].sv.cfu_conc[sim.ti-1] *
                   sim.demographics['environmentalpool'].pars.volume)
            sim.demographics['environmentalpool'].sv.cfu_conc[sim.ti-1] = (val + seasonal_cfu) / sim.demographics['environmentalpool'].pars.volume
            self.results['seasonal_cfu'][self.ti] = seasonal_cfu
        return

## - Vaccination interventions

class RoutineDelivery(ss.Intervention):
    """
    Base class for any intervention that uses routine delivery

    Args:
        args: Variable length argument list.
        kwargs (dict): Arbitrary keyword arguments that are passed to ss.Intervention

    Keyword Args:
        start_year (float, optional): The start year of intervention.
        end_year (float, optional): The end year of intervention, non-inclusive.
           This means that if start_year=yyyy, and end_year=start_year+dt, the intervention is applied only on 1 timestep
        age_pars (dict): a dictionary with keys 'min_age' and 'max_age' to determine the age group who is eligible.
        prob (float, array-like, optional): The coverage probability. Promoted to be an array if not already.
        prob_type (str, optional): whether the probability/coverage represents an annual, per time step
         or per intervention duration probability. Default is per "timestep" (if prob_type=None, or prob_type="timestep").

         Available prob_types: ["annual", "timestep", "interval", "age_based"]

         prob_type age_based:
             To achieve X% coverage of the entire eligible population over Y years,
             we need to set each agent to have X% probability of being vaccinated
             over the "period" that they are eligible based on their age.

             X% may not be exactlty achieved depeding on other feedback mechanisms,
             but for the purposes of this intervention we ignore vital dyanmics.

             This prob_type works only in the case that the duration of the
             intervention is longer than the 'duration' or age range of eligibility.

             This is typically the case when we want to create regular, routine
             vaccination (like vaccinating every 'Z' year old child in perpetuity).
             the age eligibility could be
                 min_age = Z-dt
                 max_age = Z+dt

    Returns:
        None
    """

    def __init__(self, *args, start_year=None, end_year=None, age_pars=None, prob=None, prob_type=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.start_year = start_year
        self.end_year = end_year
        self.age_pars = ss.Pars(age_pars)
        self.prob = sc.promotetoarray(prob)
        self.prob_type = prob_type if prob_type is not None else "timestep"  # Determines the period over which the probability/coverage has been defined
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder - initialize delivery
        self._dt = None
        self._timevec = None

        # Validate inputs
        avail_prob_types = ["annual", "timestep", "interval", "age_based"]
        if self.prob_type not in avail_prob_types:
            raise ValueError(f"Invalid prob_type: {prob_type}. Must be one of {avail_prob_types}.")

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        # If start_year and end_year are not provided, figure them out from the provided years or the sim
        if self.start_year is None: self.start_year = sim.pars.start
        if self.end_year is None:   self.end_year = sim.pars.stop
        self._dt = sim.pars.dt  # TODO: need to eventually replace with own timestep, but not initialized yet since super().init_pre() hasn't been called
        self._timevec = sim.t.timevec

        self._validate_time_parameters()
        self._configure_time_attributes()
        self._conform_prob()
        self._calculate_dt_probability()
        return

    def _validate_time_parameters(self):
        self.dur_years = self.end_year - self.start_year
        self.dur_timepoints = np.round((self.end_year - self.start_year)/self._dt).astype(int)
        if self.dur_timepoints < 1:
            errormsg = 'Start and end years must be at least one timestep (dt) apart.'
            raise ValueError(errormsg)

        if not (any(np.isclose(self.start_year, self._timevec)) and any(np.isclose(self.end_year, self._timevec))):
            errormsg = 'Years must be within simulation start and end dates.'
            raise ValueError(errormsg)

        # Validate age parameters
        age_timepoints = np.round((self.age_pars.max_age - self.age_pars.min_age)/self._dt).astype(int)
        if age_timepoints < 1:
            errormsg = f'Min and max age must must be at least one timestep (dt: {self._dt}) apart.'
            raise ValueError(errormsg)
        self.eligibility_interval = self.age_pars.max_age - self.age_pars.min_age
        return

    def _configure_time_attributes(self):
        # Determine the timepoints at which the intervention will be applied
        self.start_point = sc.findnearest(self._timevec-self.start_year, 0.0)
        self.end_point   = sc.findnearest(self._timevec-self.end_year, 0.0)
        self.timepoints  = sc.inclusiverange(self.start_point, self.end_point-1).astype(int) # TODO: when integrating with self.t, check if -1 still needed.
        self._timevec    = sc.inclusiverange(self.start_year, self.end_year, self._dt) # TODO: integrate with self.t
        return

    def _conform_prob(self):
        """" Make an array of probabilities to match the period of time the intervention is defined over"""
        # Get the probability input into a format compatible with timepoints
        if len(self.prob) == 1:
            self.prob = np.array([self.prob[0]] * len(self.timepoints))
        return

    def _calculate_dt_probability(self):
        # Adjust the probability by the sim's timestep, if it's an annual probability
        match self.prob_type:
            case "annual":
                # Assumes units of time in sim.timevec are in years, and so is dt, or
                # assumes that sim.timevec and dt are in the same units.
                # The probability an agent will receive a vaccine over the period of 1 year
                self.n_timesteps_per_prob_interval = round(1.0/self._dt) # TODO: integrate with time parameters,
            case "interval":
                # The probability an agent will receive a vaccine over the intervention interval defined
                # by start_year and stop_year
                self.n_timesteps_per_prob_interval = len(self.timepoints)
            case "timestep":
                # The probability an agent will receive a vaccine at each time step
                self.n_timesteps_per_prob_interval = 1
            case "eligibility_interval":
                # The probability an agent will receive a vaccine over the period where they are eligible
                # The eligibility_interval is defined by [min_age, max_age]
                # This prob_type will produce the wrong results if ageing is disabled.
                # Coverage will likely be lower than the value specified if vital dynamics are disabled,
                # depending on when in the simulation the intervention is applied.
                # This approach gets, on average, to x% coverage over the intervention interval
                # if birth rates are similar to deaths rates, ie population size is in a 'steady' state

                # If the intervention interval is < than eligibility interval, then the
                # total effective coverage can be expected to be approximtely
                # prob/(eligibility_interval/intevention_interval). This is equivalent to
                # having a sampling period lower than the nyquist frequency in signal processing
                # We could even warn the user, and suggest that intervention_period
                # should be at least == eligibility_interval
                self.n_timesteps_per_prob_interval = round(self.eligibility_interval / self._dt)
            case "age_based":
                # A 'hybrid', or defined-by parts, approach. The intervention qualitatively changes
                # behaviour depending on the relative 'sizes' of eligibility_interval and
                # intervention interval, and forces the user specified 'coverage' over the intervention period.
                # If eligibility_interval <= intervention_interval, then this should effectively
                # behave like "eligibility_interval", otherwise like "interval"

                if self.eligibility_interval <= self.dur_years:
                    self.n_timesteps_per_prob_interval = self.eligibility_interval / self._dt
                else:
                    # For the same input parameters start/stop years and age pars,
                    # this prob_type will provide different results to those generated by
                    # "prob_type=eligibility_interval".
                    # eligibility_interval > intervention_interval
                    self.n_timesteps_per_prob_interval = len(self.timepoints)

        self.prob = 1 - (1 - self.prob) ** (1.0 / self.n_timesteps_per_prob_interval)
        return


class vaccination_with_waning(RoutineDelivery):
    """
    An intervention that handles a vaccination with waning.

    NOTE: this case is a bit special because it agreggates vaccine protection
    and the immune system. Immunity waning could be handled in the disease, or
    there could be a module that is called immune_system and it could have
    different responses depending on a product.

    This intervention does not use a product (ss.Product)

    The "vaccine" administered here induces immune-evoked impulse response,
    that is modelled with a box-exponential model (ie, immunity can be constant
    at ve0 for a duration and then it wanes exponentially).

    Args:
         prob              (float/arr) : probability of eligible population getting vaccinated, by default it is interepreted as an annual probability
         booster1_prob     (float)     : conditional probability of receiving first booster dose given that an individual has received their first routine dose
         booster2_prob     (float)     : conditional probability of receiving second booster dose given that an individual has received their first routine dose
         booster1_interval (float)     : the interval of time in years between an individual receiving their routine dose and their first booster
         booster2_interval (float)     : the interval of time in years between an individual receiving their routine dose and their second booster
         imm_decay         (ss.Dist)   : a starsim Distribution that will return the approrpriate value of immunity decay (time constant)
         imm_ve0           (ss.Dist)   : a starsim Distribution that will return the appropriate value of immunity ve0 (maximum acquired immune response at time of vaccination)
         imm_constant_dur  (ss.Dist)   : a starsim Distribution that will return the appropriate value of duration at constant ve0, after which acquired immunity starts waning
         imm_draw_fn       (callable)  : a function that tells the intervention how to get the right parameters from from the ss.Dist parameters imm_decay, imm_ve0, imm_constant_dur
         label             (str)       : the name of vaccination strategy
         kwargs            (dict)      : passed to Intervention()
    """
    def __init__(self, *args, booster1_prob=0.0, booster2_prob=0.0, booster1_interval=None, booster2_interval=None,
                 imm_decay=ss.constant(v=tyi.imm_decay_by_age()),
                 imm_ve0=ss.constant(v=tyi.imm_ve0_by_age()),
                 imm_constant_dur=ss.constant(v=tyi.imm_constant_dur_by_age()),
                 imm_draw_fn=None,
                 imm_draw_fn_kwargs=None,
                 label=None, debug=False, **kwargs):
        # **kwargs: years=None, start_year=None, end_year=None, prob=None, prob_type=None,
        super().__init__(*args, **kwargs) # CK: TODO: refactor with define_pars
        self.label = label
        self.booster1_prob = sc.toarray(booster1_prob)
        self.booster2_prob = sc.toarray(booster2_prob)
        self.booster1_interval = booster1_interval  # TODO SOON: ss.years(booster1_interval) # number of years betweem 1st dose and 1st booster dose
        self.booster2_interval = booster2_interval  # TODO SOON: ss.years(booster1_interval) # number of years betweem 1st dose and 1st booster dose
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        self.eligibility = self.age_eligibility
        self.vaccinated = ss.BoolArr('vaccinated')                             # keep track of who has been vaccinated
        self.tested = ss.BoolArr('tested')
        self.ti_vaccinated = ss.FloatArr('ti_vaccinated')                      # Keep track of when the agent received their last vaccine in timesteps
        self.t_vaccinated = ss.FloatArr('t_vaccinated', default=np.nan)        # time (year) of most recent vaccination
        self.a_vaccinated = ss.FloatArr('a_vaccinated', default=np.nan)        # age at vaccination
        self.t_to_booster1 = ss.FloatArr('t_to_booster1', default=np.nan)      # time until needing the booster
        self.t_to_booster2 = ss.FloatArr('t_to_booster2', default=np.nan)      # time until needing the booster
        self.n_doses = ss.FloatArr('n_doses')                                  # number of doses received by each agent
        self.imm_ve0 = ss.FloatArr('imm_ve0', default=0.0)                     # Maximum protection at t=0 of receiving a vaccine
        self.imm_constant_dur = ss.FloatArr('imm_constant_dur', default=0.0)   # Duration of constant immunity in days, assuming the model is a box-exponential model
        self.imm_decay = ss.FloatArr('imm_decay', default=np.inf)              # Decay time constant, in days, one value per age bin of interest

        self.imm_decay_dist = imm_decay  # Decay time constant, in days, one value per age bin of interest
        self.imm_ve0_dist = imm_ve0      # Maximum protection at t=0 of receiving a vaccine
        self.imm_constant_dur_dist = imm_constant_dur  # Duration at constant level of immunity ve0 before waning starts
        self.imm_draw_fn = self.imm_draw_from_constant if imm_draw_fn is None else imm_draw_fn
        self.imm_draw_fn_kwargs = {} if imm_draw_fn_kwargs is None else imm_draw_fn_kwargs

        # Debug - track more things
        self.debug = debug

        # Validate inputs
        # error if only booster2 info given and not booster1
        if (self.booster1_interval == None) and (
                self.booster2_interval != None):
            raise ValueError(
                "booster2 should only be implemented if booster1 is also implemented. Please provide value for booster1_interval")

        # error if booster1/booster2 prob>0 but interval is None
        if (self.booster1_interval == None) and (self.booster1_prob > 0):
            raise ValueError(
                f"Booster 1 coverage {booster1_prob} is non-zero, but no booster interval `booster1_interval` was provided.")

        if (self.booster2_interval == None) and (self.booster2_prob > 0):
            raise ValueError(
                f"Booster 2 coverage {booster2_prob} is non-zero, but no booster interval `booster2_interval` was provided.")

        # booster1_interval must be shorter than booster2_interval if both exist
        if (self.booster1_interval != None) and (
                self.booster2_interval != None) and (
                self.booster1_interval >= self.booster2_interval):
            raise ValueError(
                f"Time to first booster {booster1_interval} should be less than time to second booster {booster2_interval}")

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.booster1_prob = self.booster1_prob * np.ones(shape=len(self.prob))
        self.booster2_prob = self.booster2_prob * np.ones(shape=len(self.prob))
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('cum_doses', shape=(self.sim.t.npts,), dtype=float,
                      label="Cumulative Number of Doses"),
            ss.Result('new_doses', shape=(self.sim.t.npts,), dtype=float,
                      label="New Doses Delivered"))

        # Mostly to test that we're counting things correctly. Meant to be used with vital dynamics disabled,
        # especially births, as we're setting the size to be that of the population at t=start of sim
        if self.debug:
            self.results += ss.Result('immunity', shape=(
            self.sim.t.npts, self.sim.pars["n_agents"]), dtype=float,
                                      label="Acquired Immunity")
        return

    def age_eligibility(self, sim):
        is_eligible = (sim.people.age >= self.age_pars.min_age) & (sim.people.age < self.age_pars.max_age)
        return is_eligible

    def step_acquired_immunity(self, sim, uids):
        """
        This is the model of the the dynamics of an individual's immunity
        response to receiving a vaccination. The acquired immunity wanes over
        time with an exponential decay.
        """
        module = sim.diseases.typhoid
        t_vaccinated = self.t_vaccinated[uids]    # Time, in calendar years, when the individual received the vaccine. t_0 in the waning equation.

        # Age-dependent immunity parameters
        ve0 = self.imm_ve0[uids]
        constant_ve0_dur = self.imm_constant_dur[uids]
        decay = self.imm_decay[uids]
        module.immunity_acquired[uids] = np.clip(ve0 * tyum.box_exponential(np.float32(sim.t.now('year')), t_vaccinated,
                                                                            constant_ve0_dur, decay),
                                                 a_min=0.0, a_max=1.0)
        return

    def imm_draw_from_constant(self, uids, **kwargs):
        """ Draw the correct values dependent on the agent's age at vaccination"""
        self.imm_ve0[uids] = self.imm_ve0_dist.pars.v(self.a_vaccinated[uids])
        self.imm_constant_dur[uids] = self.imm_constant_dur_dist.pars.v(self.a_vaccinated[uids])
        self.imm_decay[uids] = self.imm_decay_dist.pars.v(self.a_vaccinated[uids])
        return

    def step(self):
        sim = self.sim
        sim_year = sim.t.now('year')
        vaccinated_uids = self.vaccinated.uids

        if sim.ti in self.timepoints:
            self.t_to_booster1[self.t_to_booster1 > 0.0] -= self.sim.t.dt
            self.t_to_booster2[self.t_to_booster2 > 0.0] -= self.sim.t.dt
            ti_rel = sc.findinds(self.timepoints, sim.ti)[0] # ti relative to the start of the intervention
            prob = self.prob[ti_rel]  # Get the proportion of people who will be tested this timestep
            is_eligible = self.check_eligibility()  # Check eligibility by age for first dose
            # Select never vaccinated
            is_eligible_not_vax = (is_eligible) & ~(self.vaccinated)
            self.coverage_dist.set(p=prob)
            new_accept_uids = self.coverage_dist.filter(is_eligible_not_vax)
            if len(new_accept_uids):
                # Update people's state and dates
                self.ti_vaccinated[new_accept_uids] = sim.ti
                self.vaccinated[new_accept_uids] = True
                self.t_vaccinated[new_accept_uids] = sim_year
                self.a_vaccinated[new_accept_uids] = sim.people.age[new_accept_uids]
                self.n_doses[new_accept_uids] = 1
                self.t_to_booster1[new_accept_uids] = self.booster1_interval   # set the timer to get the booster
                self.t_to_booster2[new_accept_uids] = self.booster2_interval   # set the timer to get the booster
                self.imm_draw_fn(new_accept_uids)

            # Select eligible for a booster
            booster1_prob = self.booster1_prob[ti_rel]
            if booster1_prob > 0.0:
                is_eligible_booster = (self.vaccinated) & (self.n_doses == 1) & (self.t_to_booster1 <= 0.0) # For boosters we do not filter by age
                self.coverage_dist.set(p=booster1_prob)
                new_booster_uids = self.coverage_dist.filter(is_eligible_booster)
                self.ti_vaccinated[new_booster_uids] = sim.ti
                self.t_vaccinated[new_booster_uids] = sim_year
                self.a_vaccinated[new_booster_uids] = sim.people.age[new_booster_uids]
                self.t_to_booster1[is_eligible_booster] = np.inf # reset time for those eligible for booster, regardless of receipt
                self.n_doses[new_booster_uids] += 1
                self.imm_draw_fn(new_booster_uids)

            # Select eligible for second booster
            booster2_prob = self.booster2_prob[ti_rel]
            if booster2_prob > 0.0:
                is_eligible_booster = (self.vaccinated) & (self.n_doses >= 1) & (self.t_to_booster2 <= 0.0) # For boosters we do not filter by age - and second booster only conditional on receipt of first routine dose, not first booster
                self.coverage_dist.set(p=booster2_prob)
                new_booster_uids = self.coverage_dist.filter(is_eligible_booster)
                self.ti_vaccinated[new_booster_uids] = sim.ti
                self.t_vaccinated[new_booster_uids] = sim_year
                self.a_vaccinated[new_booster_uids] = sim.people.age[new_booster_uids]
                self.t_to_booster2[is_eligible_booster] = np.inf # reset time for those eligible for booster, regardless of receipt
                self.n_doses[new_booster_uids] += 1
                self.imm_draw_fn(new_booster_uids)
                
            vaccinated_uids = self.vaccinated.uids

        # Update immunity_acquired
        self.step_acquired_immunity(sim, vaccinated_uids)

        # Relative susceptibility is between 0-1, 1: susceptible; 0: invulnerable. This
        # variable is used in transmission dynamics
        # immunity_acquired is defined to provide a mechanism for immunity waning dynamics to exist.
        # This is also a value between 0-1. 1: perfectly immune/invulnerable; 0: no immunity, and it
        # is often referred to as, vaccine efficacy and vaccine efficacy waning
        # sim.diseases.typhoid.susceptibility[vaccinated_uids] = (1.0 - sim.diseases.typhoid.immunity_acquired[vaccinated_uids])

        # Update results
        ti_sim = sim.ti
        self.results["new_doses"][ti_sim] = np.count_nonzero(self.ti_vaccinated == ti_sim)
        self.results["cum_doses"][ti_sim] = np.sum(self.results["new_doses"][:ti_sim])

        if self.debug:
            ti_sim = sim.ti
            self.results["immunity"][ti_sim, :] = sim.diseases.typhoid.immunity_acquired[:]
        return self.vaccinated.uids
