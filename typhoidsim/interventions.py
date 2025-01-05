"""
Define Typhoid-specific treatments and diagnostics (interventions). It includes
the intervention (eg, campaign that finds eligible people) and also products.
"""
import numpy as np
import sciris as sc
import starsim as ss

from .patterns import Pattern
from .defaults import days_per_year
from . import utils_math as tyum
from .products import typhoid_test


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
        self.ti_positive = ss.FloatArr('ti_positive')
        self.validate_eligibility()
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
            self.outcomes = self.product.administer(sim.people, tested_uids)  # Actually administer the diagnostic, and get the uids of those who tested positive
        return tested_uids

    def step(self):
        """ Where everything happens at each time step"""
        sim = self.sim
        accept_uids = ss.uids()

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

    def __init__(self, start_year=None, dur=None, efficacy=None, *args, **kwargs):
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
    def init_pre(self, sim):
        super().init_pre(sim)
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
        if self.sim.year >= self.start and len(self.time):
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


class vaccination_with_waning(ss.RoutineDelivery):
    """
    An intervention that handles a vaccination with waning.
    NOTE: this case is a bit special because it agreggates vaccine protection and the immune
    system. Immunity waning could be handled in the disease, or there could be a module that is
    called immune_system and it could have different responses depending on the product.
    Maybe it's the job for a connector?

    Args:
         prob           (float/arr) : probability of eligible population getting vaccinated, by default it is interepreted as an annual probability
         annual_prob    (bool)      : whether prob represents an annual probbability or a per-time-step proability
         booster_prob   (float)     : conditional probability of receiving a boster dose given that an individual has received their first dose
         dose_interval  (float)     : the interval of time in years between an individual receiving their first dose and their booster
         age_pars       (dict)      : a dictionary with min_age and max_age to determine the age group who is eligible
         label          (str)       : the name of vaccination strategy
         kwargs         (dict)      : passed to Intervention()
    """
    def __init__(self, *args, booster_prob=0.0, dose_interval=None, label=None, age_pars=None, debug=False, **kwargs):
        # **kwargs: years=None, start_year=None, end_year=None, prob=None, annual_prob=True,
        super().__init__(*args, **kwargs) # CK: TODO: refactor with define_pars
        self.label = label
        self.booster_prob = sc.toarray(booster_prob)
        self.dose_interval = dose_interval  # TODO SOON: ss.years(dose_interval) # number of years betweem 1st dose and booster dose
        self.age_pars = ss.Pars(age_pars)
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        self.eligibility = self.age_eligibility
        self.vaccinated = ss.State('vaccinated')
        self.t_vaccinated = ss.FloatArr('t_vaccinated', default=np.nan)  # time (year) of vaccination
        self.a_vaccinated = ss.FloatArr('a_vaccinated', default=np.nan)  # aged at vaccination
        self.t_to_booster = ss.FloatArr('t_to_booster', default=np.nan)  # time until needing the booster
        self.n_doses = ss.FloatArr('n_doses')
        self.debug = debug
        return

    def init_results(self):
        super().init_results()
        self.booster_prob = self.booster_prob * np.ones(shape=len(self.prob)) # CK: TODO: move this?
        # Test without new people being born
        if self.debug:
            self.define_results(
                ss.Result('immunity', shape=(self.sim.npts, self.sim.pars["n_agents"]), dtype=float, label="Acquired Immunity")
            )
        return

    def age_eligibility(self, sim):
        is_eligible = (sim.people.age >= self.age_pars.min_age) & (sim.people.age < self.age_pars.max_age)
        return is_eligible

    def update_acquired_immunity(self, sim, uids):
        """
        This is the immunity response to a vaccine. The acquired immunity wanes
        over time.
        """
        module = sim.diseases.typhoid
        t_vaccinated = self.t_vaccinated[uids] # Time, in calendar years, when the individual received the vaccine. t_0 in the waning equation.
        a_vaccinated = self.a_vaccinated[uids]
        max_immunity = module.imm_peak(a_vaccinated)
        fixed_immunity = module.imm_fixed_dur(a_vaccinated)
        decay = module.imm_waning_time(a_vaccinated)
        module.immunity_acquired[uids] = np.clip(max_immunity * tyum.box_exponential(sim.year, t_vaccinated, fixed_immunity, decay), a_min=0.0, a_max=1.0)
        return

    def step(self):
        """
        Deliver the diagnostics by finding who's eligible, and apply the product, only once.
        """
        sim = self.sim
        sim_year = sim.t.now('year')
        vaccinated_uids = self.vaccinated.uids
        if sim_year >= self.start_year and sim.year <= self.end_year:
            self.t_to_booster[self.t_to_booster > 0.0] -= self.dt
            ti = sc.findinds(self.timepoints, sim.ti)[0]
            prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
            is_eligible = self.check_eligibility()  # Check eligibility by age for first dose
            # Select never vaccinated
            is_eligible_not_vax = (is_eligible) & ~(self.vaccinated)
            self.coverage_dist.set(p=prob)
            new_accept_uids = self.coverage_dist.filter(is_eligible_not_vax)
            if len(new_accept_uids):
                # Update people's state and dates
                self.vaccinated[new_accept_uids] = True
                self.t_vaccinated[new_accept_uids] = sim_year
                self.a_vaccinated[new_accept_uids] = sim.people.age[new_accept_uids]
                self.n_doses[new_accept_uids] = 1
                self.t_to_booster[new_accept_uids] = self.dose_interval   # set the timer to get the booster

            # Select eligible for a booster
            booster_prob = self.booster_prob[ti]
            if booster_prob > 0.0:
                is_eligible_booster = (self.vaccinated) & (self.n_doses == 1) & (self.t_to_booster <= 0.0) # For boosters we do not filter by age
                self.coverage_dist.set(p=booster_prob)
                new_booster_uids = self.coverage_dist.filter(is_eligible_booster)
                self.t_vaccinated[new_booster_uids] = sim_year
                self.a_vaccinated[new_booster_uids] = sim.people.age[new_booster_uids]
                self.t_to_booster[new_booster_uids] = np.inf # reset time for those who received the booster
                self.n_doses[new_booster_uids] += 1

            vaccinated_uids =  self.vaccinated.uids
        self.update_acquired_immunity(sim, vaccinated_uids)
        # TODO: confirm with EES team how acquired immunity enters the expression for p_infc
        sim.diseases.typhoid.rel_sus[vaccinated_uids] = 1.0 - sim.diseases.typhoid.immunity_acquired[vaccinated_uids]

        if self.debug:
            self.results["immunity"][ti, :] = sim.diseases.typhoid.immunity_acquired[:]
        return self.vaccinated.uids
