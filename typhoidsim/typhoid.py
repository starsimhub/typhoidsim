"""
Typhoid model.
"""

import numpy as np

import sciris as sc
import starsim as ss

import typhoidsim.utils as tyu
import typhoidsim.patterns as typ
import typhoidsim.defaults as tyd
import typhoidsim.utils_math as tyum

__all__ = ["Typhoid"]


class Typhoid(ss.Infection):
    """
    Typhoid module that includes the natural history of the disease in a human
    agent and also environmental 'state' variables and parameters that
    capture the growth and decay of S. typhii bacteria in contaminated resources.

    This is an early-stage 'monolithic' implementation.

    The parent class ss.Infection has the following states

    ss.BoolArr('susceptible', default=True) -- for typhoid we may need to reset
    this to False
    ss.BoolArr('infected'), -- default False
    ss.FloatArr('rel_sus', default=1.0),
    ss.FloatArr('rel_trans', default=1.0),
    ss.FloatArr('ti_infected'),)

    # TODO: add link to specification document
    Unless otherwise specified, parameters come from: XXX

    """

    def __init__(self, pars=None, *args, **kwargs):
        """Initialize with parameters"""
        super().__init__()
        self.default_pars(
            # Initial conditions and transmissibility beta
            beta=1.0,
            init_prev=ss.bernoulli(0.01),

            # NATURAL HISTORY PARAMETERS
            # From immune (never exposed) to susceptible
            p_imm2sus_6m=ss.bernoulli(p=0.14),  # Proportion of immune population at 6 months that moves to susceptible state
            p_imm2sus_3y=ss.bernoulli(p=0.29),  # Proportion of immune population at 3 years that moves to susceptible state
            p_imm2sus_6y=ss.bernoulli(p=0.61),  # Proportion of immune population at 6 years that moves to susceptible state
            p_imm2sus=ss.bernoulli(p=self.susceptibility_prob_function),
            sus_saturation_age=20.0,  # Age (years) after which agents are 100% susceptible
            sus_age_exposure_slope=1.0,

            # Prepatent stage, the parameters of the distribution of durations is CFU-dose dependent
            prep_dur_dpars=tyu.load_dataset("prepatent_dur_dist_pars"),   # CFU dose-dependent duration distribution parameters, in days. Stratified in 3 levels (low, medium and high)
            prep_dur_fun=tyum.double_sigmoid_tanh,                        # Function to represent the 3 levels of each prepatent duration distribution parameter as s continous function
            dur_prep_dist=ss.lognorm_ex(mean=self.prepatent_mean_dur_function,
                                        stdev=self.prepatent_std_dur_function),

            cfu_lo_me=5_050_000,   # Threshold cfu value that distinguishes whether to use the 'low dose' (for cfu_dose <= cfu_lo_me) or 'medium dose' mean/std duration (cfu_dose > cfu_lo_me).
            cfu_me_hi=55_000_000,  # Threshold cfu value that distinguishes whether to use the 'medium dose' (for cfu_dose <= cfu_me_hi) or 'high dose' mean/std duration (cfu_dose > cfu_lo_me).

            # Symptomatic stage, (acute and sublinical)
            p_acute=ss.bernoulli(p=0.234),  # Prob of becoming acute ()
            # Age-dependent duration distribution parameters
            symp_dur_th_age=30.0,        # Symptomatic duration age threshold.
            # For people aged less than threshold
            symp_dur_mean_le=1.172,   # Symptomatic duration mean if age < age_threshold, in weeks.
            symp_dur_std_le=0.483,    # Symptomatic duration std  if age < age_threshold, in weeks.
            # For people aged threshold value and over
            symp_dur_mean_geq=1.172,  # Symptomatic duration mean if age >= age_threshold, in weeks.
            symp_dur_std_geq=0.788,   # Symptomatic duration std if age  >= age_threshold, in weeks.
            dur_symp_dist=ss.lognorm_ex(mean=self.symp_dur_mean,
                                        stdev=self.symp_dur_std),     # Symptomatic (acute or subclinical duration), depends on age, expressed in weeks.
            dur_wait2treatment=ss.lognorm_ex(mean=2.33219066, stdev=0.5430),  # (Relative to acute onset) day of treatment-seeking for acute cases, in days.

            # Long-term stages
            p_cpg=0.15,    # Base prob of becoming chronic after subclinical or acute with gallstones. Same for female and male, but does not have to be.
            d_chro=ss.bernoulli(p=self.chronic_gall_prob_function),    # Prob of becoming chronic carrier from acute or clinical infection modulated by gallstone prevalence
            p_gall=tyu.load_dataset("gallstone_probs"),    # Probability of having gallstones by age and sex
            gall_prev=tyu.load_dataset("gallstone_prev"),  # Biological sex gallstone prevalence (expressed in fraction of the population, value between 0 and 1)
            p_death=ss.bernoulli(p=0.01),   # Probability of dying from acute, context dependent, and by default set to something zero or something very small

            # IMMUNE SYSTEM-WITHIN HOST PARAMETERS
            # Infectiousness parameters
            tai=40_000,  # Typhoid acute infectiousness, represents number of colony-forming units of S. typhi, for an average human that has 3500 mL of blood, this is about 11 CFU/mL
            tpri=0.4,    # Typhoid relative (to acute) prepatent infectiousness
            tsri=0.8,    # Typhoid relative (to acute) subclinic infectiousness
            tcri=0.1,    # Typhoid relative (to acute) chronic infectiousness
            tppi=0.99,   # Decrease in susceptibility per infection (exponential decrease)
            p_resp=ss.bernoulli(p=self.response_prob_function),

            # ENVIRONMENT PARAMETERS
            environmental_transmission=True,  # Whether to allow for environmental transmission or not
            # Tranmission parameters
            transmission=ss.Pars(
                beta=1.0,  # Beta environment
                # Interaction parameters between people and environment
                env2ppl_exposure_rate=ss.poisson(lam=0.5),  # Poisson rate determining the daily number of exposures for environment route
                env2ppl_p_inf=ss.bernoulli(p=self.infection_prob_function),
                # Interaction parameters for the contact route (people 2 people)
                ppl2ppl_exposure_rate=ss.poisson(lam=0.18),
                ppl2ppl_p_inf=ss.bernoulli(p=self.infection_prob_function),
            ),
        )
        self.update_pars(pars, **kwargs)

        # Parametrisation of prepatent duration distribution parameters (ie, mean and std are functions of CFU dose)
        self.partial_prep_dur_mean,  self.partial_prep_dur_std = self.prepare_partial_prep_funs()

        # Boolean states
        self.add_states(
            # Infection life cycle states
            # Susceptible & infected are added automatically, here we add the rest
            ss.BoolArr("immune", True, label="Completely Immune"),
            ss.BoolArr("prepatent", label="Prepatent"),  # Also known as exposed state (incubation stage)
            ss.BoolArr("acute", label="Acute"),
            ss.BoolArr("subclinical", label="Subclinical"),
            ss.BoolArr("chronic", label="Chronic"),
            ss.BoolArr("recovered", label="Recovered"),

            # States that track immunity-related quantities or variables
            # and depend on infection states
            ss.FloatArr("n_exposures", 0, label="Number of Exposures"),    # average daily exposures from a given source/route
            ss.FloatArr("cfu_dose", 0, label="Exposure amount (CFUs)"),    # exposure amount (acquisition phase, "doses" of bacteria that the target host takes as input from sources of contagion)
            ss.FloatArr("infectiousness", 0, label="Infectiousness"),      # average number of cfu during different stages of the disease (infected phase, within host). Could be rel_trans?
            ss.FloatArr("n_infections", 0, label="Number of Infections"),  # number of infections over the lifespan of this agent
            ss.FloatArr("p_chronic", label="p(chronic)"),                       # probability of becoming chronic
            ss.FloatArr("immunity", 1, label="Immunity Level"),            # Blocking effect factor due to immunity to typhoid, value between 0 (blocking new infections) and 1 (completely vulnerable). Maybe we need a more descriptive name.

            # States that track timing of events
            ss.FloatArr("ti_susceptible", label="Start of susceptible state"),
            ss.FloatArr("ti_prepatent", label="Start of prepatent stage"),
            ss.FloatArr("ti_subclinical", label="Start of subclinical stage"),
            ss.FloatArr("ti_acute", label="Start of acute stage"),
            ss.FloatArr("ti_seek_trtmnt", label="Time to seek treatment"),
            ss.FloatArr("ti_chronic", label="Start of chronic stage"),
            ss.FloatArr("ti_recovered", label="Time of recovery"),
            ss.FloatArr("ti_dead", label="Time of death"),
        )

        return

    def prepare_partial_prep_funs(self):
        """
        Partially evaluate functions that return parameters (mean and std) of
        the distribution of prepatatent durations.
        """
        from functools import partial

        th1 = self.pars.cfu_lo_me
        th2 = self.pars.cfu_me_hi
        pdpars = self.pars.prep_dur_dpars

        return (partial(self.pars.prep_dur_fun,
                        l1=pdpars["mean_dur"]["lo"],
                        l2=pdpars["mean_dur"]["me"],
                        l3=pdpars["mean_dur"]["hi"],
                        x_12=th1, x_23=th2),
                partial(self.pars.prep_dur_fun,
                        l1=pdpars["std_dur"]["lo"],
                        l2=pdpars["std_dur"]["me"],
                        l3=pdpars["std_dur"]["hi"],
                        x_12=th1, x_23=th2))

    @property
    def infectious(self):
        return self.infected

    @property
    def asymptomatic(self):
        return self.prepatent | self.subclinical | self.chronic

    @property
    def symptomatic(self):
        return self.acute

    def init_results(self):
        """
        Initialise Result objects
        """
        super().init_results()
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, "new_susceptible", npts, dtype=int, label="New Susceptible"),
            ss.Result(self.name, "new_prepatent", npts, dtype=int, label="New Prepatent"),
            ss.Result(self.name, "new_acute", npts, dtype=int, label="New Acute"),
            ss.Result(self.name, "new_subclinical", npts, dtype=int, label="New Subclinical"),
            ss.Result(self.name, "new_chronic", npts, dtype=int, label="New Chronic"),
            ss.Result(self.name, "new_recovered", npts, dtype=int, label="New Recovered"),
            ss.Result(self.name, "new_deaths", npts, dtype=int, label="New Dead"),
        ]
        return

    def init_pre(self, sim):
        """ Initialise objects and valid before simulation run"""
        super().init_pre(sim)
        self.validate_environment()
        return

    def validate_environment(self):
        """
        Validate environment
        """
        demographic_modules = self.sim.demographics
        if demographic_modules is not None and len(demographic_modules) > 0:
            if self.pars.environmental_transmission:
                #TODO: this is a temporary quick way to check that we have the only available environemental module, available in the sim
                demographic_modules["environmentalpool"]
        return

    def init_post(self):
        """
        Set initial values for states and new cases. This could involve passing in a full
        set of initial conditions, or using init_prev (initial prevalence), or other.

        Note that this is different to initialization of the Arr objects i.e.,
        creating their dynamic array, linking them to a People instance.
        That should have already taken place by the time this method is called.
        """
        # NOTE: Typhoid may assume that all individuals are born into an
        # a class where they cannot get infected, and then
        # move to the susceptible class at probabilities
        # for each age. The ss.Infection class set the self.susceptible state
        # to True by default, so here reset this array to False
        self.make_impervious()

        if self.pars.init_prev is None and self.pars.environment.init_prev is None:
            return

        if self.pars.init_prev is not None:
            # Initial cases
            initial_cases = self.pars.init_prev.filter()
            self.set_prognoses(initial_cases)
        self.progress_to_prepatent(self.sim.ti)   # Set the correct level of infectiousness of initial cases
        return

    def make_impervious(self):
        """
        Individuals that are created through births in the model should start out in a
        fully immune state.
        #TODO: This may not be needed any more if Typhoid is derived directly
        from starsim.Disease rather than from Infection, which by default
        assumes every agent starts in a susceptible state.
        """
        eligible = self.sim.people.age < self.pars.sus_saturation_age
        self.immune[eligible] = True
        self.susceptible[eligible] = False

    # Methods that are specific to a single stage of infection
    def make_susceptible(self):
        """
        From Gauld et al. 2018:
        'Our model assumes all individuals are born into an unexposed/immune class
        and move to the susceptible class at probabilities for each age.
        Specifically, at each *month of age* a fitted curve determines the
        probability of an individual entering the susceptible class.

        The curve is anchored at 0% exposure at birth, and 100% exposure at age
        20 years, with a free slope parameter (S) determining the concavity/shape
        of the function (Fig 2B).'


        From:
        https://github.com/jgauld/DtkTrunk/blob/Typhoid-Ongoing/Eradication/SusceptibilityTyphoid.cpp

        NOTE:
        Fraction of children that become susceptible upon reaching a certain
        age threhsold.

        This is referred to as age-specific immunity.
        """

        never_exposed = (self.immune).uids
        self.susceptible[never_exposed] = self.pars.p_imm2sus.rvs(never_exposed)
        self.immune[never_exposed] = ~self.susceptible[never_exposed]
        return

    def increase_childhood_susceptibility(self):
        """
        Age-based susceptiblity used in the Santiago de Chile case.
        Not currently used by default
        """
        # Santiago case:
        _6m = 0.5 * tyd.days_per_year  # in days
        _3y = 3.0 * tyd.days_per_year  # in days
        _6y = 6.0 * tyd.days_per_year  # in days

        # Detect 'age' anniversaries
        uids_6m = ((self.sim.people.age >= _6m) &
                  ((self.sim.people.age - self.sim.dt) < _6m)).uids

        uids_3y = ((self.sim.people.age >= _3y) &
                   ((self.sim.people.age - self.sim.dt) < _3y)).uids

        uids_6y = ((self.sim.people.age >= _6y) &
                   ((self.sim.people.age - self.sim.dt) < _6y)).uids

        self.susceptible[uids_6m] = self.pars.p_imm2sus_6m(uids_6m)
        self.immune[uids_6m] = ~self.susceptible[uids_6m]

        self.susceptible[uids_3y] = self.pars.p_imm2sus_3y(uids_3y)
        self.immune[uids_3y] = ~self.susceptible[uids_3y]

        self.susceptible[uids_6y] = self.pars.p_imm2sus_6y(uids_6y)
        self.immune[uids_6y] = ~self.susceptible[uids_6y]
        return

    def update_death(self, uids):
        """Reset states for dead agents"""
        for state in [
            "susceptible",
            "infected",
            "prepatent",
            "acute",
            "subclinical",
            "chronic",
            "recovered",
        ]:
            self.statesdict[state][uids] = False
        self.statesdict["immune"][uids] = True
        return

    # Update progression of disease, handle transitions
    def update_pre(self):
        """
        Update the progression of the disease -- handles disease
        state transitions. In the typical simulation flow this method is called
        before propagating the infection/disease is propagated via
        make_new_cases()
        """

        ti = self.sim.ti  # current timestep
        # Check who becomes susceptible in this timestep age 0-20
        self.make_susceptible()

        # Age-based susceptibility in children <= 6 years old
        # self.increase_childhood_susceptibility()

        # The infection life cycle or natural history flow
        # handles transitions between any two infection states or stages
        self.progress_to_prepatent(ti)  # Incubation period
        self.progress_to_diseased(ti)   # Both acute and subclinical
        self.progress_to_chronic(ti)
        self.progress_to_recovered(ti)
        self.progress_to_susceptible(ti)
        self.progress_to_dead(ti)
        return

    # Methods that handle transitions between states
    def progress_to_prepatent(self, ti):
        susc2prep = (self.prepatent & (self.ti_prepatent <= ti)).uids
        self.immune[susc2prep] = False
        self.susceptible[susc2prep] = False
        # N_i: number of prior infections,
        # used to determine the probability of becoming
        # infected upon exposure (1-P)**N_i,
        # that provides almost sterilising immunity
        # in hyper-endemic settings,
        # but we may want to incorporate a mechanism
        # to wane naturally acquired immunity.
        self.n_infections[susc2prep] += 1.0
        self.infectiousness[susc2prep] = self.pars.tai * self.pars.tpri

    def progress_to_diseased(self, ti):
        # Progress prepatent -> acute
        prep2acute = (self.prepatent & (self.ti_acute <= ti)).uids
        self.acute[prep2acute] = True
        self.prepatent[prep2acute] = False
        self.infectiousness[prep2acute] = self.pars.tai

        # Progress prepatent -> subclinical
        prep2subcl = (self.prepatent & (self.ti_subclinical <= ti)).uids
        self.subclinical[prep2subcl] = True
        self.prepatent[prep2subcl] = False
        self.infectiousness[prep2subcl] = self.pars.tai * self.pars.tsri

    def progress_to_chronic(self, ti):
        # Progress acute -> chronic
        acu2chro = (self.acute & (self.ti_chronic <= ti)).uids
        self.chronic[acu2chro] = True
        self.acute[acu2chro] = False
        self.infectiousness[acu2chro] = self.pars.tai * self.pars.tcri

        # Progress subclinical -> chronic
        sub2chro = (self.subclinical & (self.ti_chronic <= ti)).uids
        self.chronic[sub2chro] = True
        self.subclinical[sub2chro] = False
        # https://github.com/starsimhub/typhoidsim/issues/53
        self.infectiousness[sub2chro] = self.pars.tai * self.pars.tsri * self.pars.tcri

    def progress_to_dead(self, ti):
        # Trigger deaths
        deaths = (self.ti_dead <= ti).uids
        if len(deaths):
            self.sim.people.request_death(deaths)
        pass

    def progress_to_recovered(self, ti):
        # handle acute pathway
        acu2rec = (self.acute & (self.ti_recovered <= ti)).uids
        self.recovered[acu2rec] = True
        self.acute[acu2rec] = False
        self.infected[acu2rec] = False
        self.infectiousness[acu2rec] = 0.0

        # handle subclinical pathway
        sub2rec = (
            self.subclinical & (self.ti_recovered <= ti)).uids
        self.recovered[sub2rec] = True
        self.subclinical[sub2rec] = False
        self.infected[sub2rec] = False
        self.infectiousness[sub2rec] = 0.0

    def progress_to_susceptible(self, ti):
        # Make agents susceptible again
        rec2suc = (self.recovered & (self.ti_susceptible <= ti)).uids
        self.susceptible[rec2suc] = True
        self.recovered[rec2suc] = False
        self.update_immunity(rec2suc)

    # Methods that handle durations/duration pars that are dependent on other variables/states
    def get_prepatent_duration_by_exposure(self, uids):
        """ Get durations in number of timesteps"""
        dt = self.sim.dt
        dur_prep = self.pars.dur_prep_dist.rvs(uids.size)
        # Return in number of timesteps with units (1 timestep / day)
        return sc.randround(dur_prep/dt)

    def get_acute_duration_by_age(self, uids):
        """
        acute and subclinical durations?, though that would prevent
        further differentiating between those two stages (ie, if the
        if we wanted to change the 'threshold' age in one of the
        stages but not the other. )
        """
        p = self.pars
        dt = self.sim.dt
        dur_acu = p.dur_symp_dist.rvs(uids.size)
        return sc.randround(dur_acu * (tyd.days_per_week / dt))

    def get_subclinical_duration_by_age(self, uids):
        p = self.pars
        dt = self.sim.dt
        dur_scl = p.dur_symp_dist.rvs(uids.size)
        return sc.randround(dur_scl * (tyd.days_per_week / dt))

    @staticmethod
    def prepatent_mean_dur_function(module, sim, uids):
        """
        Returns a mean duration parameter for every agent based on the cfu_dose,
        they have been exposed to. Assumes the parameter will be used by a
        lognormal_ex distribution.
        """
        # NOTE: this check is necesary other wise lognorm_ex fails at initialisation
        # when the module has not been initialised and uids are None
        if uids is None:
            mu = []
        else:
            mu = module.partial_prep_dur_mean(module.cfu_dose[uids])
        return np.array(mu)

    @staticmethod
    def prepatent_std_dur_function(module, sim, uids):
        """
        Returns a stdev duration parameter for every agent based on the cfu_dose,
        they have been exposed to. Assumes the parameter will be used by a
        lognormal_ex distribution.
        """
        if uids is None:
            std = []
        else:
            std = module.partial_prep_dur_std(module.cfu_dose[uids])
        return np.array(std)

    @staticmethod
    def susceptibility_prob_function(module, sim, uids):
        """
        Estimate the age-dependent probability of becoming susceptible
        """
        mpars = module.pars
        p_sus = tyum.sigmoid(sim.people.age[uids], mpars.sus_saturation_age,
                             mpars.sus_age_exposure_slope)
        return np.array(p_sus)

    @staticmethod
    def chronic_gall_prob_function(module, sim, uids):
        """
        Assumes gallstone probabilities and prevalence are defined.
        This scales p_cpg using prob of having gallstones and gallstone prevalence.
        """
        mpars = module.pars
        age_ints = tyu.digitize_ages_1yr(sim.people.age[uids])
        p_chro = mpars.p_cpg * \
                 mpars.p_gall[age_ints, sim.people.female[uids].astype(int)] * \
                 mpars.gall_prev[age_ints, sim.people.female[uids].astype(int)]
        return np.array(p_chro)

    @staticmethod
    def chronic_prob_function(module, sim, uids):
        """
        Does not use gallstone probabilities and prevalence.
        Uses directly p_cpg as p_chro.

        This method can be useful when either we do not have age and sex
        gallstone prob and prevalence distributions, or we want to simplify
        calibration by reducing degrees of freedom.
        """
        mpars = module.pars
        p_chro = mpars.p_cpg
        return np.array(p_chro)

    def will_become_chronic_carrier(self, uids):
        """Determine who will become a chronic carrier"""
        return self.pars.d_chro.filter(uids)

    def set_prognoses(self, uids, source_uids=None):
        """
        Here we define the whole natural history for every agent
        that has been infected, specifically when agents transition from
        one stage (state) to the next. The progression of this natural
        history can be altered by the environment, interventions, or
        other diseases.
        """
        p = self.pars
        ti = self.sim.ti
        dt = self.sim.dt

        # Set value of states associated to being infected, and record events
        self.susceptible[uids] = False
        self.immune[uids] = False
        self.infected[uids] = True
        self.prepatent[uids] = True
        self.ti_prepatent[uids] = ti
        self.ti_infected[uids] = ti

        # Durations returned by functions are in units of "number of timesteps"
        # Set duration of prepatent state, by defining when they will
        # progress to the next state (either acute or sublinical)
        dur_pre = ti + self.get_prepatent_duration_by_exposure(uids)

        # Acute and Subclinical stages: Determine who will become acute and who will become subclinical
        acu_scl = p.p_acute.filter(uids, both=True)
        acute_uids, subcl_uids = acu_scl

        # Set prepatent duration of those who will become acute
        self.ti_acute[acute_uids] = ti + dur_pre[np.isin(uids, acute_uids)]

        # Set prepatent duration of those who will become subclinical
        self.ti_subclinical[subcl_uids] = ti + dur_pre[np.isin(uids, subcl_uids)]

        # If treatment applied, this is when acute cases would seek treatment, relative
        # to the onset of acute stage. This variable captures human behaviour
        dur_wait = sc.randround(p.dur_wait2treatment.rvs(acute_uids) / dt)
        self.ti_seek_trtmnt[acute_uids] = self.ti_acute[acute_uids] + dur_wait

        # Chronic/carrier stage: Determine who becomes a (chronic) carrier from acute and sublclinical
        acu2chro_uids = self.will_become_chronic_carrier(acute_uids)
        scl2chro_uids = self.will_become_chronic_carrier(subcl_uids)

        dur_acu = self.get_acute_duration_by_age(acu2chro_uids)
        self.ti_chronic[acu2chro_uids] = self.ti_acute[acu2chro_uids] + dur_acu
        dur_scl = self.get_subclinical_duration_by_age(scl2chro_uids)
        self.ti_chronic[scl2chro_uids] = self.ti_subclinical[scl2chro_uids] + dur_scl

        # Death: From the acute cases, determine who can die because they don't become carriers
        can_die_uids = np.setdiff1d(acute_uids, acu2chro_uids)

        # From the acutes who do not become carriers, determine who recovers and who dies
        will_die = p.p_death.rvs(can_die_uids)
        acu2rec_uids = can_die_uids[~will_die]

        # Recovery: Get sublinical cases that recover because they won't become carriers
        scl2rec_uids = np.setdiff1d(subcl_uids, scl2chro_uids)
        will_recover_uids = acu2rec_uids.concat(scl2rec_uids)

        # Determine when non-carriers recover and become susceptible again,
        # NOTE: we do not have to track a recovered state, we can simply output results
        # that track the 'concept' of a recovered state
        dur_acu = self.get_acute_duration_by_age(acu2rec_uids)
        dur_scl = self.get_subclinical_duration_by_age(scl2rec_uids)
        self.ti_recovered[acu2rec_uids] = self.ti_acute[acu2rec_uids]       + dur_acu
        self.ti_recovered[scl2rec_uids] = self.ti_subclinical[scl2rec_uids] + dur_scl

        # NOTE: typhoid can get very low mortality (in particular with treatment),
        # so there is a high chance of getting empty dead_uids. If that happens,
        # the line below may seg fault . Just in case check first.
        dead_uids = can_die_uids[will_die]
        if dead_uids.size:
            dur_acu = self.get_acute_duration_by_age(dead_uids)
            self.ti_dead[dead_uids] = self.ti_acute[dead_uids] + dur_acu

        self.ti_susceptible[will_recover_uids] = self.ti_recovered[will_recover_uids] + 1  # recover in the next time step, just to make things tidy
        return

    #  Transmission-realated methods - interaction between agents and "else" (other agents)
    #  or the environment
    def make_new_cases(self):
        """
        Handle transmission of the infections, includes contact and transmission
        routes
        """
        # From EMOD:
        # Contagion in the contact route is 100% per timestep (1 day in the typhoid model)
        # Contagion is a level of CFU transmitted by the the pool of contagion to a target
        new_cases_c, _, _ = super().make_new_cases()
        # Make sure new cases due to contagion contact route get assigned the correct
        # dose of cfu to determine their prepatent duration
        # From EMOD: Currently, all infections from the Contact route are assumed to be a
        # high dose prepatent duration, meaning that the characteristic dose a
        # target agent receives has to be set to be at least self.pars.cfu_me_hi + 1
        self.cfu_dose[new_cases_c] = self.pars.cfu_me_hi + 1
        # Make sure new cases are correctly set up as prepatent (ie, infectiousness levels, etc)
        self.progress_to_prepatent(self.sim.ti)
        # NOTE/TODO: confirm whether self.pars.transmission.ppl2ppl_exposure_rate.rvs(uids.size)*dt,
        # refers to average daily number of 'contacts' in the contact route

        self.make_new_cases_environmental()

        return

    def make_new_cases_environmental(self):
        """
        1. infected individuals shed into the environment (environmental contagion pool grows ↑↑)
        2. individuals get exposed by the environment (increases their n_exposures)
        3. Bacteria in the environment die at a specific rate (contagion pool in environment decays ↓↓)
        """
        trans_pars = self.pars.transmission

        # Skip all of this if there is no tranmission,
        # TODO: if environmental transmission is 0, then this parameter should also scale shedding?
        if trans_pars.beta == 0:
            return []

        ti = self.sim.ti
        dt = self.sim.dt
        environment = self.sim.demographics['environmentalpool']

        # Determine who gets infected from environment. Multiply by rel_sus, as many interventions will target this parameter
        # This means an agent can become unsusceptible because of an external factor.
        susc = self.susceptible.asnew(self.susceptible * self.rel_sus)
        susc_uids = (susc).uids

        # Increase cfu doses in susceptible people by exposing them to the environment
        # TODO: check whether the multiplication by dt makes sense. I think it does in particular if dt < 1 day
        self.n_exposures[susc_uids] = trans_pars.env2ppl_exposure_rate.rvs(susc_uids.size) * dt
        self.cfu_dose[susc_uids] = self.sim.demographics['environmentalpool'].sv.cfu_level[ti - 1]  * self.n_exposures[susc_uids]  # beta us used to simulate reduction in exposure amount due to behavioural changes

        ## The distribution trans_pars.env2ppl_p_inf(p=fun()), where fun() is
        # infection_prob_function(), which calls self.drc(). This assesses
        # the immunity responses of the hosts (drc) due to a certain amount of
        # exposure doses (cfu_dose). Then, infection_...() it estimates
        # a probability of infection.
        got_infected = trans_pars.env2ppl_p_inf(susc_uids)
        new_cases = susc_uids[got_infected]
        if len(new_cases):
            self.set_prognoses(new_cases, source_uids=None)
            self.progress_to_prepatent(ti)

        # Infectious individuals shed contagion into the contagion pool.
        # Reduction in shedding can happen due to per-agent interventions (reduces individual level of infectiousness),
        # or due to sanitation interventions.
        shedded_cfu = environment.pars.transmission.ppl2env_shedding_rate * (
                self.rel_trans[self.infected] * self.infectiousness[
            self.infected]).sum()

        # CFU level increases due to people shedding into the environment
        self.sim.demographics['environmentalpool'].sv.cfu_level[
            ti - 1] += shedded_cfu


        return new_cases

    @staticmethod
    def response_prob_function(module, sim, uids):
        p_resp = module.drc(module.cfu_dose[uids])
        return np.array(p_resp)

    @staticmethod
    def infection_prob_function(module, sim, uids):
        # Evoke an immunity-like response
        p_resp = module.drc(module.cfu_dose[uids])
        p_infc = 1.0 - (1.0 - module.immunity[uids] * p_resp) ** module.n_exposures[uids]  # total number of n_exposures per unit of time? total?
        return np.array(p_infc)

    @staticmethod
    def symp_dur_mean(module, sim, uids):
        """_
        Age-dependent mean of the distribution of durations of the
        symptomatic stage. Assumes a lognormal_ex distribution.
        """
        if uids is None:
            mean_arr = np.array([])
        else:
            th_age = module.pars.symp_dur_th_age
            mean_arr = np.ones(1 if isinstance(uids, int) else uids.size)
            mean_arr[sim.people.age[uids] < th_age]  = module.pars.symp_dur_mean_le
            mean_arr[sim.people.age[uids] >= th_age] = module.pars.symp_dur_mean_geq
        return mean_arr

    @staticmethod
    def symp_dur_std(module, sim, uids):
        """
        Age-dependent standard deviation of the distribution of durations of the
        symptomatic stage. Assumes a lognormal_ex distribution.
        """
        if uids is None:
            std_arr = np.array([])
        else:
            th_age = module.pars.symp_dur_th_age
            std_arr = np.zeros(1 if isinstance(uids, int) else uids.size)
            std_arr[sim.people.age[uids] < th_age]  = module.pars.symp_dur_std_le
            std_arr[sim.people.age[uids] >= th_age] = module.pars.symp_dur_std_geq
        return std_arr

    def update_immunity(self, uids):
        """
        Acquired immunity. Note that the more infections, the lower the number
        immunity associated with immunity -- this number acts as an
        attenuation factor.
        """
        # TPPI: Typhoid Protection Per Infection
        self.immunity[uids] = (1.0 - self.pars.tppi)**self.n_infections[uids]
        # NOTE: We could add a mechanisms for immunity waning here
        return

    def drc(self, cfu_dose, alpha=0.175, n50=1.11e6):
        """
        The probability of infection is mediated by the dose-response curve (drc),
        taking in the contagion population as a value of colony-forming units (CFU)
        and returning a probability of infection.

        The independent variable `cfu_dose` can be modulated by seasonality
        factors.

        The DRC is a beta-binomial curve fitted the historical challenge
        data by QMRA (Enger, 2013), where:

        P(response) = 1- [1 + cfu_dose * (2^(1/ α)- 1)/N50] ^(-α)

        # TODO: parameterise this function via pars. Also this function could
        be user-defined if the environment was a separate module.
        """
        p_response = 1.0 - (1.0 + cfu_dose * ((2.0**(1.0/alpha) - 1.0)/n50))**(-alpha)
        return p_response

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.sim.ti
        res.new_susceptible[ti] = np.count_nonzero(self.ti_susceptible == ti)
        res.new_prepatent[ti] = np.count_nonzero(self.ti_prepatent == ti)
        res.new_acute[ti] = np.count_nonzero(self.ti_acute == ti)
        res.new_subclinical[ti] = np.count_nonzero(self.ti_subclinical == ti)
        res.new_chronic[ti] = np.count_nonzero(self.ti_chronic == ti)
        res.new_recovered[ti] = np.count_nonzero(self.ti_recovered == ti)
        res.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        return
