"""
Typhoid model.
"""

import numpy as np

import sciris as sc
import starsim as ss

import typhoidsim.utils as tyu
import typhoidsim.defaults as tyd
import typhoidsim.utils_math as tyum

ss_int_ = ss.dtypes.int

# The disease module
__all__ = ["Typhoid"]
# Context-specific functions that can be used as parameters of the Typhoid module
__all__ += ["unexp2sus_childhood_prob_function_gauld2018",]


class Typhoid(ss.Disease):
    """
    Typhoid module that includes the natural history of the disease in a human
    agent.
    """

    def __init__(self, pars=None, *args, **kwargs):
        """Initialize with parameters"""
        super().__init__()
        self.define_pars(
            # Initial conditions
            # Seeded cases at t=0
            init_prev=ss.bernoulli(p=0.005),

            # Initial susceptibility (assumed state of the population, not everyone is naive)
            init_seroprev=ss.bernoulli(p=0.0),  # % of people who were previously infected
            init_prior_infc=ss.poisson(lam=self.prior_infections_by_age),

            # NATURAL HISTORY PARAMETERS
            # From never exposed/invulnerable to susceptible
            p_unexp2sus_6m=ss.bernoulli(p=0.14),  # Proportion of never exposed/completely immune population at 6 months that moves to susceptible state
            p_unexp2sus_3y=ss.bernoulli(p=0.29),  # Proportion of never exposed/completely immune population at 3 years that moves to susceptible state
            p_unexp2sus_6y=ss.bernoulli(p=0.61),  # Proportion of never exposed/completely immune population at 6 years that moves to susceptible state

            # Default mechanism to move agents from never exposed/invulnerable to susceptible to exposure
            p_unexp2sus=ss.bernoulli(p=self.unexp2sus_prob_gauld2018),
            unexp2sus_saturation_age=20.0,    # Age after which 100% of people are susceptible to exposure
            unexp2sus_slope=1.0,              # Slope of the sigmoid function (slope > -0.5), determines the proportion of people aged X who will be in the susceptible state

            # Prepatent stage, the parameters of the distribution of durations is CFU-dose dependent
            prep_dur_dpars=tyu.load_dataset("prepatent_dur_dist_pars"),   # CFU dose-dependent duration distribution parameters, in days. Stratified in 3 levels (low, medium and high)
            prep_dur_fun=tyum.double_sigmoid_tanh,                        # Function to represent the 3 levels of each prepatent duration distribution parameter as a continuous function
            dur_prep_dist=ss.lognorm_ex(mean=self.prepatent_mean_dur_function,
                                        std=self.prepatent_std_dur_function),

            cfu_lo_me=5_050_000.0,   # Threshold CFU value to determine whether to use the 'low dose' (for cfu_dose <= cfu_lo_me) or 'medium dose'  (cfu_dose > cfu_lo_me) mean & std duration parameters for prepatent duration distribution.
            cfu_me_hi=55_000_000.0,  # Threshold CFU value to determine whether  to use the 'medium dose' (for cfu_dose <= cfu_me_hi) or 'high dose' (cfu_dose > cfu_lo_me) mean & std duration parameters for prepatent duration distribution.

            # Infected/Diseased stage, (acute and subclinical)
            p_acute=ss.bernoulli(p=0.16),  # Prob of becoming acute
            # Age-dependent acute and subclinical duration distribution parameters
            inf_dur_th_age=30.0,     # Duration of clinically diagnosable cases (acute/subclinical) age threshold.
            # For people aged less than threshold
            inf_dur_mean_le=1.172,   # Acute/subclinical duration mean if age < age_threshold, in weeks.
            inf_dur_std_le=0.483,    # Acute/subclinical duration std  if age < age_threshold, in weeks.
            # For people aged threshold value and over
            inf_dur_mean_geq=1.172,  # Acute/subclinical duration mean if age >= age_threshold, in weeks.
            inf_dur_std_geq=0.788,   # Acute/subclinical duration std if age  >= age_threshold, in weeks.
            dur_inf_dist=ss.lognorm_ex(mean=self.inf_dur_mean,
                                       std=self.inf_dur_std),     #  acute or subclinical duration, depends on age, expressed in weeks.
            dur_wait2treatment=ss.lognorm_ex(mean=2.33219066, std=0.5430),  # (Relative to acute onset) day of treatment-seeking for acute cases, in days.

            # Long-term stages
            # Chronic
            p_cpg=0.15,    # Base prob of becoming chronic after subclinical or acute given gallstones. Same for female and male, but does not have to be.
            d_chro=ss.bernoulli(p=self.chronic_gall_prob_function),    # Prob of becoming chronic carrier from acute or clinical infection modulated by gallstone prevalence
            p_gall=tyu.load_dataset("gallstone_probs"),    # Probability of having gallstones by age and sex
            # gall_prev=tyu.load_dataset("gallstone_prev"),  # Biological sex gallstone prevalence (expressed in fraction of the population, value between 0 and 1)
            p_rec=ss.bernoulli(p=0.0),           # Prob of recovering from chronic state. Default 0 means everyone becomes a chronic carrier (duration of chronic state is infinite)
            dur_chro_dist=ss.constant(v=102.0),  # Duration of chronic state, in weeks. 102 weeks is slightly over 2 years. In a sim shorter than this duration, it behaves as if chronic carriers never recover,

            # Recovered
            dur_rec_dist=ss.constant(v=1.0),  # Duration of recovered state, in days.
            # TODO: do we need a parameter that embodies the probability of clinical immunity after acute infection?

            # Death
            p_death=ss.bernoulli(p=0.01),   # Probability of dying from acute, context dependent, and by default set to something zero or something very small

            # IMMUNE SYSTEM-WITHIN HOST PARAMETERS
            # Infectiousness parameters
            tai=ss.constant(v=40_000),  # Typhoid acute infectiousness, represents number of colony-forming units of S. typhi, for an average human that has 3500 mL of blood, this is about 11 CFU/mL
            tpri=0.5,      # Typhoid relative (to acute) prepatent infectiousness
            tsri=1.0,      # Typhoid relative (to acute) subclinic infectiousness
            tcri=0.241,    # Typhoid relative (to acute) chronic infectiousness
            tppi=ss.constant(v=0.98),     # Decrease in susceptibility per infection (exponential decrease)
            drc_alpha=0.175,  # parameter in the Dose Response Curve to environmental exposure
            drc_n50=1.11e6,   # parameter in the Dose Response Curve to environmental exposure

            # ENVIRONMENT PARAMETERS
            has_environment=None,
            # Transmission parameters
            transmission=ss.Pars(
                # Behavioural interaction parameters between people and environment
                env2ppl_p_inf=ss.bernoulli(p=self.infection_prob_function_env),
                # Behavioural interaction parameters between people and other people in their contact networks
                exposure2contact_rate=1.0,  # Rate determining the daily number of exposures for the contact route
                ppl2ppl_p_inf=ss.bernoulli(p=self.infection_prob_function_contact),  # The probabilities will be updated for each agent based on the interaction with their contacts
                # Multiroute transmission
                p_route=ss.uniform(),  # NOTE: currently unused, but stub for transmission route selection. See: https://github.com/starsimhub/typhoidsim/issues/102
                teer=ss.constant(v=2.0),  # distribution of the parameter lambda
            ),

            beta=None, # NOTE: Typhoid does not have/ does not use beta, but starsim's networks expect this parameter to exist.
                   # Its value will be updated to be dt during validation, so the net effect is a beta=1 per time step.
        )
        self.update_pars(pars, **kwargs)

        # Parametrisation of prepatent duration distribution parameters (ie, mean and std are functions of CFU dose)
        self.partial_prep_dur_mean, self.partial_prep_dur_std = self.prepare_partial_prep_funs()

        # Boolean states
        self.define_states(
            # Infection life cycle states
            ss.State("susceptible", default=False, label="Susceptible"),
            ss.State("infected", default=False, label="Infectious"),
            ss.State("unexposed", default=True, label="Unexposed"),  # People are born into this state, naive and never exposed
            ss.State("prepatent", default=False, label="Prepatent"),  # Also known as exposed state (incubation stage)
            ss.State("acute", default=False, label="Acute"),
            ss.State("subclinical", default=False, label="Subclinical"),
            ss.State("chronic", default=False, label="Chronic"),
            ss.State("recovered", default=False, label="Recovered/Been Infected"),
            ss.State("infected_ever", default=False, label="Ever Infected"),

            # States that track immunity-related quantities or variables
            # and depend on infection states
            ss.FloatArr("n_exposures", default=0.0, label="Number of Exposures"),            # average number of exposures from a given source/route over the interval of one timestep (usually 1 day)
            ss.FloatArr("exposure_amount", default=0.0, label="Number of Exposures"),        # average (number of exposures * volume) from a given source/route over the interval of one timestep (usually 1 day)
            ss.FloatArr("cfu_dose_per_exposure", default=0.0, label="Single environmental exposure amount (CFUs)"),
            ss.FloatArr("cfu_dose", default=0.0, label="Exposure amount (CFUs)"),            # contagion amount in number of CFUs (acquisition phase, "doses" of bacteria that the target host takes as input from sources of contagion)
            ss.FloatArr("infectiousness", 0.0, label="Infectiousness"),                      # average number of CFUs during different stages of the disease (infected phase, within host).
            ss.FloatArr("n_infections", 0.0, label="Number of Infections"),                  # number of infections over the lifespan of this agent
            ss.FloatArr("susceptibility", default=1.0, label="Susceptibility Level"),        # blocking effect factor due to immunity to typhoid, value between 0 (blocking new infections) and 1 (completely vulnerable). Maybe we need a more descriptive name.
            ss.FloatArr("immunity_acquired", default=0.0, label="Acquired Immunity Level"),  # Acquired/evoked immune protection against infection due to vaccinations, a value between 0 and 1

            # Track agent attributes
            ss.FloatArr("tai", default=self.pars.tai, label="Typhoid Acute Infectiousness"),   # track individual value of TAI
            ss.FloatArr("tppi", default=self.pars.tppi, label="Typhoid Protection Per Infection"),

            # Track some probabilities; some are not  used now but will become important in multi-route transmission
            ss.FloatArr("p_resp", default=0.0, label="Probability of response to infection"),  # The probability of having a response to pathogens, usually a term involved in determining p_infc
            ss.FloatArr("p_infc", default=0.0, label="Probability of Infection"),      # Track probability of infection
            ss.FloatArr("p_route", default=0.0, label="Probability Route Draw"),       # Probability to determine which route will be the route if infection
            ss.FloatArr("infc_origin", label="Origin of infection"),                   # Track origin of infection

            ss.FloatArr("rel_sus", default=1.0, label="Relative susceptibility"),
            ss.FloatArr("rel_trans", default=1.0, label="Relative transmissibility"),
            ss.FloatArr("eff_sus", default=0.0, label="Effective susceptibility"),     # Track effective susceptibility to infection, the combination of rel_sus, susceptibility and (1-aqcuired_immunity)

            # States that track timing of events
            ss.FloatArr("ti_firstinfected", label="Time of first infection"),
            ss.FloatArr("ti_infected", label="Time of infection"),
            ss.FloatArr("ti_susceptible", label="Start of susceptible state"),
            ss.FloatArr("ti_prepatent", label="Start of prepatent stage"),
            ss.FloatArr("ti_subclinical", label="Start of subclinical stage"),
            ss.FloatArr("ti_acute", label="Start of acute stage"),
            ss.FloatArr("ti_seek_trtmnt", label="Time to seek treatment"),
            ss.FloatArr("ti_chronic", label="Start of chronic stage"),
            ss.FloatArr("ti_recovered", label="Time of recovery"),
            ss.FloatArr("ti_dead", label="Time of death"),

        )

        self.n_initial_cases = None
        self.n_infections_historical = None

        return

    def init_pre(self, sim):
        """ Initialise objects and valid before simulation run"""
        super().init_pre(sim)
        self.validate_beta()
        self.validate_environment()
        return

    def validate_beta(self): # CK: TODO: remove
        """
        Perform any parameter validation
        """
        networks = self.sim.networks
        if networks is not None and len(networks) > 0:

            if 'beta' not in self.pars:
                errormsg = f'Disease {self.name} is missing beta; pars are: {sc.strjoin(self.pars.keys())}'
                raise sc.KeyNotFoundError(errormsg)

            if self.pars.beta is None:
                #NOTE: Typhoid does not have/ does not use beta, but starsim's networks
                # expect this parameter to exist. To cancel out the effect we set
                # beta to be the time step, such that when beta_per_dt is calculated
                # the effective value is 1.0
                self.pars.beta = 1./self.t.dt

            # If beta is a scalar, apply this bi-directionally to all networks
            if sc.isnumber(self.pars.beta):
                β = self.pars.beta
                self.pars.beta = sc.objdict(
                    {k: [β, β] for k in networks.keys()})

            # If beta is a dict, check all entries are bi-directional
            elif isinstance(self.pars.beta, dict):
                for k, β in self.pars.beta.items():
                    if sc.isnumber(β):
                        self.pars.beta[k] = [β, β]
        return

    def init_post(self): # CK: TODO: remove
        """
        Set initial values for all agent states, and seed new cases if initial
        prevalence is provided.

        This function could involve passing in a full set of initial conditions,
        or using init_prev (initial prevalence), or other.
        """
        super().init_post() # CK: check
        # NOTE: Typhoid assumes that all individuals are born into an
        # a class where they cannot get infected (called unexposed here) and then
        # move to the susceptible class at probabilities for each age.

        # Determines which individuals enter the susceptible class. The age-based
        # mechanism is specified in the typhoid module parameters as
        #     p_unexp2sus = ss.bernoulli(p=self.unexp2susc_prob_function)
        #     p_unexp2sus = ss.bernoulli(p=self.unexp2susc_prob_gauld2018),
        self.make_susceptible()

        # Of the people who are susceptible now, select those who we assume were infected before t=0
        prior_cases = self.pars.init_seroprev.filter((self.susceptible).uids)
        self.n_infections[prior_cases] = self.pars.init_prior_infc.rvs(prior_cases)
        self.update_susceptibility(prior_cases)
        self.n_infections_historical = 0.0

        if len(prior_cases):
            self.n_infections_historical = self.n_infections[prior_cases].sum()

        self.n_initial_cases = 0.0
        if self.pars.init_prev is not None:
            eligible_uids =  np.setdiff1d((self.susceptible).uids, prior_cases)
            initial_cases = self.pars.init_prev.filter(eligible_uids)
            self.set_prognoses(initial_cases)
            self.progress_to_prepatent(self.sim.ti)  # Incubation period
            self.ti_prepatent[initial_cases] = -2
            self.n_initial_cases = len(initial_cases)
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
        self.define_results(
            ss.Result("prevalence", dtype=float, scale=False, label="Prevalence"),
            ss.Result("new_infections", dtype=int, scale=True, label="New Infections"),
            ss.Result("cum_infections", dtype=int, scale=True, label="Cumulative Infections"),
            ss.Result("new_susceptible", dtype=int, scale=True, label="New Susceptible"),
            ss.Result("new_prepatent", dtype=int, scale=True, label="New Prepatent"),
            ss.Result("new_acute", dtype=int, scale=True, label="New Acute"),
            ss.Result("cum_acute", dtype=int, scale=True, label="Cumulative Acute"),
            ss.Result("new_subclinical", dtype=int, scale=True, label="New Subclinical"),
            ss.Result("new_chronic", dtype=int, scale=True, label="New Chronic"),
            ss.Result("new_recovered", dtype=int, scale=True, label="New Recovered"),
            ss.Result("new_deaths", dtype=int, scale=True, label="New Dead"),
            ss.Result("cum_deaths", dtype=int, scale=True, label="Cumulative Dead"),
        )
        return

    def _check_betas(self):
        """ Check that there's a network for each beta key """
        # Ensure keys are lowercase
        if isinstance(self.pars.beta, dict):  # TODO: check if needed
            self.pars.beta = {k.lower(): v for k, v in self.pars.beta.items()}

        # Create a mapping between beta and networks, and populate it
        betapars = self.pars.beta
        betamap = sc.objdict()
        netkeys = list(self.sim.networks.keys())
        if netkeys: # Skip if no networks
            for bkey in betapars.keys():
                orig_bkey = bkey[:]
                if bkey in netkeys:  # TODO: CK: could tidy up logic
                    betamap[bkey] = betapars[orig_bkey]
                else:
                    if 'net' not in bkey:
                        bkey += 'net'  # Add 'net' suffix if not already there
                    if bkey in netkeys:
                        betamap[bkey] = betapars[orig_bkey]
                    else:
                        errormsg = f'No network for beta parameter "{bkey}"; your beta should match network keys:\n{sc.newlinejoin(netkeys)}'
                        raise ValueError(errormsg)
        return betamap

    def validate_environment(self):
        """
        Validate environment
        """
        demographic_modules = self.sim.demographics
        environmental_key = "environmentalpool"
        if demographic_modules is not None and len(demographic_modules) > 0:
            try:
                if environmental_key in demographic_modules.keys():
                    self.pars.has_environment = True
            except sc.KeyNotFoundError:
                self.pars.has_environment = False
                msg = f"{environmental_key} module not found. Will run simulation without environmental transmission."
                ss.warn(msg)
        return

    # Methods that are specific to a single stage of infection
    def make_susceptible(self):
        """
        Our model assumes all individuals are born into an unexposed, completely
        immune state and move to the susceptible class based on some probability.

        The mechanism that moves individuals from one state to the other,
        can depend on age and/or other factors.

        By default, we do not assume any age-specific structure. Thus, agents are
        born into the unexposed state and are moved immediately to the susceptible
        state.

        However, there are a couple of predefined age-specific susceptibility
        probability functions implemented:
         - unexp2sus_youth_prob_function_gauld2018()
         - unexp2sus_childhood_prob_function_gauld2018()

        These functions can be passed as arguments to the Typhoid parameter. For
        instance:

           >> p_unexp2sus=ss.bernoulli(p=unexp2sus_prob_gauld2018),
        or
           >> p_unexp2sus=ss.bernoulli(p=unexp2sus_childhood_prob_function_gauld2018),

        See Also:
        https://github.com/jgauld/DtkTrunk/blob/Typhoid-Ongoing/Eradication/SusceptibilityTyphoid.cpp

        """
        never_exposed = (self.unexposed).uids
        self.susceptible[never_exposed] = self.pars.p_unexp2sus.rvs(never_exposed)
        self.unexposed[never_exposed] = ~self.susceptible[never_exposed]
        self.ti_susceptible[never_exposed] = self.ti  # Save the timing people became susceptible
        return

    def step_die(self, uids):
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
            self.statesdict[f"ti_{state}"][uids] = np.nan
        self.statesdict["unexposed"][uids] = False
        return

    # Update progression of disease, handle transitions
    def step_state(self):
        """
        Update the progression of the disease -- handles disease state transitions.
        In a typical simulation flow this method is called before propagating
        the infection/disease is propagated via make_new_cases()
        """
        ti = self.ti  # current timestep

        # If the population does not age, then over a long simulation, eventually all people aged 0-20 will transition to the susceptible state.
        self.make_susceptible()

        # Age-based susceptibility in children <= 6 years old
        # TODO: age-based susceptibility depends on the use-case
        # self.increase_childhood_susceptibility()

        # The infection life cycle or natural history flow
        # handles transitions between any two infection states or stages
        self.progress_to_diseased(ti)   # Both acute and subclinical
        self.progress_to_chronic(ti)
        self.progress_to_recovered(ti)
        self.progress_to_susceptible(ti)
        self.progress_to_dead(ti)
        return

    # Methods that handle transitions between states
    def progress_to_prepatent(self, ti):
        susc2prep = (self.prepatent & (self.ti_prepatent == ti)).uids
        self.prepatent[susc2prep] = True
        self.susceptible[susc2prep] = False
        self.unexposed[susc2prep] = False
        self.infected[susc2prep] = True
        self.infected_ever[susc2prep] = True
        # N_i: number of prior infections,
        # used to determine the probability of becoming
        # infected upon exposure (1-P)**N_i,
        # that provides almost sterilising immunity
        # in hyper-endemic settings,
        # but we may want to incorporate a mechanism
        # to wane naturally acquired immunity.
        self.n_infections[susc2prep] = self.n_infections[susc2prep] + 1.0
        self.infectiousness[susc2prep] = self.tai[susc2prep] * self.pars.tpri

    def progress_to_diseased(self, ti):
        # Progress prepatent -> acute
        prep2acute = (self.prepatent & (self.ti_acute <= ti)).uids
        self.acute[prep2acute] = True
        self.prepatent[prep2acute] = False
        self.infectiousness[prep2acute] = self.tai[prep2acute]

        # Progress prepatent -> subclinical
        prep2subcl = (self.prepatent & (self.ti_subclinical <= ti)).uids
        self.subclinical[prep2subcl] = True
        self.prepatent[prep2subcl] = False
        self.infectiousness[prep2subcl] = self.tai[prep2subcl] * self.pars.tsri

    def progress_to_chronic(self, ti):
        # Progress acute -> chronic
        acu2chro = (self.acute & (self.ti_chronic <= ti)).uids
        self.chronic[acu2chro] = True
        self.acute[acu2chro] = False
        self.infectiousness[acu2chro] = self.tai[acu2chro] * self.pars.tcri

        # Progress subclinical -> chronic
        sub2chro = (self.subclinical & (self.ti_chronic <= ti)).uids
        self.chronic[sub2chro] = True
        self.subclinical[sub2chro] = False
        self.infectiousness[sub2chro] = self.tai[sub2chro] * self.pars.tcri

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
        sub2rec = (self.subclinical & (self.ti_recovered <= ti)).uids
        self.recovered[sub2rec] = True
        self.subclinical[sub2rec] = False
        self.infected[sub2rec] = False
        self.infectiousness[sub2rec] = 0.0

    def progress_to_susceptible(self, ti):
        # Make agents susceptible again
        rec2suc = (self.recovered & (self.ti_susceptible <= ti)).uids
        self.susceptible[rec2suc] = True
        self.recovered[rec2suc] = False
        self.update_susceptibility(rec2suc)

    # Methods that handle durations/duration pars that are dependent on other variables/states
    def get_prepatent_duration_by_exposure(self, uids):
        """ Get durations in number of timesteps"""
        dt = self.t.dt
        dur_prep = self.pars.dur_prep_dist.rvs(uids).astype(float)  # in days
        dur_prep = dur_prep * tyd.day2year  # in years
        return sc.randround(dur_prep / dt)  # in number of timesteps

    def get_acute_duration_by_age(self, uids):
        """
        Duration of the acute stage
        """
        p = self.pars
        dt = self.t.dt
        dur_acu = p.dur_inf_dist.rvs(uids) * tyd.days_per_week  # in days
        dur_acu = dur_acu * tyd.day2year  # in years
        return sc.randround(dur_acu / dt)  # in number of timesteps

    def get_subclinical_duration_by_age(self, uids):
        """
        Determine duration of the sublinical stage
        """
        p = self.pars
        dt = self.t.dt
        dur_scl = p.dur_inf_dist.rvs(uids) * tyd.days_per_week  # in days
        dur_scl = dur_scl * tyd.day2year
        return sc.randround(dur_scl / dt)

    def get_wait_duration(self, uids):
        """
        Determine how many days a person in the acute stage would wait before
        seeking treatment
        """
        p = self.pars
        dt = self.t.dt
        dur_wait = p.dur_wait2treatment.rvs(uids).astype(float)
        dur_wait = dur_wait * tyd.day2year
        return sc.randround(dur_wait / dt)

    def get_recovered_duration(self, uids):
        p = self.pars
        dt = self.t.dt
        dur_rec = p.dur_rec_dist.rvs(uids)  # duration in days
        dur_rec = dur_rec * tyd.day2year                  # duration in years
        return sc.randround(dur_rec / dt)        # duration in integer number of timesteps

    def get_chronic_duration(self, uids):
        """
        Determine duration of chronic stage
        See: https://github.com/starsimhub/typhoidsim/issues/66
        """
        p = self.pars
        dt = self.t.dt
        dur_chro = p.dur_chro_dist.rvs(uids) * tyd.days_per_week  # duration in in days
        dur_chro = dur_chro * tyd.day2year        # duration in years
        return sc.randround(dur_chro / dt)        # duration in integer number of timesteps


    def get_tai_vals(self, uids):
        """
        Value of typhoid individual infectiousness
        """
        p = self.pars
        # TODO: we may need clipping if tai is defined as a normal distribution
        # to avoid having negative values
        tai_vals = p.tai.rvs(uids)
        return tai_vals

    @staticmethod
    def tai_mean(module, sim, uids):
        """
        Make an array of values for the mean of a normal distribution.
        Any function with the same signature can be passed as the values to
        the parameters of a distribution.
        """
        if uids is None:
            mu = []
        else:
            mu = module.pars.tai_mean * np.ones(len(uids))
        return np.array(mu)

    @staticmethod
    def tai_std(module, sim, uids):
        """
        Make an array of values for the std of a normal distribution
        Any function with the same signature can be passed as the values to
        the parameters of a distribution.
        """
        if uids is None:
            std = []
        else:
            std = module.pars.tai_std * np.ones(len(uids))
        return np.array(std)

    @staticmethod
    def prepatent_mean_dur_function(module, sim, uids):
        """
        Returns a mean duration parameter for every agent based on the cfu_dose,
        they have been exposed to. Assumes the parameter will be used by a
        lognormal_ex distribution.
        """
        # NOTE: this check is necessary otherwise lognorm_ex fails at initialization
        # when the module has not been initialized and uids are None
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
    def unexp2susc_prob_function(module, sim, uids):
        """
        Simple (default) mechanism that defines the probability of
        moving from the unexposed stat to the susceptible state.
        This mechanism moves everyone to the susceotible state.
        """
        p_sus = np.ones(len(uids))
        return p_sus

    @staticmethod
    def unexp2sus_prob_gauld2018(module, sim, uids):
        """
        Estimate the age-dependent probability of transistioning from
        unexposed to susceptible. From Gauld et al 2018, Fig. 2B.

        Args:
            module: a starsim (disease) Module
            sim: the starsim Sim object (fully initialized)
            uids: the uids of the eligible people

            # Parameters for age-based transition from unexposed to susceptible (Gauld et al. 2018)
            sus_saturation_age=20.0,     # Age (years) after which agents are 100% susceptible
            sus_age_exposure_slope=1.0,  # Called typhoid_exposure_lambda in emod

        Returns:
            p_sus (array): array of probabilities for every agent in uids.

        EMOD:
        if( ( age < twenty_years_old_days ) && mod_acquire == 0 )
        {
                    float lambda = IndividualHumanTyphoidConfig::typhoid_exposure_lambda;

                    perc2 = 1.0f - ((twenty_years_old_days - age) / (age*lambda + twenty_years_old_days ));
                    perc1 = 1.0f - ((twenty_years_old_days - (age-1)) / ((age-1)*lambda + twenty_years_old_days ));
                    perc = (perc2 - perc1) / (1 - perc1); <--- probability
        }

        From Gauld 2018:
        Specifically, at each month of age a fitted curve determines the
        probability of an individual entering the susceptible class.

        But the EMOD code does not seem to assess at every month of age, rather
        at every time step of 1 day.
        """
        sat_age = module.pars.unexp2sus_saturation_age
        slope = module.pars.unexp2sus_slope
        p2 = tyum.sigmoid(sim.people.age[uids], sat_age, slope)
        p1 = tyum.sigmoid(sim.people.age[uids] - sim.t.dt, sat_age, slope)
        if module.ti == 0:
            p_sus = p2
        else:
            p_sus = ((p2 - p1) / (1.0 - p1))
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
                 mpars.p_gall[age_ints, sim.people.female[uids].astype(int)]
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

    def set_prognoses(self, uids, sources=None):
        """
        Here we define the whole natural history for every agent that has been infected.
        Specifically, we define *how long* each agent will stay in each stage
        of the disease, thus define *when* agents transition from one stage (state)
        to the next. This function also uses the probability distributions passes
        as parameters of the model to determine whol will become acute/sublinical/chronic, etc.

        The progression of this natural history can be altered by the environment,
        interventions, or other diseases.
        """
        p = self.pars
        ti = self.ti

        # Set value of parameters who are distributions
        self.tai[uids] = self.get_tai_vals(uids)
        self.tppi[uids] = p.tppi.rvs(uids)

        # Set value of states associated to being infected, and record events
        self.ti_prepatent[uids] = ti
        self.ti_infected[uids] = ti
        self.ti_firstinfected[uids[self.n_infections[uids]==0]] = ti
        self.prepatent[uids] = True

        # Durations returned by functions are in units of "number of timesteps"
        # Set duration of prepatent state, by defining when they will
        # progress to the next state (either acute or sublinical)
        dur_pre = self.get_prepatent_duration_by_exposure(uids)

        # Acute and Subclinical stages: Determine who will become acute and who will become subclinical
        acu_scl = p.p_acute.filter(uids, both=True)
        acute_uids, subcl_uids = acu_scl

        # Set prepatent duration of those who will become acute
        self.ti_acute[acute_uids] = ti + dur_pre[np.isin(uids, acute_uids)]

        # Set prepatent duration of those who will become subclinical
        self.ti_subclinical[subcl_uids] = ti + dur_pre[np.isin(uids, subcl_uids)]

        # If treatment applied, this is when acute cases would seek treatment, relative
        # to the onset of acute stage. This variable captures human behaviour
        dur_wait = self.get_wait_duration(acute_uids)
        self.ti_seek_trtmnt[acute_uids] = self.ti_acute[acute_uids] + dur_wait

        # Chronic/carrier stage: Determine who becomes a (chronic) carrier from acute and sublclinical
        acu2chro_uids = self.will_become_chronic_carrier(acute_uids)
        scl2chro_uids = self.will_become_chronic_carrier(subcl_uids)

        dur_acu = self.get_acute_duration_by_age(acu2chro_uids)
        self.ti_chronic[acu2chro_uids] = self.ti_acute[acu2chro_uids] + dur_acu
        dur_scl = self.get_subclinical_duration_by_age(scl2chro_uids)
        self.ti_chronic[scl2chro_uids] = self.ti_subclinical[scl2chro_uids] + dur_scl

        # Determine which carriers will recover and when
        # See: https://github.com/starsimhub/typhoidsim/issues/66
        chro2rec_uids = acu2chro_uids[p.p_rec.rvs(acu2chro_uids)].concat(scl2chro_uids[p.p_rec.rvs(scl2chro_uids)])
        dur_chro = self.get_chronic_duration(chro2rec_uids)
        self.ti_recovered[chro2rec_uids] = self.ti_chronic[chro2rec_uids] + dur_chro

        # Death:        # From the acutes who do not become carriers, determine who recovers and who dies
        can_die_uids = np.setdiff1d(acute_uids, acu2chro_uids)
        will_die = p.p_death.rvs(can_die_uids)
        acu2rec_uids = can_die_uids[~will_die]

        # Recovery: Get sublinical cases that recover because they won't become carriers
        scl2rec_uids = np.setdiff1d(subcl_uids, scl2chro_uids)
        will_recover_uids = acu2rec_uids.concat(scl2rec_uids).concat(chro2rec_uids)

        # Determine when non-carriers who recover will become susceptible again,
        dur_acu = self.get_acute_duration_by_age(acu2rec_uids)
        dur_scl = self.get_subclinical_duration_by_age(scl2rec_uids)
        self.ti_recovered[acu2rec_uids] = self.ti_acute[acu2rec_uids]       + dur_acu
        self.ti_recovered[scl2rec_uids] = self.ti_subclinical[scl2rec_uids] + dur_scl

        # NOTE: typhoid can get very low mortality (in particular with treatment),
        # so there is a high chance of getting empty dead_uids. If that happens,
        # the line below may seg fault (or maybe that was bad luck).
        # Just in case check first.
        dead_uids = can_die_uids[will_die]
        if dead_uids.size:
            dur_acu = self.get_acute_duration_by_age(dead_uids)
            self.ti_dead[dead_uids] = self.ti_acute[dead_uids] + dur_acu

        dur_rec = self.get_recovered_duration(will_recover_uids)
        self.ti_susceptible[will_recover_uids] = self.ti_recovered[will_recover_uids] + dur_rec
        return

    #  Transmission-realated methods - interaction between agents and "else" (other agents)
    #  or the environment
    def step(self):
        """
        Handle transmission of pathogens and who becomes infected,
        includes all transmission routes. This method is called by the Sim object.
        """
        # NOTE: mostly for tracking purposes at the moment
        self.eff_sus[:] = self.rel_sus * self.susceptibility * (1.0 - self.immunity_acquired)
        self.make_new_cases_sequential()
        return

    def make_new_cases_sequential(self):
        """
        This function exists to allow for testing different mechanisms
        that handle multiroute transmission.
        """
        self.make_new_cases_contact()
        self.make_new_cases_environmental()
        return

    def make_new_cases_contact(self):
        """
        Add new cases of module, through transmission, incidence, etc.
        Common-random-number-safe transmission code works by mapping edges onto
        slots.

        The direct or contact transmission route. The probability of infection
        is calculated for each infector-infectee pair (not summed across all
        possible exposures an infectee experiences).

        See diagram for Model A in https://github.com/starsimhub/typhoidsim/issues/90

        Implicitly, at each time step, the number of "exposures" for a src-trg
        pair is equal to 1.

        In starsim we use beta_per_dt, the per time step probability that a
        contact between two susceptible people results in transmission.

        However, historically in EMOD-world this per-time-step probability
        is called exposure2contact_rate. For time being, and to faciliate mapping
        between past EMOD-based simulations, and starsim-based sims, we enforce
        beta_per_dt=1, and keep exposure2contct_rate. Eventually,
        we can move onto just using beta_per_dt.
        """
        new_cases = []
        sources = []
        networks = []
        betamap = self._check_betas()
        dt = self.t.dt

        # TODO: Handle in disease module and then use standard starsim transmission
        rel_sus   = self.rel_sus.asnew(self.susceptible * self.rel_sus) \
                    * self.susceptibility * (1.0 - self.immunity_acquired)
        rel_trans = self.rel_trans.asnew(self.infectious * self.rel_trans) \
                    * self.infectiousness / self.tai * self.pars.transmission.exposure2contact_rate / tyd.day2year * dt

        for i, (nkey,route) in enumerate(self.sim.networks.items()):
            nk = ss.standardize_netkey(nkey)
            if isinstance(route, (ss.MixingPool, ss.MixingPools)):
                target_uids = route.compute_transmission(rel_sus, rel_trans, betamap[nk])
                new_cases.append(target_uids)
                sources.append(np.full(len(target_uids), dtype=ss.dtypes.float, fill_value=np.nan))
                networks.append(np.full(len(target_uids), dtype=ss.dtypes.int, fill_value=i))
            elif isinstance(route, ss.Network) and len(route): # Skip networks with no edges
            
                # standard network keys above don't include net, but betamap expects net suffix
                # need to standardize around one or the other - hacky fix for now
                if 'net' not in nk:
                    nk += 'net'  # Add 'net' suffix if not already there
                
                edges = route.edges
                p1p2b0 = [edges.p1, edges.p2, betamap[nk][0]] # Person 1, person 2, beta 0
                p2p1b1 = [edges.p2, edges.p1, betamap[nk][1]] # Person 2, person 1, beta 1
                for src, trg, beta in [p1p2b0, p2p1b1]:
                    if beta: # Skip networks with no transmission
                        # beta_per_dt = route.net_beta(disease_beta=beta) # Compute beta for this network and timestep
                        # randvals = ss.multi_random().rvs() # Generate a new random number based on the two other random numbers
                        # args = (src, trg, rel_trans, rel_sus, beta_per_dt, randvals) # Set up the arguments to calculate transmission
                        # target_uids, source_uids = self.compute_transmission(*args) # Actually calculate it

                        # the communitynet in tyohpidsim does not behave like typical ss Route
                        # therefore calculating transmission here rather than through compute_transmission
                        self.p_resp[trg] = rel_trans[src]

                        # INFECTION OUTCOME: Decide who gets infected/
                        # self.pars.transmission.ppl2ppl_p_inf will calculate p_infc = rel_sus[trg] * susceptibility[trg] * self.p_resp[trg]
                        new_cases_bool = self.pars.transmission.ppl2ppl_p_inf(trg)
                        
                        new_cases.append(trg[new_cases_bool])
                        sources.append(src[new_cases_bool])
                        networks.append(np.full(np.count_nonzero(new_cases_bool), dtype=ss_int_, fill_value=i))

        new_cases = ss.uids.cat(new_cases)
        if len(new_cases):
            # "Adjust" cfu_dose of agents who become infected at this time step.
            # This will guarantee the high dose mu/sigma parameters for prepatent duration
            # From EMOD: Currently, all infections from the Contact route are assumed to be a
            # high dose prepatent duration, meaning that the characteristic dose a
            # target agent receives has to be set to be a value larger than self.pars.cfu_me_hi
            self.cfu_dose[new_cases] = self.pars.cfu_me_hi + 0.1 * self.pars.cfu_me_hi
            self.set_prognoses(new_cases, sources=None)
            self.progress_to_prepatent(self.sim.ti)  # Incubation period
            self.infc_origin[new_cases] = tyd.TransmissionRoute.CONTACT.value

        return new_cases, sources, networks

    def make_new_cases_environmental(self):
        """
        At each time step:
        1. Individuals get exposed by the environment (env->ppl)
            - They receive a cfu dose, which depends on:
                 - the cfu concentration in the environment, and
                 - the exposures amouint to the environment.

        2. New individuals may become infected, based on the cfu dose received
            and their past history with the disease. This is mediated by the
            Dose Response Curve. See self.drc(),

        3. Individuals shed bacteria/CFU to the environment. How many CFUs are
            shedded depends on two main factors:
                - shedding rate, a single factor that depends on the environment
                   as it represents level of sanitation and/or collective
                   change in behaviour.
                - each agent's infectiousness, which can be modulated by
                treatment interventions.
        """
        if not self.pars.has_environment:
            return []
        ti = self.ti
        dt = self.t.dt
        n_agents = np.count_nonzero(self.sim.people.alive) # current N, number of agents currently alive
        exposure_volume = 1.0 / n_agents  # CFUs in the environment need to be proportionally distributed among agents, so we don't have the case where number of CFUs received by agents > CFU's available in the environment
        environment = self.sim.demographics['environmentalpool']
        env_trans_pars = environment.pars.transmission
        trans_pars = self.pars.transmission


        # Determine who gets infected from environment. Multiply by rel_sus, as many interventions will target this parameter
        # This means an agent can become unsusceptible because of an external factor.
        susc = self.susceptible.asnew(self.susceptible * self.rel_sus * self.susceptibility * (1.0-self.immunity_acquired))
        susc_uids = (susc).uids

        # EXPOSURE: Increase cfu doses in susceptible people by exposing them to the environment
        # NOTE: left exposure amount and n_exposures for interpretability, they should be identical
        # TODO: check that we need to redraw the value every day
        teer = trans_pars.teer.rvs(susc_uids.size)
        # update lambda, now one value per agent
        environment.pars.transmission.env2ppl_exposure_rate.pars.lam = teer
        self.exposure_amount[susc_uids] = ((environment.pars.transmission.env2ppl_exposure_rate.rvs(susc_uids.size) / tyd.day2year) * dt)  # exposure amount expressed in (n_exposures * volume) on the time interval "dt"
        self.n_exposures[susc_uids] = self.exposure_amount[susc_uids] / environment.pars.volume  # env volume=1. Units of n_exposures [counts] =  (counts * volume) / volume

        self.cfu_dose_per_exposure[susc_uids] = environment.sv.cfu_conc[ti - 1] * exposure_volume * env_trans_pars.rel_trans   # Units exposure_amount [# of pathogens] =  cfu_conc [pathogens/volume] * (n_exposures * volume) --> total pathogens
        # INFECTION: The distribution trans_pars.env2ppl_p_inf(p=fun()), where fun() is
        # infection_prob_function(), which calls self.drc(). This assesses
        # the immunity responses of the hosts (drc) due to a certain amount of
        # exposure doses (cfu_dose). Then, infection_...() it estimates
        # a probability of infection.
        got_infected = trans_pars.env2ppl_p_inf(susc_uids)
        new_cases = susc_uids[got_infected]
        if len(new_cases):
            # Set the level of cfu_dose, as this is used to determine the parameters of the distribution that sets prepatent duration of new cases
            self.cfu_dose[new_cases] = self.cfu_dose_per_exposure[new_cases]
            self.set_prognoses(new_cases, sources=None)
            self.progress_to_prepatent(self.sim.ti)  # Incubation period
            self.infc_origin[new_cases] = tyd.TransmissionRoute.ENVIRONMENT.value

        # SHEDDING: Transmission people->environment:
        #     Infectious individuals shed contagion into the contagion pool.
        #     Reduction in shedding can happen due to per-agent interventions (reduces individual level of infectiousness),
        #     or due to sanitation interventions ().
        # TODO: shedding would be interpreted as shedding per unit volume
        effective_shedding = ((environment.pars.transmission.shedding_rate / tyd.day2year) * dt)   # transform to yearly rate, then multiply by dt to get the effective shedding on the time interval dt in change/volume
        shedded_cfu = (self.rel_trans[self.infected] * self.infectiousness[self.infected]).sum()   # number of CFUs
        current_level = environment.sv.cfu_conc[ti] * environment.pars.volume + shedded_cfu * effective_shedding

        # CFU level increases due to people shedding into the environment
        environment.sv.cfu_conc[ti] = current_level / environment.pars.volume
        environment.sv.cfu_conc_buffer[(ti+1) % environment.buffer_isteps] = environment.sv.cfu_conc[ti]
        return new_cases
    @staticmethod
    def infection_prob_function_env(module, sim, uids):
        """
        Calculate the probability of infection for environmental route
        In EMOD (https://github.com/jgauld/DtkTrunk/blob/Typhoid-Ongoing/Eradication/IndividualTyphoid.cpp):
        NonNegativeFloat infects = 1.0f-pow( 1.0f + exposure * ( pow( 2.0f, (1/alpha) ) -1.0f )/N50, -alpha ); // Dose-response for prob of infection
        prob = 1.0f - pow(1.0f - immunity * infects * ira, number_of_exposures);
        """
        # Evoke an immunity-like response
        p_resp = module.drc(module.cfu_dose_per_exposure[uids])
        effective_susceptibility = module.rel_sus[uids] * module.susceptibility[uids] * (1.0 - module.immunity_acquired[uids])  # effective susceptibility due to relative changes (interventions), naturally acquired immunity and vaccine-acquired immunity
        p_infc = 1.0 - (1.0 - effective_susceptibility * p_resp) ** module.n_exposures[uids]  # total number of n_exposures per unit of time
        return np.array(p_infc)

    @staticmethod
    def infection_prob_function_contact(module, sim, uids):
        p_infc = module.rel_sus[uids] * module.susceptibility[uids] * (1.0 - module.immunity_acquired[uids]) * module.p_resp[uids]
        return np.array(p_infc)

    @staticmethod
    def inf_dur_mean(module, sim, uids):
        """_
        Age-dependent mean of the distribution of durations of the
        acute or subclinical stage. Assumes a lognormal_ex distribution.
        """
        if uids is None:
            mean_arr = np.array([])
        else:
            th_age = module.pars.inf_dur_th_age
            mean_arr = np.ones(1 if isinstance(uids, int) else uids.size)
            mean_arr[sim.people.age[uids] < th_age]  = module.pars.inf_dur_mean_le
            mean_arr[sim.people.age[uids] >= th_age] = module.pars.inf_dur_mean_geq
        return mean_arr

    @staticmethod
    def inf_dur_std(module, sim, uids):
        """
        Age-dependent standard deviation of the distribution of durations of the
        acute or subclinical stage. Assumes a lognormal_ex distribution.
        """
        if uids is None:
            std_arr = np.array([])
        else:
            th_age = module.pars.inf_dur_th_age
            std_arr = np.zeros(1 if isinstance(uids, int) else uids.size)
            std_arr[sim.people.age[uids] < th_age]  = module.pars.inf_dur_std_le
            std_arr[sim.people.age[uids] >= th_age] = module.pars.inf_dur_std_geq
        return std_arr

    @staticmethod
    def prior_infections_by_age(module, sim, uids):
        """
        TODO: [WIP] palceholder - the functional form is just a toy example
        Function to modulate probability of prior n_infections by age at t=0,
        but would need to discuss functional form/how to parameterize.
        """
        if uids is None:
            lam_arr = np.array([])
        else:
            max_lam = 1.0 # average value (lambda) of prior n_infections as age -> infty
            lam_arr = tyum.sigmoid(sim.people.age[uids], tyd.max_age, 1.0)*max_lam
        return lam_arr

    def update_susceptibility(self, uids):
        """
        Susceptibility die to acquired immunity following infection.
        The more infections, the lower the number.
        """
        # TPPI: Typhoid Protection Per Infection
        self.susceptibility[uids] = (1.0 - self.tppi[uids])**self.n_infections[uids]
        # NOTE: We could add a mechanisms for immunity waning here
        return

    def drc(self, cfu_dose):
        """
        The probability of infection due to environmental exposure is mediated by
        the dose-response curve (drc), taking in the contagion population as a
        value of colony-forming units (CFU)
        and returning a probability of infection.

        Here `cfu_dose` is the CFU dose received from the environment.

        The DRC is a beta-binomial curve fitted the historical challenge
        data by QMRA (Enger, 2013), where:

        P(response) = 1- [1 + cfu_dose * (2^(1/ α)- 1)/N50] ^(-α)
        and self.drc() could be a method to be defined by the user.
        """
        p_response = 1.0 - (1.0 + cfu_dose * ((2.0**(1.0/self.pars.drc_alpha) - 1.0)/self.pars.drc_n50))**(-self.pars.drc_alpha)
        return p_response

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.ti
        n = np.count_nonzero(self.sim.people.alive)
        res.prevalence[ti] = res.n_infected[ti] / n
        res.new_infections[ti] = np.count_nonzero(self.ti_prepatent == ti)
        if ti == 0:
            res.new_infections[ti] += np.count_nonzero(self.ti_prepatent == -2)  # Count initial cases
        res.cum_infections[ti] = np.sum(res['new_infections'][:ti+1])
        res.new_susceptible[ti] = np.count_nonzero(self.ti_susceptible == ti)
        res.new_prepatent[ti] = np.count_nonzero(self.ti_prepatent == ti)
        res.new_acute[ti] = np.count_nonzero(self.ti_acute == ti)
        res.cum_acute[ti] = np.sum(res['new_acute'][:ti+1])
        res.new_subclinical[ti] = np.count_nonzero(self.ti_subclinical == ti)
        res.new_chronic[ti] = np.count_nonzero(self.ti_chronic == ti)
        res.new_recovered[ti] = np.count_nonzero(self.ti_recovered == ti)
        res.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        res.cum_deaths[ti] = np.sum(res['new_deaths'][:ti+1])

        return


def unexp2sus_childhood_prob_function_gauld2018(module, sim, uids):
    """
    Estimate the age-dependent probability of transitioning from
    unexposed to susceptible. From Gauld et al 2018, Fig. 2B.

    Age-specific immunity. Individuals that are created through births in the model
    start out in a fully immune state. For the Santiago-site simulation,
    there are three ages that individuals can move from immune (unexposed)
    to susceptible:
     - 6 months (no children under 6 months of age can be infected).
         - Need to define the distribution parameter
             p_unexp2sus_6m=ss.bernoulli(p=0.14)
     - 3 years
         - Need to define the distribution parameter
             p_unexp2sus_3y=ss.bernoulli(p=0.29),
     - 6 years
         - Need to define the distribution parameter
            p_unexp2sus_6y=ss.bernoulli(p=0.61)

     At each of these ages, a proportion of the remaining unexposed
     population will be moved to the susceptible population determined
     by the calibrated values below

    Args:
        module: a starsim (disease) Module
        sim: the starsim Sim object (fully initialized)
        uids: the uids of the eligible people

    Returns:
        p_sus (array): array of probabilities for every agent in uids.
    """

    p_sus = np.zeros(len(uids))

    # Santiago de Chile cas
    _6m = 0.5  # age in years, assumes end of maternal antibodies at 6 months
    _3y = 3.0  # age in years
    _6y = 6.0  # age in years

    # Detect whether people have reached/crossed their 'age' anniversaries
    became_6m = tyu.detect_age_anniversary(sim, _6m)
    became_3y = tyu.detect_age_anniversary(sim, _3y)
    became_6y = tyu.detect_age_anniversary(sim, _6y)
    p_sus[became_6m] = module.pars.p_unexp2sus_6m(uids[became_6m]).astype(float)
    p_sus[became_3y] = module.pars.p_unexp2sus_3y(uids[became_3y]).astype(float)
    p_sus[became_6y] = module.pars.p_unexp2sus_3y(uids[became_6y]).astype(float)
    return p_sus
