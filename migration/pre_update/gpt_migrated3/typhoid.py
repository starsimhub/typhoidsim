
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
            # Initial conditions and transmissibility beta
            init_prev=ss.bernoulli(p=0.005),

            # NATURAL HISTORY PARAMETERS
            # From never exposed/invulnerable to susceptible
            p_unexp2sus_6m=ss.bernoulli(p=0.14),
            p_unexp2sus_3y=ss.bernoulli(p=0.29),
            p_unexp2sus_6y=ss.bernoulli(p=0.61),

            # Default mechanism to move agents from never exposed/invulnerable to susceptible to exposure
            p_unexp2sus=ss.bernoulli(p=self.unexp2sus_prob_gauld2018),
            unexp2sus_saturation_age=ss.years(20.0),
            unexp2sus_slope=1.0,

            # Prepatent stage, the parameters of the distribution of durations is CFU-dose dependent
            prep_dur_dpars=tyu.load_dataset("prepatent_dur_dist_pars"),
            prep_dur_fun=tyum.double_sigmoid_tanh,
            dur_prep_dist=ss.lognorm_ex(mean=self.prepatent_mean_dur_function,
                                        std=self.prepatent_std_dur_function),

            cfu_lo_me=5_050_000.0,
            cfu_me_hi=55_000_000.0,

            # Infected/Diseased stage, (acute and sublinical)
            p_acute=ss.bernoulli(p=0.16),
            inf_dur_th_age=ss.years(30.0),
            inf_dur_mean_le=ss.years(1.172 * tyd.days_per_week),
            inf_dur_std_le=ss.years(0.483 * tyd.days_per_week),
            inf_dur_mean_geq=ss.years(1.172 * tyd.days_per_week),
            inf_dur_std_geq=ss.years(0.788 * tyd.days_per_week),
            dur_inf_dist=ss.lognorm_ex(mean=self.inf_dur_mean,
                                       std=self.inf_dur_std),
            dur_wait2treatment=ss.lognorm_ex(mean=ss.years(2.33219066 * tyd.day2year), std=0.5430),

            # Long-term stages
            # Chronic
            p_cpg=0.15,
            d_chro=ss.bernoulli(p=self.chronic_gall_prob_function),
            p_gall=tyu.load_dataset("gallstone_probs"),
            p_rec=ss.bernoulli(p=0.0),
            dur_chro_dist=ss.constant(v=ss.years(102.0 * tyd.days_per_week * tyd.day2year)),

            # Recovered
            dur_rec_dist=ss.constant(v=ss.years(1.0 * tyd.day2year)),

            # Death
            p_death=ss.bernoulli(p=0.01),

            # IMMUNE SYSTEM-WITHIN HOST PARAMETERS
            # Infectiousness parameters
            tai=40_000.0,
            tpri=0.5,
            tsri=1.0,
            tcri=0.241,
            tppi=0.98,
            drc_alpha=0.175,
            drc_n50=1.11e6,

            # ENVIRONMENT PARAMETERS
            has_environment=None,
            # Tranmission parameters
            transmission=ss.Pars(
                env2ppl_p_inf=ss.bernoulli(p=self.infection_prob_function_env),
                exposure2contact_rate=1.0,
                ppl2ppl_p_inf=ss.bernoulli(p=self.infection_prob_function_contact),
                p_route=ss.uniform()
            ),

             beta=None,
        )
        self.update_pars(pars, **kwargs)

        # Parametrisation of prepatent duration distribution parameters
        self.partial_prep_dur_mean,  self.partial_prep_dur_std = self.prepare_partial_prep_funs()

        # Boolean states
        self.define_states(
            ss.State("susceptible", default=False, label="Susceptible"),
            ss.State("infected", default=False, label="Infectious"),
            ss.State("unexposed", default=True, label="Unexposed"),
            ss.State("prepatent", default=False, label="Prepatent"),
            ss.State("acute", default=False, label="Acute"),
            ss.State("subclinical", default=False, label="Subclinical"),
            ss.State("chronic", default=False, label="Chronic"),
            ss.State("recovered", default=False, label="Recovered/Been Infected"),
            ss.State("infected_ever", default=False, label="Ever Infected"),

            ss.FloatArr("n_exposures", default=0.0, label="Number of Exposures"),
            ss.FloatArr("exposure_amount", default=0.0, label="Number of Exposures"),
            ss.FloatArr("cfu_dose_per_exposure", default=0.0, label="Single environmental exposure amount (CFUs)"),
            ss.FloatArr("cfu_dose", default=0.0, label="Exposure amount (CFUs)"),
            ss.FloatArr("infectiousness", 0.0, label="Infectiousness"),
            ss.FloatArr("n_infections", 0.0, label="Number of Infections"),
            ss.FloatArr("susceptibility", default=1.0, label="Susceptibility Level"),
            ss.FloatArr("p_resp", default=0.0, label="Probability of responset to infection"),
            ss.FloatArr("p_infc", default=0.0, label="Probability of Infection"),
            ss.FloatArr("p_route", default=0.0, label="Probability Route Draw"),
            ss.FloatArr("infc_origin", label="Origin of infection"),

            ss.FloatArr("rel_sus", default=1.0, label="Relative susceptibility"),
            ss.FloatArr("rel_trans", default=1.0, label="Relative transmissibility"),

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

        # Define random number generators for make_new_cases
        self.trans_rng = ss.multi_random('source', 'target')

        return

    def init_pre(self, sim):
        """ Initialise objects and validate before simulation run"""
        super().init_pre(sim)
        self.validate_beta()
        self.validate_environment()
        return

    def validate_beta(self):
        """
        Perform any parameter validation
        """
        networks = self.sim.networks
        if networks is not None and len(networks) > 0:

            if 'beta' not in self.pars:
                errormsg = f'Disease {self.name} is missing beta; pars are: {sc.strjoin(self.pars.keys())}'
                raise sc.KeyNotFoundError(errormsg)

            if self.pars.beta is None:
                self.pars.beta = ss.time_ratio(unit1='year', dt1=1.0, unit2=self.sim.t.unit, dt2=self.sim.t.dt, as_int=False)

            self.validate_beta(run_checks=True)
        return

    def init_post(self):
        """
        Set initial values for all agent states, and seed new cases if initial
        prevalence is provided.
        """
        self.make_susceptible()

        if self.pars.init_prev is not None:
            initial_cases = self.pars.init_prev.filter()
            self.set_prognoses(initial_cases)
        self.progress_to_prepatent(self.sim.ti)
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
            ss.Result("new_susceptible", dtype=int, scale=True, label="Newly Susceptible"),
            ss.Result("new_prepatent", dtype=int, scale=True, label="Newly Prepatent"),
            ss.Result("new_acute", dtype=int, scale=True, label="Newly Acute"),
            ss.Result("cum_acute", dtype=int, scale=True, label="Cumulative Acute"),
            ss.Result("new_subclinical", dtype=int, scale=True, label="Newly Subclinical"),
            ss.Result("new_chronic", dtype=int, scale=True, label="Newly Chronic"),
            ss.Result("new_recovered", dtype=int, scale=True, label="Newly Recovered"),
            ss.Result("new_deaths", dtype=int, scale=True, label="Newly Dead"),
            ss.Result("cum_deaths", dtype=int, scale=True, label="Cumulative Dead"),
        )
        return

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
        """
        never_exposed = (self.unexposed).uids
        self.susceptible[never_exposed] = self.pars.p_unexp2sus.rvs(never_exposed)
        self.unexposed[never_exposed] = ~self.susceptible[never_exposed]
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
        self.statesdict["unexposed"][uids] = False
        return

    # Update progression of disease, handle transitions
    def step_state(self):
        """
        Update the progression of the disease -- handles disease state transitions.
        """
        ti = self.sim.ti

        self.make_susceptible()

        # The infection life cycle or natural history flow
        self.progress_to_prepatent(ti)
        self.progress_to_diseased(ti)
        self.progress_to_chronic(ti)
        self.progress_to_recovered(ti)
        self.progress_to_susceptible(ti)
        self.progress_to_dead(ti)
        return

    # Methods that handle transitions between states
    def progress_to_prepatent(self, ti):
        susc2prep = (self.prepatent & (self.ti_prepatent <= ti)).uids
        self.unexposed[susc2prep] = False
        self.susceptible[susc2prep] = False
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
        self.infectiousness[sub2chro] = self.pars.tai * self.pars.tcri

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
        self.update_immunity(rec2suc)

    # Methods that handle durations/duration pars that are dependent on other variables/states
    def get_prepatent_duration_by_exposure(self, uids):
        """ Get durations in number of timesteps"""
        dt = self.sim.t.dt
        dur_prep = self.pars.dur_prep_dist.rvs(uids.size).astype(float)  # in days
        dur_prep = dur_prep * tyd.day2year  # in years
        return sc.randround(dur_prep / dt)  # in number of timesteps

    def get_acute_duration_by_age(self, uids):
        """
        Duration of the acute stage
        """
        p = self.pars
        dt = self.sim.t.dt
        dur_acu = p.dur_inf_dist.rvs(uids.size) * tyd.days_per_week  # in days
        dur_acu = dur_acu * tyd.day2year  # in years
        return sc.randround(dur_acu / dt)  # in number of timesteps

    def get_subclinical_duration_by_age(self, uids):
        """
        Determine duration of the sublinical stage
        """
        p = self.pars
        dt = self.sim.t.dt
        dur_scl = p.dur_inf_dist.rvs(uids.size) * tyd.days_per_week  # in days
        dur_scl = dur_scl * tyd.day2year
        return sc.randround(dur_scl / dt)

    def get_wait_duration(self, uids):
        """
        Determine how many days a person in the acute stage would wait before
        seeking treatment
        """
        p = self.pars
        dt = self.sim.t.dt
        dur_wait = p.dur_wait2treatment.rvs(uids).astype(float)
        dur_wait = dur_wait * tyd.day2year
        return sc.randround(dur_wait / dt)

    def get_recovered_duration(self, uids):
        p = self.pars
        dt = self.sim.t.dt
        dur_rec = p.dur_rec_dist.rvs(uids.size)  # duration in days
        dur_rec = dur_rec * tyd.day2year                  # duration in years
        return sc.randround(dur_rec / dt)        # duration in integer number of timesteps

    def get_chronic_duration(self, uids):
        """
        Determine duration of chronic stage
        See: https://github.com/starsimhub/typhoidsim/issues/66
        """
        p = self.pars
        dt = self.sim.t.dt
        dur_chro = p.dur_chro_dist.rvs(uids.size) * tyd.days_per_week  # duration in in days
        dur_chro = dur_chro * tyd.day2year        # duration in years
        return sc.randround(dur_chro / dt)        # duration in integer number of timesteps

    @staticmethod
    def prepatent_mean_dur_function(module, sim, uids):
        """
        Returns a mean duration parameter for every agent based on the cfu_dose,
        they have been exposed to. Assumes the parameter will be used by a
        lognormal_ex distribution.
        """
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
        """
        sat_age = module.pars.unexp2sus_saturation_age.to('year').values
        slope = module.pars.unexp2sus_slope
        p2 = tyum.sigmoid(sim.people.age[uids], sat_age, slope)
        p1 = tyum.sigmoid(sim.people.age[uids] - sim.t.dt_year, sat_age, slope)
        if sim.ti == 0:
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
        """
        p = self.pars
        ti = self.sim.ti
        dt = self.sim.t.dt

        # Set value of states associated to being infected, and record events
        self.susceptible[uids] = False
        self.unexposed[uids] = False
        self.infected[uids] = True
        self.infected_ever[uids] = True
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
        """
        new_cases = []
        sources = []
        networks = []
        betamap = self.validate_beta()
        dt = self.sim.t.dt

        for i, (nkey, net) in enumerate(self.sim.networks.items()):
            if not len(net):
                break

            nbetas = betamap[nkey]
            edges = net.edges

            # Relative Transmissibility: Relevant for sources
            rel_trans = self.rel_trans.asnew(self.infectious * self.rel_trans)

            p1p2b0 = [edges.p1, edges.p2, nbetas[0]]
            for src, trg, beta in [p1p2b0]:

                if beta == 0:
                    continue

                self.p_resp[trg] = (self.infectiousness[src] / self.pars.tai) * rel_trans[src] * ((self.pars.transmission.exposure2contact_rate / tyd.day2year) * dt)

                new_cases_bool = self.pars.transmission.ppl2ppl_p_inf(trg)

                new_cases.append(trg[new_cases_bool])
                sources.append(src[new_cases_bool])
                networks.append(np.full(np.count_nonzero(new_cases_bool), dtype=ss_int_, fill_value=i))

        if len(new_cases) and len(sources):
            new_cases = ss.uids.cat(new_cases)
            new_cases, inds = new_cases.unique(return_index=True)
            sources = ss.uids.cat(sources)[inds]
            networks = np.concatenate(networks)[inds]
        else:
            new_cases = np.empty(0, dtype=int)
            sources = np.empty(0, dtype=int)
            networks = np.empty(0, dtype=int)

        if len(new_cases):
            self.cfu_dose[new_cases] = self.pars.cfu_me_hi + 0.1 * self.pars.cfu_me_hi
            self.set_prognoses(new_cases, source_uids=None)
            self.progress_to_prepatent(self.sim.ti)
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
        ti = self.sim.ti
        dt = self.sim.t.dt
        n_agents = self.sim.pars["n_agents"]
        exposure_volume = 1.0 / n_agents
        environment = self.sim.demographics['environmentalpool']
        env_trans_pars = environment.pars.transmission
        trans_pars = self.pars.transmission

        susc = self.susceptible.asnew(self.susceptible * self.rel_sus)
        susc_uids = (susc).uids

        self.exposure_amount[susc_uids] = ((environment.pars.transmission.env2ppl_exposure_rate.rvs(susc_uids.size) / tyd.day2year) * dt)
        self.n_exposures[susc_uids] = self.exposure_amount[susc_uids] / environment.pars.volume

        self.cfu_dose_per_exposure[susc_uids] = environment.sv.cfu_conc[ti - 1] * exposure_volume * env_trans_pars.rel_trans

        got_infected = trans_pars.env2ppl_p_inf(susc_uids)
        new_cases = susc_uids[got_infected]
        if len(new_cases):
            self.cfu_dose[new_cases] = self.cfu_dose_per_exposure[new_cases]
            self.set_prognoses(new_cases, source_uids=None)
            self.progress_to_prepatent(ti)
            self.infc_origin[new_cases] = tyd.TransmissionRoute.ENVIRONMENT.value

        effective_shedding = ((environment.pars.transmission.shedding_rate / tyd.day2year) * dt)
        shedded_cfu = (self.rel_trans[self.infected] * self.infectiousness[self.infected]).sum()
        current_level = environment.sv.cfu_conc[ti] * environment.pars.volume + shedded_cfu * effective_shedding

        environment.sv.cfu_conc[ti] = current_level / environment.pars.volume
        environment.sv.cfu_conc_buffer[(ti+1) % environment.buffer_isteps] = environment.sv.cfu_conc[ti]
        return new_cases

    @staticmethod
    def infection_prob_function_env(module, sim, uids):
        """
        Calculate the probability of infection for environmental route
        """
        p_resp = module.drc(module.cfu_dose_per_exposure[uids])
        p_infc = 1.0 - (1.0 - module.rel_sus[uids] * module.susceptibility[uids] * p_resp) ** module.n_exposures[uids]
        return np.array(p_infc)

    @staticmethod
    def infection_prob_function_contact(module, sim, uids):
        p_infc = module.rel_sus[uids] * module.susceptibility[uids] * module.p_resp[uids]
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
            th_age = module.pars.inf_dur_th_age.to('year').values
            mean_arr = np.ones(1 if isinstance(uids, int) else uids.size)
            mean_arr[sim.people.age[uids] < th_age]  = module.pars.inf_dur_mean_le.to('year').values
            mean_arr[sim.people.age[uids] >= th_age] = module.pars.inf_dur_mean_geq.to('year').values
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
            th_age = module.pars.inf_dur_th_age.to('year').values
            std_arr = np.zeros(1 if isinstance(uids, int) else uids.size)
            std_arr[sim.people.age[uids] < th_age]  = module.pars.inf_dur_std_le.to('year').values
            std_arr[sim.people.age[uids] >= th_age] = module.pars.inf_dur_std_geq.to('year').values
        return std_arr

    def update_immunity(self, uids):
        """
        Susceptibility due to acquired immunity following infection.
        The more infections, the lower the number.
        """
        self.susceptibility[uids] = (1.0 - self.pars.tppi)**self.n_infections[uids]
        return

    def drc(self, cfu_dose):
        """
        The probability of infection due to environmental exposure is mediated by
        the dose-response curve (drc), taking in the contagion population as a
        value of colony-forming units (CFU)
        and returning a probability of infection.
        """
        p_response = 1.0 - (1.0 + cfu_dose * ((2.0**(1.0/self.pars.drc_alpha) - 1.0)/self.pars.drc_n50))**(-self.pars.drc_alpha)
        return p_response

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.sim.ti
        n = np.count_nonzero(self.sim.people.alive)
        res.prevalence[ti] = res.n_infected[ti] / n
        res.new_infections[ti] = np.count_nonzero(self.ti_infected == ti)
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
    Estimate the age-dependent probability of transistioning from
    unexposed to susceptible. From Gauld et al 2018, Fig. 2B.
    """

    p_sus = np.zeros(len(uids))

    _6m = ss.years(0.5).values
    _3y = ss.years(3.0).values
    _6y = ss.years(6.0).values

    became_6m = _detect_age_anniversary(sim, _6m)
    became_3y = _detect_age_anniversary(sim, _3y)
    became_6y = _detect_age_anniversary(sim, _6y)
    p_sus[became_6m] = module.pars.p_unexp2sus_6m(uids[became_6m]).astype(float)
    p_sus[became_3y] = module.pars.p_unexp2sus_3y(uids[became_3y]).astype(float)
    p_sus[became_6y] = module.pars.p_unexp2sus_3y(uids[became_6y]).astype(float)
    return p_sus


def _detect_age_anniversary(sim, age_anniversary):
    """
    Detect people who crossed a specific age_anniversary, does not
    have to be birthday necesarrily.
    """
    reached_anniv = (((sim.people.age - sim.t.dt_year) < age_anniversary) &
                      (sim.people.age >= age_anniversary))
    return reached_anniv


def _detect_birthday(sim):
    """
    Detect who had their birthdays. Assumes the time step dt is less than
    or equal to 1.
    """
    prev_int_age = sim.people.age - sim.t.dt_year
    current_int_age = sim.people.age
    had_bday = (current_int_age.astype(int) - prev_int_age.astype(int)) == 1
    return had_bday


def stratify_parameter_by_age(age_bin_edges, par_bin_values):
    """
    Returns a callable that, given an age, returns the value of a parameter
    assigned to the age bin the given age falls into.

    Args:
        age_bin_edges (np.ndarray): The edges of the age bins. Should be in
            ascending order.
        par_bin_values (np.ndarray): The parameter values assigned to each age bin.
            Should be the of length age_bin_edges - 1.

    Returns:
        age_bin_function (callable): A function that takes an age
            and returns the parameter value for the bin that the age falls into.

    age_bin_edges = np.array([0, 2, 5, 120])
    par_bin_values = np.array([904.4, 240.9, 0.0])
    age_stratified_parameter = stratify_parameter_by_age(age_bin_edges, par_bin_values)
    age_stratified_parameter(25)  # should return 0.0
    """
    def age_stratified_parameter(age):
        bin_index = tyu.digitize_ages(age, age_bin_edges)
        return par_bin_values[bin_index]

    return age_stratified_parameter
