"""
Typhoid model.

"""

import numpy as np
import starsim as ss

__all__ = ['Typhoid']


class Typhoid(ss.Infection):
    """
    Typhoid
    """

    def __init__(self, pars=None, *args, **kwargs):
        """ Initialize with parameters """
        super().__init__()
        self.default_pars(
            # Initial conditions and beta
            beta=1.0,  # Placeholder value
            init_prev=ss.bernoulli(0.0),

            # Natural history parameters, all specified in days
            dur_exp2inf=ss.lognorm_ex(mean=0.0, stdev=0.0),
            dur_asymp2rec=ss.uniform(low=0.0, high=0.),
            dur_symp2rec=ss.lognorm_ex(mean=0.0, stdev=0.0),
            dur_symp2dead=ss.lognorm_ex(mean=0.0, stdev=0.0),
            p_death=ss.bernoulli(p=0.0),
            p_symp=ss.bernoulli(p=0.0),
            asymp_trans=0.00,
            p_chronic=0.0, # Chronic carrier
            p_temporary=0.0, # Temporary carrier

            # Environmental parameters
            beta_env=0.0,
            # Infectious dose in water sufficient to produce infection in X exposed
            shedding_rate=0.0,
            # Rate at which infectious people shed bacteria to the environment (per day),
            decay_rate=0.0,
            # Rate at which bacteria in the environment dies (per day),
            p_env_transmit=ss.bernoulli(p=0),
            # Probability of environmental transmission - filled out later
            long_ccvt=None,
        )
        self.update_pars(pars, **kwargs)

        # Boolean states
        self.add_states(
            # Susceptible & infected are added automatically, here we add the rest
            ss.BoolArr('exposed'),
            ss.BoolArr('asymptomatic'),
            ss.BoolArr('symptomatic'),
            ss.BoolArr('chronic'),
            ss.BoolArr('recovered'),

            # Timepoint states
            ss.FloatArr('ti_exposed'),
            ss.FloatArr('ti_symptomatic'),
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('ti_dead'),
        )
        return

    @property
    def infectious(self):
        return self.infected | self.exposed

    @property
    def asymptomatic(self):
        return self.infected & ~self.symptomatic


    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'new_deaths', npts, dtype=int),
            ss.Result(self.name, 'cum_deaths', npts, dtype=int),
            ss.Result(self.name, 'env_prevalence', npts, dtype=float),
            ss.Result(self.name, 'env_concentration', npts, dtype=float),
        ]
        return

    def update_pre(self):
        """
        """

        # Check if someone becomes susceptible in this timestep


        # Progress exposed -> infected
        ti = self.sim.ti  # current timestep


        infected = (self.exposed & (self.ti_infected <= ti)).uids

        self.infected[infected] = True

        # Progress infected -> symptomatic
        symptomatic = (self.infected & (self.ti_symptomatic <= ti)).uids
        self.symptomatic[symptomatic] = True

        # Progress symptomatic -> recovered
        recovered = (self.infectious & (self.ti_recovered <= ti)).uids
        self.exposed[recovered] = False
        self.infected[recovered] = False
        self.symptomatic[recovered] = False
        self.recovered[recovered] = True

        # Progress to chronic given gallstones, need to define what happens for
        # people without gallstones and in case the NCD has not een defiend.
        has_gallstones = (self.sim.people.gallstones.affected).uids
        self.chronic[has_gallstones] = self.pars.p_chronic.filter(has_gallstones)

        # Trigger deaths
        deaths = (self.ti_dead <= ti).uids
        if len(deaths):
            self.sim.people.request_death(deaths)

        self.environment.update()

        return

    def calculate_environmental_prevalence(self):
        """
        Calculate environmental prevalence long-cycle
        CCVT
        """
        p = self.pars
        r = self.results
        ti = self.sim.ti

        n_symptomatic  = self.symptomatic.sum()
        n_asymptomatic = self.asymptomatic.sum()
        old_prev = self.results.env_prev[ti - 1]

        new_bacteria = p.shedding_rate * (n_symptomatic + p.asymp_trans * n_asymptomatic)
        old_bacteria = old_prev * (1 - p.decay_rate)

        r.env_prevalence[ti] = new_bacteria + old_bacteria
        r.env_concentration[ti] = r.env_prev[ti] / (r.env_prev[ti] + p.half_sat_rate)
        return

    def set_prognoses(self, uids, source_uids=None):
        """ Set prognoses for those who get infected """
        super().set_prognoses(uids, source_uids)
        ti = self.sim.ti
        dt = self.sim.dt

        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = ti

        p = self.pars

        # Determine when exposed become infected

        # Determine who becomes symptomatic and when

        # Determine who dies and when

        # Determine when agents recover

        # Determine if someone becomes chronic carrier

        # Determine if someone becomes temporary carrier

        return

    def make_new_cases(self):
        """ Add short-cycle transmission and long-cycle transmission transmission """
        # Make new cases via person-to-person transmission
        super().make_new_cases()

        new_cases = self._environmental_transmission()
        #new_cases = environmental_transmission(self.sim.people, self, self.pars.long_ccvt, self.sim.ti)

        if new_cases.any():
            self.set_prognoses(new_cases, source_uids=None)
        return

    def _environmental_transmission(self):
        # Make new cases via indirect transmission
        pars = self.pars
        res = self.results
        p_transmit = res.env_conc[self.sim.ti] * pars.beta_env
        pars.p_env_transmit.set(p=p_transmit)
        new_cases = pars.p_env_transmit.filter(self.sim.people.uid[self.susceptible])
        return new_cases


    def update_death(self, uids):
        """ Reset states for dead agents """
        for state in ['susceptible', 'exposed', 'infected', 'symptomatic', 'recovered']:
            self.statesdict[state][uids] = False
        return

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.sim.ti
        res.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        res.cum_deaths[ti] = np.sum(res.new_deaths[:ti + 1])
        return


    def make_new_cases_long_cycle(self):
        pass


    def make_susceptible(self):
        pass




def environmental_transmission(people, disease, contaminated_environment, current_ti):
    """
    Transmission and make new cases can form the basis for a Propagation class, which would be
    a specific class of the more abstract Connector
    """
    # Make new cases via indirect transmission
    pars = environment.pars
    p_transmit = pars.beta * contaminated_environment.concentration[current_ti]
    pars.p_transmit.set(p=p_transmit)
    new_cases = pars.p_env_transmit.filter(people.uid[disease.susceptible])
    return new_cases

