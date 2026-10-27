"""
Typhoid model: the natural history of typhoid in each person, transmission by
contact and through the environment, and the environment itself.
"""

import numpy as np
import sciris as sc
import starsim as ss

import typhoidsim.utils as tyu
import typhoidsim.utils_math as tyum

__all__ = ['Typhoid']

prepatent_levels = tyu.load_dataset('prepatent_dur_dist_pars') # The parameters of the prepatent period at a low, medium, and high dose


def by_dose(levels, dose, lo_me, me_hi):
    """ A parameter of the prepatent period by dose: the low dose's, the medium dose's above `lo_me`, or the high dose's above `me_hi` (EMOD's three levels; Glynn et al. 1995) """
    return levels['lo'] + (levels['me'] - levels['lo'])*(dose > lo_me) + (levels['hi'] - levels['me'])*(dose > me_hi)


class Typhoid(ss.Disease):
    """
    Typhoid (*S.* Typhi), after EMOD-Typhoid (Gauld et al. 2018)

    People are born unexposed (protected by maternal antibodies, or not yet exposed),
    and become susceptible with age. An infection starts with a prepatent (incubation)
    period, which is followed by an acute (symptomatic) or a subclinical infection; after
    either, people recover, become chronic carriers (more likely with age, and for women,
    with gallstones), or (after an acute infection) die. Carriers stay infectious for life
    unless `p_rec` is given. Each infection leaves people less susceptible (`tppi`).

    People are infected in two ways: by contact with infectious people (`contact`), over
    the sim's networks (or, with none, by homogeneous mixing), and through the environment
    (`environment`): infectious people shed colony-forming units (CFUs) into the environment
    (`cfu`, a count, not people), which decay, and susceptible people are exposed to them,
    with a dose-response curve. Either route can be switched off: `exposure2contact_rate=0`
    for contact, and `teer=0` for the environment.

    Parameters are given by keyword, e.g. `ty.Typhoid(tppi=0.1, init_seroprev=0.8)`. The
    results are the number in each state, the flow of each transition (e.g. `contact`,
    `onset.acute`, `acute_end.chronic`), `infections` (by either route), and `prevalence`.

    Args:
        init_prev (float): the share of people infected (prepatent) at the start
        init_seroprev (float): the share of the susceptible at the start who have been infected before, a Poisson number of times with a mean that rises with age
    """
    # The states: people are born unexposed (see first)
    unexposed   = ss.State(label='Unexposed')
    susceptible = ss.State(label='Susceptible')
    prepatent   = ss.State(label='Prepatent')
    acute       = ss.State(label='Acute')
    subclinical = ss.State(label='Subclinical')
    chronic     = ss.State(label='Chronic carrier')
    recovered   = ss.State(label='Recovered')
    cfu         = ss.State(people=False, label='CFUs in the environment') # A count, not people
    first       = 'unexposed'

    # Becoming susceptible with age (Gauld et al. 2018, Fig. 2B)
    unexp2sus_saturation_age = ss.par(20.0, doc='Age (years) by which everyone has become susceptible')
    unexp2sus_slope          = ss.par(1.0, doc='Slope of the share susceptible by age (> -0.5): the larger, the sooner people become susceptible')

    # Infectiousness and susceptibility
    tai  = ss.par(40_000, doc='Typhoid acute infectiousness: CFUs shed per day by someone with acute typhoid')
    tpri = ss.par(0.5, doc='Infectiousness of prepatent infection, relative to acute')
    tsri = ss.par(1.0, doc='Infectiousness of subclinical infection, relative to acute')
    tcri = ss.par(0.241, doc='Infectiousness of chronic carriers, relative to acute')
    tppi = ss.par(0.98, doc='Typhoid protection per infection: susceptibility is (1 - tppi)**(number of infections)')

    # The contact route
    exposure2contact_rate = ss.par(ss.perday(1.0), doc='Rate of infection per contact with someone with acute typhoid')

    # The environmental route
    teer          = ss.par(ss.perday(2.0), doc='Typhoid environmental exposure rate: exposures to the environment per person')
    env_rel_trans = ss.par(1e-3, doc='Share of the CFUs per person in the environment received in one exposure (mEL in Gauld et al. 2018)')
    drc_alpha     = ss.par(0.175, doc='Shape of the dose-response curve (beta-Poisson; Enger 2013)')
    drc_n50       = ss.par(1.11e6, doc='Dose (CFUs) at which half of those exposed are infected')
    shedding_rate = ss.par(ss.perday(1.0), doc='Rate at which infectious people shed into the environment, relative to tai')
    decay_rate    = ss.par(ss.perday(0.3), doc='Rate at which CFUs in the environment decay')

    # Prepatent (incubation) period, lognormal with parameters (of the log of the duration in days) that depend on the dose
    cfu_lo_me = ss.par(5_050_000.0, doc='Dose (CFUs) above which the prepatent period is that of a medium dose')
    cfu_me_hi = ss.par(55_000_000.0, doc='Dose (CFUs) above which the prepatent period is that of a high dose (as are all infections by contact)')

    # Acute and subclinical infection
    p_acute       = ss.par(0.16, doc='Probability that an infection is acute (otherwise subclinical)')
    inf_dur_mean  = ss.par(1.172, doc='Mean of the log of the duration (weeks) of acute and subclinical infection')
    inf_dur_sigma = ss.par(ss.by(age={0: 0.483, 30: 0.788}), doc='Standard deviation of the log of the duration (weeks) of acute and subclinical infection, by age')
    p_death       = ss.par(0.01, doc='Probability of death after acute infection, for those who do not become carriers')

    # Chronic carriers
    p_cpg    = ss.par(0.15, doc='Probability of becoming a chronic carrier, given gallstones')
    p_gall   = ss.par(tyu.gallstone_probs(), doc='Probability of having gallstones, by age and sex')
    p_rec    = ss.par(0.0, doc='Probability that a chronic carrier recovers, after dur_chro (by default, carriers never recover)')
    dur_chro = ss.par(ss.constant(ss.weeks(102)), doc='Time to recovery of the carriers who recover')
    dur_rec  = ss.par(ss.constant(ss.days(1)), doc='Time from recovery to being susceptible again')

    # Becoming susceptible: the share susceptible by age a is 1 - (A - a)/(k*a + A), which is everyone by age A; it is the share at the start, and this is its hazard
    A, k, age = unexp2sus_saturation_age, unexp2sus_slope, ss.ppl.age
    share_susceptible = np.minimum(1 - (A - age)/(k*age + A), 1)
    become_susceptible = unexposed >> susceptible @ ss.peryear(1)*(k + 1)*A/((k*age + A)*np.maximum(A - age, 1e-9))

    # Infection, by either route, multiplied by the susceptibility left after past infections
    n_infections = ss.attr(1, on=prepatent.enter, combine='sum', label='Number of infections')
    contact = susceptible >> prepatent @ (1 - tppi)**n_infections*exposure2contact_rate*(tpri*prepatent + acute + tsri*subclinical + tcri*chronic)
    dose = env_rel_trans*cfu.n/(unexposed + susceptible + prepatent + acute + subclinical + chronic + recovered).n # CFUs per exposure to the environment
    p_response = 1 - (1 + dose*(2**(1/drc_alpha) - 1)/drc_n50)**(-drc_alpha) # Dose-response: the probability of infection per exposure
    environment = ss.Transition(susceptible, prepatent, rate=(1 - tppi)**n_infections*teer*p_response, acquisition=True) # Exposures are Poisson
    infections = ss.outcome(contact + environment, label='Infections')

    # Shedding into the environment, by each person's infectiousness (so less by those treated), and decay
    shedding = ss.null >> cfu @ shedding_rate*tai*(tpri*prepatent.n_trans + acute.n_trans + tsri*subclinical.n_trans + tcri*chronic.n_trans)
    decay = cfu >> ss.null @ decay_rate

    # The course of an infection; the prepatent period's parameters are by the dose of the infection: by contact, a high dose (as in EMOD),
    # and from the environment, the dose received then, from the low to the high dose's in two steps at cfu_lo_me and cfu_me_hi
    infection_dose = ss.attr({contact: 1.1*cfu_me_hi, environment: dose}, label='Dose of the latest infection (CFUs)')
    prepatent_mean  = by_dose(prepatent_levels['mean_dur'], infection_dose, cfu_lo_me, cfu_me_hi)
    prepatent_sigma = by_dose(prepatent_levels['std_dur'], infection_dose, cfu_lo_me, cfu_me_hi)
    dur_prepatent = ss.par(ss.lognorm_im(mean=prepatent_mean, sigma=prepatent_sigma, unit='day'), doc='Prepatent period: lognormal, with the parameters (of the log of the days) by the dose of the infection')
    onset = ss.Transition(prepatent, [acute, subclinical], p=[p_acute, None], dur=dur_prepatent)
    dur_inf = ss.par(ss.lognorm_im(mean=inf_dur_mean, sigma=inf_dur_sigma, unit='week'), doc='Duration of acute and subclinical infection: lognormal, with the parameters of the log of the weeks')
    acute_end = ss.Transition(acute, [chronic, recovered, ss.null], p=[p_cpg*p_gall, None, (1 - p_cpg*p_gall)*p_death], dur=dur_inf)
    subclinical_end = ss.Transition(subclinical, [chronic, recovered], p=[p_cpg*p_gall, None], dur=dur_inf)
    carrier_end = ss.Transition(chronic, [recovered, ss.stay], p=[p_rec, None], dur=dur_chro)
    waning = recovered >> susceptible @ dur_rec

    # Results
    infected = prepatent + acute + subclinical + chronic
    prevalence = infected/(unexposed + susceptible + prepatent + acute + subclinical + chronic + recovered)

    def __init__(self, init_prev=0.005, init_seroprev=0.0, **kwargs):
        super().__init__(init_prev=init_prev, init=dict(susceptible=self.share_susceptible), **kwargs) # At the start, the share of each age who would have become susceptible by then
        self.init_seroprev = init_seroprev
        self.init_sero = ss.random() # Who has been infected before the start
        self.n_prior = ss.poisson(lam=lambda self, sim, uids: tyum.sigmoid(sim.people.age[uids], 120, 1.0)) # How many times (a placeholder: the mean rises with age)
        return

    def init_post(self):
        """ At the start, a share of the susceptible (`init_seroprev`) have been infected before """
        super().init_post()
        if self.init_seroprev: # [PROPOSAL TY3] ss.attr(..., init=...) for this, once its expression can read the person's own state (a bare state there is the share of everyone in it)
            susceptible = self.susceptible.uids
            prior = susceptible[self.init_sero.rvs(susceptible) < self.init_seroprev]
            self['n_infections'][prior] = self.n_prior.rvs(prior)
        return
