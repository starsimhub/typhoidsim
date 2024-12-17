
"""
Define Typhoid-specific treatments and diagnostics (interventions). It includes
the intervention (eg, campaign that finds eligible people) and also products.
"""
import numpy as np
import sciris as sc
import starsim as ss
import typhoidsim

from .patterns import Pattern
from .defaults import day2year, days_per_year
from . import utils_math as tyum

# Interventions
# Diagnostics
__all__  = ['BaseTest']
# Treatments applied to people
__all__ += ['AcuteTreatment', 'InfectionClearance', 'VaccinationWithWaning']
# Interventions applied to the environment or environmental transmission
__all__ += ['SheddingReduction', 'EnvironmentalCleanup', 'EnvironmentalExposureReduction',
            'EnvironmentalSeasonality', 'EnvironmentalTrapezoidalModulation']
# Interventions that are not treatments but change some of the agents properties
__all__ += ['BehavioralChange']
# Products
__all__ += ['InfectiousnessRedux', 'InfectiousnessClearance', 'BlockingVaccine',
            'TyphoidVaccine']


# -- Treatments
class AcuteTreatment(ss.Intervention):
    """
    Treat acute (symptomatic) subjects.
    """

    def __init__(self, product=None, prob=1.0, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.coverage_dist = ss.bernoulli(p=self.prob)
        self.treated = ss.State('treated')
        self.results = ss.Results(self.name)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result('n_treated', dtype=int, label='Number Treated')
        self.initialized = True
        return

    def step(self):
        new_candidates = self.get_eligible()
        receive_uids = self.coverage_dist.filter(new_candidates)
        newly_treated = len(receive_uids)
        if newly_treated:
            self.product.administer(self.sim.people, receive_uids)
            self.treated[receive_uids] = True
        self.results.n_treated[self.ti] = newly_treated
        return

    def get_eligible(self):
        acute_uids = self.sim.people.typhoid.acute.uids
        seeks_treatment = (self.sim.people.typhoid.ti_seek_trtmnt == self.ti).uids
        new_candidates = acute_uids.intersect(seeks_treatment)
        return new_candidates


class InfectionClearance(ss.Intervention):
    """
    Clears an infected individual's Typhoid infection.
    """

    def __init__(self, product=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.eligibility = eligibility
        self._parse_product(product)
        self.treated = ss.State('treated')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result('n_treated', dtype=int, label='Number Treated')
        self.results += ss.Result('n_started_tr', dtype=int, label='Number Started Treatment')
        self.initialized = True
        return

    def step(self):
        new_patients, under_treatment = self.get_eligible()
        newly_treated = len(new_patients)
        all_treated = newly_treated + len(under_treatment)

        if len(new_patients):
            self.product.administer(self.sim, new_patients)
            self.treated[new_patients] = True

        if len(under_treatment):
            self.product.administer(self.sim, under_treatment)

        self.results.n_started_tr[self.ti] = newly_treated
        self.results.n_treated[self.ti] = all_treated

        treated = ss.uids.cat(new_patients, under_treatment)
        cleared_uids = treated.intersect((self.sim.people.typhoid.infectiousness < 0).uids)
        if len(cleared_uids):
            self.sim.people.typhoid.infectiousness[cleared_uids] = 0.0

            for state in [
                "prepatent",
                "acute",
                "subclinical",
                "chronic",
            ]:
                self.sim.people.typhoid[state][cleared_uids] = False

            for state in [
                "ti_prepatent",
                "ti_acute",
                "ti_seek_trtment",
                "ti_subclinical",
                "ti_chronic",
                "ti_dead",
            ]:
                self.sim.people.typhoid[state][cleared_uids] = np.nan

            self.sim.people.typhoid.recovered[cleared_uids] = True
            self.sim.people.typhoid.ti_recovered[cleared_uids] = self.ti
            self.sim.people.typhoid.ti_susceptible[cleared_uids] = self.ti + 1
            self.sim.people.typhoid.update_results()

        return

    def get_eligible(self):
        infected_uids = self.sim.people.typhoid.infected.uids
        under_treatment = self.sim.people.infection_clearance.treated.uids
        untreated = (~self.sim.people.infection_clearance.treated).uids
        old_patients = infected_uids.intersect(under_treatment)
        new_patients = infected_uids.intersect(untreated)
        return new_patients, old_patients


# -- Diagnostics
class BaseTest(ss.Intervention):
    """
    By default, find who is a chronic typhoid carrier.
    """

    def __init__(self, prob_t=1.0, prob_tp=1.0, eligibility=None, eligibility_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.prob_t = sc.promotetoarray(prob_t)
        self.prob_tp = sc.promotetoarray(prob_tp)
        self.eligibility = eligibility
        self.eligibility_kwargs = eligibility_kwargs
        self.coverage_dist = ss.bernoulli(p=self.prob_t)
        self.test_dist = ss.bernoulli(p=self.prob_tp)
        self.tested = ss.State('tested', default=False)
        self.positive = ss.State('positive', default=False)
        self.ti_tested = ss.FloatArr('ti_tested')
        self.ti_positive = ss.FloatArr('ti_positive')
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result('new_positive', dtype=int, label='New Positive')
        self.results += ss.Result('new_tested', dtype=int, label='New Tested')
        self.results += ss.Result('positivity', dtype=float, label='Positivity')

        if self.eligibility is None:
            self.eligibility = self.check_eligibility
        return

    def step(self):
        self.check_still_positive()
        eligible_uids = self._eligibility()
        self.coverage_dist.set(p=self.prob_t)
        self.test_dist.set(p=self.prob_tp)
        tested_uids = self.coverage_dist.filter(eligible_uids)
        infected_uids = self.sim.people.typhoid.infected.uids
        are_pos_uids = ss.uids(np.intersect1d(infected_uids, tested_uids))
        tested_pos_uids = self.test_dist.filter(are_pos_uids)
        self.tested[tested_uids] = True
        self.positive[tested_pos_uids] = True
        self.ti_tested[tested_uids] = self.ti
        self.ti_positive[tested_pos_uids] = self.ti
        self.results.new_tested[self.ti] = len(tested_uids)
        self.results.new_positive[self.ti] = len(tested_pos_uids)
        self.results.positivity[self.ti] = sc.safedivide(self.results.new_positive[self.ti], self.results.new_tested[self.ti])
        return tested_uids

    def check_still_positive(self):
        infected_uids = (~self.sim.people.typhoid.infected).uids
        self.positive[infected_uids] = False
        return

    def check_eligibility(self):
        chronic_uids = (self.sim.people.typhoid.chronic & ~self.tested).uids
        return chronic_uids

    def _eligibility(self):
        if callable(self.eligibility) and self.eligibility_kwargs is not None:
            return self.eligibility(self.sim, **self.eligibility_kwargs)
        elif callable(self.eligibility) and self.eligibility_kwargs is None:
            return self.eligibility(self.sim)
        else:
            return self.eligibility


# -- Environmental interventions
class WASH(ss.Intervention):
    """
    An environmental intervention that targets one of three different
    factors in transmission.
    """

    def __init__(self, start_year=None, dur=None, efficacy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start_year
        self.dur = dur
        self.efficacy_pattern = efficacy
        self.efficacy = None
        self.end = None
        self.time = None
        self.ti = 0
        self.target_baseline = None
        self.target_attr_path = None
        self.target_attr = None
        return

    def init_pre(self, sim):
        start_year = sim.pars.start if self.start is None else self.start
        duration = sim.pars.dur if self.dur is None else self.dur
        self.time = sc.inclusiverange(0, duration, sim.t.dt)
        self.results += ss.Result('efficacy', dtype=float)
        self.results += ss.Result('effective_value', dtype=float)

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
        self.results.effective_value[self.ti] = self.target_baseline
        if self.sim.year >= self.start and len(self.time):
            self.efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            self._set_target_val_par((1.0 - self.efficacy) * self.target_baseline)
            self._update_results()
            self.ti += 1
        return

    def _update_results(self):
        self.results.efficacy[self.ti] = self.efficacy
        self.results.effective_value[self.ti] = (1.0 - self.efficacy) * self.target_baseline
        return


class SheddingReduction(WASH):
    """
    Simulates sanitation interventions such as latrines and sewage disposal.
    """

    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['demographics', 'environmentalpool', 'pars', 'transmission', 'shedding_rate']
        self.target_baseline = super()._get_target_baseline()
        return


class EnvironmentalExposureReduction(WASH):
    """
    Results in a reduction in frequency of exposure.
    """

    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['diseases', 'typhoid', 'pars', 'transmission', 'exposure2env_rate', 'lam']
        self.target_baseline = super()._get_target_baseline()
        return


class EnvironmentalTrapezoidalModulation(WASH):
    """
    Results in a reduction of the relative exposure to the environment.
    """

    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['demographics', 'environmentalpool', 'pars', 'transmission', 'rel_trans']
        self.target_baseline = super()._get_target_baseline()
        return

    def step(self):
        self.efficacy = 0.0
        self.results.effective_value[self.ti] = self.target_baseline
        if self.sim.year >= self.start and len(self.time):
            time_days = self.sim.year * days_per_year
            self.efficacy = self.efficacy_pattern(time_days)
            self.ti += 1
        self._set_target_val_par(self.efficacy * self.target_baseline)
        self._update_results()
        return

    def _update_results(self):
        self.results.efficacy[self.ti] = self.efficacy
        self.results.effective_value[self.ti] = self.efficacy * self.target_baseline
        return


class EnvironmentalCleanup(WASH):
    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['demographics', 'environmentalpool', 'sv', 'cfu_conc']
        return

    def step(self):
        if self.sim.year >= self.start and len(self.time):
            efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            val = super()._get_target_val_arr(self.ti-1)
            r_val = (1.0 - efficacy) * val
            super()._set_target_val_arr(self.ti-1, r_val)
            self.results.efficacy[self.ti] = efficacy
            self.ti += 1
        return


class BehavioralChange(WASH):
    """
    Simulates reduction in exposure amount due to behavioral changes.
    """

    def init_pre(self, sim):
        super().init_pre(sim)
        self.target_attr_path = ['diseases', 'typhoid', 'rel_sus']
        return

    def step(self):
        if self.sim.year >= self.start and len(self.time):
            efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            val = super()._get_target_baseline()
            super()._set_target_val_par((1.0 - efficacy)*val)
            self.results.efficacy[self.ti] = efficacy
            self.ti += 1
        return


# Environmental modulation
class EnvironmentalSeasonality(ss.Intervention):
    """
    Use the mechanism of interventions to increase the number of CFUs in the environment.
    """
    def __init__(self, start_year=None, dur=None, seasonal_pattern=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start_year
        self.dur = dur
        self.pattern = seasonal_pattern
        self.end_day = None
        self.time = None
        self.ti = 0
        self.results = ss.Results(self.name)
        return

    def init_pre(self, sim):
        start_year = sim.pars.start if self.start is None else self.start
        duration = sim.pars.dur if self.dur is None else self.dur
        self.time = sc.inclusiverange(0, duration, sim.t.dt)
        self.results += ss.Result('seasonal_cfu', dtype=float, label='Seasonal CFUs')
        self.initialized = True
        return

    def step(self):
        if self.sim.year >= self.start and len(self.time):
            seasonal_cfu = self.pattern(self.time[0])
            self.time = self.time[1:]
            val = (self.sim.demographics['environmentalpool'].sv.cfu_conc[self.ti-1] *
                   self.sim.demographics['environmentalpool'].pars.volume)
            self.sim.demographics['environmentalpool'].sv.cfu_conc[self.ti-1] = (val + seasonal_cfu) / self.sim.demographics['environmentalpool'].pars.volume
            self.results.seasonal_cfu[self.ti] = seasonal_cfu
            self.ti += 1
        return


# -- Products
class InfectiousnessRedux(ss.Product):
    """
    Reduction in infectiousness.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(multiplier=0.5)
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        people.typhoid.infectiousness[uids] *= self.pars.multiplier
        return


class InfectiousnessClearance(ss.Product):
    """
    Reduction in infectiousness.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(clearance_rate=0.2)
        self.update_pars(pars, **kwargs)
        return

    def step(self):
        self.initialized = True
        return

    def administer(self, sim, uids):
        clearance = sim.people.typhoid.infectiousness[uids] * (self.pars.clearance_rate / day2year) * sim.t.dt
        sim.people.typhoid.infectiousness[uids] -= clearance
        return


class VaccinationWithWaning(ss.RoutineDelivery):
    """
    An intervention that handles a vaccination with waning.
    """

    def __init__(self, *args, booster_prob=0.0, dose_interval=None, label=None, age_pars=None, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.booster_prob = sc.promotetoarray(booster_prob)
        self.dose_interval = dose_interval
        self.age_pars = ss.Pars(age_pars)
        self.coverage_dist = ss.bernoulli(p=0)
        self.eligibility = self.age_eligibility
        self.vaccinated = ss.State('vaccinated')
        self.t_vaccinated = ss.FloatArr('t_vaccinated', default=np.nan)
        self.a_vaccinated = ss.FloatArr('a_vaccinated', default=np.nan)
        self.t_to_booster = ss.FloatArr('t_to_booster', default=np.nan)
        self.n_doses = ss.FloatArr('n_doses')
        self.debug = debug
        if self.debug:
            self.results = ss.Results(self.name)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.booster_prob = self.booster_prob * np.ones(shape=len(self.prob))
        if self.debug:
            self.results += ss.Result('immunity', dtype=float, label='Acquired Immunity')
        return

    def age_eligibility(self):
        is_eligible = (self.sim.people.age >= self.age_pars.min_age) & (self.sim.people.age < self.age_pars.max_age)
        return is_eligible

    def update_acquired_immunity(self, uids):
        module = self.sim.diseases.typhoid
        t_vaccinated = self.t_vaccinated[uids]
        a_vaccinated = self.a_vaccinated[uids]
        max_immunity = module.imm_peak(a_vaccinated)
        fixed_immunity = module.imm_fixed_dur(a_vaccinated)
        decay = module.imm_waning_time(a_vaccinated)
        module.immunity_acquired[uids] = np.clip(max_immunity * tyum.box_exponential(self.sim.year, t_vaccinated, fixed_immunity, decay), a_min=0.0, a_max=1.0)
        return

    def step(self):
        vaccinated_uids = self.vaccinated.uids
        if self.sim.year >= self.start_year and self.sim.year <= self.end_year:
            self.t_to_booster[self.t_to_booster > 0.0] -= self.sim.t.dt
            ti = sc.findinds(self.timepoints, self.ti)[0]
            prob = self.prob[ti]
            is_eligible = self.check_eligibility()
            is_eligible_not_vax = (is_eligible) & ~(self.vaccinated)
            self.coverage_dist.set(p=prob)
            new_accept_uids = self.coverage_dist.filter(is_eligible_not_vax)
            if len(new_accept_uids):
                self.vaccinated[new_accept_uids] = True
                self.t_vaccinated[new_accept_uids] = self.sim.year
                self.a_vaccinated[new_accept_uids] = self.sim.people.age[new_accept_uids]
                self.n_doses[new_accept_uids] = 1
                self.t_to_booster[new_accept_uids] = self.dose_interval

            booster_prob = self.booster_prob[ti]
            if booster_prob > 0.0:
                is_eligible_booster = (self.vaccinated) & (self.n_doses == 1) & (self.t_to_booster <= 0.0)
                self.coverage_dist.set(p=booster_prob)
                new_booster_uids = self.coverage_dist.filter(is_eligible_booster)
                self.t_vaccinated[new_booster_uids] = self.sim.year
                self.a_vaccinated[new_booster_uids] = self.sim.people.age[new_booster_uids]
                self.t_to_booster[new_booster_uids] = np.inf
                self.n_doses[new_booster_uids] += 1

            vaccinated_uids =  self.vaccinated.uids
        self.update_acquired_immunity(vaccinated_uids)
        self.sim.diseases.typhoid.rel_sus[vaccinated_uids] = 1.0 - self.sim.diseases.typhoid.immunity_acquired[vaccinated_uids]

        if self.debug:
            self.results.immunity[self.ti, :] = self.sim.diseases.typhoid.immunity_acquired[:]
        return self.vaccinated.uids


class BlockingVaccine(ss.Product):
    """
    An Acquisition Blocking vaccine that impacts the overall probability of infection after exposure.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(efficacy=1.0)
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        people.typhoid.susceptibility[uids] -= self.pars.efficacy * people.typhoid.immunity[uids]
        return


class TyphoidVaccine(ss.Vx):
    """
    Create a vaccine product that affects the probability of infection.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            efficacy=0.9,
            leaky=True
        )
        self.update_pars(pars, **kwargs)
        return

    def administer(self, people, uids):
        if self.pars.leaky:
            people.typhoid.rel_sus[uids] *= 1 - self.pars.efficacy
        else:
            people.typhoid.rel_sus[uids] *= np.random.binomial(1, 1 - self.pars.efficacy, len(uids))
        return
