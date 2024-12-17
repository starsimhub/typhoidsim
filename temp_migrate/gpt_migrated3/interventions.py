
"""
Define Typhoid-specific treatments and diagnostics (interventions). It includes
the intervention (eg, campaign that finds eligible people) and also products.
"""
import numpy as np
import sciris as sc
import starsim as ss

# Interventions
# Diagnostics
__all__  = ['base_test']
# Treatments applied to people
__all__ += ['acute_treatment', 'infection_clearence', 'vaccination_with_waning']
# Interventions applied to the environment or environmental transmission
__all__ += ['shedding_reduction', 'environmental_cleanup', 'environmental_exposure_reduction',
            'environmental_seasonality', 'environmental_trapezoidal_modulation']
# Interventions that are not treatments but change some of the agents properties
__all__ += ['behavioral_change']
# Products
__all__ += ['infectiousness_redux', 'infectiousness_clearence', 'blocking_vaccine',
            'typhoid_vaccine']


# -- Treatments
class acute_treatment(ss.Intervention):
    """
    Treat acute (symptomatic) subjects.

    This is expected to use the "infectiousness_redux" product, which
    results in an effective reduction or blocking in shedding, by reducing
    an agent's infectiousness level.
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
        self.define_results(ss.Result('n_treated', dtype=int, label="Number Treated"))
        return

    def step(self):
        new_candidates = self.get_eligible()
        receive_uids = self.coverage_dist.filter(new_candidates)
        newly_treated = len(receive_uids)
        if newly_treated:
            self.product.administer(receive_uids)
            self.treated[receive_uids] = True
        self.results['n_treated'][self.ti] = newly_treated
        return

    def get_eligible(self):
        acute_uids = (self.sim.people.typhoid.acute).uids
        seeks_treatment = (self.sim.people.typhoid.ti_seek_trtmnt == self.ti).uids
        new_candidates = acute_uids.intersect(seeks_treatment)
        return new_candidates


class infection_clearence(ss.Intervention):
    """
    Clears an infected individual's Typhoid infection. It will clear Typhoid
    infections of all types (prepatent, acute, subclinical, chronic)
    """

    def __init__(self, product=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.eligibility = eligibility
        self._parse_product(product)
        self.treated = ss.State('treated')
        return

    def init_pre(self):
        super().init_pre()
        self.define_results(
            ss.Result('n_treated', dtype=int, label="Number Treated"),
            ss.Result('n_started_tr', dtype=int, label="Started Treatment")
        )
        return

    def step(self):
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

        treated = ss.uids.cat(new_patients, under_treatment)
        cleared_uids = treated.intersect((self.sim.people.typhoid.infectiousness < 0).uids)
        if len(cleared_uids):
            self.sim.people.typhoid.infectiousness[cleared_uids] = 0.0
            for state in ["prepatent", "acute", "subclinical", "chronic"]:
               self.sim.people.typhoid[state][cleared_uids] = False
            for state in ["ti_prepatent", "ti_acute", "ti_seek_trtment", "ti_subclinical", "ti_chronic", "ti_dead"]:
                self.sim.people.typhoid[state][cleared_uids] = np.nan
            self.sim.people.typhoid["recovered"][cleared_uids] = True
            self.sim.people.typhoid["ti_recovered"][cleared_uids] = self.ti
            self.sim.people.typhoid["ti_susceptible"][cleared_uids] = self.ti + 1
            self.sim.people.typhoid.update_results()
        return

    def get_eligible(self):
        infected_uids = (self.sim.people.typhoid.infected).uids
        under_treatment = (self.treated).uids
        untreated = (~self.treated).uids
        old_patients = infected_uids.intersect(under_treatment)
        new_patients = infected_uids.intersect(untreated)
        return new_patients, old_patients


# -- Diagnostics
class base_test(ss.Intervention):
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

    def init_pre(self):
        super().init_pre()
        self.define_results(
            ss.Result('new_positive', dtype=int, label="New Positive"),
            ss.Result('new_tested', dtype=int, label="New Tested"),
            ss.Result('positivity', dtype=float, label="Positivity")
        )
        if self.eligibility is None:
            self.eligibility = self.check_eligibility
        return

    def step(self):
        self.check_still_positive()
        eligible_uids = self._eligibility()
        self.coverage_dist.set(p=self.prob_t)
        self.test_dist.set(p=self.prob_tp)
        tested_uids = self.coverage_dist.filter(eligible_uids)
        infected_uids = (self.sim.people.typhoid.infected).uids
        are_pos_uids = ss.uids(np.intersect1d(infected_uids, tested_uids))
        tested_pos_uids = self.test_dist.filter(are_pos_uids)
        self.tested[tested_uids] = True
        self.positive[tested_pos_uids] = True
        self.ti_tested[tested_uids] = self.ti
        self.ti_positive[tested_pos_uids] = self.ti
        self.results['new_tested'][self.ti] = len(tested_uids)
        self.results['new_positive'][self.ti] = len(tested_pos_uids)
        self.results['positivity'][self.ti] = sc.safedivide(self.results['new_positive'][self.ti], self.results['new_tested'][self.ti])
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
        self.dur   = dur
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
        self.start = sc.ifelse(self.start, sim.pars.start)
        self.dur = sc.ifelse(self.dur, sim.pars.dur)
        self.time = sc.inclusiverange(0, self.dur, sim.t.dt)
        self.define_results(
            ss.Result('efficacy', dtype=float),
            ss.Result('effective_value', dtype=float)
        )
        if self.efficacy_pattern is None:
            raise ValueError('No efficacy value or pattern specified')
        if sc.isnumber(self.efficacy_pattern):
            self.efficacy_pattern = ss.Pattern("efficacy", pars={'efficacy': self.efficacy_pattern})
        return

    def _get_target_baseline(self, sim):
        attr = sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        val = getattr(attr, target_attr)
        return val

    def _get_target_val_arr(self, sim, idx):
        attr = sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        val = getattr(attr, target_attr[idx])
        return val

    def _set_target_val_par(self, sim, val):
        attr = sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        setattr(attr, target_attr, val)
        return

    def _set_target_val_arr(self, sim, idx, val):
        attr = sim
        for attr_name in self.target_attr_path[:-1]:
            attr = getattr(attr, attr_name)
        target_attr = self.target_attr_path[-1]
        target_attr[idx] = val
        return

    def step(self):
        self.results['effective_value'][self.ti] = self.target_baseline
        if self.now('year') >= self.start and len(self.time):
            self.efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            self._set_target_val_par(self.sim, (1.0 - self.efficacy) * self.target_baseline)
            self._update_results()
            self.ti += 1
        return

    def _update_results(self):
        self.results['efficacy'][self.ti] = self.efficacy
        self.results['effective_value'][self.ti] = (1.0 - self.efficacy) * self.target_baseline
        return


class shedding_reduction(WASH):
    """
    Simulates sanitation interventions such as latrines and sewage disposal.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_pre(self):
        super().init_pre()
        self.target_attr_path = ['demographics', 'environmentalpool', 'pars', 'transmission', 'shedding_rate']
        self.target_baseline = super()._get_target_baseline(self.sim)
        return

    def step(self):
        super().step()
        return


class environmental_exposure_reduction(WASH):
    """
    Results in a reduction in frequency of exposure due to interventions such
    as dietary changes, crop irrigation, health inspections of food vendors.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_pre(self):
        super().init_pre()
        self.target_attr_path = ['diseases', 'typhoid', 'pars', 'transmission', 'exposure2env_rate', 'lam']
        self.target_baseline = super()._get_target_baseline(self.sim)
        return

    def step(self):
        super().step()
        return


class environmental_trapezoidal_modulation(WASH):
    """
    Results in a reduction of the relative exposure to the environment
    due crop irrigation, health inspections of food vendors.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_pre(self):
        super().init_pre()
        self.target_attr_path = ['demographics', 'environmentalpool', 'pars', 'transmission', 'rel_trans']
        self.target_baseline = self._get_target_baseline(self.sim)
        return

    def step(self):
        self.efficacy = 0.0
        self.results['effective_value'][self.ti] = self.target_baseline
        if self.now('year') >= self.start and len(self.time):
            time_days = self.now('year') * ss.time.time_units.year
            self.efficacy = self.efficacy_pattern(time_days)
            self.ti += 1
        self._set_target_val_par(self.sim, self.efficacy * self.target_baseline)
        self._update_results()
        return

    def _update_results(self):
        self.results['efficacy'][self.ti] = self.efficacy
        self.results['effective_value'][self.ti] = self.efficacy * self.target_baseline
        return


class environmental_cleanup(WASH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_pre(self):
        super().init_pre()
        self.target_attr_path = ['demographics', 'environmentalpool', 'sv', 'cfu_conc']
        return

    def step(self):
        if self.now('year') >= self.start and len(self.time):
            efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            val = self._get_target_val_arr(self.sim, self.ti-1)
            r_val = (1.0 - efficacy) * val
            self._set_target_val_arr(self.sim, self.ti-1, r_val)
            self.results['efficacy'][self.ti] = efficacy
            self.ti += 1
        return


class behavioral_change(WASH):
    """
    Simulates reduction in exposure amount that may be due to behavioral
    changes (washing vegetables, handwashing).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_pre(self):
        super().init_pre()
        self.target_attr_path = ['diseases', 'typhoid', 'rel_sus']
        return

    def step(self):
        if self.now('year') >= self.start and len(self.time):
            efficacy = self.efficacy_pattern(self.time[0])
            self.time = self.time[1:]
            val = self._get_target_baseline(self.sim)
            self._set_target_val_par(self.sim, (1.0 - efficacy)*val)
            self.results['efficacy'][self.ti] = efficacy
            self.ti += 1
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
        self.pattern = seasonal_pattern
        self.time = None
        self.ti = 0
        self.results = ss.Results(self.name)
        return

    def init_pre(self):
        self.start = sc.ifelse(self.start, self.sim.pars.start)
        self.dur = sc.ifelse(self.dur, self.sim.pars.dur)
        self.time = sc.inclusiverange(0, self.dur, self.sim.t.dt)
        self.define_results(
            ss.Result('seasonal_cfu', dtype=float, label="Seasonal CFUs")
        )
        return

    def step(self):
        if self.now('year') >= self.start and len(self.time):
            seasonal_cfu = self.pattern(self.time[0])
            self.time = self.time[1:]
            val = (self.sim.demographics['environmentalpool'].sv.cfu_conc[self.ti-1] *
                   self.sim.demographics['environmentalpool'].pars.volume)
            self.sim.demographics['environmentalpool'].sv.cfu_conc[self.ti-1] = (val + seasonal_cfu) / self.sim.demographics['environmentalpool'].pars.volume
            self.results['seasonal_cfu'][self.ti] = seasonal_cfu
            self.ti += 1
        return


# -- Products
class infectiousness_redux(ss.Product):
    """
    Reduction in infectiousness. This product is applied to acute cases only
    and results in a reduction or blocking in shedding.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(multiplier=0.5)
        self.update_pars(pars, **kwargs)
        return

    def administer(self, uids):
        self.sim.people.typhoid.infectiousness[uids] *= self.pars.multiplier
        return


class infectiousness_clearence(ss.Product):
    """
    Reduction in infectiousness. This product is applied to acute cases only
    and results in a reduction or blocking in shedding.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(clearence_rate=0.2)
        self.update_pars(pars, **kwargs)
        return

    def administer(self, uids):
        clearence = self.sim.people.typhoid.infectiousness[uids] * (self.pars.clearence_rate / ss.peryear(1)) * self.sim.t.dt_year
        self.sim.people.typhoid.infectiousness[uids] -= clearence
        return


class vaccination_with_waning(ss.RoutineDelivery):
    """
    An intervention that handles a vaccination with waning.
    """

    def __init__(self, *args, prob=None, label=None, age_pars=None, waning_pars=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = sc.promotetoarray(prob)
        self.label = label
        self.vaccinated = ss.State('vaccinated')
        self.ty_vaccinated = ss.FloatArr('ty_vaccinated')
        self.coverage_dist = ss.bernoulli(p=0)
        self.age_pars = age_pars
        self.vax_pars = waning_pars
        return

    def eligibility(self):
        is_eligible = (self.sim.people.age >= self.age_pars.min_age) & (self.sim.people.age < self.age_pars.max_age).uids
        return is_eligible

    def step(self):
        if self.now('year') >= self.start_year and self.now('year') <= self.end_year:
            ti = sc.findinds(self.timepoints, self.ti)[0]
            prob = self.prob[ti]
            is_eligible = self.eligibility()
            is_eligible_not_vax = (is_eligible & ~self.vaccinated)
            self.coverage_dist.set(p=prob)
            new_accept_uids = self.coverage_dist.filter(is_eligible_not_vax)
            if len(new_accept_uids):
                self.vaccinated[new_accept_uids] = True
                self.ty_vaccinated[new_accept_uids] = self.now('year')
            if len(self.vaccinated.uids):
                current_year = self.now('year') * np.ones(len(self.vaccinated.uids))
                box_durs = self.vax_pars['box_duration'] * np.ones(len(self.vaccinated.uids))
                decay_constant = self.vax_pars['decay_time_constant'] * np.ones(len(self.vaccinated.uids))
                efficacy = self.vax_pars['efficacy'] * ss.box_exponential(current_year, self.ty_vaccinated[self.vaccinated.uids], box_durs, decay_constant)
                self.sim.diseases.typhoid.rel_sus[self.vaccinated.uids] = 1.0 - efficacy
        return self.vaccinated.uids


class blocking_vaccine(ss.Product):
    """
    An Acquisition Blocking vaccine that impacts the overall probability of infection after exposure.
    """

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(efficacy=1.0)
        self.update_pars(pars, **kwargs)
        return

    def administer(self, uids):
        self.sim.people.typhoid.susceptibility[uids] -= self.pars.efficacy * self.sim.people.typhoid.immunity[uids]
        return


class typhoid_vaccine(ss.Vx):
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

    def administer(self, uids):
        if self.pars.leaky:
            self.sim.people.typhoid.rel_sus[uids] *= 1 - self.pars.efficacy
        else:
            self.sim.people.typhoid.rel_sus[uids] *= np.random.binomial(1, 1 - self.pars.efficacy, len(uids))
        return
