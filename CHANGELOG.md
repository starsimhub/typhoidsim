# What's new

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".


## Version 2.0.0 (unreleased)
- Ported the codebase from Starsim v3 to Starsim v4.
- `ty.Typhoid` is a declared disease: its states (`unexposed`, `susceptible`, `prepatent`, `acute`, `subclinical`, `chronic`, `recovered`) and transitions are written in the class, parameters are given by keyword (e.g. `ty.Typhoid(tppi=0.1, init_prev=0.001)`) as numbers or typed values rather than distributions, and the results are named after the states and transitions (e.g. `typhoid.infections`, `typhoid.contact`, `typhoid.environment`, `typhoid.onset.acute`).
- The environment is part of the disease: `ty.EnvironmentalPool` is removed, and its CFUs are the result `typhoid.cfu`, with parameters `shedding_rate`, `decay_rate`, `env_rel_trans`, and `teer`. `exposure2contact_rate=0` switches off contact transmission and `teer=0` environmental transmission; with no network, contact transmission is homogeneous mixing.
- Interventions are Starsim deliveries and modifiers: vaccines are `ss.Vaccine` with `ss.vaccinate()`, tests are `ss.Diagnostic` with `ss.screen()`, and treatments `ss.Treatment` with `ss.treat()`; WASH interventions and seasonality are `ss.modify()` of a parameter (e.g. `ss.modify('typhoid.pars.shedding_rate', multiply=0.7, start=2005)`) or a seasonal parameter (`ss.seasonal()`). `ty.acute_treatment()` and `ty.vaccination_with_waning()` remain, and now return Starsim deliveries.
- Calibration uses `sim.fit()`, with bounded parameters (`ss.par(0.15, bounds=[0.05, 0.4])`) and data passed to the sim.
- Results by age or sex come from stratifying the sim (`ss.Sim(..., stratify=dict(age=[0, 5, 15, 60]))`), replacing the histogram monitors.
- Removed: `ty.EnvironmentalPool`; `ty.Pattern` and `ty.StateVariable` (write a Python function of the sim instead); the products (`ty.typhoid_vaccine`, `ty.blocking_vaccine`, `ty.typhoid_test`, `ty.infectiousness_redux`, `ty.infectiousness_clearence`); `ty.routine_acute_screening`, `ty.infection_clearence`, and the environmental interventions (`ty.shedding_reduction`, `ty.behavioral_change`, `ty.environmental_cleanup`, `ty.environmental_exposure_reduction`, `ty.environmental_seasonality`, `ty.environmental_trapezoidal_modulation`); the calibration helpers (`ty.euclidean`, `ty.weighted_euclidean`, `ty.normalized_median_absolute_error`, `ty.calib_to_df`); `ty.states_consistency_monitor`, `ty.histograms_by_age_sex_monitor`, and `ty.histogram_by_vaccination_status`; and `ty.plot_sim` and `ty.plot_calib` (use `sim.plot()` and `fit.plot()`).
- *Regression information*: The move to Starsim v4 changes the random number streams, so results agree with v3 results statistically, not exactly. A sim with no network now has contact transmission (by homogeneous mixing); use `ty.Typhoid(exposure2contact_rate=0)` for none.


## Version 1.0.0 (2026-06-07)
- Ported the codebase from Starsim v2.3 to Starsim v3.
- Regenerated `tests/baseline.json` under Starsim v3.
- Converted documentation to Quarto.
- *Regression information*: The move to Starsim v3 changes the CRN/RNG streams, so simulation outputs differ from v2.3 results (short runs match within a few percent; long runs diverge through chaotic amplification).


## Version 0.32.0 (2025-07-08)
- Updating immune waning to follow box-gamma decay (reducible to previous exponential decay with shape=1)


## Version 0.6.0 (2024-11-22)
- Bugfixes to seasonal modulation
- Improved histogram by age and sex analyzer
- Improved base test intervention
- Backport Calibration classes from starsim 2.2.0


## Version 0.4.0 (2024-10-24)
- Environment level of pathogens is measured in CFU concentration
- Environment has a new parameter `volume`
- Lots of code tidying up
- First test script for interventions added for CI workflows
- Requires starsim 1.0.3


## Version 0.3.0 (2024-10-05)
- New community network with age mixing
- New functionality to use location-specific age distribution
- Introduces dependency to synthpops
- Requires starsim 1.0.3


## Version 0.2.X (2024-07-28)
- New tutorials
- New WASH interventions
- Requires starsim 1.1.0


## Version 0.1.0 (2024-07-04)
- Update age-based susceptibility dynamics.
- Requires starsim 0.5.9


## Version 0.0.1 (2024-05-10)
- Initial version. Repository draft.
- Requires starsim 0.5.2
