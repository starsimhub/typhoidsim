# What's new

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".


## Version 1.0.1 (2026-11-01)
- Require Starsim 3.3.1 or later, which added `ss.uids.concatenate()`, and below 4 (`starsim>=3.3.1,<4`), since this version does not run on Starsim 4.
- Updated for Starsim 3.7, which no longer allows `float()` on an `ss.TimePar`: durations and timeline values now use `.years`. Results are unchanged for a given Starsim version when `dt` is in years.
- Fixed the Sciris interlinks URL in the docs, and changed `dt` in the vaccination and interventions tutorials to 1/52 so routine delivery can find the start and end years on the timestep grid.
- `test_base_test` now compares against the realized initial prevalence, since `init_prev` only applies to agents who are susceptible at t=0.
- Regenerated `tests/baseline.json` under Starsim 3.7.2.
- *Regression information*: The new baseline differs because `ss.RandomNet` changed in Starsim 3.5 and 3.6, not because of any change here: in the baseline sim, cumulative infections are 11.5% lower than under Starsim 3.4. To recover the previous network behavior, use `ss.RandomExactNet`.


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
