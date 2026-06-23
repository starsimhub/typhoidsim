# What's new

All notable changes to the codebase are documented in this file. Changes that may result in differences in model output, or are required in order to run an old parameter set with the current version, are flagged with the term "Regression information".


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
