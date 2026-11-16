### Calibrations can follow how a look changes colour with brightness
- `calibrate-lightroom fit --value-divisions 4,6 --folds 5` also tries hue/saturation maps whose correction changes with brightness, for looks such as Capture One's ProStandard that make dark oranges fuller and bright ones paler.
- A brightness-dependent map is kept only when it beats the single-step map on held-out photos in every fold and stays smooth; otherwise the fit writes the single-step map and says why. Without the option the fit is unchanged.
