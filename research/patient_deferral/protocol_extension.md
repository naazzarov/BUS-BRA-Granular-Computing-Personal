# Follow-up analysis protocol

The primary five-fold ranking results were available when this follow-up was designed. The primary cohort, risk-ranking models, endpoint, and result file are frozen. The analyses below are secondary and use the same source cohort.

## Frozen threshold replay

For each official outer fold, reserve the first validation block of a four-way shuffled `StratifiedGroupKFold` split of the outer training set (seed `6040 + outer fold`) for threshold calibration. Fit each original backbone and both error rankers on the remaining development data. Generate the development scores for ranker fitting by three-way grouped cross-fitting with seed `7040 + outer fold`. Fit the backbone on the entire development block for calibration and held-out scoring. For each rule and backbone, freeze the smallest calibration score that retains at least 90% of paired calibration patients, including any ties at the boundary. Apply that cutoff to the official held-out fold. Calibration diagnoses do not enter the cutoff calculation. This empirical rule does not provide guaranteed test coverage or risk control.

Report counts of eligible and retained patients by diagnosis, errors among retained patients, and false negatives among retained malignant patients, together with per-fold and pooled achieved coverage. Report conditional paired percentile bootstrap intervals for benign coverage minus malignant coverage using 5,000 patient resamples within the outer folds (seed `20261125`). The bootstrap holds models and thresholds fixed.

## Acquisition-device transport

If the official BUS-BRA release provides per-image equipment identifiers that join to the frozen input, pool patients by acquisition device and hold one device out at a time. For each test device, require at least 50 paired patients and at least 10 paired patients per diagnosis. Exclude patients with ambiguous or discordant acquisition device from this analysis only. Include every other unambiguous patient's records in training, including patients from devices below the test eligibility threshold. Apply the development and calibration procedure above with split number derived from alphabetical device order. Acquisition device does not enter the predictors. Report device-stratified held-out AUGRC, realized coverage and retained errors. This is a shift analysis within BUS-BRA. Small or unresolved device strata are reported as unavailable for transport estimation.

## Scope

External validation requires an independent cohort with linked patient identifiers, paired images, a comparable outcome, and a compatible feature extraction. A separate scanner within BUS-BRA is insufficient for that claim. The study reports predictive performance on retrospective annotated data and does not estimate downstream outcomes or utility after review.

## Public-input sensitivity analysis

The original ranking and cohort composition were known when defining this analysis. Rerun the five-fold ranking comparison with an alternate descriptor set that can be rebuilt directly from the official BUS-BRA archive without the handoff table. Use grayscale pixels divided by 255 and the nonzero foreground of the provided lesion mask. For each masked image compute the mean, population standard deviation, and 10th, 25th, 50th, 75th, and 90th intensity percentiles. Compute the mean absolute intensity difference across foreground horizontal and vertical neighbor pairs, using zero if no such pairs exist. Also compute mask width, height, area in pixels, and the fraction of the mask bounding box occupied by foreground. Use no data-dependent feature filtering. Keep the original nested splits, weighting, backbones, risk models, and conditional bootstrap. Report accuracy, ROC area, and AUGRC contrast with its interval for both backbones. This tests sensitivity to the input representation on the same patients.
