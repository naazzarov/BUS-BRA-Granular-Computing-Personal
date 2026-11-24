# arXiv submission metadata

**Title**

Does Image Disagreement Help Decide Which Breast Ultrasound Predictions to Defer?

**Authors, in order**

Bahodir Nazarov, Zineb Gharib, Fatima Zazoul, Luo Chuan

**Suggested primary category**

Computer Science → Machine Learning (cs.LG)

**Suggested Comments field**

9 pages, 2 figures, 5 tables

**Keywords for the manuscript or other venues**

Breast ultrasound, selective classification, deferral, risk–coverage, image disagreement

**Abstract**

Purpose: We test whether disagreement between two breast ultrasound image scores helps rank patient-level errors after accounting for confidence in the average score.

Methods: We use 1,875 image records from 1,064 BUS-BRA patient groups, including 811 two-image groups. We compare confidence-only and confidence-plus-disagreement error rankers using patient-disjoint nested splits with logistic regression and histogram gradient boosting. Secondary analyses test frozen cutoffs, held-out scanners, and an alternate feature set generated from the public archive.

Results: Adding disagreement changed area under the generalized risk-coverage curve by -0.011 percentage points with logistic regression (paired patient-bootstrap 95% interval [-0.056, 0.033]) and by -0.045 points with gradient boosting ([-0.131, 0.038]). With a confidence-only cutoff set to retain at least 90% of separate calibration patients, the logistic model retained 90.6% of test patients overall, including 83.4% of malignant and 94.1% of benign cases. The benign-minus-malignant coverage gap was 10.7 points (95% interval [6.1, 15.5]).

Conclusion: The incremental benefit of absolute score disagreement was uncertain in this cohort. The frozen cutoff retained fewer malignant cases proportionally, supporting diagnosis-specific reporting alongside overall coverage.

## Before release

- Have all four authors approve the paper, author order, contributions, and disclosure wording. Add affiliations and the correspondence address if the authors want them in the paper. Confirm applicable institutional, funding, and conflict-of-interest statements with the authors.
- The manuscript links to the [private GitHub research branch](https://github.com/naazzarov/BUS-BRA-Granular-Computing-Personal/tree/research/patient-deferral/research/patient_deferral). Confirm whether readers will receive access to the analysis code or an approved public code snapshot. Keep the handoff feature table, patient-level records, individual predictions, and fitted models in authorized storage.
- Choose the arXiv distribution license with the authors and verify category eligibility in the submitting account. [arXiv licenses](https://info.arxiv.org/help/license/index.html) · [Endorsement](https://info.arxiv.org/help/endorsement.html)
- Upload the TeX source ZIP and review arXiv's processed PDF, metadata, and references before submitting. [Submission overview](https://info.arxiv.org/help/submit/index.html) · [TeX instructions](https://info.arxiv.org/help/submit_tex.html)

The local nine-page PDF was compiled from the extracted upload archive. Source ZIP SHA-256 is `cad4d24c1cc6a7ec3c3d9eb067045768c88bb23643deebf32f5ac26de37ec49b`. Verify the page count in the arXiv preview. [arXiv metadata requirements](https://info.arxiv.org/help/prep.html)

The study is a new analysis of selective prediction. The earlier feature-selection submission has its own moderation decision and appeal process. [arXiv appeals](https://info.arxiv.org/help/moderation/appeals.html)
