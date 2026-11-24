# Patient-level deferral in breast ultrasound

Does disagreement between two image scores improve patient-level error ranking beyond confidence in their average? This secondary BUS-BRA study evaluates that question with logistic regression and histogram gradient boosting on patient-disjoint folds. It also measures whom a frozen cutoff retains and how it transfers to held-out acquisition devices.

The primary logistic estimate for adding disagreement was −0.011 percentage points in area under the generalized risk–coverage curve (conditional 95% interval −0.056 to 0.033). The confidence-only logistic cutoff retained 90.6% of held-out patients overall, including 83.4% of malignant and 94.1% of benign cases. The same public cohort underlies all of the analyses.

[Manuscript source](manuscript/main.tex) · [arXiv source ZIP](submission/arxiv-source.zip) · [Submission metadata and preflight](submission/metadata.md)

## Reproduce the analyses

Run from the directory containing this README in a Python 3.11 environment. The results were generated with Python 3.11.15. Set the two placeholder paths to local inputs. The first two runs use the exact, authorized derived CSV from the team handoff. All three runners write aggregate JSON and check the input version before fitting. Download `BUSBRA.zip` from the [official BUS-BRA release](https://zenodo.org/records/8231412) and cite the [dataset paper](https://doi.org/10.1002/mp.16812) when reusing it.

```bash
python -m pip install -r requirements.txt
python -m pytest -q

features=/path/to/BUSBRA_features_ready.csv
archive=/path/to/BUSBRA.zip
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

python evaluate.py --dataset "$features" --output results.json
python evaluate_extension.py --dataset "$features" --official-archive "$archive" \
  --output extension_results.json
python evaluate_public.py --official-archive "$archive" --output public_results.json
python plot_results.py
python render_tables.py
python build_submission.py
```

The third run reconstructs 12 descriptors from public images and lesion masks, so it requires only the official archive. The code computes descriptors and patient-linked predictions in memory. The three result files contain fold-, device-, or cohort-level summaries.

| Analysis | Plan | Aggregate output |
| --- | --- | --- |
| Main five-fold radiomic analysis | [Primary protocol](protocol.md) | [Primary results](results.json) |
| Frozen cutoffs and held-out scanners | [Follow-up protocol](protocol_extension.md) | [Follow-up results](extension_results.json) |
| Alternate descriptors from public images and masks | [Public-input protocol](protocol_extension.md#public-input-sensitivity-analysis) | [Public-input results](public_results.json) |

## Build the manuscript

The upload ZIP has four files at its root. `main.tex` includes the bibliography, `audit_tables.tex` contains the aggregate fold rows, and `partitions.pdf` and `results.pdf` are the vector figures. Rebuild the ZIP after changing the manuscript, tables, or figures. Unpack it in an empty directory and run `pdflatex main.tex` twice. Tectonic can also compile the source locally. Compare the arXiv-generated preview with the reviewed PDF before final submission. [arXiv TeX instructions](https://info.arxiv.org/help/submit_tex.html)
