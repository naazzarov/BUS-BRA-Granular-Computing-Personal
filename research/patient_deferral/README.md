# Patient-level deferral in breast ultrasound

This study tests whether disagreement between two images helps rank patient-level prediction errors beyond confidence in the average image score. The analysis uses the frozen BUS-BRA feature table and saves aggregate results.

## Reproduce the run

Use Python 3.11.15. Install the pinned packages in a virtual environment. Run from the directory that contains this README.

~~~bash
python -m pip install -r requirements.txt
python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python evaluate.py --dataset /path/to/BUSBRA_features_ready.csv --output results.json
python plot_results.py
~~~

The data path requires the exact, authorized derived BUS-BRA table from the team handoff. The runner verifies its SHA-256 before training. Dataset background is available in the [BUS-BRA paper](https://doi.org/10.1002/mp.16812) and [source record](https://doi.org/10.5281/zenodo.8231412).

[Analysis plan](protocol.md) · [Aggregate outcomes](results.json) · [Manuscript source](manuscript/main.tex)

## Before submitting to arXiv

1. Review the analysis and manuscript with the contributing authors. Confirm names, affiliations, author order, data-use permissions and any required institutional wording.
2. Preserve the exact code snapshot and decide how readers can access it. Keep the derived table, identifiers, individual predictions and any private artifacts in authorized storage. Include only approved source code and aggregate results in a public release.
3. Add the confirmed author information and public code link to the manuscript. Confirm the wording of the AI-assistance disclosure.
4. Review the original moderation notice before choosing a category or submitting the new work. Check endorsement if required. [arXiv moderation](https://info.arxiv.org/help/moderation/index.html) · [Endorsement](https://info.arxiv.org/help/endorsement.html)
5. Prepare an upload ZIP containing main.tex and results.png from the manuscript directory. The references are embedded in the TeX source. Verify arXiv renders the same content before final submission. [arXiv TeX instructions](https://info.arxiv.org/help/submit_tex.html)

The local rerun reproduced the aggregate results byte for byte with the same environment. Four synthetic tests cover the risk calculation, ties at the coverage boundary, patient aggregation, and split overlap checks.
