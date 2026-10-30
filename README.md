# Bean Shape Lab

A small, reproducible experiment for classifying dry-bean varieties from **16
precomputed shape measurements**. It combines a from-scratch multinomial logistic
regression with a FastAPI inference endpoint. It does not accept images or perform
segmentation, and is a research demonstration rather than a production sorter.

## Architecture

UCI Dry Bean archive → checksum verification → ARFF measurements → stratified
train/validation/test split → train-only standardization and logistic regression →
portable JSON coefficients → FastAPI /predict. Inference uses a stable softmax and
never loads pickle. Python 3.11–3.12, NumPy, SciPy, scikit-learn, FastAPI, uv.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```sh
uv sync --locked
uv run uvicorn bean_shape.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' --data @examples/request.json
```

The sample is the mean training feature vector, not an individual bean. The actual
recorded response is in [examples/response.json](examples/response.json), with label
`SIRA` and probability approximately `0.8885`. `/docs` provides OpenAPI docs.

`POST /predict` takes exactly 16 finite numbers between -1e9 and 1e9 in this order:
`Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`,
`Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness`,
`Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4`.
`AspectRation` is the spelling in the source ARFF. Feature semantics and units are
in the [UCI dataset record](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
The response contains `label`, ordered `labels`, and corresponding `probabilities`.
Malformed inputs return 422; a missing model returns 503. `/health` reports model
availability, not production readiness. Values must use the source measurement
conventions; the endpoint does not verify physical plausibility.

## Reproduce and test

The checked-in JSON model is ready to use. To re-download the dataset and reproduce
the experiment (the original training code is identified in the manifest):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m bean_shape.training \
  --training-commit "$(git rev-parse HEAD)" --output /tmp/bean-shape-model
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The command downloads the 4.5 MB UCI archive to ignored `data/`, rejects checksum
mismatches, and writes model, metrics and provenance to the output directory. Model
hashes may differ across platforms/library versions due to numerical differences.
For exact provenance use the recorded code commit and lockfile. The test split should
not be used for model selection. CPU experiment timing includes local archive read,
training and evaluation: 0.493 s, peak process RSS 156,928 KiB on Linux. Timing is
specific to this run; see [metrics](model/metrics.json) and
[manifest](model/training-manifest.json).

## Results and limitations

The fixed-hyperparameter model was fitted on 8,166 rows; validation and test contain
2,722 and 2,723 rows, respectively. Training-only preprocessing; stratified random
split, seed 42. On the held-out test set: **accuracy 0.9210**, **macro F1 0.9346**;
the training-majority baseline gets **0.2607** accuracy. The largest confusion is
DERMASON → SIRA (59 of 710), followed by SIRA → DERMASON (43 of 527).
Per-class metrics and confusion matrix are recorded in the model directory.

**Important:** 19 distinct identical feature vectors occur in both train and test,
which may inflate this estimate. This split does not hold out farms, lots, cameras,
or acquisition sessions; external performance is unknown. No calibration or
out-of-distribution detection is performed. Probabilities are not guaranteed to be
calibrated. See the [model card](model/README.md) before reuse.

## Data and license

Koklu and Özkan, *Dry Bean* (2020), UCI Machine Learning Repository,
[DOI 10.24432/C50S4B](https://doi.org/10.24432/C50S4B), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the authors and
cite the dataset and accompanying paper. Raw data are not distributed in this repo.
Code and model are under the [MIT License](LICENSE); the dataset retains its own
license and attribution requirements.
