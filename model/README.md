---
license: mit
library_name: scikit-learn
pipeline_tag: tabular-classification
---
# Bean Shape Lab model card

A seven-class multinomial logistic regression trained from scratch on UCI Dry Bean
morphological measurements, with training-only standardization. The JSON artifact
contains coefficients, intercepts, scaler statistics, class order and feature order;
it is not a pickle. Use the accompanying repository for inference.

## Data and license

Koklu and Özkan, *Dry Bean* (2020), UCI Machine Learning Repository,
[DOI 10.24432/C50S4B](https://doi.org/10.24432/C50S4B),
[dataset record](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit the dataset authors and cite the dataset and accompanying paper when reusing it.
The original data are not redistributed here. The model and code are MIT licensed;
this does not change the dataset license.

## Evaluation

Stratified random 60/20/20 train/validation/test split, seed 42, with 8,166 / 2,722 /
2,723 rows. Fixed C=1 and no tuning. Validation accuracy 0.9243; held-out test accuracy
0.9210 and macro F1 0.9346; training-majority baseline accuracy 0.2607. The test
confusion matrix and per-class results are in `metrics.json`.

Notably, exact feature vectors appear across split boundaries: 19 distinct vectors in
both train and test. This can inflate the reported score. A random split is not a
farm-, lot-, camera-, or time-held-out evaluation. Other measurement methods and
conditions may perform worse. The model is an educational research demo, not a
production seed sorter; no calibration, out-of-distribution detector, image pipeline
or operational guarantees are provided. Probabilities are not guaranteed calibrated.

Run provenance, dataset checksum, model checksum, split and training commit are in
`training-manifest.json`; local execution is CPU-only. No personal data are used.
