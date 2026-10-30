import numpy as np
from sklearn.datasets import make_classification

from bean_shape.training import evaluate, split_indices, train_classifier


def test_split_is_stratified_disjoint_and_reproducible():
    y = np.repeat(np.arange(3), 100)
    train, val, test = split_indices(y)
    assert (len(train), len(val), len(test)) == (180, 60, 60)
    assert len(set(train) | set(val) | set(test)) == len(y)
    assert all(len(set(y[index])) == 3 for index in (train, val, test))
    assert np.array_equal(train, split_indices(y)[0])


def test_fitted_preprocessing_and_baseline_metrics():
    x, y = make_classification(
        n_samples=300, n_features=16, n_informative=10, n_classes=3, random_state=7
    )
    train, _val, test = split_indices(y)
    scaler, classifier = train_classifier(x[train], y[train])
    np.testing.assert_allclose(scaler.mean_, x[train].mean(axis=0))
    scores = evaluate(classifier, scaler, x[test], y[test], y[train])
    assert 0 <= scores["accuracy"] <= 1
    assert 0 <= scores["majority_accuracy"] <= 1
    assert sum(map(sum, scores["confusion_matrix"])) == len(test)


def test_uci_arff_uses_aspect_ration_spelling(tmp_path):
    import zipfile

    from bean_shape.training import load_data

    names = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRation",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4",
    ]
    arff = "@RELATION beans\n" + "".join(f"@ATTRIBUTE {n} NUMERIC\n" for n in names)
    arff += "@ATTRIBUTE Class {SEKER,SIRA}\n@DATA\n" + ",".join(["1"] * 16) + ",SEKER\n"
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("DryBeanDataset/Dry_Bean_Dataset.arff", arff)
    x, y = load_data(archive)
    assert x.shape == (1, 16)
    assert y.tolist() == ["SEKER"]


def test_dataset_checksum_rejects_unexpected_data(tmp_path):
    import pytest

    from bean_shape.training import verify_archive

    archive = tmp_path / "data.zip"
    archive.write_bytes(b"not the source dataset")
    with pytest.raises(ValueError, match="checksum"):
        verify_archive(archive)
