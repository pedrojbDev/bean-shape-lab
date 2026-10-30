"""Reproducible training of a seven-class bean morphology classifier."""

import argparse
import hashlib
import io
import json
import platform
import resource
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .model import export_model

URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
ARCHIVE_SHA256 = "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"
FEATURES = [
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


def split_indices(y):
    indices = np.arange(len(y))
    train, rest = train_test_split(indices, test_size=0.4, stratify=y, random_state=42)
    val, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=42)
    return train, val, test


def train_classifier(x, y):
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    classifier.fit(scaler.transform(x), y)
    return scaler, classifier


def evaluate(classifier, scaler, x, y, y_train):
    prediction = classifier.predict(scaler.transform(x))
    labels = classifier.classes_
    values, counts = np.unique(y_train, return_counts=True)
    majority = values[np.argmax(counts)]
    return {
        "accuracy": accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro"),
        "majority_accuracy": float(np.mean(y == majority)),
        "majority_class": str(majority),
        "labels": labels.tolist(),
        "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
        "classification_report": classification_report(
            y, prediction, labels=labels, output_dict=True, zero_division=0
        ),
    }


def verify_archive(archive):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Dataset archive checksum does not match the pinned UCI snapshot")
    return digest


def load_data(archive):
    with zipfile.ZipFile(archive) as z:
        data = z.read("DryBeanDataset/Dry_Bean_Dataset.arff")
    rows, _ = arff.loadarff(io.StringIO(data.decode("utf-8")))
    x = np.column_stack([rows[name] for name in FEATURES]).astype(float)
    y = np.array([value.decode("utf-8") for value in rows["Class"]])
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/dry-bean.zip"))
    parser.add_argument("--output", type=Path, default=Path("model"))
    parser.add_argument("--training-commit", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read(10_000_001)
        if len(content) > 10_000_000:
            raise ValueError("Dataset archive exceeds expected maximum size")
        args.archive.write_bytes(content)
    archive_hash = verify_archive(args.archive)
    x, y = load_data(args.archive)
    train, val, test = split_indices(y)
    scaler, classifier = train_classifier(x[train], y[train])
    model = export_model(scaler, classifier, classifier.classes_.tolist())
    model["features"] = FEATURES
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = args.output / "model.json"
    artifact.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "seed": 42,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "validation": evaluate(classifier, scaler, x[val], y[val], y[train]),
        "test": evaluate(classifier, scaler, x[test], y[test], y[train]),
        "training_and_evaluation_seconds": time.monotonic() - started,
        "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "training_type": "from-scratch",
        "training_commit": args.training_commit,
        "dataset": "UCI Dry Bean (2020), DOI:10.24432/C50S4B",
        "dataset_license": "CC-BY-4.0",
        "dataset_url": URL,
        "dataset_archive_sha256": archive_hash,
        "split_method": "stratified random 60/20/20, seed 42; train-only scaler",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "model": "multinomial logistic regression, C=1",
    }
    (args.output / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "test_accuracy": metrics["test"]["accuracy"],
                "test_macro_f1": metrics["test"]["macro_f1"],
                "majority_accuracy": metrics["test"]["majority_accuracy"],
                "seconds": metrics["training_and_evaluation_seconds"],
                "artifact_sha256": manifest["artifact_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
