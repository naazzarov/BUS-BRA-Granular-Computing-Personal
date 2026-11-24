"""Replay frozen deferral cutoffs and test within-cohort device transport."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from evaluate import (
    DATA_SHA, FOLDS, PRIMARY_PAIR, assert_disjoint, augrc, fit_base, fit_rejector, load, summarize,
)


TARGET = 0.90
DRAWS = 5000
DEVICE_RECORD_MD5 = "1f8b2be6476d58fc97bfb5e5a1ea9bab"
RULE_COLUMNS = {PRIMARY_PAIR[0]: ["margin"], PRIMARY_PAIR[1]: ["margin", "gap"]}


def frozen_cutoff(scores: np.ndarray) -> float:
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration needs finite scores from paired patients")
    rank = int(np.ceil(TARGET * len(scores))) - 1
    return float(np.partition(scores, rank)[rank])


def replay(name: str, data: pd.DataFrame, predictors: list[str],
           train: np.ndarray, test: np.ndarray, split_number: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = data[predictors].to_numpy(dtype=float)
    y = data.diagnosis.to_numpy()
    groups = data.patient_id.to_numpy()
    weights = 1 / data.groupby("patient_id").patient_id.transform("size").to_numpy()
    assert_disjoint(train, test, groups)
    calibration_split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=6040 + split_number)
    dev_rel, cal_rel = next(calibration_split.split(X[train], y[train], groups[train]))
    dev, cal = train[dev_rel], train[cal_rel]
    assert_disjoint(dev, cal, groups)
    inner_split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=7040 + split_number)
    inner = np.full(len(dev), np.nan)
    for fit_rel, valid_rel in inner_split.split(X[dev], y[dev], groups[dev]):
        fit, valid = dev[fit_rel], dev[valid_rel]
        assert_disjoint(fit, valid, groups)
        model = fit_base(name, X[fit], y[fit], weights[fit])
        inner[valid_rel] = model.predict_proba(X[valid])[:, 1]
    if not np.isfinite(inner).all():
        raise ValueError("Some development images lack grouped out-of-fold predictions")

    development = summarize(groups[dev], y[dev], inner).query("n_images == 2")
    model = fit_base(name, X[dev], y[dev], weights[dev])
    calibration = summarize(groups[cal], y[cal], model.predict_proba(X[cal])[:, 1]).query("n_images == 2").copy()
    held_out = summarize(groups[test], y[test], model.predict_proba(X[test])[:, 1]).query("n_images == 2").copy()
    for rule, columns in RULE_COLUMNS.items():
        ranker = fit_rejector(development, columns)
        calibration_scores = ranker.predict_proba(calibration[columns])[:, 1]
        cutoff = frozen_cutoff(calibration_scores)
        calibration[rule + "_accept"] = calibration_scores <= cutoff
        held_out[rule] = ranker.predict_proba(held_out[columns])[:, 1]
        held_out[rule + "_accept"] = held_out[rule] <= cutoff
    return held_out, calibration


def counts(rows: pd.DataFrame, rule: str) -> dict:
    accepted = rows[rule + "_accept"]
    malignant = rows.label.eq(1)
    return {
        "n": len(rows), "accepted": int(accepted.sum()),
        "n_malignant": int(malignant.sum()), "accepted_malignant": int((accepted & malignant).sum()),
        "accepted_errors": int((accepted & rows.error.eq(1)).sum()),
        "accepted_false_negatives": int((accepted & malignant & rows.predicted.eq(0)).sum()),
    }


def gap_interval(blocks: list[pd.DataFrame]) -> dict:
    rng = np.random.default_rng(20261125)
    totals = np.zeros((DRAWS, 2 + 2 * len(RULE_COLUMNS)), dtype=int)
    for block in blocks:
        label = block.label.to_numpy(dtype=bool)
        accepted = [block[rule + "_accept"].to_numpy() for rule in RULE_COLUMNS]
        columns = [~label, label]
        for keep in accepted:
            columns.extend((~label & keep, label & keep))
        values = np.column_stack(columns)
        idx = rng.integers(len(block), size=(DRAWS, len(block)))
        totals += values[idx].sum(axis=1)
    if (totals[:, :2] == 0).any():
        raise ValueError("A bootstrap draw lacks a diagnosis class")
    return {
        rule: np.quantile(totals[:, 2 + 2*i] / totals[:, 0] - totals[:, 3 + 2*i] / totals[:, 1],
                          [.025, .975]).tolist()
        for i, rule in enumerate(RULE_COLUMNS)
    }


def block_result(held_out: pd.DataFrame, calibration: pd.DataFrame) -> dict:
    return {
        "calibration_pairs": len(calibration),
        "rules": {rule: {
            **counts(held_out, rule),
            "calibration_accepted": int(calibration[rule + "_accept"].sum()),
            "augrc": augrc(held_out.error.to_numpy(), held_out[rule].to_numpy()),
        } for rule in RULE_COLUMNS},
    }


def officially_grouped(name: str, data: pd.DataFrame, predictors: list[str]) -> dict:
    records = []
    blocks = []
    folds = data.fold5_kfold.to_numpy()
    for fold in FOLDS:
        held_out, calibration = replay(name, data, predictors, np.flatnonzero(folds != fold),
                                       np.flatnonzero(folds == fold), fold)
        records.append({"fold": fold, **block_result(held_out, calibration)})
        blocks.append(held_out)
    pooled = pd.concat(blocks)
    if len(pooled) != 811 or not pooled.index.is_unique:
        raise ValueError("Expected 811 unique paired patients in the held-out test set")
    intervals = gap_interval(blocks)
    return {
        "folds": records,
        "pooled": {rule: {**counts(pooled, rule), "coverage_gap_interval": intervals[rule]}
                   for rule in RULE_COLUMNS},
    }


def load_release(archive: Path) -> tuple[pd.DataFrame, str]:
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != DEVICE_RECORD_MD5:
            raise ValueError("The BUS-BRA archive does not match Zenodo record 8231412")
    with ZipFile(archive) as source:
        metadata_bytes = source.read("BUSBRA/bus_data.csv")
        with source.open("BUSBRA/bus_data.csv") as stream:
            metadata = pd.read_csv(stream, usecols=["ID", "Case", "Pathology", "Device"])
        with source.open("BUSBRA/5-fold-cv.csv") as stream:
            folds = pd.read_csv(stream, usecols=["ID", "kFold"])
    verified = metadata.merge(folds, on="ID", validate="one_to_one")
    if len(verified) != 1875 or verified.Case.nunique() != 1064:
        raise ValueError("Unexpected image or patient counts in the public metadata")
    return verified, hashlib.sha256(metadata_bytes).hexdigest()


def with_devices(data: pd.DataFrame, archive: Path) -> tuple[pd.DataFrame, str]:
    verified, metadata_sha = load_release(archive)
    joined = data.merge(verified, left_on="image_id", right_on="ID", how="left", validate="one_to_one")
    if joined.Case.isna().any() or not joined.Case.map("case_{:04d}".format).eq(joined.patient_id).all():
        raise ValueError("Official case identifiers do not match the frozen table")
    if not joined.Pathology.map({"benign": 0, "malignant": 1}).eq(joined.diagnosis).all():
        raise ValueError("Official diagnoses do not match the frozen table")
    if not joined.kFold.eq(joined.fold5_kfold).all():
        raise ValueError("Official folds do not match the frozen table")
    return joined, metadata_sha


def device_transport(data: pd.DataFrame, predictors: list[str], names: tuple[str, ...]) -> dict:
    per_patient = data.groupby("patient_id").agg(
        device=("Device", "first"), n_devices=("Device", lambda rows: rows.nunique(dropna=False)),
        n_images=("Device", "size"), label=("diagnosis", "first"),
    )
    resolved = per_patient.device.notna() & per_patient.n_devices.eq(1)
    valid_patients = per_patient.loc[resolved]
    eligible_data = data[data.patient_id.isin(valid_patients.index)].reset_index(drop=True)
    records = []
    for position, (device, patients) in enumerate(valid_patients.groupby("device"), start=1):
        paired = patients.query("n_images == 2")
        malignant = int(paired.label.sum())
        eligible = len(paired) >= 50 and min(malignant, len(paired) - malignant) >= 10
        record = {"device": device, "patients": len(patients), "pairs": len(paired),
                  "paired_malignant": malignant, "transport_eligible": eligible}
        if eligible:
            test_mask = eligible_data.Device.eq(device).to_numpy()
            train, test = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
            record["models"] = {
                name: block_result(*replay(name, eligible_data, predictors, train, test, position))
                for name in names
            }
        records.append(record)
    return {"excluded_patients": int((~resolved).sum()), "devices": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, predictors = load(args.dataset)
    device_data, metadata_sha = with_devices(data, args.official_archive)
    names = ("logistic", "gradient_boosting")
    result = {
        "data_sha256": DATA_SHA, "official_archive_md5": DEVICE_RECORD_MD5,
        "device_metadata_sha256": metadata_sha, "bootstrap_draws": DRAWS,
        "nominal_calibration_coverage": TARGET,
        "threshold_replay": {name: officially_grouped(name, data, predictors) for name in names},
        "held_out_device": device_transport(device_data, predictors, names),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
