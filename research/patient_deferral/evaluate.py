"""Evaluate selective prediction on repeated image records in BUS-BRA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_SHA = "7f22dccf4be1f8920a0e9fbd46ae31ed814f1be138e55dc936b00b76d67324ef"
FOLDS = (1, 2, 3, 4, 5)
RULES = ("mean_margin", "worst_margin", "learned_margin", "learned_margin_gap")
PRIMARY_PAIR = ("learned_margin", "learned_margin_gap")


def load(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("The supplied CSV differs from the frozen BUS-BRA table")
    data = pd.read_csv(path)
    predictors = [col for col in data if col.startswith(("original_", "wavelet-"))]
    if len(data) != 1875 or len(predictors) != 288 or not data.image_id.is_unique:
        raise ValueError("Unexpected BUS-BRA dimensions or image keys")
    groups = data.groupby("patient_id")
    if groups.size().value_counts().to_dict() != {2: 811, 1: 253}:
        raise ValueError("Unexpected number of imaging records per patient")
    if not groups.diagnosis.nunique().eq(1).all():
        raise ValueError("The patient-group endpoint requires one diagnosis per patient")
    if not groups.fold5_kfold.nunique().eq(1).all():
        raise ValueError("A patient appears in multiple outer folds")
    return data, predictors


def assert_disjoint(train: np.ndarray, test: np.ndarray, group: np.ndarray) -> None:
    if np.intersect1d(group[train], group[test]).size:
        raise ValueError("A patient appears on both sides of a split")


def fit_base(name: str, X: np.ndarray, y: np.ndarray, weight: np.ndarray):
    if name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000),
        )
        return model.fit(X, y, standardscaler__sample_weight=weight,
                         logisticregression__sample_weight=weight)
    model = HistGradientBoostingClassifier(
        max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, random_state=42,
    )
    return model.fit(X, y, sample_weight=weight)


def summarize(group: np.ndarray, y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = pd.DataFrame({"patient": group, "label": y, "score": scores})
    rows["image_margin"] = abs(rows.score - 0.5)
    patients = rows.groupby("patient", sort=True).agg(
        label=("label", "first"), n_images=("score", "size"),
        mean_score=("score", "mean"), high=("score", "max"),
        low=("score", "min"), min_margin=("image_margin", "min"),
    )
    patients["margin"] = abs(patients.mean_score - 0.5)
    patients["gap"] = patients.high - patients.low
    patients["predicted"] = (patients.mean_score >= 0.5).astype(int)
    patients["error"] = (patients.predicted != patients.label).astype(int)
    return patients


def fit_rejector(patients: pd.DataFrame, columns: list[str]):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    return model.fit(patients[columns], patients.error)


def run_fold(name: str, fold: int, data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    X = data[predictors].to_numpy(dtype=float)
    y = data.diagnosis.to_numpy()
    groups = data.patient_id.to_numpy()
    weights = 1 / data.groupby("patient_id").patient_id.transform("size").to_numpy()
    train = np.flatnonzero(data.fold5_kfold.to_numpy() != fold)
    test = np.flatnonzero(data.fold5_kfold.to_numpy() == fold)
    assert_disjoint(train, test, groups)

    split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + fold)
    inner_predictions = np.full(len(train), np.nan)
    for train_rel, valid_rel in split.split(X[train], y[train], groups[train]):
        fit, valid = train[train_rel], train[valid_rel]
        assert_disjoint(fit, valid, groups)
        model = fit_base(name, X[fit], y[fit], weights[fit])
        inner_predictions[valid_rel] = model.predict_proba(X[valid])[:, 1]
    if not np.isfinite(inner_predictions).all():
        raise ValueError("Some outer-training images did not receive inner predictions")
    inner = summarize(groups[train], y[train], inner_predictions)
    pairs = inner.loc[inner.n_images == 2]
    confidence = fit_rejector(pairs, ["margin"])
    disagreement = fit_rejector(pairs, ["margin", "gap"])

    model = fit_base(name, X[train], y[train], weights[train])
    outer_score = model.predict_proba(X[test])[:, 1]
    outer = summarize(groups[test], y[test], outer_score)
    outer["fold"] = fold
    outer["mean_margin"] = -outer.margin
    outer["worst_margin"] = -outer.min_margin
    outer["learned_margin"] = confidence.predict_proba(outer[["margin"]])[:, 1]
    outer["learned_margin_gap"] = disagreement.predict_proba(outer[["margin", "gap"]])[:, 1]
    return outer


def augrc(errors: np.ndarray, risk_score: np.ndarray) -> float:
    # Low-risk patients are accepted first. Average ranks integrate score ties.
    n = len(errors)
    admission_rank = rankdata(risk_score, method="average")
    return float(errors @ (n - admission_rank + 0.5) / (n * n))


def fixed_coverage(rows: pd.DataFrame, rule: str, target: float) -> dict:
    # Expectations over a uniform tie break at the empirical coverage boundary.
    k = max(1, int(np.ceil(target * len(rows))))
    threshold = np.sort(rows[rule].to_numpy())[k - 1]
    below = rows[rule].lt(threshold).to_numpy()
    tied = rows[rule].eq(threshold).to_numpy()
    weights = below.astype(float) + tied * ((k - below.sum()) / tied.sum())
    malignant = rows.label.eq(1).to_numpy()
    false_negative = malignant & rows.predicted.eq(0).to_numpy()
    return {
        "n_patients": len(rows), "accepted": k,
        "accepted_errors": float(weights @ rows.error.to_numpy()),
        "accepted_malignant": float(weights @ malignant),
        "n_malignant": int(malignant.sum()),
        "accepted_false_negatives": float(weights @ false_negative),
    }


def bootstrap(pairs: pd.DataFrame, draws: int) -> list[float]:
    rng = np.random.default_rng(20261124)
    deltas = np.zeros(draws)
    for _, block in pairs.groupby("fold"):
        errors = block.error.to_numpy()
        baseline, enriched = (block[rule].to_numpy() for rule in PRIMARY_PAIR)
        for i, indices in enumerate(rng.integers(len(block), size=(draws, len(block)))):
            deltas[i] += (
                augrc(errors[indices], enriched[indices])
                - augrc(errors[indices], baseline[indices])
            ) / len(FOLDS)
    return np.quantile(deltas, [.025, .975]).tolist()


def evaluate(name: str, data: pd.DataFrame, predictors: list[str], draws: int) -> dict:
    patients = pd.concat([run_fold(name, f, data, predictors) for f in FOLDS])
    if len(patients) != 1064 or not patients.index.is_unique:
        raise ValueError("Expected one held-out prediction per patient")
    pairs = patients.loc[patients.n_images == 2]
    folds = []
    for fold, rows in pairs.groupby("fold"):
        folds.append({
            "fold": int(fold), "n": len(rows), "n_malignant": int(rows.label.sum()),
            "accuracy": float(1 - rows.error.mean()),
            "auc": float(roc_auc_score(rows.label, rows.mean_score)),
            "augrc": {rule: augrc(rows.error.to_numpy(), rows[rule].to_numpy()) for rule in RULES},
            "operating_points": {
                rule: {str(target): fixed_coverage(rows, rule, target) for target in (.8, .9)}
                for rule in RULES
            },
        })
    summaries = {
        rule: {
            "augrc": float(np.mean([f["augrc"][rule] for f in folds])),
            "operating_points": {
                str(target): {
                    key: sum(f["operating_points"][rule][str(target)][key] for f in folds)
                    for key in ("n_patients", "accepted", "accepted_errors",
                                "accepted_malignant", "n_malignant", "accepted_false_negatives")
                } for target in (.8, .9)
            },
        } for rule in RULES
    }
    baseline, enriched = PRIMARY_PAIR
    return {
        "folds": folds, "rules": summaries,
        "mean_delta": summaries[enriched]["augrc"] - summaries[baseline]["augrc"],
        "paired_percentile_interval": bootstrap(pairs, draws),
        "full_coverage": {
            str(int(k)): {"patients": len(block), "malignant": int(block.label.sum()),
                          "accuracy": float(1 - block.error.mean()),
                          "auc": float(roc_auc_score(block.label, block.mean_score))}
            for k, block in patients.groupby("n_images")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, predictors = load(args.dataset)
    draws = 5000
    result = {"data_sha256": DATA_SHA, "bootstrap_draws": draws}
    result.update({name: evaluate(name, data, predictors, draws)
                   for name in ("logistic", "gradient_boosting")})
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
