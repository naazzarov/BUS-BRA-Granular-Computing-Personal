"""Repeat the risk-ranking analysis with features from public BUS-BRA images and masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from evaluate import evaluate
from evaluate_extension import DEVICE_RECORD_MD5, DRAWS, load_release

FEATURES = (
    "intensity_mean", "intensity_sd", "intensity_p10", "intensity_p25", "intensity_p50",
    "intensity_p75", "intensity_p90", "adjacent_difference", "mask_width", "mask_height",
    "mask_area", "mask_fill",
)


def describe(image: np.ndarray, mask: np.ndarray) -> tuple[float, ...]:
    if image.ndim != 2 or image.shape != mask.shape or not mask.any():
        raise ValueError("Each image needs an aligned nonempty binary lesion mask")
    pixels = image.astype(float) / 255
    within = pixels[mask]
    foreground_y, foreground_x = np.nonzero(mask)
    width = int(np.ptp(foreground_x)) + 1
    height = int(np.ptp(foreground_y)) + 1
    h_pairs = mask[:, :-1] & mask[:, 1:]
    v_pairs = mask[:-1, :] & mask[1:, :]
    adjacent_diff = np.concatenate((np.abs(np.diff(pixels, axis=1))[h_pairs],
                                    np.abs(np.diff(pixels, axis=0))[v_pairs]))
    return (
        within.mean(), within.std(), *np.percentile(within, [10, 25, 50, 75, 90]),
        adjacent_diff.mean() if adjacent_diff.size else 0.0,
        width, height, within.size, within.size / (width * height),
    )


def public_features(archive: Path) -> pd.DataFrame:
    metadata, _ = load_release(archive)
    descriptors = []
    with ZipFile(archive) as source:
        for image_id in metadata.ID:
            mask_id = image_id.replace("bus_", "mask_", 1)
            with source.open(f"BUSBRA/Images/{image_id}.png") as stream, Image.open(stream) as image:
                pixels = np.asarray(image.convert("L"))
            with source.open(f"BUSBRA/Masks/{mask_id}.png") as stream, Image.open(stream) as mask:
                foreground = np.asarray(mask.convert("L")) != 0
            descriptors.append(describe(pixels, foreground))
    groups = metadata.assign(diagnosis=metadata.Pathology.map({"benign": 0, "malignant": 1}))
    groups = groups.rename(columns={"Case": "patient_id", "kFold": "fold5_kfold"})
    groups = groups[["patient_id", "diagnosis", "fold5_kfold"]]
    by_patient = groups.groupby("patient_id")
    if (groups.diagnosis.isna().any() or not by_patient.diagnosis.nunique().eq(1).all()
            or not by_patient.fold5_kfold.nunique().eq(1).all()
            or by_patient.size().value_counts().to_dict() != {2: 811, 1: 253}):
        raise ValueError("Public metadata violates the original patient or label boundaries")
    return pd.concat([groups, pd.DataFrame(descriptors, columns=FEATURES)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = public_features(args.official_archive)
    result = {"official_archive_md5": DEVICE_RECORD_MD5, "features": FEATURES, "bootstrap_draws": DRAWS}
    result.update({name: evaluate(name, data, list(FEATURES), DRAWS)
                   for name in ("logistic", "gradient_boosting")})
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
