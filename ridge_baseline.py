"""Generic grouped Ridge baseline; array inputs have no dataset-specific schema."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_grouped_ridge(features, targets, groups, *, alpha=1.0, test_size=0.25, seed=7):
    """Fit scaling on training rows only, with entirely disjoint test groups."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    groups = np.asarray(groups)
    if targets.ndim == 1:
        targets = targets[:, None]
    if features.ndim != 2 or targets.ndim != 2 or groups.ndim != 1:
        raise ValueError("features/targets must be matrices and groups must be one-dimensional")
    if len(features) != len(targets) or len(features) != len(groups) or len(features) < 4:
        raise ValueError("aligned features, targets and groups with at least four rows are required")
    if not features.shape[1] or not targets.shape[1] or not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("feature and target matrices must be nonempty and finite")
    if groups.dtype.kind not in "iuUS" or len(np.unique(groups)) < 2:
        raise ValueError("groups must contain at least two integer or string identifiers")
    if not np.isfinite(alpha) or alpha <= 0 or not 0 < test_size < 1:
        raise ValueError("alpha must be positive and test_size must lie between zero and one")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(features, targets, groups))
    if len(test) < 2:
        raise ValueError("at least two held-out rows are needed for R²")
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features[train], targets[train])
    prediction = model.predict(features[test])
    score = float(r2_score(targets[test], prediction, multioutput="variance_weighted"))
    return model, train, test, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True, help="NPZ containing features, targets and groups")
    parser.add_argument("--output", type=Path, required=True, help="External output JSON")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.resolve() == args.arrays.resolve():
        parser.error("output must differ from input")
    with np.load(args.arrays, allow_pickle=False) as arrays:
        model, train, test, score = fit_grouped_ridge(arrays["features"], arrays["targets"], arrays["groups"],
                                                    alpha=args.alpha, test_size=args.test_size, seed=args.seed)
    record = {"train_rows": len(train), "test_rows": len(test), "r2": score, "seed": args.seed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
