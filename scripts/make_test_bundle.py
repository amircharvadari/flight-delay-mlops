"""Write a small serving bundle from synthetic data (CI container test, I-24).

    uv run python scripts/make_test_bundle.py <out_dir>

The model is trained on the synthetic batches of the test suite. It is only
used to check that the container starts and answers; it is not a project model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_models import labelled_df

from flightdelay.models import FlightDelayModel
from flightdelay.serving import frozen_threshold, write_bundle


def main() -> int:
    out = Path(sys.argv[1])
    model = FlightDelayModel("lightgbm").fit_frame(
        labelled_df(n=800, seed=1), {"train_from": "2019-01", "train_to": "2019-01",
                                     "train_months": 1})
    scores = model.predict_proba(labelled_df(n=400, seed=2, month=2)).to_numpy()
    write_bundle(model, out, frozen_threshold(model, scores, "2019-02"),
                 {"model_version": "ci-synthetic"})
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
