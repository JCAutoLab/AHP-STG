import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

DATASETS = {
    "SD": ROOT / "data" / "SD" / "flowsd.npz",
    "CA": ROOT / "data" / "CA" / "flowca.npz",
    "GBA": ROOT / "data" / "GBA" / "flowgba.npz",
    "GLA": ROOT / "data" / "GLA" / "flowgla.npz",
}

INPUT_LEN = 12
OUTPUT_LEN = 12
TRAIN_RATIO = 0.6
TEST_RATIO = 0.2
DAY_PERIOD = 96
WEEK_PERIOD = DAY_PERIOD * 7
BATCH_SIZE = 32


class MetricAccumulator:
    def __init__(self, horizon):
        self.horizon = horizon
        self.abs_sum = np.zeros(horizon + 1, dtype=np.float64)
        self.sq_sum = np.zeros(horizon + 1, dtype=np.float64)
        self.pct_sum = np.zeros(horizon + 1, dtype=np.float64)
        self.count = np.zeros(horizon + 1, dtype=np.float64)

    def update(self, pred, label):
        diff = pred - label
        mask = label != 0
        abs_err = np.abs(diff)
        sq_err = diff * diff
        pct_err = np.zeros_like(abs_err, dtype=np.float32)
        np.divide(abs_err, label, out=pct_err, where=mask)

        for idx in range(self.horizon):
            step_mask = mask[:, idx, :]
            self.abs_sum[idx] += abs_err[:, idx, :][step_mask].sum(dtype=np.float64)
            self.sq_sum[idx] += sq_err[:, idx, :][step_mask].sum(dtype=np.float64)
            self.pct_sum[idx] += pct_err[:, idx, :][step_mask].sum(dtype=np.float64)
            self.count[idx] += np.count_nonzero(step_mask)

        self.abs_sum[-1] += abs_err[mask].sum(dtype=np.float64)
        self.sq_sum[-1] += sq_err[mask].sum(dtype=np.float64)
        self.pct_sum[-1] += pct_err[mask].sum(dtype=np.float64)
        self.count[-1] += np.count_nonzero(mask)

    def to_dict(self):
        count = np.maximum(self.count, 1.0)
        mae = self.abs_sum / count
        rmse = np.sqrt(self.sq_sum / count)
        mape = self.pct_sum / count
        return {
            "mae": [float(x) for x in mae],
            "rmse": [float(x) for x in rmse],
            "mape": [float(x) for x in mape],
            "valid_count": [int(x) for x in self.count],
        }


def _split_indices(total_steps):
    num_train = int(total_steps * TRAIN_RATIO)
    num_test = int(total_steps * TEST_RATIO)
    border1 = total_steps - num_test - INPUT_LEN
    border2 = total_steps
    sample_count = border2 - border1 - INPUT_LEN - OUTPUT_LEN + 1
    return border1, sample_count


def _load_values(path):
    values = np.load(path)["data"][..., :1]
    return np.squeeze(values, axis=-1).astype(np.float32, copy=False)


def _window_values(values, starts, offsets):
    return values[starts[:, None] + offsets[None, :], :]


def _predictions(values, sample_starts, horizon_offsets):
    history_offsets = np.arange(INPUT_LEN)
    target_starts = sample_starts + INPUT_LEN
    last_value = values[sample_starts + INPUT_LEN - 1, :]
    first_value = values[sample_starts, :]
    history = _window_values(values, sample_starts, history_offsets)

    daily = _window_values(values, target_starts - DAY_PERIOD, horizon_offsets)
    weekly = _window_values(values, target_starts - WEEK_PERIOD, horizon_offsets)
    residual = last_value - values[sample_starts + INPUT_LEN - 1 - DAY_PERIOD, :]
    slope = (last_value - first_value) / max(INPUT_LEN - 1, 1)
    trend_scale = (horizon_offsets + 1).astype(np.float32)

    return {
        "persistence": np.repeat(last_value[:, None, :], OUTPUT_LEN, axis=1),
        "recent_mean": np.repeat(history.mean(axis=1, keepdims=True), OUTPUT_LEN, axis=1),
        "daily_seasonal": daily,
        "weekly_seasonal": weekly,
        "seasonal_residual": daily + residual[:, None, :],
        "linear_trend": last_value[:, None, :] + trend_scale[None, :, None] * slope[:, None, :],
    }


def evaluate_dataset(dataset, path):
    values = _load_values(path)
    border1, sample_count = _split_indices(values.shape[0])
    horizon_offsets = np.arange(OUTPUT_LEN)
    accumulators = {name: MetricAccumulator(OUTPUT_LEN) for name in [
        "persistence",
        "recent_mean",
        "daily_seasonal",
        "weekly_seasonal",
        "seasonal_residual",
        "linear_trend",
    ]}

    for start in range(0, sample_count, BATCH_SIZE):
        local = np.arange(start, min(start + BATCH_SIZE, sample_count), dtype=np.int64)
        sample_starts = border1 + local
        label = _window_values(values, sample_starts + INPUT_LEN, horizon_offsets)
        for name, pred in _predictions(values, sample_starts, horizon_offsets).items():
            accumulators[name].update(pred, label)

    results = {name: acc.to_dict() for name, acc in accumulators.items()}
    ranked = sorted(results, key=lambda name: results[name]["mae"][-1])
    return {
        "dataset": dataset,
        "path": str(path),
        "timesteps": int(values.shape[0]),
        "nodes": int(values.shape[1]),
        "test_samples": int(sample_count),
        "best_baseline": ranked[0],
        "methods": results,
    }


def main():
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "train_ratio": TRAIN_RATIO,
            "test_ratio": TEST_RATIO,
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "day_period": DAY_PERIOD,
            "week_period": WEEK_PERIOD,
            "batch_size": BATCH_SIZE,
            "metric": "masked MAE/RMSE/MAPE with label != 0, matching lib.utils.metric",
        },
        "datasets": {},
    }
    for dataset, path in DATASETS.items():
        payload["datasets"][dataset] = evaluate_dataset(dataset, path)
        best = payload["datasets"][dataset]["best_baseline"]
        mae = payload["datasets"][dataset]["methods"][best]["mae"][-1]
        print(f"{dataset}: best={best}, MAE={mae:.4f}")

    out = RESULT_DIR / "local_baseline_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
