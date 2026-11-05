from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
DECIMAL3 = re.compile(r"-?\d+\.\d{3}\Z")


def write_jsonl_line(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def load_dataframe(path: str, sheet: str | int) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, sheet_name=sheet)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError("Input must be an .xlsx, .xls or .csv file")


def fit_train_channel_scaler(x: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Matching sklearn StandardScaler (population variance; constant std=>1)."""
    train = x[:train_end].astype(np.float64)
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=0)
    scale = np.where(std > 0, std, 1.0)
    return mean, scale


def window_revin(u_history: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[L,C] -> [L,C], [C], [C]; each feature independently over L.

    Nonaffine RevIN, variance unbiased=False, eps *inside* sqrt, as in
    torch.sqrt(torch.var(x, unbiased=False) + eps).
    """
    mean = np.mean(u_history, axis=0)
    stdev = np.sqrt(np.mean((u_history - mean) ** 2, axis=0) + eps)
    return (u_history - mean) / stdev, mean, stdev


def fmt_matrix(x: np.ndarray, fmt: str = "%.3f") -> str:
    return "\n".join(" ".join(fmt % float(v) for v in row) for row in x)


def fmt_timed_history(timestamps: np.ndarray, z_history: np.ndarray, fmt: str) -> str:
    return "\n".join(
        f"{time_value} " + " ".join(fmt % float(v) for v in row)
        for time_value, row in zip(timestamps, z_history)
    )


def target_start_indices(segment_start: int, segment_end: int, L: int, H: int, stride: int):
    """Starts of forecast targets, fully contained in one chronological split."""
    first = max(L, segment_start)
    last = segment_end - H
    if first <= last:
        yield from range(first, last + 1, stride)


def make_record(
    split: str,
    t0: int,
    u: np.ndarray,
    timestamps: np.ndarray,
    features: list[str],
    L: int,
    H: int,
    eps: float,
    fmt: str,
    include_assistant: bool,
) -> tuple[dict, dict]:
    hist_idx = t0 - L
    hist = u[hist_idx:t0]
    target_std = u[t0:t0 + H]
    z_hist, mu, sigma = window_revin(hist, eps)
    # IMPORTANT: SFT targets must be in the SAME window-RevIN coordinates as input.
    z_target = (target_std - mu) / sigma
    sid = f"{split}_{t0:08d}_L{L}_H{H}"
    nvars = len(features)
    system = (
        "You are a time series forecasting assistant. "
        f"Forecast the next {H} time steps for {nvars} variables in this exact order: "
        + ", ".join(features) + ". "
        f"Return exactly {H} lines enclosed in one pair of square brackets. "
        f"Each line must contain exactly {nvars} finite decimal numbers, each with "
        "exactly three digits after the decimal point, separated by single spaces. "
        "Negative signs and decimal points are allowed. "
        "Use newline characters between time steps. "
        "Do not output timestamps, explanations, or any other text."
    )
    record = {
        "sample_id": sid,
        "system": system,
        "user": "[Context (per-window RevIN-normalized data):\n"
        + fmt_timed_history(timestamps[hist_idx:t0], z_hist, fmt) + "]",
        "meta": {
            "hist_start": str(timestamps[hist_idx]),
            "hist_end": str(timestamps[t0 - 1]),
            "tgt_start": str(timestamps[t0]),
            "tgt_end": str(timestamps[t0 + H - 1]),
            "historical_row_start": int(hist_idx),
            "target_row_start": int(t0),
            "target_row_end_exclusive": int(t0 + H),
            "num_variables": int(nvars),
            "lookback": int(L),
            "horizon": int(H),
            "feature_order": features,
            "input_scale": "per-window channel-wise RevIN",
            "generated_output_scale": "per-window channel-wise RevIN",
        },
    }
    if include_assistant:
        record["assistant"] = "[" + fmt_matrix(z_target, fmt) + "]"
    # Private reference file: not passed into LLM prompts, especially at test time.
    ref = {
        "sample_id": sid,
        "split": split,
        "target_row_start": int(t0),
        "target_row_end_exclusive": int(t0 + H),
        "window_mean_in_standardized_space": mu.tolist(),
        "window_std_in_standardized_space": sigma.tolist(),
        "horizon": H,
        "num_variables": nvars,
    }
    return record, ref


def run_prepare(args: argparse.Namespace) -> None:
    if args.lookback <= 0 or args.horizon <= 0 or args.stride <= 0:
        raise ValueError("lookback, horizon and stride must be positive")
    if args.eps <= 0:
        raise ValueError("eps must be > 0")
    if not (0 < args.train_ratio < 1 and 0 <= args.val_ratio < 1):
        raise ValueError("Invalid train_ratio or val_ratio")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be below 1")
    if args.decimals != 3:
        raise ValueError("This prompt/parser contract currently requires 3 decimal places")

    df = load_dataframe(args.data_path, args.sheet)
    required = [args.time_col, *args.feature_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}; found: {list(df.columns)}")
    timestamps = df[args.time_col].astype(str).to_numpy()
    x = df[args.feature_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("NaN or infinity in input values; handle missing data first")
    n, c = x.shape
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + args.val_ratio))
    if args.val_ratio == 0:
        val_end = train_end
    if train_end <= args.lookback + args.horizon:
        raise ValueError("Training section too short for lookback + horizon")
    if n - val_end < args.horizon:
        raise ValueError("Test section too short for forecast horizon")
    if args.val_ratio > 0 and val_end - train_end < args.horizon:
        raise ValueError("Validation section too short for forecast horizon")

    train_mean, train_scale = fit_train_channel_scaler(x, train_end)
    u = (x - train_mean) / train_scale
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f"{prefix}_standardized_data.npz",
        data_std=u.astype(np.float64),
        train_mean=train_mean,
        train_scale=train_scale,
        train_end=np.int64(train_end),
        val_end=np.int64(val_end),
        feature_cols=np.array(args.feature_cols),
    )

    boundaries = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n),
    }
    counts = {}
    for split, (seg_start, seg_end) in boundaries.items():
        if seg_start == seg_end:
            continue
        input_path = f"{prefix}_{split}_inputs.jsonl"
        ref_path = f"{prefix}_{split}_references.jsonl"
        sft_path = f"{prefix}_{split}_sft.jsonl"
        counts[split] = 0
        # SFT labels are needed for train & validation only, never written for test.
        emit_sft = split != "test"
        with open(input_path, "w", encoding="utf-8") as fout, \
             open(ref_path, "w", encoding="utf-8") as fre:
            if emit_sft:
                sf = open(sft_path, "w", encoding="utf-8")
            else:
                sf = None
            try:
                for t0 in target_start_indices(seg_start, seg_end, args.lookback, args.horizon, args.stride):
                    rec, ref = make_record(
                        split, t0, u, timestamps, args.feature_cols,
                        args.lookback, args.horizon, args.eps, "%.3f", emit_sft,
                    )
                    if sf is not None:
                        write_jsonl_line(sf, rec)
                    # Inputs never include assistant targets.
                    rec.pop("assistant", None)
                    write_jsonl_line(fout, rec)
                    write_jsonl_line(fre, ref)
                    counts[split] += 1
            finally:
                if sf is not None:
                    sf.close()

    summary = {
        "original_data_path": str(Path(args.data_path).resolve()),
        "n_rows": n,
        "num_features": c,
        "feature_order": args.feature_cols,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "stride": args.stride,
        "fit_scaler_rows": [0, train_end],
        "splits_target_rows_half_open": {
            key: list(value) for key, value in boundaries.items()
        },
        "split_row_date_ranges": {
            key: [str(timestamps[start]), str(timestamps[end-1])] if start < end else []
            for key, (start, end) in boundaries.items()
        },
        "sample_counts": counts,
        "train_standardizer_mean": train_mean.tolist(),
        "train_standardizer_scale": train_scale.tolist(),
        "revin_eps_inside_sqrt": args.eps,
        "revin_statistics": "per sample, per channel, along lookback axis only",
        "prompt_numeric_scale": "window-RevIN-normalized training-standardized data",
        "sft_assistant_scale": "window-RevIN-normalized, rounded to 3 decimals",
        "evaluation_scale": "training-set-standardized values after inverse window RevIN",
        "test_predictions_are_excluded_from_prompts": True,
        "warning": "If you change the training/validation proportions or normalization from the original experiments, main-table results must be recomputed.",
    }
    with open(f"{prefix}_protocol.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"prefix": str(prefix), "row_boundaries": summary["splits_target_rows_half_open"],
                      "counts": counts, "scoring_scale": summary["evaluation_scale"]},
                     ensure_ascii=False, indent=2))


def parse_prediction(output: str, horizon: int, nvar: int) -> np.ndarray:
    """Strictly parse [H lines, C three-decimal finite signed numbers]."""
    if not isinstance(output, str):
        raise ValueError("prediction output is not a string")
    raw = output.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    lines = raw.split("\n")
    if len(lines) != horizon:
        raise ValueError(f"Expected {horizon} lines; got {len(lines)}")
    result = np.empty((horizon, nvar), dtype=np.float64)
    for i, line in enumerate(lines):
        # Preserve original whitespace contract (exactly one space between values).
        cells = line.split(" ")
        if len(cells) != nvar or any(not DECIMAL3.fullmatch(s) for s in cells):
            raise ValueError(f"Invalid numeric row {i+1}: expected {nvar} three-decimal values")
        result[i] = [float(s) for s in cells]
    if not np.isfinite(result).all():
        raise ValueError("Prediction contains non-finite values")
    return result


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def run_evaluate(args: argparse.Namespace) -> None:
    prefix = Path(args.out_prefix)
    with np.load(f"{prefix}_standardized_data.npz", allow_pickle=False) as npz:
        u = np.array(npz["data_std"], dtype=np.float64)
    refpath = Path(f"{prefix}_{args.split}_references.jsonl")
    refs = {r["sample_id"]: r for r in iter_jsonl(refpath)}
    if not refs:
        raise ValueError(f"No reference samples in {refpath}")
    preds = {}
    for p in iter_jsonl(Path(args.predictions)):
        sid = p.get("sample_id")
        if not sid:
            raise ValueError("Each prediction must include sample_id")
        if sid in preds:
            raise ValueError(f"Duplicate prediction sample_id: {sid}")
        preds[sid] = p.get("output", p.get("response", p.get("prediction")))
    missing = set(refs) - set(preds)
    extra = set(preds) - set(refs)
    if missing or extra:
        raise ValueError(f"Prediction/reference ID mismatch: missing={len(missing)}, extra={len(extra)}; "
                         f"examples missing={list(sorted(missing))[:3]}, extra={list(sorted(extra))[:3]}")

    sum_sq = 0.0
    sum_abs = 0.0
    n_elements = 0
    invalid = []
    per_sample = []
    for sid, ref in refs.items():
        try:
            z_pred = parse_prediction(preds[sid], ref["horizon"], ref["num_variables"])
        except ValueError as exc:
            invalid.append((sid, str(exc)))
            if args.invalid_policy == "error":
                raise ValueError(f"Invalid prediction sample {sid}: {exc}") from exc
            continue
        mu = np.array(ref["window_mean_in_standardized_space"])[None, :]
        sigma = np.array(ref["window_std_in_standardized_space"])[None, :]
        # Undo ONLY the inner window-RevIN. DO NOT inverse the train StandardScaler.
        pred_std = z_pred * sigma + mu
        start, end = ref["target_row_start"], ref["target_row_end_exclusive"]
        true_std = u[start:end]
        err = pred_std - true_std
        sse = float(np.sum(err**2))
        sae = float(np.sum(np.abs(err)))
        n = err.size
        sum_sq += sse
        sum_abs += sae
        n_elements += n
        per_sample.append({"sample_id": sid, "mse": sse / n, "mae": sae / n})
    if not n_elements:
        raise ValueError("No valid predictions to score")
    metrics = {
        "split": args.split,
        "total_samples": len(refs),
        "scored_samples": len(per_sample),
        "invalid_samples": len(invalid),
        "valid_rate": len(per_sample) / len(refs),
        "mse_standardized": sum_sq / n_elements,
        "mae_standardized": sum_abs / n_elements,
        "evaluation_definition": "z_pred -> inverse window RevIN -> TRAIN-StandardScaler space; compare with full-precision standardized target; global element-weighted mean",
        "invalid_policy": args.invalid_policy,
    }
    with open(f"{prefix}_{args.split}_scores.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(f"{prefix}_{args.split}_per_sample_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "mse", "mae"])
        writer.writeheader()
        writer.writerows(per_sample)
    if invalid:
        with open(f"{prefix}_{args.split}_invalid_predictions.jsonl", "w", encoding="utf-8") as f:
            for sid, reason in invalid:
                write_jsonl_line(f, {"sample_id": sid, "reason": reason})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if invalid:
        print("WARNING: invalid outputs were omitted from numerical averages. Do not treat "
              "valid-only metrics as complete-test performance; disclose coverage/invalid rate.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Create normalized SFT and inference files")
    prep.add_argument("--data-path", required=True, help="Original .xlsx / .csv in physical units")
    prep.add_argument("--sheet", default=0, help="Excel sheet name or index (default 0)")
    prep.add_argument("--time-col", default="date")
    prep.add_argument("--feature-cols", nargs="+", default=DEFAULT_FEATURES)
    prep.add_argument("--lookback", type=int, default=96)
    prep.add_argument("--horizon", type=int, default=96)
    prep.add_argument("--stride", type=int, default=1)
    prep.add_argument("--out-prefix", required=True)
    prep.add_argument("--train-ratio", type=float, default=0.7,
                      help="Default 0.7: training rows for scaler fitting & SFT; 0.1 validation and 0.2 test")
    prep.add_argument("--val-ratio", type=float, default=0.1,
                      help="Default 0.1; for old 80/20 split use --train-ratio 0.8 --val-ratio 0")
    prep.add_argument("--eps", type=float, default=1e-5)
    prep.add_argument("--decimals", type=int, default=3)
    eva = sub.add_parser("evaluate", help="Evaluate model outputs after inverse window RevIN")
    eva.add_argument("--out-prefix", required=True, help="Same path prefix used in prepare")
    eva.add_argument("--predictions", required=True, help="JSONL: {sample_id, output}")
    eva.add_argument("--split", choices=["train", "val", "test"], default="test")
    eva.add_argument("--invalid-policy", choices=["error", "skip"], default="error",
                     help="error (default): forbid silently dropping malformed forecasts")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        # Allow numeric sheet indices passed from CLI.
        if isinstance(args.sheet, str) and args.sheet.isdecimal():
            args.sheet = int(args.sheet)
        run_prepare(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
