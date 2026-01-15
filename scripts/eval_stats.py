from __future__ import annotations

import json
import math
import os
import random
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def resolve_seeds(
    base_seed: Optional[int],
    seeds: Optional[Sequence[int]],
    num_seeds: int,
    seed_step: int = 1,
) -> List[int]:
    if seeds:
        resolved = list(dict.fromkeys(int(seed) for seed in seeds))
    else:
        if base_seed is None:
            base_seed = 0
        if num_seeds <= 0:
            raise ValueError("num_seeds must be positive when seeds are not provided.")
        resolved = [int(base_seed) + seed_step * idx for idx in range(num_seeds)]
    if len(resolved) < 3:
        raise ValueError("Statistical evaluation requires at least 3 unique seeds.")
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(mean), float(std)


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def bootstrap_mean_ci(
    values: Sequence[float],
    num_samples: int,
    confidence: float,
    rng: random.Random,
) -> Optional[Tuple[float, float]]:
    if num_samples <= 0 or not values:
        return None
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1.")
    count = len(values)
    means: List[float] = []
    for _ in range(num_samples):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        means.append(statistics.fmean(sample))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = int(alpha * num_samples)
    high_idx = int((1.0 - alpha) * num_samples) - 1
    low_idx = max(0, min(low_idx, num_samples - 1))
    high_idx = max(0, min(high_idx, num_samples - 1))
    return float(means[low_idx]), float(means[high_idx])


def _student_t_pdf(x: float, df: int) -> float:
    numerator = math.gamma((df + 1.0) / 2.0)
    denominator = math.sqrt(df * math.pi) * math.gamma(df / 2.0)
    return (numerator / denominator) * (1.0 + (x * x) / df) ** (-(df + 1.0) / 2.0)


def _student_t_cdf(t_stat: float, df: int) -> float:
    # Numerical integration fallback for the Student-t CDF when scipy is unavailable.
    if t_stat == 0.0:
        return 0.5
    x = abs(t_stat)
    steps = min(100000, max(2000, int(2000 * x)))
    if steps % 2 == 1:
        steps += 1
    h = x / steps
    total = _student_t_pdf(0.0, df) + _student_t_pdf(x, df)
    for idx in range(1, steps):
        coeff = 4.0 if idx % 2 == 1 else 2.0
        total += coeff * _student_t_pdf(idx * h, df)
    integral = total * h / 3.0
    cdf = 0.5 + math.copysign(integral, t_stat)
    return max(0.0, min(1.0, cdf))


def paired_ttest(values_a: Sequence[float], values_b: Sequence[float]) -> Dict[str, float]:
    if len(values_a) != len(values_b):
        raise ValueError("paired_ttest expects equal-length samples.")
    pairs = []
    for a, b in zip(values_a, values_b):
        if a is None or b is None:
            continue
        a_val = float(a)
        b_val = float(b)
        if math.isnan(a_val) or math.isnan(b_val):
            continue
        pairs.append((a_val, b_val))
    n = len(pairs)
    if n < 2:
        return {"n": int(n), "t_stat": float("nan"), "p_value": float("nan")}
    diffs = [a - b for a, b in pairs]
    mean_diff = statistics.fmean(diffs)
    std_diff = statistics.stdev(diffs) if n > 1 else 0.0
    if std_diff == 0.0:
        t_stat = float("inf") if mean_diff != 0.0 else 0.0
        p_value = 0.0 if mean_diff != 0.0 else 1.0
        return {
            "n": int(n),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_diff": float(mean_diff),
        }
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    try:
        from scipy import stats  # type: ignore

        result = stats.ttest_rel(values_a, values_b, nan_policy="omit")
        return {
            "n": int(n),
            "t_stat": float(result.statistic),
            "p_value": float(result.pvalue),
            "mean_diff": float(mean_diff),
        }
    except Exception:
        cdf = _student_t_cdf(abs(t_stat), n - 1)
        p_value = 2.0 * (1.0 - cdf)
        return {
            "n": int(n),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_diff": float(mean_diff),
        }


def load_seed_summaries(output_dir: str, seeds: Iterable[int]) -> Tuple[List[dict], List[int]]:
    summaries: List[dict] = []
    missing: List[int] = []
    for seed in seeds:
        path = os.path.join(output_dir, f"seed_{seed}", "summary.json")
        if not os.path.isfile(path):
            missing.append(seed)
            continue
        with open(path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["seed"] = seed
        summaries.append(summary)
    return summaries, missing
