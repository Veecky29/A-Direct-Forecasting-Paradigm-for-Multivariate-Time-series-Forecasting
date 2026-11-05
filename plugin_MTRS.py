from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from swift.plugin import ORM, orms

FORMAT_PENALTY = -5.0
LENGTH_PENALTY_MODE = 'relative'
REWARD_SPACE = 'standardized'
STRICT_DECIMAL_PLACES = 3

_NUMBER = re.compile(r'^-?(?:0|[1-9]\d*)\.\d{' + str(STRICT_DECIMAL_PLACES) + r'}$')


@dataclass
class Parsed:
    rows: list[Optional[list[float]]]
    envelope_ok: bool

    @property
    def any_format_error(self) -> bool:
        return not self.envelope_ok or not self.rows or any(row is None for row in self.rows)


def _to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict) and isinstance(completion.get('content'), str):
        return completion['content']
    return ''


def _parse_row_tokens(tokens: list[str], c: int, *, strict: bool) -> Optional[list[float]]:
    if len(tokens) != c:
        return None
    out = []
    for token in tokens:
        if strict and _NUMBER.fullmatch(token) is None:
            return None
        try:
            value = float(token)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value):
            return None
        out.append(value)
    return out


def _parse_matrix(text: Any, c: int, *, completion: bool) -> Parsed:
    if c <= 0:
        raise ValueError('num_variables must be a positive integer')
    if not isinstance(text, str):
        if not completion and isinstance(text, list):
            data = text
            rows = []
            if data and not isinstance(data[0], (list, tuple)):
                data = [[v] for v in data]
            for row in data:
                if not isinstance(row, (list, tuple)):
                    rows.append(None)
                    continue
                rows.append(_parse_row_tokens([str(x) for x in row], c, strict=False))
            return Parsed(rows=rows, envelope_ok=True)
        text = _to_text(text)

    raw = text.strip()
    envelope_ok = raw.startswith('[') and raw.endswith(']')

    if not completion and envelope_ok:
        try:
            data = json.loads(raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        except (ValueError, TypeError):
            data = None
        if isinstance(data, list):
            if data and not isinstance(data[0], list):
                data = [[v] for v in data]
            rows = []
            for row in data:
                if not isinstance(row, list):
                    rows.append(None)
                else:
                    rows.append(_parse_row_tokens([str(x) for x in row], c, strict=False))
            return Parsed(rows=rows, envelope_ok=True)


    body = raw
    if body.startswith('['):
        body = body[1:]
    if body.endswith(']'):
        body = body[:-1]
    body = body.strip('\r\n')
    if not body.strip():
        return Parsed(rows=[], envelope_ok=envelope_ok)

    rows: list[Optional[list[float]]] = []
    for line in body.splitlines():
        line = line.strip()
        tokens = line.split()
        rows.append(_parse_row_tokens(tokens, c, strict=completion))
    return Parsed(rows=rows, envelope_ok=envelope_ok)


def _as_batch(value: Any, n: int, name: str) -> list:
    if isinstance(value, (str, dict)):
        if n == 1:
            return [value]
        raise ValueError(f'{name} must supply one value per completion (expected {n})')
    if isinstance(value, (list, tuple)) and len(value) == n:
        return list(value)
    if n == 1 and value is not None:
        return [value]
    raise ValueError(f'{name}: expected {n} values but got {type(value).__name__}')


def _get_solutions(solution: Any, kwargs: dict, n: int) -> list:
    if solution is None:
        for key in ('solution', 'ground_truth', 'target', 'gt', 'labels'):
            if kwargs.get(key) is not None:
                solution = kwargs[key]
                break
    if solution is None:
        raise ValueError('MTRS requires a top-level `solution` column for every GRPO sample')
    return _as_batch(solution, n, 'solution')


def _get_optional_stat(kwargs: dict, keys: tuple[str, ...], n: int) -> Optional[list]:
    for key in keys:
        if kwargs.get(key) is not None:
            return _as_batch(kwargs[key], n, key)
    return None


def _std_vector(value: Any, c: int) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError('revin_std string must be a JSON numeric array') from exc
    if not isinstance(value, (list, tuple)) or len(value) != c:
        raise ValueError(f'revin_std must contain exactly {c} per-channel statistics')
    out = []
    for v in value:
        try:
            s = float(v)
        except (ValueError, TypeError):
            raise ValueError('revin_std contains a non-numeric value') from None
        if not math.isfinite(s) or s <= 0:
            raise ValueError('revin_std must be finite and strictly positive')
        out.append(s)
    return out


def _context(completions: list, solution: Any, kwargs: dict):
    n = len(completions)
    solutions = _get_solutions(solution, kwargs, n)
    horizons = []
    channels = []
    references = []
    predictions = []
    hvals = _get_optional_stat(kwargs, ('horizon', 'H'), n)
    cvals = _get_optional_stat(kwargs, ('num_variables', 'C'), n)
    for i, gt in enumerate(solutions):
        if cvals is None:
            raise ValueError('MTRS requires a top-level `num_variables` column')
        c = int(cvals[i])
        truth = _parse_matrix(gt, c, completion=False)
        if not truth.envelope_ok or not truth.rows or any(row is None for row in truth.rows):
            raise ValueError(f'Invalid reference solution at completion index {i}')
        h = int(hvals[i]) if hvals is not None else len(truth.rows)
        if h <= 0 or len(truth.rows) != h:
            raise ValueError(f'Ground-truth horizon mismatch at completion index {i}')
        channels.append(c)
        horizons.append(h)
        references.append(truth)
        predictions.append(_parse_matrix(completions[i], c, completion=True))
    return horizons, channels, references, predictions


def _scores(completions, solution, kwargs):
    """Single source of truth; Swift independently invokes all three wrappers."""
    horizons, channels, references, predictions = _context(completions, solution, kwargs)
    n = len(completions)
    stat_list = _get_optional_stat(
        kwargs,
        ('revin_std', 'window_std_in_standardized_space'),
        n,
    )
    if REWARD_SPACE == 'standardized' and stat_list is None:
        raise ValueError(
            'standardized-space MSE requires per-sample `revin_std` column '
            '(the historical-window RevIN std in training-standardized space)'
        )
    if REWARD_SPACE not in ('standardized', 'revin'):
        raise ValueError(f'Unknown REWARD_SPACE: {REWARD_SPACE}')

    result = []
    for i, (h, c, truth, pred) in enumerate(zip(horizons, channels, references, predictions)):
        fmt_reward = FORMAT_PENALTY if pred.any_format_error else 0.0
        if LENGTH_PENALTY_MODE == 'relative':
            len_reward = -abs(len(pred.rows) - h) / h
        elif LENGTH_PENALTY_MODE == 'per_step':
            len_reward = -float(abs(len(pred.rows) - h))
        else:
            raise ValueError(f'Unknown LENGTH_PENALTY_MODE: {LENGTH_PENALTY_MODE}')
        std_vec = _std_vector(stat_list[i], c) if stat_list is not None and REWARD_SPACE == 'standardized' else [1.0] * c

        squared_errors = []
        for t in range(min(len(pred.rows), h)):
            pred_row = pred.rows[t]
            if pred_row is None:
                continue
            gt_row = truth.rows[t]
            row_errors = []
            try:
                for j in range(c):
                    e = (pred_row[j] - gt_row[j]) * std_vec[j]
                    squared = e * e
                    if not math.isfinite(squared):
                        raise OverflowError('squared prediction error overflow')
                    row_errors.append(squared)
            except (OverflowError, ValueError):
                fmt_reward = FORMAT_PENALTY
                continue
            squared_errors.extend(row_errors)
        if squared_errors:
            mse_reward = -math.fsum(v / len(squared_errors) for v in squared_errors)
        else:
            mse_reward = 0.0
        result.append((float(fmt_reward), float(len_reward), float(mse_reward)))
    return result


class TSFormatORM(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        return [r[0] for r in _scores(completions, solution, kwargs)]


class TSLengthORM(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        return [r[1] for r in _scores(completions, solution, kwargs)]


class TSMSEORM(ORM):
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        return [r[2] for r in _scores(completions, solution, kwargs)]


orms['ts_format'] = TSFormatORM
orms['ts_length'] = TSLengthORM
orms['ts_mse'] = TSMSEORM
