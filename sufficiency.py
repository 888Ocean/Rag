"""Transparent evidence-sufficiency score for the answer-versus-refuse decision.

Two signals, positive weights that sum to 1, one threshold. The score is the
weighted sum. A signal pushes toward answering when its normalized value
clears that same threshold, and toward refusing when it falls short.
`dense_evidence` is the production refusal gate; this score is advisory there
and decisive only in the comparison CLI.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

__all__ = [
    "PRODUCTION_MIN_SCORE",
    "tokenize",
    "dense_evidence",
    "Signal",
    "SufficiencyConfig",
    "DEFAULT_CONFIG",
    "CoverageDetail",
    "Measurement",
    "Contribution",
    "Assessment",
    "measure",
]

PRODUCTION_MIN_SCORE = 0.60

_DENSE_FLOOR = 0.35
_DENSE_CEILING = 0.85
_FUNCTION_CHARS = frozenset(
    "的地得了着过之"
    "吗呢吧啊呀么"
    "什怎哪谁几多少何为"
    "是有在要会能可该应需"
    "我你他她它们这那此其每些个"
    "和与及或而且并把被给对从向跟以于到"
    "都也还就才又再很太更最"
)
_WEIGHT_SUM_TOLERANCE = 1e-9
_DECISION_TOLERANCE = 1e-9
_RUN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[a-z0-9]+")


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    lowered = text.lower()
    spans: list[tuple[str, int, int]] = []
    for match in _RUN.finditer(lowered):
        run = match.group()
        start = match.start()
        if _CJK.fullmatch(run):
            if len(run) == 1:
                spans.append((run, start, start + 1))
            else:
                spans.extend(
                    (run[index : index + 2], start + index, start + index + 2)
                    for index in range(len(run) - 1)
                )
        else:
            spans.append((run, start, match.end()))
    return spans


def tokenize(text: str) -> list[str]:
    """BM25 tokenizer: lowercased latin/digit runs, CJK character bigrams."""
    return [token for token, _start, _end in _token_spans(text)]


def dense_evidence(candidates: Sequence[dict], min_score: float) -> list[dict]:
    """Production gate: keep candidates whose dense score clears min_score."""
    return [
        item
        for item in candidates
        if item.get("dense_score") is not None and item["dense_score"] >= min_score
    ]


def _finite_dense(candidates: Sequence[Mapping]) -> list[float]:
    values: list[float] = []
    for item in candidates:
        score = item.get("dense_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        if not math.isfinite(score):
            continue
        values.append(float(score))
    return values


class Signal(str, Enum):
    TOP_DENSE = "top_dense"
    PASSAGE_COVERAGE = "passage_coverage"

    @property
    def label(self) -> str:
        if self is Signal.TOP_DENSE:
            return "语义相似度"
        if self is Signal.PASSAGE_COVERAGE:
            return "问题要点覆盖"
        raise AssertionError(self)


def _as_weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("信号权重必须是有限正数")
    return float(value)


@dataclass(frozen=True)
class SufficiencyConfig:
    """The only list of signals that enter the score. Every weight is positive."""

    weights: Mapping[Signal, float]
    threshold: float

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("至少要有一个信号")
        cleaned: dict[Signal, float] = {}
        for key, weight in self.weights.items():
            if not isinstance(key, Signal):
                raise ValueError(f"未知信号：{key!r}")
            number = _as_weight(weight)
            if number <= 0:
                raise ValueError("信号权重必须大于 0")
            cleaned[key] = number
        ordered = {signal: cleaned[signal] for signal in Signal if signal in cleaned}
        if abs(sum(ordered.values()) - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError("信号权重之和必须为 1")
        threshold = self.threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("阈值必须是 0 到 1 之间的数")
        if not math.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("阈值必须是 0 到 1 之间的数")
        object.__setattr__(self, "weights", MappingProxyType(ordered))
        object.__setattr__(self, "threshold", float(threshold))


# Chosen on eval/dev.jsonl only, before any held-out score was computed.
# Grid: accuracy, then false-answer rate, then false-refusal rate, then fewer signals.
DEFAULT_CONFIG = SufficiencyConfig(
    weights={Signal.TOP_DENSE: 0.45, Signal.PASSAGE_COVERAGE: 0.55},
    threshold=0.55,
)


@dataclass(frozen=True)
class CoverageDetail:
    covered: int
    total: int
    missing: tuple[str, ...]
    node_id: str | None
    source: str | None

    @property
    def fraction(self) -> float:
        return self.covered / self.total if self.total else 0.0


@dataclass(frozen=True)
class _Unit:
    kind: str
    start: int
    end: int
    text: str
    covered_by: frozenset[str]


def _content_units(question: str) -> list[_Unit]:
    spans = _token_spans(question)
    lowered = question.lower()
    tokens_at: dict[int, set[str]] = {}
    for token, start, end in spans:
        if _LATIN.fullmatch(token):
            continue
        for position in range(start, end):
            tokens_at.setdefault(position, set()).add(token)
    units: list[_Unit] = []
    seen_cjk: set[int] = set()
    for token, start, end in spans:
        if _LATIN.fullmatch(token):
            units.append(_Unit("latin", start, end, token, frozenset((token,))))
            continue
        for position in range(start, end):
            if position in seen_cjk or lowered[position] in _FUNCTION_CHARS:
                continue
            seen_cjk.add(position)
            units.append(
                _Unit(
                    "cjk",
                    position,
                    position + 1,
                    lowered[position],
                    frozenset(tokens_at.get(position, ())),
                )
            )
    units.sort(key=lambda unit: unit.start)
    return units


def _missing_fragments(units: Sequence[_Unit], terms: set[str]) -> tuple[str, ...]:
    missing: list[str] = []
    seen: set[str] = set()
    buffer: list[str] = []
    last_end = -1

    def flush() -> None:
        if not buffer:
            return
        text = "".join(buffer)
        buffer.clear()
        if text not in seen:
            seen.add(text)
            missing.append(text)

    for unit in units:
        covered = bool(unit.covered_by & terms)
        if unit.kind == "cjk" and not covered:
            if buffer and unit.start != last_end:
                flush()
            buffer.append(unit.text)
            last_end = unit.end
            continue
        flush()
        if unit.kind == "latin" and not covered and unit.text not in seen:
            seen.add(unit.text)
            missing.append(unit.text)
    flush()
    return tuple(missing)


def _coverage(question: str, candidates: Sequence[Mapping]) -> CoverageDetail:
    units = _content_units(question)
    best_count = -1
    best: Mapping | None = None
    best_terms: set[str] = set()
    for candidate in candidates:
        terms = set(tokenize(str(candidate.get("text") or "")))
        count = sum(1 for unit in units if unit.covered_by & terms)
        if count > best_count:
            best_count = count
            best = candidate
            best_terms = terms
    if best is None:
        return CoverageDetail(0, len(units), _missing_fragments(units, set()), None, None)
    covered = best_count if best_count > 0 else 0
    node_id = best.get("node_id")
    source = best.get("source")
    return CoverageDetail(
        covered,
        len(units),
        _missing_fragments(units, best_terms),
        None if node_id is None else str(node_id),
        None if source is None else str(source),
    )


@dataclass(frozen=True)
class Measurement:
    top_dense: float | None
    coverage: CoverageDetail

    def raw(self, signal: Signal) -> float | None:
        if signal is Signal.TOP_DENSE:
            return self.top_dense
        if signal is Signal.PASSAGE_COVERAGE:
            return self.coverage.fraction
        raise AssertionError(signal)

    def value(self, signal: Signal) -> float:
        if signal is Signal.TOP_DENSE:
            if self.top_dense is None:
                return 0.0
            span = _DENSE_CEILING - _DENSE_FLOOR
            scaled = (self.top_dense - _DENSE_FLOOR) / span
            return min(1.0, max(0.0, scaled))
        if signal is Signal.PASSAGE_COVERAGE:
            return self.coverage.fraction
        raise AssertionError(signal)

    def assess(self, config: SufficiencyConfig) -> Assessment:
        contributions = tuple(
            Contribution(signal, self.raw(signal), self.value(signal), weight)
            for signal, weight in config.weights.items()
        )
        coverage = self.coverage if Signal.PASSAGE_COVERAGE in config.weights else None
        return Assessment(contributions, config.threshold, coverage)


def measure(question: str, candidates: Sequence[Mapping]) -> Measurement:
    """Signal values for one question. Does not mutate candidates."""
    dense_values = _finite_dense(candidates)
    top_dense = max(dense_values) if dense_values else None
    return Measurement(top_dense, _coverage(question, candidates))


@dataclass(frozen=True)
class Contribution:
    signal: Signal
    raw: float | None
    value: float
    weight: float

    @property
    def points(self) -> float:
        return self.weight * self.value


def _direction(value: float, threshold: float) -> str:
    if value > threshold + _DECISION_TOLERANCE:
        return "up"
    if value + _DECISION_TOLERANCE < threshold:
        return "down"
    return "flat"


_DIRECTION_LABELS = {"up": "推高", "down": "压低", "flat": "持平"}


@dataclass(frozen=True)
class Assessment:
    contributions: tuple[Contribution, ...]
    threshold: float
    coverage: CoverageDetail | None

    @property
    def score(self) -> float:
        return sum(item.points for item in self.contributions)

    @property
    def sufficient(self) -> bool:
        return self.score + _DECISION_TOLERANCE >= self.threshold

    def to_dict(self) -> dict:
        def rounded(value: float | None) -> float | None:
            return None if value is None else round(float(value), 4)

        coverage = None
        if self.coverage is not None:
            coverage = {
                "covered": self.coverage.covered,
                "total": self.coverage.total,
                "missing": list(self.coverage.missing),
                "node_id": self.coverage.node_id,
                "source": self.coverage.source,
            }
        return {
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "sufficient": self.sufficient,
            "signals": [
                {
                    "signal": item.signal.value,
                    "label": item.signal.label,
                    "raw": rounded(item.raw),
                    "value": round(item.value, 4),
                    "weight": round(item.weight, 4),
                    "points": round(item.points, 4),
                    "direction": _direction(item.value, self.threshold),
                    "direction_label": _DIRECTION_LABELS[
                        _direction(item.value, self.threshold)
                    ],
                }
                for item in self.contributions
            ],
            "coverage": coverage,
        }
