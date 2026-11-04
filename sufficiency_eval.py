"""Compare the dense refusal gate with the sufficiency scorer.

Selection reads only eval/dev.jsonl. The default command scores the frozen
constants on dev and held-out and does not search.

    python sufficiency_eval.py --select
    python sufficiency_eval.py

Decision rule, fixed before looking at held-out: higher accuracy, then lower
false-answer rate, then lower false-refusal rate, then fewer signals, then a
larger gap between the score and the threshold. One held-out case moves every
rate by 1/n. A one-case gap is the resolution of that set, not a stable win.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from sufficiency import (
    DEFAULT_CONFIG,
    PRODUCTION_MIN_SCORE,
    Assessment,
    Measurement,
    Signal,
    SufficiencyConfig,
    dense_evidence,
    measure,
)

ROOT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "eval" / "dev.jsonl"
HELDOUT_PATH = ROOT / "eval" / "heldout.jsonl"
RETRIEVAL_MODE = "hybrid"
TOP_K = 4

# Chosen on eval/dev.jsonl only, before any held-out score was computed.
DEV_BASELINE_THRESHOLD = 0.615

_BASELINE_GRID = tuple(round(0.35 + 0.005 * i, 3) for i in range(101))
_SCORER_THRESHOLD_GRID = tuple(round(0.01 * i, 2) for i in range(101))
_WEIGHT_UNITS = 20


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    should_answer: bool
    section_id: str
    tags: tuple[str, ...]


def load_cases(path: Path) -> list[Case]:
    """Parse a split file. Disabled rows are skipped."""
    cases: list[Case] = []
    seen: set[str] = set()
    text = Path(path).read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_number} 行不是 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path} 第 {line_number} 行不是对象")
        if payload.get("enabled", True) is False:
            continue
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError(f"{path} 第 {line_number} 行问题为空")
        should_answer = payload.get("should_answer")
        if not isinstance(should_answer, bool):
            raise ValueError(f"{path} 第 {line_number} 行 should_answer 必须是布尔值")
        case_id = str(payload.get("id", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"{path} 第 {line_number} 行用例 ID 无效或重复")
        seen.add(case_id)
        tags = payload.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list):
            tags = []
        cases.append(
            Case(
                case_id,
                question,
                should_answer,
                str(payload.get("section_id", "all")) or "all",
                tuple(str(tag) for tag in tags if str(tag).strip()),
            )
        )
    return cases


@dataclass(frozen=True)
class RetrievedCase:
    case: Case
    candidates: tuple[dict, ...]
    measurement: Measurement


class Outcome:
    __slots__ = ("should_answer", "answered")

    def __init__(self, should_answer: bool, answered: bool):
        self.should_answer = should_answer
        self.answered = answered


@dataclass(frozen=True)
class DecisionMetrics:
    """Counts are the source of truth. An empty denominator is None, not 1."""

    correct_answers: int
    correct_refusals: int
    false_answers: int
    false_refusals: int

    def __post_init__(self) -> None:
        counts = (
            self.correct_answers,
            self.correct_refusals,
            self.false_answers,
            self.false_refusals,
        )
        if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
            raise ValueError("混淆计数必须是非负整数")
        if sum(counts) == 0:
            raise ValueError("没有可统计的用例")

    @property
    def n(self) -> int:
        return (
            self.correct_answers
            + self.correct_refusals
            + self.false_answers
            + self.false_refusals
        )

    @property
    def accuracy(self) -> float:
        return (self.correct_answers + self.correct_refusals) / self.n

    @property
    def _predicted_refusals(self) -> int:
        return self.correct_refusals + self.false_refusals

    @property
    def _should_refuse(self) -> int:
        return self.correct_refusals + self.false_answers

    @property
    def _should_answer(self) -> int:
        return self.correct_answers + self.false_refusals

    @property
    def refusal_precision(self) -> float | None:
        if self._predicted_refusals == 0:
            return None
        return self.correct_refusals / self._predicted_refusals

    @property
    def refusal_recall(self) -> float | None:
        if self._should_refuse == 0:
            return None
        return self.correct_refusals / self._should_refuse

    @property
    def false_answer_rate(self) -> float | None:
        if self._should_refuse == 0:
            return None
        return self.false_answers / self._should_refuse

    @property
    def false_refusal_rate(self) -> float | None:
        if self._should_answer == 0:
            return None
        return self.false_refusals / self._should_answer


def decision_metrics(outcomes: Iterable[Outcome]) -> DecisionMetrics:
    correct_answers = correct_refusals = false_answers = false_refusals = 0
    for outcome in outcomes:
        if outcome.answered and outcome.should_answer:
            correct_answers += 1
        elif outcome.answered and not outcome.should_answer:
            false_answers += 1
        elif not outcome.answered and outcome.should_answer:
            false_refusals += 1
        else:
            correct_refusals += 1
    return DecisionMetrics(correct_answers, correct_refusals, false_answers, false_refusals)


@dataclass(frozen=True)
class CaseRow:
    case_id: str
    should_answer: bool
    top_dense: float | None
    production_answered: bool
    baseline_answered: bool
    assessment: Assessment


@dataclass(frozen=True)
class Comparison:
    baseline_threshold: float
    config: SufficiencyConfig
    production: DecisionMetrics
    baseline: DecisionMetrics
    scorer: DecisionMetrics
    always_answer: DecisionMetrics
    always_refuse: DecisionMetrics
    rows: tuple[CaseRow, ...]

    def _paired(self, baseline_flag: str) -> tuple[int, int]:
        scorer_only = baseline_only = 0
        for row in self.rows:
            base_ok = getattr(row, baseline_flag) == row.should_answer
            scorer_ok = row.assessment.sufficient == row.should_answer
            if scorer_ok and not base_ok:
                scorer_only += 1
            elif base_ok and not scorer_ok:
                baseline_only += 1
        return scorer_only, baseline_only

    @property
    def scorer_only_correct(self) -> int:
        return self._paired("baseline_answered")[0]

    @property
    def baseline_only_correct(self) -> int:
        return self._paired("baseline_answered")[1]


def compare(
    rows: Sequence[RetrievedCase],
    baseline_threshold: float,
    config: SufficiencyConfig,
) -> Comparison:
    case_rows: list[CaseRow] = []
    for row in rows:
        assessment = row.measurement.assess(config)
        case_rows.append(
            CaseRow(
                row.case.id,
                row.case.should_answer,
                row.measurement.top_dense,
                bool(dense_evidence(row.candidates, PRODUCTION_MIN_SCORE)),
                bool(dense_evidence(row.candidates, baseline_threshold)),
                assessment,
            )
        )
    frozen = tuple(case_rows)

    def outcomes(answered_of) -> list[Outcome]:
        return [Outcome(row.should_answer, answered_of(row)) for row in frozen]

    return Comparison(
        baseline_threshold,
        config,
        decision_metrics(outcomes(lambda row: row.production_answered)),
        decision_metrics(outcomes(lambda row: row.baseline_answered)),
        decision_metrics(outcomes(lambda row: row.assessment.sufficient)),
        decision_metrics(outcomes(lambda _row: True)),
        decision_metrics(outcomes(lambda _row: False)),
        frozen,
    )


@dataclass(frozen=True)
class Selected:
    rule: object
    metrics: DecisionMetrics
    margin: float


def _rate(value: float | None) -> float:
    return 0.0 if value is None else value


def _rank_key(metrics: DecisionMetrics, signal_count: int, margin: float) -> tuple:
    return (
        -metrics.accuracy,
        _rate(metrics.false_answer_rate),
        _rate(metrics.false_refusal_rate),
        signal_count,
        -margin,
    )


def _baseline_margin(rows: Sequence[RetrievedCase], threshold: float) -> float:
    gaps = [
        abs(row.measurement.top_dense - threshold)
        for row in rows
        if row.measurement.top_dense is not None
    ]
    return min(gaps) if gaps else 0.0


def select_baseline(
    rows: Sequence[RetrievedCase], grid: Sequence[float] = _BASELINE_GRID
) -> Selected:
    if not rows:
        raise ValueError("没有可用于选择的用例")
    best: Selected | None = None
    best_key: tuple | None = None
    for threshold in grid:
        outcomes = [
            Outcome(row.case.should_answer, bool(dense_evidence(row.candidates, threshold)))
            for row in rows
        ]
        metrics = decision_metrics(outcomes)
        margin = _baseline_margin(rows, threshold)
        key = _rank_key(metrics, 1, margin)
        if best_key is None or key < best_key:
            best_key = key
            best = Selected(float(threshold), metrics, margin)
    assert best is not None
    return best


def _scorer_grid() -> list[SufficiencyConfig]:
    configs: list[SufficiencyConfig] = []
    signals = list(Signal)
    for size in range(1, len(signals) + 1):
        for subset in combinations(signals, size):
            if len(subset) == 1:
                weightings = [{subset[0]: 1.0}]
            else:
                weightings = []
                for units in range(1, _WEIGHT_UNITS):
                    weights = {
                        subset[0]: units / _WEIGHT_UNITS,
                        subset[1]: (_WEIGHT_UNITS - units) / _WEIGHT_UNITS,
                    }
                    weightings.append(weights)
            for weights in weightings:
                for threshold in _SCORER_THRESHOLD_GRID:
                    configs.append(SufficiencyConfig(weights, threshold))
    return configs


def select_scorer(
    rows: Sequence[RetrievedCase], grid: Iterable[SufficiencyConfig] | None = None
) -> Selected:
    if not rows:
        raise ValueError("没有可用于选择的用例")
    best: Selected | None = None
    best_key: tuple | None = None
    for config in grid if grid is not None else _scorer_grid():
        gaps: list[float] = []
        outcomes: list[Outcome] = []
        for row in rows:
            assessment = row.measurement.assess(config)
            outcomes.append(Outcome(row.case.should_answer, assessment.sufficient))
            gaps.append(abs(assessment.score - config.threshold))
        metrics = decision_metrics(outcomes)
        margin = min(gaps) if gaps else 0.0
        key = _rank_key(metrics, len(config.weights), margin)
        if best_key is None or key < best_key:
            best_key = key
            best = Selected(config, metrics, margin)
    assert best is not None
    return best


def _retrieve_all(cases: Sequence[Case]) -> list[RetrievedCase]:
    os.environ.setdefault("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from evidence_core import EvidenceCopilot

    engine = EvidenceCopilot(ROOT)
    rows: list[RetrievedCase] = []
    for case in cases:
        candidates, _seconds = engine.retrieve(
            case.question,
            TOP_K,
            lambda _value, _message: None,
            case.section_id,
            RETRIEVAL_MODE,
        )
        frozen = tuple(candidates)
        rows.append(RetrievedCase(case, frozen, measure(case.question, frozen)))
    return rows


def _format_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _format_metrics(name: str, metrics: DecisionMetrics) -> str:
    cells = [
        name,
        _format_rate(metrics.accuracy),
        _format_rate(metrics.refusal_precision),
        _format_rate(metrics.refusal_recall),
        _format_rate(metrics.false_answer_rate),
        _format_rate(metrics.false_refusal_rate),
        str(metrics.correct_answers),
        str(metrics.false_answers),
        str(metrics.false_refusals),
        str(metrics.correct_refusals),
        str(metrics.n),
    ]
    return "| " + " | ".join(cells) + " |"


def _config_text(config: SufficiencyConfig) -> str:
    weights = ", ".join(
        f"{signal.value}={weight:.2f}" for signal, weight in config.weights.items()
    )
    return f"sufficiency[{weights}] t={config.threshold:.2f}"


def _config_literal(config: SufficiencyConfig) -> str:
    parts = ", ".join(
        f"Signal.{signal.name}: {weight:.2f}" for signal, weight in config.weights.items()
    )
    return (
        "DEFAULT_CONFIG = SufficiencyConfig(\n"
        f"    weights={{{parts}}},\n"
        f"    threshold={config.threshold:.2f},\n"
        ")"
    )


def format_comparison(title: str, report: Comparison) -> str:
    header = (
        "| rule | accuracy | refusal_precision | refusal_recall | false_answer_rate | "
        "false_refusal_rate | TA | FA | FR | TR | n |"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    versus_tuned_scorer, versus_tuned_base = report._paired("baseline_answered")
    versus_default_scorer, versus_default_base = report._paired("production_answered")
    lines = [
        title,
        header,
        separator,
        _format_metrics("always-answer", report.always_answer),
        _format_metrics("always-refuse", report.always_refuse),
        _format_metrics(f"baseline-default t={PRODUCTION_MIN_SCORE:.2f}", report.production),
        _format_metrics(f"baseline-dev t={report.baseline_threshold:.3f}", report.baseline),
        _format_metrics(_config_text(report.config), report.scorer),
        (
            f"paired vs baseline-dev: scorer_only_correct={versus_tuned_scorer} "
            f"baseline_only_correct={versus_tuned_base} "
            f"(delta cases={versus_tuned_scorer - versus_tuned_base})"
        ),
        (
            f"paired vs baseline-default: scorer_only_correct={versus_default_scorer} "
            f"baseline_only_correct={versus_default_base} "
            f"(delta cases={versus_default_scorer - versus_default_base})"
        ),
    ]
    return "\n".join(lines)


def _dense_diagnostic(rows: Sequence[RetrievedCase]) -> str:
    values = [row.measurement.top_dense for row in rows if row.measurement.top_dense is not None]
    if not values:
        return "top_dense: no finite scores"
    ordered = sorted(values)
    mid = ordered[len(ordered) // 2]
    saturated = sum(value >= 0.85 for value in values) / len(values)
    floored = sum(value <= 0.35 for value in values) / len(values)
    return (
        f"top_dense n={len(values)} min={ordered[0]:.4f} median={mid:.4f} "
        f"max={ordered[-1]:.4f} share>=0.85={saturated:.2f} share<=0.35={floored:.2f}"
    )


def _subset_winners(rows: Sequence[RetrievedCase]) -> str:
    lines = ["best config inside each signal subset (dev):"]
    signals = list(Signal)
    for size in range(1, len(signals) + 1):
        for subset in combinations(signals, size):
            subset_grid = [
                config
                for config in _scorer_grid()
                if tuple(config.weights) == subset
            ]
            chosen = select_scorer(rows, subset_grid)
            config = chosen.rule
            assert isinstance(config, SufficiencyConfig)
            lines.append(
                f"  {_config_text(config)} acc={chosen.metrics.accuracy:.4f} "
                f"FA={_format_rate(chosen.metrics.false_answer_rate)} "
                f"FR={_format_rate(chosen.metrics.false_refusal_rate)} "
                f"TA={chosen.metrics.correct_answers} FA={chosen.metrics.false_answers} "
                f"FR={chosen.metrics.false_refusals} TR={chosen.metrics.correct_refusals}"
            )
    return "\n".join(lines)


def render_split(name: str, path: Path, threshold: float, config: SufficiencyConfig) -> str:
    rows = _retrieve_all(load_cases(path))
    answers = sum(row.case.should_answer for row in rows)
    report = compare(rows, threshold, config)
    title = (
        f"split={name} n={len(rows)} answer={answers} refuse={len(rows) - answers} "
        f"file={path.name}"
    )
    return format_comparison(title, report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the dense-gate baseline against the sufficiency scorer."
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="Choose the baseline threshold and scorer config on eval/dev.jsonl only.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.select:
        rows = _retrieve_all(load_cases(DEV_PATH))
        baseline = select_baseline(rows)
        scorer = select_scorer(rows)
        config = scorer.rule
        assert isinstance(config, SufficiencyConfig)
        threshold = float(baseline.rule)
        report = compare(rows, threshold, config)
        answers = sum(row.case.should_answer for row in rows)
        text = "\n".join(
            [
                "selection split=dev only; held-out was not read",
                f"grid scorer={len(_scorer_grid())} baseline={len(_BASELINE_GRID)}",
                _dense_diagnostic(rows),
                _subset_winners(rows),
                format_comparison(
                    f"split=dev n={len(rows)} answer={answers} refuse={len(rows) - answers} selected",
                    report,
                ),
                f"DEV_BASELINE_THRESHOLD = {threshold:.3f}",
                _config_literal(config),
            ]
        )
        print(text)
        if args.out:
            args.out.write_text(text + "\n", encoding="utf-8")
        return 0

    text = "\n\n".join(
        [
            "frozen constants; this command does not select",
            f"DEV_BASELINE_THRESHOLD = {DEV_BASELINE_THRESHOLD:.3f}",
            _config_literal(DEFAULT_CONFIG),
            render_split("dev", DEV_PATH, DEV_BASELINE_THRESHOLD, DEFAULT_CONFIG),
            render_split("heldout", HELDOUT_PATH, DEV_BASELINE_THRESHOLD, DEFAULT_CONFIG),
        ]
    )
    print(text)
    if args.out:
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
