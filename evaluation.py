from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from evidence_core import EvidenceCopilot


ProgressCallback = Callable[[int, str], None]


def _clean_list(values) -> list[str]:
    if isinstance(values, str):
        values = values.split(",")
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _source_matches(expected: str, candidate: dict) -> bool:
    needle = Path(expected.casefold()).stem
    values = {
        str(candidate.get("source", "")).casefold(),
        str(candidate.get("source_path", "")).casefold(),
    }
    return any(needle == Path(value).stem or needle in value for value in values)


class EvaluationStore:
    def __init__(self, root: Path):
        self.root = root
        self.eval_dir = root / "eval"
        self.cases_path = self.eval_dir / "cases.jsonl"
        self.runs_dir = self.eval_dir / "runs"
        self.eval_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)

    def list_cases(self) -> list[dict]:
        if not self.cases_path.exists():
            return []
        cases = []
        for line_number, line in enumerate(
            self.cases_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"评测集第 {line_number} 行格式错误") from exc
            if isinstance(value, dict):
                cases.append(value)
        return cases

    def save_case(self, payload: dict) -> dict:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValueError("评测问题不能为空")
        case_id = str(payload.get("id", "")).strip() or uuid.uuid4().hex[:12]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", case_id):
            raise ValueError("用例 ID 只能包含字母、数字、下划线和连字符")
        case = {
            "id": case_id,
            "question": question,
            "section_id": str(payload.get("section_id", "all")) or "all",
            "should_answer": bool(payload.get("should_answer", True)),
            "expected_sources": _clean_list(payload.get("expected_sources", [])),
            "expected_terms": _clean_list(payload.get("expected_terms", [])),
            "tags": _clean_list(payload.get("tags", [])),
            "note": str(payload.get("note", "")).strip(),
            "enabled": bool(payload.get("enabled", True)),
        }
        cases = self.list_cases()
        existing = next((index for index, item in enumerate(cases) if item.get("id") == case_id), None)
        if existing is None:
            cases.append(case)
        else:
            cases[existing] = case
        self._write_cases(cases)
        return case

    def delete_case(self, case_id: str) -> None:
        cases = self.list_cases()
        kept = [item for item in cases if item.get("id") != case_id]
        if len(kept) == len(cases):
            raise ValueError("评测用例不存在")
        self._write_cases(kept)

    def _write_cases(self, cases: list[dict]) -> None:
        temporary = self.cases_path.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in cases),
            encoding="utf-8",
        )
        temporary.replace(self.cases_path)

    def save_run(self, run: dict) -> None:
        target = self.runs_dir / f"{run['id']}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)

    def list_runs(self, limit: int = 20) -> list[dict]:
        runs = []
        for path in sorted(self.runs_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            runs.append(
                {
                    "id": value.get("id"),
                    "created_at": value.get("created_at"),
                    "case_count": value.get("case_count", 0),
                    "config_count": len(value.get("configs", [])),
                    "best": value.get("best"),
                    "full_generation": value.get("full_generation", False),
                    "elapsed_seconds": value.get("elapsed_seconds", 0),
                }
            )
        return runs

    def get_run(self, run_id: str) -> dict:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
            raise ValueError("评测记录 ID 无效")
        path = self.runs_dir / f"{run_id}.json"
        if not path.is_file():
            raise ValueError("评测记录不存在")
        return json.loads(path.read_text(encoding="utf-8"))


class RAGEvaluator:
    def __init__(self, engine: EvidenceCopilot, store: EvaluationStore):
        self.engine = engine
        self.store = store

    def run(
        self,
        retrieval_modes: list[str],
        top_ks: list[int],
        thresholds: list[float],
        full_generation: bool = False,
        progress: ProgressCallback | None = None,
    ) -> dict:
        progress = progress or (lambda _value, _message: None)
        modes = [mode for mode in retrieval_modes if mode in {"vector", "bm25", "hybrid"}]
        top_ks = sorted({min(max(int(value), 1), 12) for value in top_ks})
        thresholds = sorted({round(min(max(float(value), 0), 1), 2) for value in thresholds})
        if not modes or not top_ks or not thresholds:
            raise ValueError("请至少选择一种检索方式、Top-K 和阈值")
        cases = [case for case in self.store.list_cases() if case.get("enabled", True)]
        if not cases:
            raise ValueError("没有可运行的评测用例")

        combinations = [
            (mode, top_k, threshold)
            for mode in modes
            for top_k in top_ks
            for threshold in thresholds
        ]
        total_steps = len(combinations) * len(cases)
        completed = 0
        started = time.perf_counter()
        progress(5, f"已读取 {len(cases)} 条评测用例")
        configs = []
        for mode, top_k, threshold in combinations:
            details = []
            for case in cases:
                if full_generation:
                    answer_result = self.engine.ask(
                        case["question"],
                        top_k,
                        threshold,
                        lambda _value, _message: None,
                        case.get("section_id", "all"),
                        mode,
                    )
                    candidates = answer_result.get("candidates", [])
                    status = answer_result.get("status", "refused")
                    answer = answer_result.get("answer", "")
                    citations_valid = answer_result.get("citations_valid")
                    timing = answer_result.get("timing", {})
                else:
                    candidates, retrieval_seconds = self.engine.retrieve(
                        case["question"],
                        top_k,
                        lambda _value, _message: None,
                        case.get("section_id", "all"),
                        mode,
                    )
                    best_dense = max(
                        (item.get("dense_score") or 0 for item in candidates), default=0
                    )
                    status = "answered" if best_dense >= threshold else "refused"
                    answer = ""
                    citations_valid = None
                    timing = {
                        "retrieval_seconds": retrieval_seconds,
                        "generation_seconds": 0,
                        "total_seconds": retrieval_seconds,
                    }

                expected_sources = _clean_list(case.get("expected_sources", []))
                source_rank = next(
                    (
                        index
                        for index, candidate in enumerate(candidates, 1)
                        if any(
                            _source_matches(expected, candidate)
                            for expected in expected_sources
                        )
                    ),
                    None,
                )
                source_hit = source_rank is not None if expected_sources else None
                should_answer = bool(case.get("should_answer", True))
                actually_answered = status == "answered"
                decision_correct = actually_answered == should_answer
                expected_terms = _clean_list(case.get("expected_terms", []))
                matched_terms = [term for term in expected_terms if term.casefold() in answer.casefold()]
                term_coverage = (
                    len(matched_terms) / len(expected_terms)
                    if full_generation and expected_terms and actually_answered
                    else None
                )
                failures = []
                if source_hit is False:
                    failures.append("未召回预期来源")
                if not decision_correct:
                    failures.append("应回答却拒答" if should_answer else "应拒答却回答")
                if full_generation and actually_answered and citations_valid is not True:
                    failures.append("引用校验失败")
                if term_coverage is not None and term_coverage < 1:
                    failures.append("答案要点不完整")
                details.append(
                    {
                        "case_id": case.get("id"),
                        "question": case["question"],
                        "should_answer": should_answer,
                        "status": status,
                        "decision_correct": decision_correct,
                        "expected_sources": expected_sources,
                        "source_hit": source_hit,
                        "source_rank": source_rank,
                        "expected_terms": expected_terms,
                        "matched_terms": matched_terms,
                        "term_coverage": term_coverage,
                        "citations_valid": citations_valid,
                        "answer": answer,
                        "failures": failures,
                        "timing": {key: round(float(value or 0), 4) for key, value in timing.items()},
                        "candidates": candidates,
                    }
                )
                completed += 1
                value = 8 + round(completed / total_steps * 84)
                progress(value, f"正在评测 {mode} · Top-{top_k} · {threshold:.2f}")
            configs.append(
                {
                    "retrieval_mode": mode,
                    "top_k": top_k,
                    "min_score": threshold,
                    "metrics": self._metrics(details),
                    "details": details,
                }
            )

        best_config = max(
            configs,
            key=lambda item: (
                item["metrics"]["composite_score"],
                -item["metrics"]["average_total_seconds"],
            ),
        )
        run = {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "case_count": len(cases),
            "full_generation": full_generation,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "best": {
                "retrieval_mode": best_config["retrieval_mode"],
                "top_k": best_config["top_k"],
                "min_score": best_config["min_score"],
                "metrics": best_config["metrics"],
            },
            "configs": configs,
        }
        progress(96, "正在保存评测报告")
        self.store.save_run(run)
        progress(100, "评测完成")
        return run

    @staticmethod
    def _metrics(details: list[dict]) -> dict:
        retrieval_details = [item for item in details if item["source_hit"] is not None]
        hit_at_k = (
            sum(1 for item in retrieval_details if item["source_hit"]) / len(retrieval_details)
            if retrieval_details
            else 0
        )
        mrr = (
            sum(1 / item["source_rank"] if item["source_rank"] else 0 for item in retrieval_details)
            / len(retrieval_details)
            if retrieval_details
            else 0
        )
        decision_accuracy = sum(item["decision_correct"] for item in details) / len(details)
        predicted_refusals = [item for item in details if item["status"] != "answered"]
        expected_refusals = [item for item in details if not item["should_answer"]]
        correct_refusals = [
            item for item in predicted_refusals if not item["should_answer"]
        ]
        refusal_precision = (
            len(correct_refusals) / len(predicted_refusals) if predicted_refusals else 1
        )
        refusal_recall = (
            len(correct_refusals) / len(expected_refusals) if expected_refusals else 1
        )
        answerable = [item for item in details if item["should_answer"]]
        false_refusal_rate = (
            sum(item["status"] != "answered" for item in answerable) / len(answerable)
            if answerable
            else 0
        )
        citation_details = [
            item for item in details if item["citations_valid"] is not None
        ]
        citation_accuracy = (
            sum(item["citations_valid"] is True for item in citation_details)
            / len(citation_details)
            if citation_details
            else None
        )
        term_details = [item for item in details if item["term_coverage"] is not None]
        term_coverage = (
            sum(item["term_coverage"] for item in term_details) / len(term_details)
            if term_details
            else None
        )
        average = lambda key: sum(item["timing"].get(key, 0) for item in details) / len(details)
        composite = hit_at_k * 0.4 + mrr * 0.25 + decision_accuracy * 0.35
        return {
            "hit_at_k": round(hit_at_k, 4),
            "mrr": round(mrr, 4),
            "decision_accuracy": round(decision_accuracy, 4),
            "refusal_precision": round(refusal_precision, 4),
            "refusal_recall": round(refusal_recall, 4),
            "false_refusal_rate": round(false_refusal_rate, 4),
            "citation_accuracy": None if citation_accuracy is None else round(citation_accuracy, 4),
            "term_coverage": None if term_coverage is None else round(term_coverage, 4),
            "average_retrieval_seconds": round(average("retrieval_seconds"), 4),
            "average_generation_seconds": round(average("generation_seconds"), 4),
            "average_total_seconds": round(average("total_seconds"), 4),
            "failure_count": sum(bool(item["failures"]) for item in details),
            "composite_score": round(composite, 4),
        }
