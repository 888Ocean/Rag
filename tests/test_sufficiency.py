import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from sufficiency import (
    Signal,
    SufficiencyConfig,
    dense_evidence,
    measure,
    tokenize,
)
from sufficiency_eval import (
    Case,
    DecisionMetrics,
    Outcome,
    RetrievedCase,
    decision_metrics,
    load_cases,
    main,
    select_baseline,
    select_scorer,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_IDS = {
    "coffee-001",
    "coffee-002",
    "coffee-003",
    "coffee-004",
    "coffee-005",
    "coffee-006",
    "coffee-007",
    "coffee-008",
    "coffee-009",
    "refuse-001",
    "refuse-002",
    "refuse-003",
    "refuse-004",
    "refuse-005",
    "refuse-006",
}


def _oracle_tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)
    output: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                output.append(token)
            else:
                output.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            output.append(token)
    return output


def _case(case_id: str, should_answer: bool) -> Case:
    return Case(case_id, case_id, should_answer, "all", ())


def _retrieved(case: Case, top_dense: float | None, covered: int, total: int) -> RetrievedCase:
    from sufficiency import CoverageDetail, Measurement

    text = "x" * covered
    measurement = Measurement(
        top_dense,
        CoverageDetail(covered, total, (), "n", "s"),
    )
    candidates = ({"dense_score": top_dense, "text": text, "node_id": "n", "source": "s"},)
    return RetrievedCase(case, candidates, measurement)


class ScorerBehaviorTest(unittest.TestCase):
    def test_tokenizer_matches_the_previous_bm25_tokenizer(self):
        samples = ["", "豆", "咖啡", "House Blend", "深度烘焙Bean12", "A1咖啡", "ABC"]
        for sample in samples:
            self.assertEqual(tokenize(sample), _oracle_tokenize(sample))

    def test_passage_coverage_uses_one_passage_and_names_the_gap(self):
        question = "美式气泡水的苏打水和萃取液比例是多少？"
        candidates = [
            {
                "node_id": "a",
                "source": "美式气泡水制作",
                "dense_score": 0.78,
                "bm25_score": 3.1,
                "text": "美式气泡水融合：加冰后倒入苏打水再加入萃取液",
            },
            {
                "node_id": "b",
                "source": "拿铁制作",
                "dense_score": 0.66,
                "bm25_score": 2.0,
                "text": "牛奶和咖啡液的比例为10:1，36g的萃取液要360ml的牛奶",
            },
        ]
        measured = measure(question, candidates)
        self.assertEqual(measured.coverage.total, 13)
        self.assertEqual(measured.coverage.covered, 11)
        self.assertEqual(measured.coverage.node_id, "a")
        self.assertEqual(measured.coverage.missing, ("比例",))
        latte_only = measure(question, candidates[1:])
        self.assertEqual(latte_only.coverage.covered, 5)
        config = SufficiencyConfig(
            {Signal.TOP_DENSE: 0.4, Signal.PASSAGE_COVERAGE: 0.6},
            0.95,
        )
        assessment = measured.assess(config)
        self.assertFalse(assessment.sufficient)
        self.assertEqual(
            [item.signal for item in assessment.contributions],
            list(config.weights),
        )
        payload = assessment.to_dict()
        self.assertEqual(payload["coverage"]["missing"], ["比例"])
        directions = {item["signal"]: item["direction"] for item in payload["signals"]}
        self.assertEqual(directions["passage_coverage"], "down")

    def test_empty_question_and_empty_candidates_cover_nothing(self):
        self.assertEqual(measure("", []).coverage.fraction, 0.0)
        self.assertIsNone(measure("研磨度", []).coverage.node_id)
        self.assertEqual(measure("研磨度", []).coverage.covered, 0)

    def test_config_rejects_bad_weights_and_thresholds(self):
        with self.assertRaises(ValueError):
            SufficiencyConfig({Signal.TOP_DENSE: 0.0, Signal.PASSAGE_COVERAGE: 1.0}, 0.5)
        with self.assertRaises(ValueError):
            SufficiencyConfig({Signal.TOP_DENSE: -0.2, Signal.PASSAGE_COVERAGE: 1.2}, 0.5)
        with self.assertRaises(ValueError):
            SufficiencyConfig({Signal.TOP_DENSE: 0.2, Signal.PASSAGE_COVERAGE: 0.2}, 0.5)
        with self.assertRaises(ValueError):
            SufficiencyConfig({}, 0.5)
        with self.assertRaises(ValueError):
            SufficiencyConfig({Signal.TOP_DENSE: 1.0}, 1.2)

    def test_score_does_not_fall_when_evidence_gets_stronger(self):
        config = SufficiencyConfig(
            {Signal.TOP_DENSE: 0.5, Signal.PASSAGE_COVERAGE: 0.5},
            0.5,
        )
        question = "研磨度是多少"
        weak = [{"text": "清洗咖啡台", "dense_score": 0.40, "node_id": "a", "source": "清洗"}]
        stronger_dense = [{**weak[0], "dense_score": 0.80}]
        covering = weak + [
            {"text": "研磨度是多少", "dense_score": 0.40, "node_id": "b", "source": "研磨"}
        ]
        weak_score = measure(question, weak).assess(config).score
        self.assertGreaterEqual(measure(question, stronger_dense).assess(config).score, weak_score)
        self.assertGreaterEqual(measure(question, covering).assess(config).score, weak_score)

    def test_dense_only_matches_the_production_gate_inside_the_anchors(self):
        threshold = 0.50
        config = SufficiencyConfig({Signal.TOP_DENSE: 1.0}, threshold)
        gate = 0.35 + 0.5 * threshold
        for dense in (None, 0.20, 0.59, 0.60, 0.90):
            candidates = [{"dense_score": dense, "text": "", "node_id": "a", "source": "s"}]
            answered = measure("问", candidates).assess(config).sufficient
            self.assertEqual(answered, bool(dense_evidence(candidates, gate)))

    def test_trivial_classifiers_have_defined_rates(self):
        labels = [True, True, False, False]
        always_answer = decision_metrics(Outcome(label, True) for label in labels)
        always_refuse = decision_metrics(Outcome(label, False) for label in labels)
        self.assertEqual(always_answer.n, 4)
        self.assertEqual(always_answer.accuracy, 0.5)
        self.assertIsNone(always_answer.refusal_precision)
        self.assertEqual(always_answer.refusal_recall, 0.0)
        self.assertEqual(always_answer.false_answer_rate, 1.0)
        self.assertEqual(always_answer.false_refusal_rate, 0.0)
        self.assertEqual(always_answer.false_answers, 2)
        self.assertEqual(always_answer.correct_answers, 2)
        self.assertEqual(always_refuse.accuracy, 0.5)
        self.assertEqual(always_refuse.refusal_precision, 0.5)
        self.assertEqual(always_refuse.refusal_recall, 1.0)
        self.assertEqual(always_refuse.false_answer_rate, 0.0)
        self.assertEqual(always_refuse.false_refusal_rate, 1.0)
        self.assertEqual(always_refuse.correct_refusals, 2)
        self.assertEqual(always_refuse.false_refusals, 2)
        answers_only = DecisionMetrics(2, 0, 0, 0)
        self.assertIsNone(answers_only.refusal_precision)
        self.assertIsNone(answers_only.refusal_recall)
        self.assertIsNone(answers_only.false_answer_rate)
        self.assertIsNone(DecisionMetrics(0, 2, 0, 0).false_refusal_rate)

    def test_selection_breaks_ties_in_the_declared_order(self):
        perfect = [
            _retrieved(_case("a", True), 0.85, 1, 1),
            _retrieved(_case("b", False), 0.35, 0, 1),
        ]
        both = SufficiencyConfig(
            {Signal.TOP_DENSE: 0.5, Signal.PASSAGE_COVERAGE: 0.5},
            0.5,
        )
        dense_only = SufficiencyConfig({Signal.TOP_DENSE: 1.0}, 0.5)
        chosen = select_scorer(perfect, [both, dense_only])
        self.assertEqual(tuple(chosen.rule.weights), (Signal.TOP_DENSE,))

        hard_negative = [
            _retrieved(_case("answer", True), 0.85, 1, 1),
            _retrieved(_case("refuse", False), 0.85, 0, 1),
        ]
        chosen = select_scorer(
            hard_negative,
            [
                SufficiencyConfig({Signal.TOP_DENSE: 1.0}, 0.5),
                SufficiencyConfig(
                    {Signal.TOP_DENSE: 0.5, Signal.PASSAGE_COVERAGE: 0.5},
                    0.55,
                ),
            ],
        )
        self.assertEqual(len(chosen.rule.weights), 2)
        self.assertEqual(chosen.metrics.false_answers, 0)
        self.assertEqual(chosen.metrics.accuracy, 1.0)

        same_decisions = [
            _retrieved(_case("a", True), 0.85, 0, 1),
            _retrieved(_case("b", False), 0.35, 0, 1),
        ]
        low_margin = SufficiencyConfig({Signal.TOP_DENSE: 1.0}, 0.20)
        high_margin = SufficiencyConfig({Signal.TOP_DENSE: 1.0}, 0.50)
        chosen = select_scorer(same_decisions, [low_margin, high_margin])
        self.assertEqual(chosen.rule.threshold, 0.50)

        tied_accuracy = [
            _retrieved(_case("a1", True), 0.50, 0, 1),
            _retrieved(_case("a2", True), 0.50, 0, 1),
            _retrieved(_case("r1", False), 0.50, 1, 1),
            _retrieved(_case("r2", False), 0.50, 1, 1),
        ]
        refuse_more = SufficiencyConfig({Signal.TOP_DENSE: 1.0}, 0.50)
        answer_more = SufficiencyConfig({Signal.TOP_DENSE: 1.0}, 0.10)
        chosen = select_scorer(tied_accuracy, [answer_more, refuse_more])
        self.assertEqual(chosen.metrics.accuracy, 0.5)
        self.assertEqual(chosen.rule.threshold, 0.50)
        self.assertEqual(chosen.metrics.false_answer_rate, 0.0)

    def test_select_refuses_a_heldout_path_argument(self):
        with self.assertRaises(SystemExit) as caught:
            main(["--split", "eval/heldout.jsonl", "--select"])
        self.assertEqual(caught.exception.code, 2)

    def test_select_loads_only_the_dev_split(self):
        seen = []

        def spy(path):
            seen.append(Path(path).name)
            raise RuntimeError("stop-before-retrieval")

        with patch("sufficiency_eval.load_cases", spy):
            with self.assertRaises(RuntimeError):
                main(["--select"])
        self.assertEqual(seen, ["dev.jsonl"])


class SplitLoaderTest(unittest.TestCase):
    def test_dev_and_heldout_are_a_balanced_partition(self):
        dev = load_cases(ROOT / "eval" / "dev.jsonl")
        heldout = load_cases(ROOT / "eval" / "heldout.jsonl")
        everything = load_cases(ROOT / "eval" / "cases.jsonl")
        dev_ids = {case.id for case in dev}
        heldout_ids = {case.id for case in heldout}
        self.assertEqual(len(dev), 32)
        self.assertEqual(len(heldout), 16)
        self.assertTrue(dev_ids.isdisjoint(heldout_ids))
        self.assertEqual(dev_ids | heldout_ids, {case.id for case in everything})
        for rows in (dev, heldout):
            answers = sum(case.should_answer for case in rows)
            self.assertEqual(answers, len(rows) - answers)
        by_id = {case.id: case for case in everything}
        self.assertTrue(ORIGINAL_IDS <= set(by_id))
        for case_id, case in by_id.items():
            if case_id in ORIGINAL_IDS:
                self.assertNotIn("generated", case.tags)
            else:
                self.assertIn("generated", case.tags)

    def test_answer_cases_are_grounded_in_data_files(self):
        files = {
            path.stem: path.read_text(encoding="utf-8")
            for path in (ROOT / "data").iterdir()
            if path.is_file() and not path.name.startswith(".")
        }
        for line in (ROOT / "eval" / "cases.jsonl").read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            if not case["should_answer"]:
                self.assertEqual(case["expected_sources"], [])
                self.assertEqual(case["expected_terms"], [])
                continue
            self.assertTrue(case["expected_sources"])
            self.assertTrue(case["expected_terms"])
            stems = [Path(source).stem for source in case["expected_sources"] if Path(source).stem in files]
            self.assertTrue(stems, case["id"])
            if "generated" in case.get("tags", []):
                self.assertEqual(len(stems), len(case["expected_sources"]), case["id"])
            for term in case["expected_terms"]:
                self.assertTrue(
                    any(
                        term in files[stem] or term.casefold() in files[stem].casefold()
                        for stem in stems
                    ),
                    case["id"],
                )


if __name__ == "__main__":
    unittest.main()
