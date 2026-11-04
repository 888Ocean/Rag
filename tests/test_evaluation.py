import tempfile
import unittest
from pathlib import Path

from evaluation import EvaluationStore, RAGEvaluator


class FakeEngine:
    def retrieve(
        self, question, top_k, progress, section_id="all", retrieval_mode="hybrid"
    ):
        if question == "answerable":
            return [
                {
                    "source": "guide.md",
                    "source_path": "guide.md",
                    "dense_score": 0.82,
                    "bm25_score": 1.2,
                    "hybrid_score": 0.98,
                    "text": "supported",
                }
            ], 0.01
        return [
            {
                "source": "unrelated.md",
                "source_path": "unrelated.md",
                "dense_score": 0.31,
                "bm25_score": 0.1,
                "hybrid_score": 0.2,
                "text": "unrelated",
            }
        ], 0.01


class EvaluationTest(unittest.TestCase):
    def test_fast_evaluation_metrics_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationStore(Path(directory))
            store.save_case(
                {
                    "id": "answerable",
                    "question": "answerable",
                    "should_answer": True,
                    "expected_sources": ["guide"],
                }
            )
            store.save_case(
                {
                    "id": "refusal",
                    "question": "refusal",
                    "should_answer": False,
                }
            )
            evaluator = RAGEvaluator(FakeEngine(), store)
            result = evaluator.run(["hybrid"], [4], [0.6])

            metrics = result["best"]["metrics"]
            self.assertEqual(metrics["hit_at_k"], 1.0)
            self.assertEqual(metrics["mrr"], 1.0)
            self.assertEqual(metrics["decision_accuracy"], 1.0)
            self.assertEqual(metrics["false_refusal_rate"], 0.0)
            self.assertEqual(len(store.list_runs()), 1)
            self.assertEqual(store.get_run(result["id"])["id"], result["id"])


if __name__ == "__main__":
    unittest.main()
