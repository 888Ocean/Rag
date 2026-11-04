import argparse
from pathlib import Path

from evidence_core import EvidenceCopilot


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Local OMLX Evidence Copilot")
    parser.add_argument("question", nargs="?", default="RAG 的核心流程是什么？")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.60)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    copilot = EvidenceCopilot(ROOT)
    if args.rebuild:
        summary = copilot.rebuild(lambda value, message: print(f"[{value:3d}%] {message}"))
        print(f"索引：{summary['documents']} 个文档，{summary['chunks']} 个片段")
    result = copilot.ask(
        args.question,
        top_k=args.top_k,
        min_score=args.min_score,
        progress=lambda value, message: print(f"[{value:3d}%] {message}"),
    )
    print("\n回答：", result["answer"])
    if result.get("reason"):
        print("原因：", result["reason"])
    print("\n证据：")
    for item in result["evidence"]:
        dense = item["dense_score"] or 0.0
        print(
            f"[{item['evidence_id']}] {item['source']} "
            f"dense={dense:.4f} bm25={item['bm25_score']:.4f} hybrid={item['hybrid_score']:.4f}"
        )
        print(item["text"][:300])
    print("\n耗时：", result["timing"])


if __name__ == "__main__":
    main()
