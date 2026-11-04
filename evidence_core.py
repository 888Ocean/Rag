from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike


ProgressCallback = Callable[[int, str], None]
REFUSAL = "资料不足，无法根据现有证据确认。"


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)
    output: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                output.append(token)
            else:
                output.extend(token[i : i + 2] for i in range(len(token) - 1))
        else:
            output.append(token)
    return output


class BM25Index:
    def __init__(self, nodes: list, k1: float = 1.5, b: float = 0.75):
        self.nodes = nodes
        self.k1 = k1
        self.b = b
        self.tokens = [_tokenize(node.text) for node in nodes]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequency: Counter = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = len(nodes)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self, query: str, limit: int, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        query_tokens = _tokenize(query)
        ranked: list[tuple[str, float]] = []
        for node, frequencies, length in zip(
            self.nodes, self.term_frequencies, self.lengths
        ):
            if allowed_ids is not None and node.node_id not in allowed_ids:
                continue
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.avg_length, 1)
                )
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                ranked.append((node.node_id, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]


class EvidenceCopilot:
    def __init__(self, root: Path):
        self.root = root
        load_dotenv(root / ".env")
        self.data_dir = Path(os.getenv("DATA_DIR", str(root / "data"))).expanduser().resolve()
        self.log_dir = root / "logs"
        self.log_path = self.log_dir / "retrieval.jsonl"
        self.catalog_path = self.data_dir / ".library.json"
        self.data_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.index = None
        self.nodes: list = []
        self.bm25: BM25Index | None = None

        model = os.getenv(
            "OMLX_MODEL", "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit"
        )
        api_base = os.getenv("OMLX_API_BASE", "http://127.0.0.1:8000/v1")
        embed_model = os.getenv(
            "EMBED_MODEL", str(root / "models" / "bge-small-zh-v1.5")
        )
        self.api_base = api_base
        Settings.llm = OpenAILike(
            model=model,
            api_base=api_base,
            api_key=os.getenv("OMLX_API_KEY", os.getenv("OPENAI_API_KEY", "local-key")),
            is_chat_model=True,
            context_window=int(os.getenv("OMLX_CONTEXT_WINDOW", "32768")),
            max_tokens=int(os.getenv("OMLX_MAX_TOKENS", "768")),
            temperature=0.1,
        )
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)

    def document_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.data_dir.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        )

    def _load_catalog(self) -> dict:
        default = {"sections": [], "documents": {}}
        if not self.catalog_path.exists():
            return default
        try:
            value = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
        value.setdefault("sections", [])
        value.setdefault("documents", {})
        return value

    def _save_catalog(self, catalog: dict) -> None:
        temporary = self.catalog_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self.catalog_path)
        except OSError as exc:
            raise PermissionError(
                f"无法保存知识分区：{self.catalog_path}。请确认 data 目录可写。"
            ) from exc

    def _relative_name(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.data_dir.resolve()))

    def _safe_document_path(self, name: str) -> Path:
        target = (self.data_dir / name).resolve()
        if self.data_dir.resolve() not in target.parents or target.name.startswith("."):
            raise ValueError("非法文档路径")
        if not target.is_file():
            raise ValueError("文档不存在")
        return target

    def library(self) -> dict:
        with self.lock:
            catalog = self._load_catalog()
            files = self.document_files()
            existing = {self._relative_name(path) for path in files}
            sections = [
                {"id": "unclassified", "name": "未分类", "system": True},
                *catalog["sections"],
            ]
            documents = []
            for path in files:
                name = self._relative_name(path)
                stat = path.stat()
                suffix = path.suffix.lower()
                documents.append(
                    {
                        "name": name,
                        "section_id": catalog["documents"].get(name, {}).get(
                            "section_id", "unclassified"
                        ),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime)
                        .astimezone()
                        .isoformat(timespec="seconds"),
                        "editable": suffix in {"", ".txt", ".md", ".csv"},
                    }
                )
            return {"sections": sections, "documents": documents}

    def create_section(self, name: str) -> dict:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("分区名称不能为空")
        with self.lock:
            catalog = self._load_catalog()
            if any(item["name"].casefold() == clean_name.casefold() for item in catalog["sections"]):
                raise ValueError("分区名称已存在")
            section = {"id": uuid.uuid4().hex[:12], "name": clean_name, "system": False}
            catalog["sections"].append(section)
            self._save_catalog(catalog)
            return section

    def rename_section(self, section_id: str, name: str) -> dict:
        clean_name = name.strip()
        if section_id == "unclassified":
            raise ValueError("系统分区不能改名")
        if not clean_name:
            raise ValueError("分区名称不能为空")
        with self.lock:
            catalog = self._load_catalog()
            if any(
                item["id"] != section_id
                and item["name"].casefold() == clean_name.casefold()
                for item in catalog["sections"]
            ):
                raise ValueError("分区名称已存在")
            section = next(
                (item for item in catalog["sections"] if item["id"] == section_id),
                None,
            )
            if section is None:
                raise ValueError("分区不存在")
            section["name"] = clean_name
            self._save_catalog(catalog)
            return section

    def get_document(self, name: str) -> dict:
        path = self._safe_document_path(name)
        library = self.library()
        info = next(item for item in library["documents"] if item["name"] == name)
        if info["editable"]:
            content = path.read_text(encoding="utf-8", errors="replace")
        else:
            documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
            content = "\n\n".join(document.text for document in documents)
        return {**info, "content": content}

    def update_document(
        self, name: str, new_name: str, section_id: str, content: str | None
    ) -> dict:
        with self.lock:
            path = self._safe_document_path(name)
            catalog = self._load_catalog()
            valid_sections = {"unclassified", *(item["id"] for item in catalog["sections"])}
            if section_id not in valid_sections:
                raise ValueError("分区不存在")
            safe_name = Path(new_name.strip()).name
            if not safe_name or safe_name.startswith("."):
                raise ValueError("文档名称无效")
            if path.suffix.lower() not in {"", ".txt", ".md", ".csv"} and Path(safe_name).suffix.lower() != path.suffix.lower():
                raise ValueError("PDF/DOCX 重命名时必须保留原扩展名")
            target = path.with_name(safe_name)
            if target != path and target.exists():
                raise ValueError("同名文档已存在")
            editable = path.suffix.lower() in {"", ".txt", ".md", ".csv"}
            if content is not None and editable:
                temporary = path.with_name(f".{path.name}.edit")
                temporary.write_text(content, encoding="utf-8")
                temporary.replace(path)
            old_key = self._relative_name(path)
            if target != path:
                path.rename(target)
            new_key = self._relative_name(target)
            metadata = catalog["documents"].pop(old_key, {})
            metadata["section_id"] = section_id
            catalog["documents"][new_key] = metadata
            self._save_catalog(catalog)
            return self.get_document(new_key)

    def assign_documents(self, names: list[str], section_id: str) -> None:
        with self.lock:
            catalog = self._load_catalog()
            valid_sections = {"unclassified", *(item["id"] for item in catalog["sections"])}
            if section_id not in valid_sections:
                raise ValueError("分区不存在")
            for name in names:
                path = self._safe_document_path(name)
                key = self._relative_name(path)
                catalog["documents"].setdefault(key, {})["section_id"] = section_id
            self._save_catalog(catalog)

    def rebuild(self, progress: ProgressCallback | None = None) -> dict:
        progress = progress or (lambda _value, _message: None)
        started = time.perf_counter()
        with self.lock:
            progress(10, "正在读取文档")
            files = self.document_files()
            if not files:
                raise ValueError("data 目录中没有可读取的文档")
            documents = SimpleDirectoryReader(input_files=[str(path) for path in files]).load_data()
            catalog = self._load_catalog()
            section_names = {
                "unclassified": "未分类",
                **{item["id"]: item["name"] for item in catalog["sections"]},
            }
            for document in documents:
                raw_path = Path(document.metadata.get("file_path", ""))
                try:
                    source_path = self._relative_name(raw_path)
                except (ValueError, OSError):
                    source_path = document.metadata.get("file_name", "unknown")
                section_id = catalog["documents"].get(source_path, {}).get(
                    "section_id", "unclassified"
                )
                document.metadata.update(
                    {
                        "source_path": source_path,
                        "section_id": section_id,
                        "section_name": section_names.get(section_id, "未分类"),
                    }
                )
            progress(35, "正在切分文档")
            splitter = SentenceSplitter(
                chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
            )
            nodes = splitter.get_nodes_from_documents(documents)
            progress(55, "正在生成向量索引")
            index = VectorStoreIndex(nodes)
            progress(80, "正在生成 BM25 索引")
            bm25 = BM25Index(nodes)
            self.nodes = nodes
            self.index = index
            self.bm25 = bm25
            elapsed = time.perf_counter() - started
            progress(100, "索引构建完成")
            return {
                "documents": len(files),
                "chunks": len(nodes),
                "seconds": round(elapsed, 3),
            }

    def _ensure_index(self, progress: ProgressCallback) -> None:
        if self.index is None or self.bm25 is None:
            self.rebuild(progress)

    def retrieve(
        self,
        question: str,
        top_k: int,
        progress: ProgressCallback,
        section_id: str | None = None,
        retrieval_mode: str = "hybrid",
    ) -> tuple[list[dict], float]:
        if retrieval_mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("不支持的检索方式")
        self._ensure_index(progress)
        started = time.perf_counter()
        candidate_limit = max(top_k * 3, 12)
        allowed_ids = (
            {
                node.node_id
                for node in self.nodes
                if node.metadata.get("section_id") == section_id
            }
            if section_id and section_id != "all"
            else None
        )
        if allowed_ids is not None and not allowed_ids:
            return [], time.perf_counter() - started
        progress(30, "正在进行语义检索")
        vector_results = self.index.as_retriever(
            similarity_top_k=len(self.nodes) if allowed_ids is not None else candidate_limit
        ).retrieve(question)
        if allowed_ids is not None:
            vector_results = [
                item for item in vector_results if item.node.node_id in allowed_ids
            ][:candidate_limit]
        progress(45, "正在进行 BM25 关键词检索")
        bm25_results = self.bm25.search(question, candidate_limit, allowed_ids)

        node_map = {node.node_id: node for node in self.nodes}
        dense_scores = {item.node.node_id: item.score for item in vector_results}
        lexical_scores = dict(bm25_results)
        fused: dict[str, float] = {}
        for rank, item in enumerate(vector_results, 1):
            fused[item.node.node_id] = fused.get(item.node.node_id, 0.0) + 1 / (60 + rank)
        for rank, (node_id, _score) in enumerate(bm25_results, 1):
            fused[node_id] = fused.get(node_id, 0.0) + 1 / (60 + rank)

        if retrieval_mode == "vector":
            ordered = [item.node.node_id for item in vector_results[:top_k]]
        elif retrieval_mode == "bm25":
            ordered = [node_id for node_id, _score in bm25_results[:top_k]]
        else:
            ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
        catalog = self._load_catalog()
        section_names = {
            "unclassified": "未分类",
            **{item["id"]: item["name"] for item in catalog["sections"]},
        }
        results = []
        for node_id in ordered:
            node = node_map[node_id]
            metadata = node.metadata or {}
            results.append(
                {
                    "node_id": node_id,
                    "source": metadata.get("file_name", "unknown"),
                    "source_path": metadata.get("source_path"),
                    "section_id": metadata.get("section_id", "unclassified"),
                    "section_name": section_names.get(
                        metadata.get("section_id", "unclassified"), "未分类"
                    ),
                    "page": metadata.get("page_label") or metadata.get("page_number"),
                    "text": node.text.strip(),
                    "dense_score": dense_scores.get(node_id),
                    "bm25_score": lexical_scores.get(node_id, 0.0),
                    "hybrid_score": fused.get(node_id, 0.0) * 30.5,
                    "retrieval_mode": retrieval_mode,
                }
            )
        return results, time.perf_counter() - started

    def ask(
        self,
        question: str,
        top_k: int = 4,
        min_score: float = 0.60,
        progress: ProgressCallback | None = None,
        section_id: str | None = None,
        retrieval_mode: str = "hybrid",
    ) -> dict:
        progress = progress or (lambda _value, _message: None)
        total_started = time.perf_counter()
        with self.lock:
            progress(8, "正在准备索引")
            candidates, retrieval_seconds = self.retrieve(
                question, top_k, progress, section_id, retrieval_mode
            )
            evidence = [
                item
                for item in candidates
                if item["dense_score"] is not None
                and item["dense_score"] >= min_score
            ]
            best_dense = max(
                (item["dense_score"] or 0.0 for item in candidates), default=0.0
            )

            if not evidence:
                result = {
                    "status": "refused",
                    "answer": REFUSAL,
                    "reason": f"最高语义分数 {best_dense:.4f}，低于阈值 {min_score:.4f}",
                    "evidence": [],
                    "candidates": candidates,
                    "citations_valid": None,
                    "timing": {
                        "retrieval_seconds": round(retrieval_seconds, 3),
                        "generation_seconds": 0.0,
                        "total_seconds": round(time.perf_counter() - total_started, 3),
                    },
                }
                progress(100, "证据不足，已安全拒答")
                self._write_log(
                    question, top_k, min_score, result, section_id, retrieval_mode
                )
                return result

            evidence_blocks = []
            for index, item in enumerate(evidence, 1):
                item["evidence_id"] = f"E{index}"
                location = item["source"]
                if item["page"]:
                    location += f", page {item['page']}"
                evidence_blocks.append(f"[E{index}] {location}\n{item['text']}")

            progress(65, "证据充足，正在生成答案")
            prompt = f"""你是 Evidence Copilot。只能根据下方本轮检索证据回答。
每个事实性结论必须引用真实证据编号，例如 [E1]。不要编造事实或引用。
若证据只能回答问题的一部分，应明确说明边界。不要声称整个知识库没有资料。
回答应简洁、完整，默认使用中文。

问题：{question}

证据：
{chr(10).join(evidence_blocks)}"""
            generation_started = time.perf_counter()
            try:
                answer = Settings.llm.complete(prompt).text.strip()
            except Exception as exc:
                raise RuntimeError(
                    f"无法连接 OMLX：{self.api_base}。请确认服务正在运行。"
                ) from exc
            generation_seconds = time.perf_counter() - generation_started

            used_ids = set(re.findall(r"\[E(\d+)\]", answer))
            valid_ids = {str(index) for index in range(1, len(evidence) + 1)}
            invalid_ids = sorted(used_ids - valid_ids)
            citations_valid = bool(used_ids) and not invalid_ids
            status = "answered" if citations_valid else "citation_error"
            if not citations_valid:
                answer = "引用校验未通过，已阻止展示未经引用支持的答案。"

            progress(92, "正在校验引用并记录日志")
            result = {
                "status": status,
                "answer": answer,
                "reason": "" if citations_valid else f"无效引用：{invalid_ids or '未引用证据'}",
                "evidence": evidence,
                "candidates": candidates,
                "citations_valid": citations_valid,
                "timing": {
                    "retrieval_seconds": round(retrieval_seconds, 3),
                    "generation_seconds": round(generation_seconds, 3),
                    "total_seconds": round(time.perf_counter() - total_started, 3),
                },
            }
            self._write_log(
                question, top_k, min_score, result, section_id, retrieval_mode
            )
            progress(100, "回答完成")
            return result

    def _write_log(
        self,
        question: str,
        top_k: int,
        min_score: float,
        result: dict,
        section_id: str | None = None,
        retrieval_mode: str = "hybrid",
    ) -> None:
        entry = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "question": question,
            "top_k": top_k,
            "min_score": min_score,
            "section_id": section_id or "all",
            "retrieval_mode": retrieval_mode,
            "status": result["status"],
            "timing": result["timing"],
            "retrieval_results": [
                {
                    key: item.get(key)
                    for key in (
                        "source",
                        "source_path",
                        "section_id",
                        "section_name",
                        "page",
                        "text",
                        "dense_score",
                        "bm25_score",
                        "hybrid_score",
                    )
                }
                for item in result["candidates"]
            ],
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def recent_logs(self, limit: int = 20) -> list[dict]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in reversed(lines) if line.strip()]
