from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from evidence_core import EvidenceCopilot
from evaluation import EvaluationStore, RAGEvaluator


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "web"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
ALLOWED_EXTENSIONS = {"", ".txt", ".md", ".pdf", ".docx", ".csv"}
ENGINE = EvidenceCopilot(ROOT)
EVALUATION_STORE = EvaluationStore(ROOT)
EVALUATOR = RAGEvaluator(ENGINE, EVALUATION_STORE)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def create_job(kind: str) -> str:
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "progress": 2,
            "message": "任务已创建",
            "result": None,
            "error": None,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    return job_id


def update_job(job_id: str, **values) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(values)


def run_job(job_id: str, action) -> None:
    def progress(value: int, message: str) -> None:
        update_job(job_id, status="running", progress=value, message=message)

    try:
        update_job(job_id, status="running", progress=5, message="任务开始")
        result = action(progress)
        update_job(
            job_id,
            status="completed",
            progress=100,
            message="任务完成",
            result=result,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            progress=100,
            message="任务失败",
            error=str(exc),
        )


def start_job(job_id: str, action) -> None:
    threading.Thread(target=run_job, args=(job_id, action), daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "EvidenceCopilot/1.0"

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def json_response(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 32 * 1024 * 1024:
            raise ValueError("请求内容过大")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self.json_response({"ok": True, "api_base": ENGINE.api_base})
            return
        if path == "/api/documents":
            documents = []
            for file in ENGINE.document_files():
                stat = file.stat()
                documents.append(
                    {
                        "name": str(file.relative_to(ENGINE.data_dir)),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                    }
                )
            self.json_response({"documents": documents})
            return
        if path == "/api/library":
            self.json_response(ENGINE.library())
            return
        if path == "/api/document":
            name = parse_qs(parsed.query).get("name", [""])[0]
            if not name:
                self.json_response({"error": "缺少文档名称"}, 400)
                return
            try:
                self.json_response(ENGINE.get_document(name))
            except ValueError as exc:
                self.json_response({"error": str(exc)}, 400)
            return
        if path == "/api/logs":
            self.json_response({"logs": ENGINE.recent_logs()})
            return
        if path == "/api/eval/cases":
            self.json_response({"cases": EVALUATION_STORE.list_cases()})
            return
        if path == "/api/eval/runs":
            self.json_response({"runs": EVALUATION_STORE.list_runs()})
            return
        if path == "/api/eval/run":
            run_id = parse_qs(parsed.query).get("id", [""])[0]
            if not run_id:
                self.json_response({"error": "缺少评测记录 ID"}, 400)
                return
            try:
                self.json_response(EVALUATION_STORE.get_run(run_id))
            except ValueError as exc:
                self.json_response({"error": str(exc)}, 400)
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                payload = dict(job) if job else None
            self.json_response(payload or {"error": "任务不存在"}, 200 if payload else 404)
            return
        self.serve_static(path)

    def serve_static(self, path: str) -> None:
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC / relative).resolve()
        if STATIC.resolve() not in target.parents and target != STATIC.resolve():
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/query":
                payload = self.read_json()
                question = str(payload.get("question", "")).strip()
                if not question:
                    raise ValueError("请输入问题")
                top_k = min(max(int(payload.get("top_k", 4)), 1), 12)
                min_score = min(max(float(payload.get("min_score", 0.60)), 0), 1)
                section_id = str(payload.get("section_id", "all"))
                job_id = create_job("query")
                start_job(
                    job_id,
                    lambda progress: ENGINE.ask(
                        question, top_k, min_score, progress, section_id, "hybrid"
                    ),
                )
                self.json_response({"job_id": job_id}, 202)
                return
            if path == "/api/rebuild":
                job_id = create_job("rebuild")
                start_job(job_id, ENGINE.rebuild)
                self.json_response({"job_id": job_id}, 202)
                return
            if path == "/api/upload":
                names = self.save_uploads()
                section_id = self.headers.get("X-Section-Id", "unclassified")
                ENGINE.assign_documents(names, section_id)
                job_id = create_job("upload")
                start_job(job_id, ENGINE.rebuild)
                self.json_response({"job_id": job_id, "files": names}, 202)
                return
            if path == "/api/sections":
                payload = self.read_json()
                self.json_response(
                    {"section": ENGINE.create_section(str(payload.get("name", "")))},
                    201,
                )
                return
            if path == "/api/sections/rename":
                payload = self.read_json()
                section = ENGINE.rename_section(
                    str(payload.get("section_id", "")), str(payload.get("name", ""))
                )
                self.json_response({"section": section})
                return
            if path == "/api/document/update":
                payload = self.read_json()
                name = str(payload.get("name", ""))
                new_name = str(payload.get("new_name", ""))
                section_id = str(payload.get("section_id", "unclassified"))
                content = payload.get("content")
                document = ENGINE.update_document(
                    name, new_name, section_id, None if content is None else str(content)
                )
                job_id = create_job("document-update")
                start_job(job_id, ENGINE.rebuild)
                self.json_response({"job_id": job_id, "document": document}, 202)
                return
            if path == "/api/eval/cases":
                case = EVALUATION_STORE.save_case(self.read_json())
                self.json_response({"case": case}, 201)
                return
            if path == "/api/eval/cases/delete":
                payload = self.read_json()
                EVALUATION_STORE.delete_case(str(payload.get("id", "")))
                self.json_response({"ok": True})
                return
            if path == "/api/eval/run":
                payload = self.read_json()
                modes = payload.get("retrieval_modes", ["hybrid"])
                top_ks = payload.get("top_ks", [4])
                thresholds = payload.get("thresholds", [0.60])
                if not isinstance(modes, list) or not isinstance(top_ks, list) or not isinstance(thresholds, list):
                    raise ValueError("评测配置格式错误")
                full_generation = bool(payload.get("full_generation", False))
                job_id = create_job("evaluation")
                start_job(
                    job_id,
                    lambda progress: EVALUATOR.run(
                        modes, top_ks, thresholds, full_generation, progress
                    ),
                )
                self.json_response({"job_id": job_id}, 202)
                return
            self.json_response({"error": "接口不存在"}, 404)
        except (ValueError, json.JSONDecodeError) as exc:
            self.json_response({"error": str(exc)}, 400)
        except Exception as exc:
            self.json_response({"error": str(exc)}, 500)

    def save_uploads(self) -> list[str]:
        content_type = self.headers.get("Content-Type", "")
        boundary_match = re.search(r"boundary=([^;]+)", content_type)
        if "multipart/form-data" not in content_type or not boundary_match:
            raise ValueError("上传格式必须是 multipart/form-data")
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError("上传内容为空或超过 30 MB")
        boundary = boundary_match.group(1).strip('"').encode()
        body = self.rfile.read(length)
        saved: list[str] = []
        for part in body.split(b"--" + boundary):
            if b"\r\n\r\n" not in part:
                continue
            raw_headers, content = part.split(b"\r\n\r\n", 1)
            match = re.search(br'filename="([^"]+)"', raw_headers)
            if not match:
                continue
            filename = Path(match.group(1).decode("utf-8", "replace")).name
            if not filename or Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                raise ValueError(f"不支持的文件类型：{filename}")
            content = content.removesuffix(b"\r\n")
            target = ENGINE.data_dir / filename
            if target.exists():
                target = ENGINE.data_dir / f"{target.stem}-{uuid.uuid4().hex[:6]}{target.suffix}"
            temporary = ENGINE.data_dir / f".{target.name}.upload"
            temporary.write_bytes(content)
            temporary.replace(target)
            saved.append(target.name)
        if not saved:
            raise ValueError("没有找到可上传的文件")
        return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Evidence Copilot web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Evidence Copilot: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
