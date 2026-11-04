# Evidence Copilot

Local RAG with an OMLX-compatible LLM, local BGE embeddings, hybrid retrieval,
cited evidence, safe refusal, timing logs, upload progress, and completion sound.

## Start the web app

```bash
git clone https://github.com/888Ocean/Rag.git
cd Rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python web_app.py
```

Open <http://127.0.0.1:8080>.

The existing CLI remains available:

```bash
python app.py --rebuild "深度烘焙豆如何调整研磨度？"
```

## Configuration

Put local configuration in `.env`:

```env
OMLX_API_BASE=http://127.0.0.1:8000/v1
OMLX_MODEL=your-model-id
OMLX_API_KEY=local-key
EMBED_MODEL=BAAI/bge-small-zh-v1.5
OMLX_MAX_TOKENS=768
CHUNK_SIZE=512
CHUNK_OVERLAP=80
# Optional: read documents from another local directory.
DATA_DIR=/absolute/path/to/data
```

Set `OMLX_MODEL` to an ID returned by your local OMLX-compatible server.
The embedding model can be a Hugging Face ID or a local model path. Add
your own documents through the web interface; local documents and generated
evaluation runs are not included in the repository.

## Evaluation center

The **Evaluation Center** includes 15 editable baseline cases and can compare
vector, BM25, and hybrid ranking across multiple Top-K and refusal-threshold
settings. Fast mode evaluates retrieval and refusal decisions without calling
the LLM. Enable full generation when you also want citation and answer-keyword
checks. Cases live in `eval/cases.jsonl`; reproducible reports are stored under
`eval/runs/`.

The web UI exposes `top_k` and the semantic refusal threshold. Hybrid ranking
uses vector retrieval plus BM25 with reciprocal-rank fusion. The refusal gate
uses the dense score because BM25 scores are not calibrated probabilities.

The knowledge library supports named sections, section renaming, upload-to-section,
document viewing, document renaming, and direct editing for text/Markdown/CSV
files. PDF and DOCX extracted text is read-only, while their filename and section
remain editable. Section metadata is attached to every indexed chunk and can be
selected as the query scope in the web UI.

Retrieval results and stage timings are appended to `logs/retrieval.jsonl`.
Uploaded files are stored in `data/`; duplicate filenames receive a unique suffix.

Run the unit test with `python -m unittest discover -s tests -v`.
