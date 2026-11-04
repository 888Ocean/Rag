const $ = (selector) => document.querySelector(selector);
const state = {
  files: [],
  audioContext: null,
  library: { sections: [], documents: [] },
  currentDocument: null,
  sectionEditing: null,
  evalCases: [],
  currentEvalCase: null,
};

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[char]);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function prepareAudio() {
  if (!state.audioContext) state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
  if (state.audioContext.state === "suspended") state.audioContext.resume();
}

function playDone() {
  if (!state.audioContext) return;
  const now = state.audioContext.currentTime;
  [523.25, 659.25, 783.99].forEach((frequency, index) => {
    const oscillator = state.audioContext.createOscillator();
    const gain = state.audioContext.createGain();
    oscillator.frequency.value = frequency;
    oscillator.type = "sine";
    gain.gain.setValueAtTime(0.0001, now + index * 0.11);
    gain.gain.exponentialRampToValueAtTime(0.08, now + index * 0.11 + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + index * 0.11 + 0.2);
    oscillator.connect(gain).connect(state.audioContext.destination);
    oscillator.start(now + index * 0.11);
    oscillator.stop(now + index * 0.11 + 0.22);
  });
}

function showProgress(value, message) {
  $("#progressPanel").classList.remove("hidden");
  $("#progressBar").style.width = `${Math.max(0, Math.min(100, value))}%`;
  $("#progressValue").textContent = `${value}%`;
  $("#progressMessage").textContent = message;
}

function hideProgress() { setTimeout(() => $("#progressPanel").classList.add("hidden"), 800); }

async function pollJob(jobId) {
  while (true) {
    const job = await api(`/api/jobs/${jobId}`);
    showProgress(job.progress, job.message);
    if (job.status === "completed") { playDone(); hideProgress(); return job.result; }
    if (job.status === "failed") throw new Error(job.error || "任务失败");
    await new Promise((resolve) => setTimeout(resolve, 650));
  }
}

async function loadHealth() {
  try {
    await api("/api/health");
    $(".status").classList.add("online");
    $("#serverStatus").textContent = "本地服务已连接";
  } catch (_) { $("#serverStatus").textContent = "本地服务不可用"; }
}

function sectionOptions(includeAll = false, selected = "") {
  const sections = state.library.sections || [];
  return `${includeAll ? '<option value="all">全部分区</option>' : ""}${sections.map((section) =>
    `<option value="${escapeHtml(section.id)}"${section.id === selected ? " selected" : ""}>${escapeHtml(section.name)}</option>`
  ).join("")}`;
}

async function loadLibrary() {
  const previousQuerySection = $("#querySection").value;
  const previousUploadSection = $("#uploadSection").value;
  state.library = await api("/api/library");
  $("#documentCount").textContent = state.library.documents.length;
  $("#sectionCount").textContent = state.library.sections.length;
  $("#querySection").innerHTML = sectionOptions(true, previousQuerySection);
  $("#uploadSection").innerHTML = sectionOptions(false, previousUploadSection || "unclassified");
  $("#documentSection").innerHTML = sectionOptions();
  $("#evalSection").innerHTML = sectionOptions(true);

  $("#sectionList").innerHTML = state.library.sections.map((section) => {
    const documents = state.library.documents.filter((doc) => doc.section_id === section.id);
    const rows = documents.length ? documents.map((doc) => `
      <div class="document" title="${escapeHtml(doc.name)}">
        <span class="document-icon">DOC</span>
        <span class="document-copy"><span class="document-name">${escapeHtml(doc.name)}</span><span class="document-size">${formatBytes(doc.size)} · ${doc.editable ? "可编辑" : "只读正文"}</span></span>
        <span class="document-actions">
          <button class="document-action" type="button" data-action="view-document" data-name="${encodeURIComponent(doc.name)}" title="查看">查看</button>
          <button class="document-action" type="button" data-action="edit-document" data-name="${encodeURIComponent(doc.name)}" title="修改">改</button>
        </span>
      </div>`).join("") : '<p class="empty-docs">暂无文档，可上传或移动到这里</p>';
    return `<details class="knowledge-section" open>
      <summary><span class="section-chevron">›</span><span class="section-name">${escapeHtml(section.name)}</span><span class="section-total">${documents.length}</span>${section.system ? "" : `<button class="section-rename" type="button" data-action="rename-section" data-section="${escapeHtml(section.id)}" data-name="${escapeHtml(section.name)}">改名</button>`}</summary>
      <div class="document-list">${rows}</div>
    </details>`;
  }).join("");
}

function selectFiles(files) {
  state.files = [...files];
  $("#selectedFiles").textContent = state.files.length ? `已选择：${state.files.map((file) => file.name).join("、")}` : "";
}

async function uploadFiles() {
  if (!state.files.length) return showError("请先选择文档");
  prepareAudio(); setBusy(true); showProgress(1, "正在上传文档");
  const form = new FormData();
  state.files.forEach((file) => form.append("files", file));
  try {
    const payload = await new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("POST", "/api/upload");
      request.setRequestHeader("X-Section-Id", $("#uploadSection").value);
      request.upload.onprogress = (event) => {
        if (event.lengthComputable) showProgress(Math.max(1, Math.round(event.loaded / event.total * 12)), "正在上传文档");
      };
      request.onload = () => {
        const data = JSON.parse(request.responseText || "{}");
        request.status >= 200 && request.status < 300 ? resolve(data) : reject(new Error(data.error || "上传失败"));
      };
      request.onerror = () => reject(new Error("上传连接失败"));
      request.send(form);
    });
    const result = await pollJob(payload.job_id);
    selectFiles([]); $("#fileInput").value = "";
    await loadLibrary();
    renderNotice(`索引完成：${result.documents} 个文档，${result.chunks} 个片段，用时 ${result.seconds}s`);
  } catch (error) { showError(error.message); }
  finally { setBusy(false); }
}

async function rebuild() {
  prepareAudio(); setBusy(true);
  try {
    const payload = await api("/api/rebuild", { method: "POST" });
    const result = await pollJob(payload.job_id);
    renderNotice(`索引完成：${result.documents} 个文档，${result.chunks} 个片段，用时 ${result.seconds}s`);
  } catch (error) { showError(error.message); }
  finally { setBusy(false); }
}

function openSectionDialog(sectionId = null, currentName = "") {
  state.sectionEditing = sectionId;
  $("#sectionDialogTitle").textContent = sectionId ? "修改分区名称" : "新建分区";
  $("#sectionNameInput").value = currentName;
  $("#sectionDialog").showModal();
  $("#sectionNameInput").focus();
}

async function saveSection() {
  const name = $("#sectionNameInput").value.trim();
  if (!name) return;
  try {
    if (state.sectionEditing) {
      await api("/api/sections/rename", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ section_id: state.sectionEditing, name }) });
    } else {
      await api("/api/sections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
    }
    $("#sectionDialog").close();
    await loadLibrary();
  } catch (error) { showError(error.message); }
}

async function openDocument(name, editMode) {
  try {
    const document = await api(`/api/document?name=${encodeURIComponent(name)}`);
    state.currentDocument = document;
    $("#dialogTitle").textContent = editMode ? "修改文档" : "查看文档";
    $("#documentName").value = document.name;
    $("#documentName").disabled = !editMode;
    $("#documentSection").innerHTML = sectionOptions(false, document.section_id);
    $("#documentSection").disabled = !editMode;
    $("#documentContent").value = document.content;
    $("#documentContent").readOnly = !editMode || !document.editable;
    $("#documentHelp").textContent = !editMode ? "查看模式不会修改原始文件。" : document.editable ? "保存后会重新构建向量与 BM25 索引。" : "PDF/DOCX 正文为只读；可以修改名称和所属分区。";
    $("#saveDocument").classList.toggle("hidden", !editMode);
    $("#documentDialog").showModal();
  } catch (error) { showError(error.message); }
}

async function saveDocument() {
  if (!state.currentDocument) return;
  prepareAudio();
  const document = state.currentDocument;
  const payload = {
    name: document.name,
    new_name: $("#documentName").value.trim(),
    section_id: $("#documentSection").value,
    content: document.editable ? $("#documentContent").value : null,
  };
  if (!payload.new_name) return;
  setBusy(true); $("#saveDocument").disabled = true;
  try {
    const response = await api("/api/document/update", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    $("#documentDialog").close();
    const result = await pollJob(response.job_id);
    await loadLibrary();
    renderNotice(`文档已保存，索引包含 ${result.chunks} 个片段。`);
  } catch (error) { showError(error.message); }
  finally { setBusy(false); $("#saveDocument").disabled = false; }
}

async function ask() {
  const question = $("#question").value.trim();
  if (!question) return showError("请输入问题");
  prepareAudio(); setBusy(true);
  try {
    const payload = await api("/api/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: Number($("#topK").value), min_score: Number($("#threshold").value), section_id: $("#querySection").value }),
    });
    const result = await pollJob(payload.job_id);
    renderResult(result); await loadLogs();
  } catch (error) { showError(error.message); }
  finally { setBusy(false); }
}

function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[E(\d+)\]/g, '<button class="citation-link" type="button" data-citation="E$1">E$1</button>');
}

function renderMarkdown(markdown = "") {
  const lines = String(markdown).replace(/\r/g, "").split("\n");
  let html = "";
  for (let index = 0; index < lines.length;) {
    const line = lines[index].trim();
    if (!line) { index += 1; continue; }
    if (/^\|.*\|$/.test(line) && index + 1 < lines.length && /^\|?[\s:|-]+\|?$/.test(lines[index + 1].trim())) {
      const rows = [];
      const parseRow = (value) => value.replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
      const header = parseRow(line); index += 2;
      while (index < lines.length && /^\|.*\|$/.test(lines[index].trim())) { rows.push(parseRow(lines[index].trim())); index += 1; }
      html += `<div class="answer-table-wrap"><table><thead><tr>${header.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
      continue;
    }
    const heading = line.match(/^(#{2,4})\s+(.+)$/);
    if (heading) { const level = heading[1].length; html += `<h${level}>${inlineMarkdown(heading[2])}</h${level}>`; index += 1; continue; }
    if (/^---+$/.test(line)) { html += "<hr>"; index += 1; continue; }
    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) { items.push(lines[index].trim().replace(/^[-*]\s+/, "")); index += 1; }
      html += `<ul>${items.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>`; continue;
    }
    if (/^\d+[.)]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\d+[.)]\s+/.test(lines[index].trim())) { items.push(lines[index].trim().replace(/^\d+[.)]\s+/, "")); index += 1; }
      html += `<ol>${items.map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ol>`; continue;
    }
    html += `<p>${inlineMarkdown(line)}</p>`; index += 1;
  }
  return html;
}

function renderResult(result) {
  const status = result.status === "answered" ? "已回答" : result.status === "refused" ? "证据不足" : "引用异常";
  const badgeClass = result.status === "answered" ? "" : result.status === "refused" ? "refused" : "error";
  const timing = result.timing || {}; const evidence = result.evidence || [];
  const citationLabel = result.citations_valid == null ? "不适用" : result.citations_valid ? "通过" : "失败";
  const candidateNote = result.status === "refused" && result.candidates?.length ? `<h3 class="evidence-title">最接近的候选片段（未作为证据）</h3>${renderEvidence(result.candidates, false)}` : "";
  $("#resultPanel").classList.remove("empty");
  $("#resultPanel").innerHTML = `
    <div class="result-head"><div><p class="eyebrow">GROUNDED RESPONSE</p><h2>回答</h2></div><span class="result-badge ${badgeClass}">${status}</span></div>
    <div class="answer">${renderMarkdown(result.answer)}</div>
    ${result.reason ? `<div class="reason">${escapeHtml(result.reason)}</div>` : ""}
    ${renderSufficiency(result.sufficiency)}
    <div class="metrics">${metric("检索", timing.retrieval_seconds)}${metric("生成", timing.generation_seconds)}${metric("总计", timing.total_seconds)}<div class="metric"><small>引用校验</small><strong>${citationLabel}</strong></div></div>
    ${evidence.length ? `<h3 class="evidence-title">引用证据</h3>${renderEvidence(evidence, true)}` : ""}${candidateNote}`;
}

function metric(label, value = 0) { return `<div class="metric"><small>${label}</small><strong>${Number(value).toFixed(2)}s</strong></div>`; }

function renderSufficiency(sufficiency) {
  if (!sufficiency || !Array.isArray(sufficiency.signals)) return "";
  const parts = sufficiency.signals.map((item) => {
    const points = Number(item.points || 0).toFixed(2);
    const direction = escapeHtml(item.direction_label || "");
    return `${escapeHtml(item.label || item.signal)} ${points}（${direction}）`;
  }).join(" + ");
  const missing = sufficiency.coverage?.missing?.length
    ? ` · 证据未覆盖：${sufficiency.coverage.missing.map((item) => escapeHtml(item)).join("、")}`
    : "";
  const verdict = sufficiency.sufficient ? "充分" : "不足";
  const score = Number(sufficiency.score || 0).toFixed(2);
  const threshold = Number(sufficiency.threshold || 0).toFixed(2);
  return `<div class="sufficiency">证据充分度 ${score} / ${threshold}（${verdict}，参考，不参与拒答）· ${parts}${missing}</div>`;
}

function renderEvidence(items, accepted) {
  return `<div class="evidence-grid">${items.map((item, index) => {
    const dense = item.dense_score == null ? "—" : Number(item.dense_score).toFixed(4);
    const label = accepted ? item.evidence_id || `E${index + 1}` : `R${index + 1}`;
    const text = item.text || ""; const isLong = text.length > 150 || text.split("\n").length > 3;
    return `<article class="evidence-card" id="evidence-${label}">
      <div class="evidence-meta"><span class="evidence-id">${label}</span><span class="evidence-source">${escapeHtml(item.source)}</span><span class="section-chip">${escapeHtml(item.section_name || "未分类")}</span><span>向量 ${dense}</span><span>BM25 ${Number(item.bm25_score || 0).toFixed(3)}</span><span>融合 ${Number(item.hybrid_score || 0).toFixed(3)}</span></div>
      <p class="evidence-text${isLong ? " collapsed" : ""}">${escapeHtml(text)}</p>${isLong ? '<button class="evidence-toggle" type="button" aria-expanded="false">展开全文</button>' : ""}
    </article>`;
  }).join("")}</div>`;
}

function commaValues(value = "") {
  return String(value).split(/[,，]/).map((item) => item.trim()).filter(Boolean);
}

function checkedValues(name, cast = (value) => value) {
  return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((input) => cast(input.value));
}

async function loadEvalCases() {
  const { cases } = await api("/api/eval/cases");
  state.evalCases = cases || [];
  const enabled = state.evalCases.filter((item) => item.enabled !== false).length;
  $("#evalCaseCount").textContent = enabled;
  $("#evalCaseList").innerHTML = state.evalCases.length ? state.evalCases.map((item, index) => `
    <button class="eval-case${item.enabled === false ? " disabled" : ""}" type="button" data-action="edit-eval-case" data-case="${escapeHtml(item.id)}">
      <span class="eval-case-index">${String(index + 1).padStart(2, "0")}</span>
      <span class="eval-case-copy"><strong>${escapeHtml(item.question)}</strong><small>${escapeHtml(item.note || (item.expected_sources || []).join("、") || "暂无备注")}</small></span>
      <span class="eval-behavior${item.should_answer ? "" : " refuse"}">${item.should_answer ? "应回答" : "应拒答"}</span>
    </button>`).join("") : '<p class="empty-docs">还没有评测用例，请先新建。</p>';
}

function openEvalCaseDialog(caseId = null) {
  const item = caseId ? state.evalCases.find((entry) => entry.id === caseId) : null;
  state.currentEvalCase = item || null;
  $("#evalCaseDialogTitle").textContent = item ? "修改评测用例" : "新建评测用例";
  $("#evalQuestion").value = item?.question || "";
  $("#evalShouldAnswer").value = String(item?.should_answer ?? true);
  $("#evalSection").innerHTML = sectionOptions(true, item?.section_id || "all");
  $("#evalSources").value = (item?.expected_sources || []).join(", ");
  $("#evalTerms").value = (item?.expected_terms || []).join(", ");
  $("#evalNote").value = item?.note || "";
  $("#evalEnabled").checked = item?.enabled !== false;
  $("#deleteEvalCase").classList.toggle("hidden", !item);
  $("#evalCaseDialog").showModal();
  $("#evalQuestion").focus();
}

async function saveEvalCase() {
  const question = $("#evalQuestion").value.trim();
  if (!question) return showError("请输入评测问题");
  const payload = {
    id: state.currentEvalCase?.id,
    question,
    should_answer: $("#evalShouldAnswer").value === "true",
    section_id: $("#evalSection").value,
    expected_sources: commaValues($("#evalSources").value),
    expected_terms: commaValues($("#evalTerms").value),
    note: $("#evalNote").value.trim(),
    tags: state.currentEvalCase?.tags || [],
    enabled: $("#evalEnabled").checked,
  };
  try {
    await api("/api/eval/cases", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    $("#evalCaseDialog").close();
    await loadEvalCases();
  } catch (error) { showError(error.message); }
}

async function deleteEvalCase() {
  if (!state.currentEvalCase || !window.confirm(`确定删除用例“${state.currentEvalCase.question}”吗？`)) return;
  try {
    await api("/api/eval/cases/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: state.currentEvalCase.id }) });
    $("#evalCaseDialog").close();
    await loadEvalCases();
  } catch (error) { showError(error.message); }
}

async function runEvaluation() {
  const retrievalModes = checkedValues("evalMode");
  const topKs = checkedValues("evalTopK", Number);
  const thresholds = checkedValues("evalThreshold", Number);
  if (!retrievalModes.length || !topKs.length || !thresholds.length) return showError("请至少选择一种检索方式、Top-K 和阈值");
  prepareAudio(); setBusy(true); showProgress(2, "正在创建评测任务");
  try {
    const payload = await api("/api/eval/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ retrieval_modes: retrievalModes, top_ks: topKs, thresholds, full_generation: $("#fullGeneration").checked }),
    });
    const result = await pollJob(payload.job_id);
    renderEvaluation(result);
    await loadEvalRuns();
  } catch (error) { showError(error.message); }
  finally { setBusy(false); }
}

function percent(value) { return `${(Number(value || 0) * 100).toFixed(1)}%`; }

function evalMetric(label, value) {
  return `<div class="eval-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function renderEvaluation(run) {
  const best = run.best || {}; const metrics = best.metrics || {};
  const bestConfig = (run.configs || []).find((item) => item.retrieval_mode === best.retrieval_mode && item.top_k === best.top_k && Number(item.min_score) === Number(best.min_score)) || (run.configs || [])[0];
  const comparison = (run.configs || []).map((item) => {
    const value = item.metrics || {};
    const active = item === bestConfig ? " ★" : "";
    return `<div class="eval-comparison-row"><span>${escapeHtml(item.retrieval_mode)} · K=${item.top_k} · ${Number(item.min_score).toFixed(2)}${active}</span><span>${percent(value.hit_at_k)}</span><span>${Number(value.mrr || 0).toFixed(3)}</span><span>${percent(value.decision_accuracy)}</span><span>${value.failure_count ?? 0}</span><span>${Number(value.average_total_seconds || 0).toFixed(3)}s</span></div>`;
  }).join("");
  const failures = (bestConfig?.details || []).filter((item) => item.failures?.length);
  const failureHtml = failures.length ? failures.map((item) => {
    const sources = (item.candidates || []).slice(0, 4).map((candidate, index) => `${index + 1}. ${candidate.source} · 向量 ${candidate.dense_score == null ? "—" : Number(candidate.dense_score).toFixed(4)}`).join("\n") || "未召回候选片段";
    return `<details class="failure-card"><summary><strong>${escapeHtml(item.question)}</strong><span class="failure-reason">${escapeHtml(item.failures.join("、"))}</span></summary><div class="failure-content"><p>预期：${item.should_answer ? "回答" : "拒答"} · 实际：${item.status === "answered" ? "回答" : "拒答"}</p><p>预期来源：${escapeHtml((item.expected_sources || []).join("、") || "无")}</p><div class="failure-sources">${escapeHtml(sources)}</div>${item.answer ? `<p><b>答案摘要：</b>${escapeHtml(item.answer.slice(0, 260))}</p>` : ""}</div></details>`;
  }).join("") : '<p class="empty-docs">最佳配置下没有发现失败用例。</p>';
  $("#evalResult").classList.remove("empty");
  $("#evalResult").innerHTML = `
    <div class="eval-summary-head"><div><p class="eyebrow">EVALUATION REPORT</p><h2>评测报告</h2><p>${run.case_count} 条用例 · ${run.configs?.length || 0} 组配置 · ${run.full_generation ? "完整生成" : "快速检索"} · ${Number(run.elapsed_seconds || 0).toFixed(2)}s</p></div><div class="best-config">最佳：${escapeHtml(best.retrieval_mode || "—")} · Top-${best.top_k || "—"} · ${best.min_score == null ? "—" : Number(best.min_score).toFixed(2)}</div></div>
    <div class="eval-metrics">${evalMetric("Hit@K", percent(metrics.hit_at_k))}${evalMetric("MRR", Number(metrics.mrr || 0).toFixed(3))}${evalMetric("决策准确率", percent(metrics.decision_accuracy))}${evalMetric("拒答召回率", percent(metrics.refusal_recall))}${evalMetric("误拒答率", percent(metrics.false_refusal_rate))}${evalMetric("平均耗时", `${Number(metrics.average_total_seconds || 0).toFixed(3)}s`)}</div>
    <div class="eval-comparison"><div class="eval-comparison-row header"><span>配置</span><span>Hit@K</span><span>MRR</span><span>决策准确率</span><span>失败</span><span>耗时</span></div>${comparison}</div>
    <h3 class="failure-heading">失败用例 · ${failures.length}</h3><div class="failure-list">${failureHtml}</div>`;
  $("#evalResult").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadEvalRuns() {
  const { runs } = await api("/api/eval/runs");
  $("#evalRunList").innerHTML = runs?.length ? runs.map((run) => {
    const best = run.best || {};
    return `<div class="eval-run"><time>${escapeHtml(run.created_at?.slice(0, 19).replace("T", " ") || "")}</time><span>${run.case_count} 用例 · ${run.config_count} 配置 · ${run.full_generation ? "完整生成" : "快速检索"}</span><span>${escapeHtml(best.retrieval_mode || "—")} / K=${best.top_k || "—"}</span><button type="button" data-action="view-eval-run" data-run="${escapeHtml(run.id)}">查看</button></div>`;
  }).join("") : '<p class="empty-docs">暂无评测历史。</p>';
}

async function viewEvalRun(runId) {
  try { renderEvaluation(await api(`/api/eval/run?id=${encodeURIComponent(runId)}`)); }
  catch (error) { showError(error.message); }
}

function renderNotice(message) { $("#resultPanel").classList.remove("empty"); $("#resultPanel").innerHTML = `<div class="empty-state"><div class="empty-mark">✓</div><h2>完成</h2><p>${escapeHtml(message)}</p></div>`; }
function showError(message) {
  const panel = $("#evalView").classList.contains("hidden") ? $("#resultPanel") : $("#evalResult");
  panel.classList.remove("empty");
  panel.innerHTML = `<div class="empty-state"><div class="empty-mark">!</div><h2>出现问题</h2><p class="error-text">${escapeHtml(message)}</p></div>`;
  hideProgress();
}
function setBusy(busy) { ["#askButton", "#uploadButton", "#rebuildButton", "#runEvaluation"].forEach((selector) => $(selector).disabled = busy); }

async function loadLogs() {
  const { logs } = await api("/api/logs");
  $("#logBody").innerHTML = logs.length ? logs.map((log) => {
    const timing = log.timing || {}; const cls = log.status === "answered" ? "ok-text" : log.status === "refused" ? "warn-text" : "error-text";
    const label = log.status === "answered" ? "已回答" : log.status === "refused" ? "已拒答" : "引用异常";
    return `<tr><td>${escapeHtml(log.timestamp?.slice(5, 19).replace("T", " ") || "")}</td><td title="${escapeHtml(log.question)}">${escapeHtml(log.question)}</td><td><span class="run-status ${cls}">${label}</span></td><td>${Number(timing.retrieval_seconds || 0).toFixed(2)}s</td><td>${Number(timing.generation_seconds || 0).toFixed(2)}s</td><td>${Number(timing.total_seconds || 0).toFixed(2)}s</td></tr>`;
  }).join("") : '<tr><td colspan="6">暂无记录</td></tr>';
}

document.addEventListener("click", (event) => {
  const action = event.target.closest("[data-action]");
  if (action) {
    event.preventDefault(); event.stopPropagation();
    if (action.dataset.action === "view-document") openDocument(decodeURIComponent(action.dataset.name), false);
    if (action.dataset.action === "edit-document") openDocument(decodeURIComponent(action.dataset.name), true);
    if (action.dataset.action === "rename-section") openSectionDialog(action.dataset.section, action.dataset.name);
    if (action.dataset.action === "edit-eval-case") openEvalCaseDialog(action.dataset.case);
    if (action.dataset.action === "view-eval-run") viewEvalRun(action.dataset.run);
    return;
  }
  const toggle = event.target.closest(".evidence-toggle");
  if (toggle) {
    const text = toggle.previousElementSibling; const expanded = toggle.getAttribute("aria-expanded") === "true";
    text.classList.toggle("collapsed", expanded); toggle.setAttribute("aria-expanded", String(!expanded)); toggle.textContent = expanded ? "展开全文" : "收起"; return;
  }
  const citation = event.target.closest(".citation-link");
  if (citation) {
    const card = document.getElementById(`evidence-${citation.dataset.citation}`);
    if (card) { card.scrollIntoView({ behavior: "smooth", block: "center" }); card.classList.add("citation-focus"); setTimeout(() => card.classList.remove("citation-focus"), 1400); }
  }
});

$("#fileInput").addEventListener("change", (event) => selectFiles(event.target.files));
$("#uploadButton").addEventListener("click", uploadFiles); $("#rebuildButton").addEventListener("click", rebuild);
$("#newSectionButton").addEventListener("click", () => openSectionDialog()); $("#askButton").addEventListener("click", ask);
$("#refreshLogs").addEventListener("click", loadLogs); $("#saveDocument").addEventListener("click", saveDocument);
$("#closeDocumentDialog").addEventListener("click", () => $("#documentDialog").close());
$("#cancelDocument").addEventListener("click", () => $("#documentDialog").close());
$("#documentDialog").addEventListener("click", (event) => { if (event.target === $("#documentDialog")) $("#documentDialog").close(); });
$("#saveSection").addEventListener("click", saveSection);
$("#closeSectionDialog").addEventListener("click", () => $("#sectionDialog").close());
$("#cancelSection").addEventListener("click", () => $("#sectionDialog").close());
$("#sectionNameInput").addEventListener("keydown", (event) => { if (event.key === "Enter") saveSection(); });
$("#topK").addEventListener("input", (event) => $("#topKValue").textContent = event.target.value);
$("#threshold").addEventListener("input", (event) => $("#thresholdValue").textContent = Number(event.target.value).toFixed(2));
$("#question").addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) ask(); });
document.querySelectorAll(".workspace-tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".workspace-tab").forEach((item) => item.classList.toggle("active", item === tab));
  document.querySelectorAll(".workspace-view").forEach((view) => view.classList.toggle("hidden", view.id !== tab.dataset.view));
  if (tab.dataset.view === "evalView") { loadEvalCases().catch((error) => showError(error.message)); loadEvalRuns().catch(() => {}); }
}));
$("#newEvalCase").addEventListener("click", () => openEvalCaseDialog());
$("#runEvaluation").addEventListener("click", runEvaluation);
$("#refreshEvalRuns").addEventListener("click", loadEvalRuns);
$("#saveEvalCase").addEventListener("click", saveEvalCase);
$("#deleteEvalCase").addEventListener("click", deleteEvalCase);
$("#closeEvalCaseDialog").addEventListener("click", () => $("#evalCaseDialog").close());
$("#cancelEvalCase").addEventListener("click", () => $("#evalCaseDialog").close());
$("#evalCaseDialog").addEventListener("click", (event) => { if (event.target === $("#evalCaseDialog")) $("#evalCaseDialog").close(); });
const dropzone = $("#dropzone");
["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
dropzone.addEventListener("drop", (event) => selectFiles(event.dataTransfer.files));

loadHealth();
loadLibrary().then(loadEvalCases).catch((error) => showError(error.message));
loadLogs().catch(() => {}); loadEvalRuns().catch(() => {});
