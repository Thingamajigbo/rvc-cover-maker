const $ = (id) => document.getElementById(id);
const api = (p, opts) => fetch(p, opts).then(async (r) => {
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
});

let currentJob = null;
let polling = null;

// ---- slider labels ----
[["pitch", "pitchVal"], ["indexRate", "indexRateVal"], ["protect", "protectVal"], ["rms", "rmsVal"]]
  .forEach(([s, v]) => $(s).addEventListener("input", () => $(v).textContent = $(s).value));

// ---- status pill ----
async function loadStatus() {
  try {
    const g = await api("/api/gpu");
    const d = g.device;
    const dev = d.cuda ? `GPU · ${d.cuda_name}` : d.mps ? "로컬 · Apple (CPU)" : "로컬 · CPU";
    $("statusText").textContent = g.busy ? "처리 중…" : dev;
    $("statusPill").className = "status-pill " + (g.busy ? "busy" : "online");
  } catch {
    $("statusText").textContent = "엔진 오프라인";
    $("statusPill").className = "status-pill";
  }
}

// ---- models ----
async function loadModels() {
  const sel = $("model"), cur = sel.value;
  const models = await api("/api/models");
  sel.innerHTML = models.length
    ? models.map(m => `<option value="${m.name}">${m.name}${m.has_index ? "" : " (index 없음)"}</option>`).join("")
    : `<option value="">모델 없음 — ＋ 로 업로드</option>`;
  if (cur) sel.value = cur;
}
$("reloadModels").onclick = loadModels;
$("openUpload").onclick = () => $("uploadBox").classList.toggle("hidden");
$("cancelUpload").onclick = () => $("uploadBox").classList.add("hidden");
$("doUpload").onclick = async () => {
  const name = $("upName").value.trim(), files = $("upFiles").files;
  if (!name || !files.length) return $("upMsg").textContent = "이름과 파일을 모두 지정하세요.";
  const fd = new FormData();
  fd.append("name", name);
  for (const f of files) fd.append("files", f);
  $("upMsg").textContent = "업로드 중…";
  try {
    const res = await api("/api/models", { method: "POST", body: fd });
    $("upMsg").textContent = `완료: ${res.name}`;
    await loadModels(); $("model").value = res.name;
    setTimeout(() => $("uploadBox").classList.add("hidden"), 800);
  } catch (e) { $("upMsg").textContent = "오류: " + e.message; }
};

// ---- panels ----
function show(area) { // 'placeholder' | 'live' | 'result' | 'error'
  $("placeholder").classList.toggle("hidden", area !== "placeholder");
  $("liveArea").classList.toggle("hidden", area !== "live");
  $("resultArea").classList.toggle("hidden", area !== "result");
  $("errorArea").classList.toggle("hidden", area !== "error");
}

// progress -> visual step index (0..4)
function stepIndex(p) {
  if (p >= 1) return 4;
  if (p >= 0.96) return 3;
  if (p >= 0.80) return 2;
  if (p >= 0.15) return 1;
  return 0;
}
function paintStepper(p, done) {
  const idx = stepIndex(p);
  document.querySelectorAll(".step").forEach((el, i) => {
    el.classList.toggle("active", i === idx && !done);
    el.classList.toggle("done", i < idx || done);
  });
}

// ---- source mode (youtube / file) ----
let coverSrc = "yt";
document.querySelectorAll(".src-tab").forEach((t) => {
  t.onclick = () => {
    coverSrc = t.dataset.src;
    document.querySelectorAll(".src-tab").forEach((x) => x.classList.toggle("active", x === t));
    $("srcYt").classList.toggle("hidden", coverSrc !== "yt");
    $("srcFile").classList.toggle("hidden", coverSrc !== "file");
  };
});

// ---- generate ----
$("generate").onclick = async () => {
  const model_name = $("model").value;
  if (!model_name) return alert("모델을 선택하세요.");
  const opts = {
    pitch: +$("pitch").value, index_rate: +$("indexRate").value,
    protect: +$("protect").value, rms_mix_rate: +$("rms").value,
    f0_method: $("f0").value, output_format: $("oformat").value,
  };

  let request;
  if (coverSrc === "file") {
    const f = $("coverFile").files[0];
    if (!f) return alert("음원 파일을 선택하세요.");
    const fd = new FormData();
    fd.append("file", f);
    fd.append("model_name", model_name);
    for (const [k, v] of Object.entries(opts)) fd.append(k, v);
    request = () => api("/api/jobs/file", { method: "POST", body: fd });
  } else {
    const youtube_url = $("yt").value.trim();
    if (!youtube_url) return alert("유튜브 링크를 입력하세요.");
    request = () => api("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ youtube_url, model_name, ...opts }),
    });
  }

  $("generate").disabled = true;
  show("live");
  $("log").textContent = "";
  $("stallBanner").classList.remove("show");
  setProgress(0, "작업 제출 중…");
  $("statusSub").textContent = "커버를 생성하고 있습니다.";
  try {
    const { job_id } = await request();
    currentJob = job_id;
    pollJob(job_id);
  } catch (e) { showError(e.message); $("generate").disabled = false; }
};

function setProgress(frac, step) {
  $("bar").style.width = Math.round(frac * 100) + "%";
  $("pct").textContent = Math.round(frac * 100) + "%";
  if (step) $("stepText").textContent = step;
  paintStepper(frac, false);
}

function pollJob(id) {
  clearInterval(polling);
  polling = setInterval(async () => {
    let job;
    try { job = await api(`/api/jobs/${id}`); } catch { return; }
    setProgress(job.progress, `${job.step} · ${Math.round(job.progress * 100)}%`);
    $("log").textContent = job.logs.slice(-50).join("\n");
    $("log").scrollTop = $("log").scrollHeight;
    $("stallBanner").classList.toggle("show", !!job.stalled);

    if (job.status === "done") { stop(); paintStepper(1, true); showResult(job.result); }
    else if (job.status === "error") { stop(); showError(job.error || "알 수 없는 오류"); }
    else if (job.status === "cancelled") { stop(); $("statusSub").textContent = "작업이 취소되었습니다."; show("placeholder"); }
  }, 1200);
  function stop() { clearInterval(polling); $("generate").disabled = false; loadStatus(); }
}

$("cancelBtn").onclick = async () => {
  if (!currentJob) return;
  $("cancelBtn").disabled = true;
  try { await api(`/api/jobs/${currentJob}/cancel`, { method: "POST" }); } catch {}
  setTimeout(() => $("cancelBtn").disabled = false, 1500);
};
$("retryBtn").onclick = async () => {
  if (!currentJob) return;
  try {
    const { job_id } = await api(`/api/jobs/${currentJob}/retry`, { method: "POST" });
    currentJob = job_id; show("live"); $("generate").disabled = true;
    setProgress(0, "재시도 중…"); pollJob(job_id);
  } catch (e) { showError(e.message); }
};

function showError(msg) {
  show("error"); $("errText").textContent = msg;
  $("statusSub").textContent = "오류가 발생했습니다.";
}
function showResult(result) {
  if (!result) return;
  show("result");
  $("statusSub").textContent = "커버가 완성되었습니다 🎉";
  $("resultTitle").textContent = result.title || "";
  $("player").src = result.url;
  $("download").href = result.url;
  $("download").setAttribute("download", result.download_name || result.filename);
  $("lrcTitle").value = (result.title || "").replace(/\(.*?Ver\)/i, "").replace(/[｜|].*/, "").trim();
}

// ---- lyrics ----
$("lrcSearch").onclick = async () => {
  const title = $("lrcTitle").value.trim(), artist = $("lrcArtist").value.trim();
  if (!title) return;
  $("lrcBox").textContent = "검색 중…";
  try {
    const r = await api(`/api/lyrics?title=${encodeURIComponent(title)}&artist=${encodeURIComponent(artist)}`);
    $("lrcBox").textContent = r.synced || r.plain || "가사를 찾지 못했습니다. 직접 붙여넣으세요.";
  } catch (e) { $("lrcBox").textContent = "검색 실패: " + e.message; }
};

// ======================= TRAINING TAB =======================
// tab switching
document.querySelectorAll(".studio-tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".studio-tab").forEach((t) => t.classList.toggle("active", t === tab));
    const p = tab.dataset.panel;
    $("coverPanel").classList.toggle("hidden", p !== "cover");
    $("trainPanel").classList.toggle("hidden", p !== "train");
  };
});

let trainJob = null;
let tPolling = null;

$("tCheck").onclick = async () => {
  const name = $("tModelName").value.trim();
  const files = $("tFiles").files;
  if (!name) return alert("모델 이름을 입력하세요.");
  if (!files.length) return alert("목소리 데이터 파일을 선택하세요.");
  const box = $("tCheckResult");
  box.classList.remove("hidden");
  box.innerHTML = "검사 중…";
  const fd = new FormData();
  fd.append("name", name);
  for (const f of files) fd.append("files", f);
  try {
    const r = await api("/api/train/check", { method: "POST", body: fd });
    const mins = (r.total_seconds / 60).toFixed(1);
    let html = `<div class="metrics"><span>파일 <b>${r.files}</b></span><span>길이 <b>${mins}분</b></span>` +
      `<span>무음 <b>${Math.round(r.silence_ratio * 100)}%</b></span><span>SR <b>${(r.sample_rates || []).join(",")}</b></span></div>`;
    if (r.warnings && r.warnings.length) html += r.warnings.map((w) => `<div class="warn-item">⚠️ ${w}</div>`).join("");
    else html += `<div class="ok-item">✓ 데이터 양호</div>`;
    box.innerHTML = html;
    $("tStart").disabled = false;
    $("tStartHint").textContent = "이제 학습을 시작할 수 있습니다.";
  } catch (e) { box.innerHTML = `<div class="warn-item">검사 실패: ${e.message}</div>`; }
};

function tShow(area) {
  $("tPlaceholder").classList.toggle("hidden", area !== "placeholder");
  $("tLiveArea").classList.toggle("hidden", area !== "live");
  $("tResultArea").classList.toggle("hidden", area !== "result");
  $("tErrorArea").classList.toggle("hidden", area !== "error");
}

$("tStart").onclick = async () => {
  const body = {
    model_name: $("tModelName").value.trim(),
    epochs: +$("tEpochs").value || 100,
    sample_rate: $("tSr").value,
    batch_size: +$("tBatch").value || 4,
  };
  $("tStart").disabled = true;
  tShow("live");
  $("tLog").textContent = "";
  $("tStallBanner").classList.remove("show");
  $("tBar").style.width = "0%"; $("tPct").textContent = "0%"; $("tStepText").textContent = "작업 제출 중…";
  $("tStatusSub").textContent = "모델을 학습하고 있습니다.";
  try {
    const { job_id } = await api("/api/train", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    trainJob = job_id;
    tPoll(job_id);
  } catch (e) { tShowError(e.message); $("tStart").disabled = false; }
};

function tPoll(id) {
  clearInterval(tPolling);
  tPolling = setInterval(async () => {
    let job;
    try { job = await api(`/api/jobs/${id}`); } catch { return; }
    const pct = Math.round(job.progress * 100);
    $("tBar").style.width = pct + "%"; $("tPct").textContent = pct + "%";
    $("tStepText").textContent = `${job.step} · ${pct}%`;
    $("tLog").textContent = job.logs.slice(-50).join("\n");
    $("tLog").scrollTop = $("tLog").scrollHeight;
    $("tStallBanner").classList.toggle("show", !!job.stalled);
    if (job.status === "done") { tStop(); tShowResult(job.result); }
    else if (job.status === "error") { tStop(); tShowError(job.error || "알 수 없는 오류"); }
    else if (job.status === "cancelled") { tStop(); $("tStatusSub").textContent = "학습이 취소되었습니다."; tShow("placeholder"); }
  }, 2000);
  function tStop() { clearInterval(tPolling); $("tStart").disabled = false; loadStatus(); }
}

$("tCancel").onclick = async () => {
  if (!trainJob) return;
  $("tCancel").disabled = true;
  try { await api(`/api/jobs/${trainJob}/cancel`, { method: "POST" }); } catch {}
  setTimeout(() => $("tCancel").disabled = false, 1500);
};

function tShowError(msg) { tShow("error"); $("tErrText").textContent = msg; $("tStatusSub").textContent = "오류가 발생했습니다."; }
function tShowResult(result) {
  tShow("result");
  $("tStatusSub").textContent = "학습이 완료되었습니다 🎉";
  $("tResultName").textContent = `모델: ${result?.name || ""}` + (result?.index ? " (index 포함)" : "");
  loadModels(); // new model now selectable in cover tab
}
$("tGoCover").onclick = () => document.querySelector('.studio-tab[data-panel="cover"]').click();

// ---- init ----
loadStatus(); loadModels();
setInterval(loadStatus, 5000);
