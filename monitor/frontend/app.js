const FILE_KEYS = [
  "mapping_api",
  "pipeline_stats",
  "llm",
  "org_pipeline_stats",
  "patent_pipeline_stats",
  "google_search",
  "serpapi",
];

const FILE_LABELS = {
  mapping_api: "API 调用",
  pipeline_stats: "ORCID 查找",
  llm: "LLM 调用",
  org_pipeline_stats: "人物报告",
  patent_pipeline_stats: "专利查询",
  google_search: "Google 搜索",
  serpapi: "SerpAPI",
};

const $ = (sel) => document.querySelector(sel);

function apiBase() {
  const raw = ($("#apiBase").value || "").trim().replace(/\/$/, "");
  return raw;
}

function apiUrl(path) {
  const base = apiBase();
  return base ? `${base}${path}` : path;
}

function toast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.style.background = isError ? "#991b1b" : "#111827";
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 4000);
}

function last7Days() {
  const days = [];
  const now = new Date();
  for (let i = 6; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(now.getDate() - i);
    days.push(d.toISOString().slice(0, 10));
  }
  return days;
}

function fmtUptime(sec) {
  if (!sec) return "—";
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  return `${d}天 ${h}小时`;
}

async function fetchJson(path) {
  const res = await fetch(apiUrl(path));
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

function renderSummary(infra) {
  const celery = infra.celery || {};
  const redis = infra.redis || {};
  const mongo = infra.mongo || {};
  const nodes = infra.nodes || {};
  const llm = infra.llm_local || {};

  const cards = [
    {
      label: "本地 LLM",
      value: llm.healthy ? "健康" : "异常",
      hint: llm.healthy
        ? `${llm.model || "—"} · ${llm.latency_ms ?? "—"} ms`
        : llm.error || `HTTP ${llm.http_status ?? "—"}`,
      healthy: llm.healthy,
    },
    {
      label: "Celery 在线",
      value: celery.ok ? `${celery.online_count || 0}` : "—",
      hint: celery.ok ? `并发槽 ${celery.total_concurrency || 0}` : celery.error,
    },
    {
      label: "活跃任务",
      value: celery.ok ? `${celery.active_tasks || 0}` : "—",
      hint: "当前执行中",
    },
    {
      label: "Redis 内存",
      value: redis.ok ? redis.used_memory_human : "—",
      hint: redis.ok ? `队列 celery:${redis.queue_celery} patent:${redis.queue_patent}` : redis.error,
    },
    {
      label: "Mongo 数据",
      value: mongo.ok ? `${mongo.data_size_gb} GB` : "—",
      hint: mongo.ok ? `存储 ${mongo.storage_size_gb} GB` : mongo.error,
    },
    {
      label: "Worker 进程",
      value: `${nodes.workers_with_celery || 0} / 4`,
      hint: "有 Celery 的节点数",
    },
  ];

  $("#summary").innerHTML = cards
    .map(
      (c) => `
    <div class="stat-card${c.healthy === true ? " stat-ok" : c.healthy === false ? " stat-bad" : ""}">
      <div class="label">${c.label}</div>
      <div class="value">${c.value ?? "—"}</div>
      <div class="hint">${c.hint || ""}</div>
    </div>`
    )
    .join("");
}

function renderCelery(infra) {
  const c = infra.celery || {};
  if (!c.ok) {
    $("#celeryTable").innerHTML = `<div class="error-box">${c.error || "采集失败"}</div>`;
    return;
  }
  const rows = (c.nodes || [])
    .map(
      (n) => `
    <tr>
      <td>${n.hostname}</td>
      <td><span class="badge ${n.online ? "ok" : "bad"}">${n.online ? "在线" : "离线"}</span></td>
      <td>${n.concurrency ?? "—"}</td>
      <td>${n.total_tasks?.toLocaleString?.() ?? n.total_tasks ?? 0}</td>
      <td>${n.active_count ?? 0}</td>
      <td>${fmtUptime(n.uptime_seconds)}</td>
    </tr>`
    )
    .join("");

  $("#celeryTable").innerHTML = `
    <table>
      <thead><tr><th>节点</th><th>状态</th><th>并发</th><th>累计任务</th><th>活跃</th><th>运行时长</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="6" class="empty-note">无数据</td></tr>'}</tbody>
    </table>`;
}

function renderNodes(infra) {
  const n = infra.nodes || {};
  const rows = (n.nodes || [])
    .map((node) => {
      const pm2 = node.pm2;
      let pm2Text = "—";
      if (pm2?.status === "online") pm2Text = `online (${pm2.restarts ?? 0} 重启)`;
      else if (pm2?.status === "no_pm2") pm2Text = "无 PM2";
      else if (pm2) pm2Text = pm2.status || "—";

      const ok = node.ok !== false && (node.celery_processes || 0) > 0;
      return `
      <tr>
        <td>${node.label || node.id}<br><span class="muted">${node.hostname || ""}</span></td>
        <td><span class="badge ${ok ? "ok" : "bad"}">${ok ? "运行中" : "异常"}</span></td>
        <td>${node.load ?? "—"}</td>
        <td>${node.memory ?? "—"}</td>
        <td>${node.celery_processes ?? 0}</td>
        <td>${node.celery_queues ?? "—"}</td>
        <td>${pm2Text}</td>
      </tr>`;
    })
    .join("");

  $("#nodesTable").innerHTML = `
    <table>
      <thead><tr><th>节点</th><th>状态</th><th>Load</th><th>内存</th><th>Celery</th><th>队列</th><th>PM2</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderRedis(infra) {
  const r = infra.redis || {};
  if (!r.ok) {
    $("#redisPanel").innerHTML = `<div class="error-box">${r.error}</div>`;
    return;
  }
  $("#redisPanel").innerHTML = `
    <dl class="kv-grid">
      <dt>内存占用</dt><dd>${r.used_memory_human} (${r.used_memory_mb} MB)</dd>
      <dt>maxmemory</dt><dd>${r.maxmemory_human}</dd>
      <dt>Key 总数</dt><dd>${r.dbsize?.toLocaleString?.() ?? r.dbsize}</dd>
      <dt>celery 队列</dt><dd>${r.queue_celery}</dd>
      <dt>patent 队列</dt><dd>${r.queue_patent}</dd>
    </dl>`;
}

function renderMongo(infra) {
  const m = infra.mongo || {};
  if (!m.ok) {
    $("#mongoPanel").innerHTML = `<div class="error-box">${m.error}</div>`;
    return;
  }
  const top = (m.collections || []).slice(0, 8);
  const rows = top
    .map(
      (c) => `
    <tr>
      <td>${c.name}</td>
      <td>${c.count?.toLocaleString?.() ?? c.count}</td>
      <td>${c.size_mb} MB</td>
      <td>${c.storage_mb} MB</td>
    </tr>`
    )
    .join("");

  $("#mongoPanel").innerHTML = `
    <dl class="kv-grid" style="margin-bottom:0.75rem">
      <dt>数据库</dt><dd>${m.db}</dd>
      <dt>逻辑大小</dt><dd>${m.data_size_gb} GB</dd>
      <dt>磁盘存储</dt><dd>${m.storage_size_gb} GB</dd>
      <dt>索引</dt><dd>${m.index_size_mb} MB</dd>
    </dl>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Collection</th><th>文档数</th><th>大小</th><th>存储</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function renderUsageCard(fileKey, item) {
  const d = item.data;
  const date = item.date;
  const has = item.has_data && d;

  let metrics = "";
  let endpoints = "";

  if (!has) {
    metrics = `<div class="empty-note">该日无数据</div>`;
  } else if (fileKey === "mapping_api" || fileKey === "google_search" || fileKey === "serpapi" || fileKey === "llm") {
    metrics = `
      <div class="usage-metrics">
        <span class="metric-label">总调用</span><span class="metric-value">${d.total?.toLocaleString?.() ?? d.total ?? 0}</span>
      </div>`;
    if (d.endpoints) {
      endpoints = `<div class="usage-endpoints">${Object.entries(d.endpoints)
        .sort((a, b) => b[1] - a[1])
        .map(([k, v]) => `<div><span>${k}</span><span>${v.toLocaleString()}</span></div>`)
        .join("")}</div>`;
    }
  } else if (fileKey === "pipeline_stats" || fileKey === "org_pipeline_stats") {
    metrics = `
      <div class="usage-metrics">
        <span class="metric-label">请求</span><span class="metric-value">${d.total_requests ?? 0}</span>
        <span class="metric-label">缓存命中</span><span class="metric-value">${d.cache_hits ?? 0}</span>
        <span class="metric-label">成功</span><span class="metric-value">${d.success ?? 0}</span>
        <span class="metric-label">未找到</span><span class="metric-value">${d.not_found ?? 0}</span>
        <span class="metric-label">错误</span><span class="metric-value">${d.error ?? 0}</span>
      </div>`;
  } else if (fileKey === "patent_pipeline_stats") {
    metrics = `
      <div class="usage-metrics">
        <span class="metric-label">请求</span><span class="metric-value">${d.total_requests ?? 0}</span>
        <span class="metric-label">成功</span><span class="metric-value">${d.success ?? 0}</span>
        <span class="metric-label">未找到</span><span class="metric-value">${d.not_found ?? 0}</span>
        <span class="metric-label">错误</span><span class="metric-value">${d.error ?? 0}</span>
        <span class="metric-label">确认专利</span><span class="metric-value">${d.confirmed_total ?? 0}</span>
      </div>`;
  }

  const days = last7Days();
  const options = days
    .map((day) => `<option value="${day}" ${day === date ? "selected" : ""}>${day}</option>`)
    .join("");

  return `
    <div class="usage-card" data-file="${fileKey}">
      <div class="usage-card-header">
        <h3>${FILE_LABELS[fileKey] || fileKey}</h3>
        <select class="usage-date-select" data-file="${fileKey}">${options}</select>
      </div>
      <div class="usage-card-body" id="usage-body-${fileKey}">
        ${metrics}
        ${endpoints}
      </div>
    </div>`;
}

function renderUsageBatch(batch) {
  const byKey = {};
  (batch.items || []).forEach((item) => {
    byKey[item.file_key] = item;
  });

  $("#usageGrid").innerHTML = FILE_KEYS.map((key) => renderUsageCard(key, byKey[key] || { file_key: key, date: batch.date, has_data: false })).join("");

  document.querySelectorAll(".usage-date-select").forEach((sel) => {
    sel.addEventListener("change", async (e) => {
      const file = e.target.dataset.file;
      const day = e.target.value;
      await loadSingleUsage(file, day);
    });
  });
}

async function loadSingleUsage(file, day) {
  const body = document.getElementById(`usage-body-${file}`);
  if (body) body.innerHTML = `<div class="empty-note">加载中…</div>`;
  try {
    const item = await fetchJson(`/api/usage?file=${encodeURIComponent(file)}&date=${encodeURIComponent(day)}`);
    const card = renderUsageCard(file, item);
    const tmp = document.createElement("div");
    tmp.innerHTML = card;
    const newBody = tmp.querySelector(".usage-card-body");
    if (body && newBody) {
      body.innerHTML = newBody.innerHTML;
      const sel = document.querySelector(`.usage-date-select[data-file="${file}"]`);
      if (sel) sel.value = day;
    }
  } catch (err) {
    if (body) body.innerHTML = `<div class="error-box">${err.message}</div>`;
    toast(`用量加载失败: ${file}`, true);
  }
}

function fillGlobalDateSelect(selected) {
  const sel = $("#globalDate");
  const days = last7Days();
  sel.innerHTML = days.map((d) => `<option value="${d}" ${d === selected ? "selected" : ""}>${d}</option>`).join("");
}

async function loadAll() {
  const btn = $("#refreshBtn");
  btn.disabled = true;
  const day = $("#globalDate").value || new Date().toISOString().slice(0, 10);

  try {
    const [infra, usage] = await Promise.all([
      fetchJson("/api/infra"),
      fetchJson(`/api/usage/batch?date=${encodeURIComponent(day)}`),
    ]);

    renderSummary(infra);
    renderCelery(infra);
    renderNodes(infra);
    renderRedis(infra);
    renderMongo(infra);
    renderUsageBatch(usage);

    document.querySelectorAll(".usage-date-select").forEach((sel) => {
      sel.value = day;
    });

    $("#lastFetch").textContent = `上次加载: ${new Date().toLocaleString()}`;
    toast("数据已更新");
  } catch (err) {
    toast(`加载失败: ${err.message}`, true);
    console.error(err);
  } finally {
    btn.disabled = false;
  }
}

function init() {
  const today = new Date().toISOString().slice(0, 10);
  fillGlobalDateSelect(today);

  $("#refreshBtn").addEventListener("click", loadAll);
  $("#globalDate").addEventListener("change", loadAll);

  loadAll();
}

init();
