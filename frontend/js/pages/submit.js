/* 作业提交 Submit */
Components.init('submit');
const C = Components;

let SAMPLES = [];
let lastReport = null;       // most recent preflight report
let lastBody = null;         // payload that produced it
let forceSubmit = false;     // bypass the warning confirmation once

async function init() {
  const funcs = await API.get('/api/functions');
  SAMPLES = await API.get('/api/samples');

  fillSelect('mapper', funcs.mappers);
  fillSelect('reducer', funcs.reducers);

  const preset = document.getElementById('preset');
  preset.innerHTML = SAMPLES.map(s => `<option value="${s.name}">${C.esc(s.name)}</option>`).join('');
  preset.addEventListener('change', () => {
    const s = SAMPLES.find(x => x.name === preset.value);
    if (s) fillFromSample(s);
    resetReport();
  });

  document.getElementById('form').addEventListener('submit', onSubmit);
  document.getElementById('btn-preflight').addEventListener('click', onPreflightOnly);
  document.getElementById('btn-force').addEventListener('click', onForceSubmit);
  document.getElementById('btn-back').addEventListener('click', resetReport);

  document.getElementById('func-list').innerHTML =
    '<h3>Map</h3>' + funcs.mappers.map(f =>
      `<div class="small" style="padding:2px 0"><span class="mono">${C.esc(f.name)}</span> — ${C.esc(f.description)}</div>`).join('') +
    '<h3 class="mt">Reduce</h3>' + funcs.reducers.map(f =>
      `<div class="small" style="padding:2px 0"><span class="mono">${C.esc(f.name)}</span> — ${C.esc(f.description)}</div>`).join('');

  loadRecent();
}

function fillSelect(id, items) {
  document.getElementById(id).innerHTML = items
    .map(f => `<option value="${f.name}">${C.esc(f.name)}</option>`).join('');
}

function fillFromSample(s) {
  document.getElementById('name').value = s.name;
  document.getElementById('mapper').value = s.mapper;
  document.getElementById('reducer').value = s.reducer;
  document.getElementById('num_map_tasks').value = s.num_map_tasks;
  document.getElementById('num_reduce_tasks').value = s.num_reduce_tasks;
  document.getElementById('input_rows').value = s.input_rows;
  document.getElementById('pattern').value = (s.params && s.params.pattern) || '';
  document.getElementById('depends_on').value = '';
  document.getElementById('simulate_failure').checked = false;
}

// ---------------------------------------------------------------------------
// Payload
// ---------------------------------------------------------------------------
function buildBody() {
  const body = {
    name: document.getElementById('name').value.trim(),
    mapper: document.getElementById('mapper').value,
    reducer: document.getElementById('reducer').value,
    num_map_tasks: parseInt(document.getElementById('num_map_tasks').value, 10),
    num_reduce_tasks: parseInt(document.getElementById('num_reduce_tasks').value, 10),
    input_rows: parseInt(document.getElementById('input_rows').value, 10),
    params: {},
  };
  const pattern = document.getElementById('pattern').value.trim();
  if (pattern) body.params.pattern = pattern;
  if (document.getElementById('simulate_failure').checked) body.params.simulate_failure = true;
  const deps = document.getElementById('depends_on').value
    .split(/[,\s]+/).map(s => s.trim()).filter(Boolean);
  if (deps.length) body.depends_on = deps;
  return body;
}

function setBusy(submitBtn, busy) {
  submitBtn.disabled = busy;
  document.getElementById('btn-preflight').disabled = busy;
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------
async function onSubmit(ev) {
  ev.preventDefault();
  await runPreflightThenSubmit(false);
}

async function onPreflightOnly() {
  await runPreflightThenSubmit(true);
}

async function onForceSubmit() {
  forceSubmit = true;
  await doSubmit(lastBody);
}

async function runPreflightThenSubmit(dryRunOnly) {
  const body = buildBody();
  lastBody = body;
  const submitBtn = document.getElementById('form').querySelector('button[type=submit]');
  setBusy(submitBtn, true);
  try {
    const report = await API.post('/api/jobs/preflight', body);
    lastReport = report;
    renderReport(report);

    if (dryRunOnly) {
      C.toast(`预检完成：${report.passed} 通过 / ${report.warnings} 警告 / ${report.failed} 失败`,
        report.failed ? 'error' : (report.warnings ? '' : 'ok'));
      return;
    }
    if (report.failed > 0) {
      C.toast(`预检未通过：${report.failed} 项失败，作业未提交`, 'error');
      return;
    }
    if (report.warnings > 0 && !forceSubmit) {
      C.toast(`预检通过但有 ${report.warnings} 个警告，请确认后再提交`, 'error');
      document.getElementById('force-bar').hidden = false;
      return;
    }
    await doSubmit(body);
  } catch (e) {
    C.toast('预检请求失败 ' + e.message, 'error');
  } finally {
    setBusy(submitBtn, false);
  }
}

async function doSubmit(body) {
  const submitBtn = document.getElementById('form').querySelector('button[type=submit]');
  setBusy(submitBtn, true);
  try {
    const job = await API.post('/api/jobs', body);
    C.toast('作业已提交 Job submitted: ' + job.job_id, 'ok');
    setTimeout(() => location.href = 'monitor.html', 600);
  } catch (e) {
    C.toast('提交被预检拒绝: ' + e.message, 'error');
    setBusy(submitBtn, false);
  }
}

function resetReport() {
  lastReport = null;
  forceSubmit = false;
  document.getElementById('report').hidden = true;
  document.getElementById('force-bar').hidden = true;
}

// ---------------------------------------------------------------------------
// Report rendering
// ---------------------------------------------------------------------------
const STATUS_UI = {
  pass: { icon: '✓', cls: 'good', label: '通过 OK' },
  warn: { icon: '!', cls: 'warn', label: '警告 WARN' },
  fail: { icon: '✕', cls: 'bad', label: '失败 FAIL' },
};
const CATEGORY_ORDER = ['params', 'data', 'shape', 'sharding', 'dependencies', 'cluster'];

function renderReport(report) {
  document.getElementById('force-bar').hidden = !(report.warnings > 0 && report.failed === 0);
  const host = document.getElementById('report');
  host.hidden = false;
  const placeholder = document.getElementById('report-placeholder');
  if (placeholder) placeholder.hidden = true;

  const verdict = report.failed > 0
    ? `<span class="badge bad">预检未通过 ${report.failed} 项失败</span>`
    : report.warnings > 0
      ? `<span class="badge warn">通过但有 ${report.warnings} 个警告</span>`
      : `<span class="badge good">全部 ${report.passed} 项检查通过，可以提交</span>`;

  const plan = report.plan || {};
  const planHtml = `
    <div class="preflight-plan small">
      <span><b>Mapper</b> ${C.esc(plan.mapper || '-')}</span>
      <span><b>Reducer</b> ${C.esc(plan.reducer || '-')}</span>
      <span><b>输入类型</b> ${C.esc(plan.input_kind || '-')}</span>
      <span><b>声明行数</b> ${C.fmtNum(plan.declared_input_rows)}</span>
      <span><b>实际记录</b> ${C.fmtNum(plan.generated_records)}</span>
      <span><b>Map 任务</b> ${C.fmtNum(plan.requested_map_tasks)} → ${C.fmtNum(plan.effective_map_tasks)}</span>
      <span><b>Reduce 任务</b> ${C.fmtNum(plan.effective_reduce_tasks)}</span>
      <span><b>前置作业</b> ${plan.depends_on && plan.depends_on.length ? plan.depends_on.map(C.esc).join(', ') : '无'}</span>
    </div>`;

  const byCat = {};
  for (const chk of report.checks) (byCat[chk.category] = byCat[chk.category] || []).push(chk);
  const groups = CATEGORY_ORDER
    .filter(cat => byCat[cat])
    .map(cat => {
      const rows = byCat[cat].map(renderCheck).join('');
      const label = (byCat[cat][0] && byCat[cat][0].category_label) || cat;
      const hasFail = byCat[cat].some(c => c.status === 'fail');
      const hasWarn = byCat[cat].some(c => c.status === 'warn');
      const headCls = hasFail ? 'bad' : (hasWarn ? 'warn' : 'good');
      return `<div class="preflight-group">
        <div class="preflight-group-head ${headCls}">${C.esc(label)}</div>
        <div class="preflight-rows">${rows}</div>
      </div>`;
    }).join('');

  host.innerHTML = `
    <div class="preflight-head flex between wrap">
      <div class="flex wrap">${verdict}
        <span class="small muted">通过 ${report.passed} · 警告 ${report.warnings} · 失败 ${report.failed}</span>
      </div>
    </div>
    ${planHtml}
    <div class="preflight-grid">${groups}</div>`;
}

function renderCheck(chk) {
  const ui = STATUS_UI[chk.status] || STATUS_UI.pass;
  const target = renderTarget(chk);
  const suggest = chk.suggestion
    ? `<div class="small muted preflight-suggest">建议: ${C.esc(chk.suggestion)}</div>` : '';
  return `<div class="preflight-row ${chk.status}">
    <span class="preflight-icon ${ui.cls}">${ui.icon}</span>
    <div class="preflight-body">
      <div class="preflight-title">${C.esc(chk.title)}</div>
      <div class="small">${C.esc(chk.message)}</div>
      ${target}
      ${suggest}
    </div>
  </div>`;
}

function fieldBadge(target) {
  if (!target || !target.field) return '';
  return `<span class="preflight-field mono">${C.esc(String(target.field))}</span>`;
}

function renderTarget(chk) {
  const t = chk.target || {};
  let html = '<div class="preflight-target small">';
  html += fieldBadge(t);

  // Offending input records / shards: make the bad data directly clickable in
  // the report rather than forcing the user to guess which record is broken.
  if (Array.isArray(t.offenders) && t.offenders.length) {
    html += '<div class="preflight-offenders"><table class="table"><thead><tr>' +
      '<th>记录 #</th><th>分片 Shard</th><th>问题 / 内容</th></tr></thead><tbody>';
    for (const o of t.offenders) {
      const where = o.shard_id
        ? `${C.esc(o.shard_id)} #${C.fmtNum(o.record_index)}`
        : `#${C.fmtNum(o.record_index)}`;
      const reason = o.reason || o.error || '';
      const rec = o.record !== undefined ? `<div class="mono preflight-rec">${C.esc(o.record)}</div>` : '';
      html += `<tr><td class="tabular">${C.fmtNum(o.record_index)}</td>` +
        `<td>${C.esc(o.shard_id || '')}</td>` +
        `<td>${C.esc(reason)}${rec}</td></tr>`;
    }
    html += '</tbody></table></div>';
  }

  if (Array.isArray(t.shard_ids) && t.shard_ids.length) {
    html += `<div class="muted">空分片: ${t.shard_ids.map(C.esc).join(', ')}</div>`;
  }
  if (Array.isArray(t.empty_partitions) && t.empty_partitions.length) {
    html += `<div class="muted">空 reduce 分区: ${t.empty_partitions.join(', ')}</div>`;
  }
  if (t.requested !== undefined && t.effective !== undefined) {
    html += `<div class="muted">请求 ${t.requested} → 实际 ${t.effective}</div>`;
  }
  html += '</div>';
  return html === '<div class="preflight-target small"></div>' ? '' : html;
}

// ---------------------------------------------------------------------------
async function loadRecent() {
  const d = await API.get('/api/jobs');
  const jobs = d.jobs || [];
  document.getElementById('recent').innerHTML = jobs.length
    ? C.table([
        { key: 'name', label: '作业 Job' },
        { key: 'status', label: '状态 Status', render: r => C.stateBadge(r.status, true) },
        { key: 'mapper', label: 'Mapper' },
        { key: 'reducer', label: 'Reducer' },
        { key: 'created_ms', label: '时间 Time', render: r => C.fmtTime(r.created_ms) },
        { key: 'link', label: '', render: r => `<a href="monitor.html">监控→</a>` },
      ], jobs)
    : C.empty();
}

init();
C.poll(loadRecent, 4000).start();
