/* 作业提交 Submit */
Components.init('submit');
const C = Components;

let SAMPLES = [];
let PREFLIGHT_OK = false;   // submit stays disabled until a clean preflight
let LAST_REPORT = null;

const GROUP_TITLES = {
  params: '参数合法性 Parameters',
  data: '输入数据 Input data',
  shape: '形状匹配 / 试运行 Shape & canary trial',
  sharding: '切分可行性 Sharding',
  dependencies: '前置作业 Dependencies',
  cluster: '集群就绪 Cluster',
  storage: '存储 Storage',
};
const GROUP_ORDER = ['params', 'data', 'shape', 'sharding', 'dependencies', 'cluster', 'storage'];
const SEV_ICON = { error: '✕', warning: '⚠', info: 'ℹ' };

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
  });

  document.getElementById('mapper').addEventListener('change', invalidatePreflight);
  document.getElementById('reducer').addEventListener('change', invalidatePreflight);
  ['num_map_tasks', 'num_reduce_tasks', 'input_rows', 'pattern', 'depends_on',
   'simulate_failure', 'name'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', invalidatePreflight);
    el.addEventListener('change', invalidatePreflight);
  });
  document.getElementById('preflight-btn').addEventListener('click', runPreflight);
  document.getElementById('form').addEventListener('submit', onSubmit);

  document.getElementById('func-list').innerHTML =
    '<h3>Map</h3>' + funcs.mappers.map(f =>
      `<div class="small" style="padding:2px 0"><span class="mono">${C.esc(f.name)}</span> — ${C.esc(f.description)}</div>`).join('') +
    '<h3 class="mt">Reduce</h3>' + funcs.reducers.map(f =>
      `<div class="small" style="padding:2px 0"><span class="mono">${C.esc(f.name)}</span> — ${C.esc(f.description)}</div>`).join('');

  loadRecent();
  togglePatternRow();
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
  document.getElementById('simulate_failure').checked = false;
  document.getElementById('pattern').value = (s.params && s.params.pattern) || 'map';
  togglePatternRow();
  invalidatePreflight();
}

function togglePatternRow() {
  const isGrep = document.getElementById('mapper').value === 'grep_mapper';
  document.getElementById('pattern-row').hidden = !isGrep;
}

function invalidatePreflight() {
  // Any edit after a clean report means the report no longer describes the form.
  PREFLIGHT_OK = false;
  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  btn.textContent = '提交作业 Submit Job';
  togglePatternRow();
}

function collectBody() {
  const body = {
    name: document.getElementById('name').value.trim(),
    mapper: document.getElementById('mapper').value,
    reducer: document.getElementById('reducer').value,
    num_map_tasks: parseInt(document.getElementById('num_map_tasks').value, 10),
    num_reduce_tasks: parseInt(document.getElementById('num_reduce_tasks').value, 10),
    input_rows: parseInt(document.getElementById('input_rows').value, 10),
    params: {},
    depends_on: Array.from(document.getElementById('depends_on').selectedOptions).map(o => o.value),
  };
  if (document.getElementById('mapper').value === 'grep_mapper') {
    const p = document.getElementById('pattern').value.trim();
    if (p) body.params.pattern = p;
  }
  if (document.getElementById('simulate_failure').checked) body.params.simulate_failure = true;
  return body;
}

async function runPreflight() {
  const btn = document.getElementById('preflight-btn');
  btn.disabled = true;
  btn.textContent = '预检中… Running…';
  try {
    const report = await API.post('/api/jobs/preflight', collectBody());
    LAST_REPORT = report;
    renderReport(report);
    PREFLIGHT_OK = report.ok;
    const submitBtn = document.getElementById('submit-btn');
    submitBtn.disabled = !report.ok;
    submitBtn.textContent = report.ok
      ? '提交作业 Submit Job'
      : `预检未通过（${report.error_count} 项错误）Fix errors first`;
    C.toast(report.ok
      ? `预检通过 Preflight passed${report.warning_count ? `，${report.warning_count} 项警告` : ''}`
      : `预检未通过：${report.error_count} 项错误 / ${report.warning_count} 项警告`,
      report.ok ? 'ok' : 'error');
  } catch (e) {
    C.toast('预检请求失败 ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '预检 / 试运行 Preflight';
  }
}

function renderReport(report) {
  document.getElementById('preflight-card').hidden = false;
  const p = report.plan_preview || {};
  const verdictCls = report.ok ? 'pf-ok' : 'pf-bad';
  const verdict = report.ok
    ? (report.warning_count
        ? `预检通过，但有 ${report.warning_count} 项警告 — 可以提交 / passed with warnings`
        : '预检通过，可以提交 / passed — safe to submit')
    : `预检未通过：${report.error_count} 项错误必须修复，${report.warning_count} 项警告 / blocked`;
  document.getElementById('preflight-summary').innerHTML =
    `<div class="pf-verdict ${verdictCls}">
       <span class="pf-verdict-icon">${report.ok ? '✓' : '✕'}</span>
       <span>${C.esc(verdict)}</span>
     </div>
     <div class="pf-preview small muted">
       ${p.input_rows != null ? `数据 dataset：<b class="tabular">${C.fmtNum(p.input_rows)}</b> 条 · ` : ''}
       ${p.num_map_tasks != null ? `<b class="tabular">${p.num_map_tasks}</b> map × <b class="tabular">${p.num_reduce_tasks}</b> reduce · ` : ''}
       ${p.input_kind ? `类型 <span class="mono">${C.esc(p.input_kind)}</span>` : ''}
     </div>`;

  const groups = report.groups || {};
  const html = GROUP_ORDER.filter(g => groups[g] && groups[g].length).map(g => {
    const items = groups[g].map(f => `
      <li class="pf-item pf-${f.severity}">
        <span class="pf-icon">${SEV_ICON[f.severity] || '•'}</span>
        <div class="pf-body">
          <div class="pf-msg">${C.esc(f.message)}</div>
          <div class="pf-loc small muted">
            ${f.field ? `<span class="pf-tag">字段 field: <code>${C.esc(f.field)}</code></span>` : ''}
            ${f.target ? `<span class="pf-tag">定位 target: <code>${C.esc(f.target)}</code></span>` : ''}
            ${f.suggestion ? `<span class="pf-suggest">建议: ${C.esc(f.suggestion)}</span>` : ''}
          </div>
        </div>
      </li>`).join('');
    return `<details class="accordion pf-group" ${groups[g].some(x => x.severity === 'error') ? 'open' : ''}>
      <summary>${C.esc(GROUP_TITLES[g] || g)}
        <span class="pf-counts">${groupCounts(groups[g])}</span>
      </summary>
      <div class="body"><ul class="pf-list">${items}</ul></div>
    </details>`;
  }).join('');
  document.getElementById('preflight-report').innerHTML = html || C.empty();
}

function groupCounts(items) {
  const n = sev => items.filter(x => x.severity === sev).length;
  const parts = [];
  if (n('error')) parts.push(`<span class="pf-badge pf-bad">${n('error')} 错误</span>`);
  if (n('warning')) parts.push(`<span class="pf-badge pf-warn">${n('warning')} 警告</span>`);
  if (n('info')) parts.push(`<span class="pf-badge pf-info">${n('info')} 信息</span>`);
  return parts.join(' ');
}

async function onSubmit(ev) {
  ev.preventDefault();
  // Server re-runs the preflight regardless; the gate here avoids a pointless
  // round trip and makes the workflow explicit in the UI.
  if (!PREFLIGHT_OK) {
    C.toast('请先通过预检 Run preflight first', 'error');
    return;
  }
  const btn = ev.target.querySelector('#submit-btn');
  btn.disabled = true;
  try {
    const job = await API.post('/api/jobs', collectBody());
    C.toast('作业已提交 Job submitted: ' + job.job_id, 'ok');
    setTimeout(() => location.href = 'monitor.html', 600);
  } catch (e) {
    C.toast('提交失败 ' + e.message, 'error');
    btn.disabled = false;
  }
}

async function loadRecent() {
  const d = await API.get('/api/jobs');
  const jobs = d.jobs || [];
  // Rebuild the dependency options only when the set of jobs changed, so a
  // 4s poll never wipes the user's multi-selection mid-form.
  const dep = document.getElementById('depends_on');
  const sig = jobs.map(j => j.job_id + ':' + j.status).join('|');
  if (dep && dep.dataset.sig !== sig) {
    const selected = new Set(Array.from(dep.selectedOptions).map(o => o.value));
    dep.innerHTML = jobs
      .map(j => `<option value="${j.job_id}"${selected.has(j.job_id) ? ' selected' : ''}>` +
                `${C.esc(j.name)} — ${C.esc(C.LABELS[j.status] || j.status)} (${j.job_id})</option>`)
      .join('');
    dep.dataset.sig = sig;
  }
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
