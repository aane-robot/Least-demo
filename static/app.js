/* ============================================================
   LEAST Demo 前端 —— 零外部依赖（自绘 SVG 图表 + 力导向布局）
   ============================================================ */
const S = { jobId: null, timer: null, result: null, src: 'syn', view: 'conv',
            csvText: null, layouts: {}, hoverNode: null };

/* ---------- 小工具 ---------- */
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const num = v => (v === null || v === undefined) ? '--' : (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(4).replace(/0+$/, '').replace(/\.$/, ''));

function switchSrc(t) {
  S.src = t;
  document.querySelectorAll('[data-tab]').forEach(e => {
    if (e.parentElement.classList.contains('tabs') && e.dataset.tab)
      e.classList.toggle('active', e.dataset.tab === t);
  });
  ['syn', 'bi', 'up'].forEach(k => $('pane-' + k).classList.toggle('active', k === t));
}
function switchView(v) {
  S.view = v;
  document.querySelectorAll('[data-vt]').forEach(e => e.classList.toggle('active', e.dataset.vt === v));
  ['conv', 'dag', 'mat', 'edges', 'rca', 'log'].forEach(k => $('view-' + k).classList.toggle('active', k === v));
  if (v === 'dag') renderDag();
  if (v === 'mat') renderMatrix();
  if (v === 'edges') renderEdges();
  if (v === 'rca') initRca();
}
function toggleHelp() { $('help').classList.toggle('open'); }
function openBench() { $('bench').classList.add('open'); pollBench(); }
function exportCsv() {
  if (!S.jobId) { alert('请先运行一次 LEAST'); return; }
  window.location.href = '/api/download/' + S.jobId;
}
function closeBench() { $('bench').classList.remove('open'); }
function onDatasetChange() {
  const d = $('p_dataset').value;
  $('bi_desc').textContent =
    d === 'flight' ? '论文 §VI-A 场景：14 个监控指标，人工构造的故障传播 DAG 作为 ground truth。'
    : d === 'chambers' ? '真实物理装置：ETH 光隧道（10,000×20，真值 39 边由装置物理直接确定，'
      + '非专家共识）。关系非线性，F1 供横向对比；d≤120 因此会同时跑 NOTEARS 对照。'
    : d === 'sachs' ? '真实数据：Sachs 蛋白信号网络（流式细胞术，853×11，真值 19 边，已 log 变换）。'
      + '真值为文献共识图，存在争议；此处 F1 仅作横向对比（LEAST≈0.32 / NOTEARS≈0.28，同管线）。'
    : '论文 §VI-C 场景：32 部电影 / 4 个潜在类型，每列一部电影、每行一个用户（已去均值）。';
}

/* ---------- 文件格式识别（CSV / BIF / DSC / NET） ---------- */
function detectFmt(text, name) {
  const low = String(text || '').slice(0, 4000).toLowerCase();
  const ext = String(name || '').split('.').pop().toLowerCase();
  const hasCpt = low.includes('probability') || low.includes('potential');
  const hasNode = low.includes('type discrete') || low.includes('states =')
    || low.includes('variable') || low.includes('node ');
  if (low.includes('belief network') || (low.includes('node ') && low.includes('type :'))) return 'dsc';
  if (low.includes('potential') && low.includes('states =')) return 'net';
  if (['csv', 'txt', 'tsv', 'dat'].includes(ext)) return (hasCpt && hasNode && low.includes('{')) ? 'bif' : 'csv';
  if (hasCpt && hasNode) return 'bif';
  if (['bif', 'dsc', 'net'].includes(ext)) return ext;
  return 'csv';
}
function onFileLoaded(text) {
  S.fileFmt = detectFmt(text, S.fileName);
  const note = $('up_note');
  if (S.fileFmt === 'csv') {
    note.innerHTML = '上传 <b>CSV 数据表</b>：无 ground truth，因此不计算 F1/P/R/SHD，只输出学到的因果图、收敛轨迹与根因路径。';
  } else {
    const nm = { bif: 'BIF', dsc: 'DSC', net: 'NET (Hugin)' }[S.fileFmt];
    note.innerHTML = '检测到 <b>' + nm + '</b> 贝叶斯网络定义文件：将按 CPT 前向采样 '
      + '<b>' + ($('p_net_n').value || 2000) + '</b> 条数据（离散状态编码为 0..k-1），'
      + '并把网络自身的边表作为 <b>ground truth</b>，因此可计算 F1 / P / R / SHD。'
      + '<br><span style="color:#a15c00">注意：这类网络是离散的，而 LEAST/NOTEARS 等线性 SEM 方法'
      + '假设连续数据，F1 通常只有 0.2–0.5（实测 alarm：LEAST 0.39 / NOTEARS 0.48；同一份代码在'
      + '连续数据上 F1≈0.8–0.9）。这是数据-模型失配的上限，不是算法问题；结果用于横向对比而非'
      + '绝对精度。建议：λ 改 0.1、τ 调小观察。</span>';
  }
}

/* ---------- 文件选择 ---------- */
document.addEventListener('DOMContentLoaded', () => {
  const f = $('file');
  f.addEventListener('change', () => {
    const file = f.files[0]; if (!file) return;
    $('fname').textContent = file.name + ' (' + file.size + ' B)';
    S.fileName = file.name;
    const r = new FileReader();
    r.onload = e => { S.csvText = e.target.result; onFileLoaded(e.target.result); };
    r.readAsText(file);
  });
  const drop = $('drop');
  ['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.style.borderColor = '#2980b9'; }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.style.borderColor = ''; }));
  drop.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0]; if (!file) return;
    $('fname').textContent = file.name;
    S.fileName = file.name;
    const r = new FileReader(); r.onload = ev => { S.csvText = ev.target.result; onFileLoaded(ev.target.result); }; r.readAsText(file);
  });
  const nn = $('p_net_n');
  if (nn) nn.addEventListener('change', () => { if (S.csvText) onFileLoaded(S.csvText); });
});

/* ---------- 参数收集 ---------- */
function collect() {
  const p = {
    mode: S.src === 'syn' ? 'synthetic' : (S.src === 'bi' ? 'builtin' : 'upload'),
    k: +$('p_k').value, alpha: +$('p_alpha').value,
    zeta: +$('p_zeta').value, lam: +$('p_lam').value, theta: +$('p_theta').value,
    eps: +$('p_eps').value, B: +$('p_B').value,
    T_o: +$('p_To').value, T_i: +$('p_Ti').value, lr: +$('p_lr').value,
    termination: $('p_term').value, warm_start: $('p_warm').checked,
    max_seconds: +$('p_maxsec').value, compare_notears: $('p_nt').checked,
    eps_grid: $('p_epsgrid').checked,
    engine: $('p_engine').value, zeta_sp: +$('p_zeta_sp').value,
    theta_sp: +$('p_theta_sp').value, sparse_threshold: +$('p_spthr').value,
  };
  p.taus = $('p_taus').value.split(',').map(s => +s.trim()).filter(v => !isNaN(v));
  const bs = +$('p_batch').value; if (bs > 0) p.batch_size = bs;
  if (S.src === 'syn') {
    p.d = +$('p_d').value; p.n = +$('p_n').value;
    p.graph_type = $('p_graph').value; p.noise_type = $('p_noise').value;
    p.degree = +$('p_degree').value; p.seed = +$('p_seed').value;
  } else if (S.src === 'bi') {
    p.dataset = $('p_dataset').value; p.n = +$('p_bi_n').value;
    p.seed = +$('p_bi_seed').value; p.anomaly = $('p_anomaly').checked;
  } else {
    if (!S.csvText) { alert('请先选择数据文件（CSV / BIF / DSC / NET）'); return null; }
    p.csv_content = S.csvText; p.has_header = $('p_header').checked;
    p.standardize = $('p_std').checked; p.tau_upload = +$('p_tau_up').value;
    p.file_name = S.fileName || '';
    p.file_format = S.fileFmt || (S.csvText ? detectFmt(S.csvText, S.fileName) : 'csv');
    p.sample_n = +$('p_net_n').value;
    p.seed = 42;
  }
  return p;
}

/* ---------- 运行 ---------- */
function runJob() {
  const p = collect(); if (!p) return;
  $('btnRun').disabled = true;
  $('btnRun').innerHTML = '<span class="spin"></span>运行中…';
  $('prog').style.width = '0%'; $('progtext').textContent = '提交任务…';
  $('metrics').innerHTML = '';
  ['ch1', 'ch2', 'mat', 'svg_true', 'svg_pred', 'edgetbl', 'edgenote', 'rca_out'].forEach(id => $(id).innerHTML = '');
  fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p) })
    .then(r => r.json()).then(d => {
      S.jobId = d.job_id; S.result = null;
      clearInterval(S.timer);
      S.timer = setInterval(poll, 700);
    }).catch(e => { alert('提交失败: ' + e); resetBtn(); });
}
function resetBtn() { $('btnRun').disabled = false; $('btnRun').textContent = '运行 LEAST'; clearInterval(S.timer); }

function poll() {
  fetch('/api/status/' + S.jobId).then(r => r.json()).then(st => {
    $('prog').style.width = (st.progress * 100).toFixed(0) + '%';
    $('progtext').textContent = st.status === 'running'
      ? '运行中 · 已完成 ' + (st.progress * 100).toFixed(0) + '%' + (st.history && st.history.length ? ' · 外层 ' + st.history[st.history.length - 1].outer : '')
      : (st.status === 'done' ? '完成' : '出错');
    if (st.log) $('logbox').textContent = st.log.join('\n');
    if (st.status === 'running' && st.history && st.history.length > 1) drawConv(st.history, true);
    if (st.status === 'done') {
      clearInterval(S.timer);
      fetch('/api/result/' + S.jobId).then(r => r.json()).then(res => {
        S.result = res; S.layouts = {};
        renderAll(); resetBtn();
      });
    } else if (st.status === 'error') {
      clearInterval(S.timer);
      $('progtext').innerHTML = '<span style="color:#c0392b">错误: ' + esc(st.error) + '</span>';
      $('logbox').textContent = (st.log || []).join('\n') + '\n' + (st.traceback || '');
      resetBtn(); switchView('log');
    }
  });
}

/* ---------- 总渲染 ---------- */
function renderAll() {
  const r = S.result; if (!r) return;
  renderMetrics(); drawConv(r.history, false); renderDag(); renderMatrix();
  renderEdges(); initRca();
  $('logbox').textContent = (r.log || []).join('\n');
  switchView(S.view);
}

function renderMetrics() {
  const r = S.result, m = r.metrics, meta = r.meta;
  const cards = [];
  const hasGT = meta.has_gt;
  const f1c = m.f1 === null ? '' : (m.f1 >= .8 ? 'good' : m.f1 >= .6 ? 'warn' : 'bad');
  const NA = '<span title="上传数据没有真实因果图（ground truth），无法计算预测准确率；F1/SHD 仅在合成数据和内置示例（自带真实图）时提供">—</span>';
  cards.push(['F1-score', hasGT ? m.f1.toFixed(4) : NA, f1c]);
  cards.push(['Precision', hasGT ? m.precision.toFixed(4) : NA, '']);
  cards.push(['Recall', hasGT ? m.recall.toFixed(4) : NA, '']);
  cards.push(['SHD', hasGT ? m.shd : NA, '']);
  const banner = hasGT ? '' :
    '<div style="grid-column:1/-1;font-size:11.5px;color:#8a6d3b;background:#fdf6e3;border:1px solid #f0e0b0;border-radius:6px;padding:6px 10px">上传数据没有真实因果图，F1 / Precision / Recall / SHD 不可计算（显示"—"是正常现象，不是运行失败）。学到的边见"因果图 / 边表 / 根因分析"标签页；图的质量可参考 h(W)（越小越接近无环）和你的领域知识。</div>';
  cards.push(['δ(W)', m.delta.toExponential(2), '']);
  cards.push(['h(W)', m.h.toExponential(2), '']);
  cards.push(['收敛', m.converged ? '是' : '否', m.converged ? 'good' : 'bad']);
  cards.push(['外层迭代', m.outer, '']);
  if (m.stop_reason)
    cards.push(['终止原因',
                `<span style="font-size:11px;line-height:1.3" title="${esc(m.stop_why || '')}">${esc(m.stop_reason)}</span>`, '']);
  cards.push(['预测边数', meta.num_pred_edges, '']);
  cards.push(['阈值 τ', m.tau, '']);
  cards.push(['运行时间', meta.algo_time.toFixed(1) + 's', '']);
  if (meta.algo_time_grid && meta.algo_runs > 1)
    cards.push(['LEAST 网格总耗时',
                `<span style="font-size:12px" title="为取得 best case 共运行 ${meta.algo_runs} 个 ε 配置的总耗时（与 NOTEARS λ 网格总耗时口径对称）">${meta.algo_time_grid.toFixed(1)}s (${meta.algo_runs}次)</span>`, '']);
  cards.push(['是 DAG', m.is_dag ? '是' : '否', m.is_dag ? 'good' : 'warn']);
  if (m.spec_r !== undefined && m.spec_r !== null)
    cards.push(['谱半径 ρ(S)',
      `<span style="font-size:12px" title="S=W∘W（阈值 τ 后的图）的真实谱半径，特征值/幂迭代直算。≈0 ⟺ 学到的加权图无环——这是比 δ 更直接的无环证据：δ 只是上界（k=5 时约松 20 倍且对稠密小权重有 ~1e-1 地板），所以 δ 停在 1e-1~1e0 不代表有环。未阈值化原始 W 的 ρ=${m.spec_r_raw === null || m.spec_r_raw === undefined ? '—' : m.spec_r_raw.toExponential(1)}（残留的亚阈值弱环，阈值化后即消失）">${m.spec_r < 1e-10 ? '≈0' : m.spec_r.toExponential(2)}</span>`,
      m.spec_r < 1e-8 ? 'good' : 'warn']);
  if (r.notears && r.notears.skipped) {
    cards.push(['NOTEARS F1', `<span style="font-size:11px" title="${esc(r.notears.skip_reason || '')}">未运行</span>`, 'warn']);
    cards.push(['NOTEARS 说明', `<span style="font-size:11px;line-height:1.3" title="${esc(r.notears.skip_reason || '')}">d>120 已跳过</span>`, 'warn']);
  } else if (r.notears) {
    cards.push(['NOTEARS F1', r.notears.f1 === null ? 'N/A' : r.notears.f1.toFixed(4), '']);
    cards.push(['NOTEARS 网格总耗时',
                `<span style="font-size:12px" title="NOTEARS λ∈{0.1,0.5} 网格的总耗时（含未中选配置）">${r.notears.time.toFixed(1)}s</span>`, '']);
    const gridBase = (meta.algo_time_grid && meta.algo_time_grid > 0) ? meta.algo_time_grid : meta.algo_time;
    cards.push(['加速比(网格口径)',
                `<span style="font-size:12px" title="NOTEARS λ网格总耗时 ÷ LEAST ε网格总耗时，双方同为 best-case-over-grid 的对称比较">${(r.notears.time / Math.max(gridBase, 1e-9)).toFixed(1)}×</span>`, 'good']);
    if (r.notears.time_single)
      cards.push(['加速比(单次)',
                  `<span style="font-size:12px" title="双方各自最佳配置的单次耗时之比（参考口径）">${(r.notears.time_single / Math.max(meta.algo_time, 1e-9)).toFixed(1)}×</span>`, '']);
  }
  $('metrics').innerHTML = banner + cards.map(c =>
    `<div class="metric"><div class="v ${c[2]}">${c[1]}</div><div class="k">${esc(c[0])}</div></div>`).join('');
}

/* ---------- 收敛曲线（自绘 SVG） ---------- */
function drawConv(hist, live) {
  if (!hist || !hist.length) return;
  const outer = hist.map(h => h.outer);
  const box1 = $('ch1'), box2 = $('ch2');
  const W = box1.clientWidth || 520, H = 230, pad = { l: 52, r: 16, t: 14, b: 30 };
  const series = [{ k: 'delta', c: '#2980b9', n: 'δ(W) 上界' }, { k: 'h', c: '#c0392b', n: 'h(W) 精确' }];
  let vals = [];
  series.forEach(s => hist.forEach(h => { const v = h[s.k]; if (v > 0) vals.push(v); }));
  if (!vals.length) vals = [1e-8, 1];
  let lo = Math.min(...vals), hi = Math.max(...vals);
  lo = Math.max(lo, 1e-12); hi = Math.max(hi, lo * 10);
  const l0 = Math.log10(lo), l1 = Math.log10(hi);
  const X = i => pad.l + (outer.length < 2 ? 0 : (i / (outer.length - 1)) * (W - pad.l - pad.r));
  const Y = v => { const t = (Math.log10(Math.max(v, 1e-12)) - l0) / (l1 - l0 || 1); return pad.t + (1 - t) * (H - pad.t - pad.b); };

  let g = `<svg viewBox="0 0 ${W} ${H}">`;
  for (let e = Math.floor(l0); e <= Math.ceil(l1); e++) {
    const y = Y(Math.pow(10, e));
    g += `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="#eef2f6"/>`;
    g += `<text x="${pad.l - 6}" y="${y + 3.5}" font-size="10" fill="#98a4b0" text-anchor="end">1e${e}</text>`;
  }
  series.forEach(s => {
    const d = hist.map((h, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(h[s.k]).toFixed(1)}`).join(' ');
    g += `<path d="${d}" fill="none" stroke="${s.c}" stroke-width="1.9"/>`;
  });
  g += `<line x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}" stroke="#d8dee6"/>`;
  g += `<text x="${W - pad.r}" y="${H - 8}" font-size="10" fill="#98a4b0" text-anchor="end">outer iteration</text>`;
  let lx = pad.l + 4;
  series.forEach(s => {
    g += `<rect x="${lx}" y="${pad.t}" width="9" height="3" fill="${s.c}"/>`;
    g += `<text x="${lx + 13}" y="${pad.t + 5}" font-size="10.5" fill="#5b6b7c">${s.n}</text>`; lx += 90;
  });
  g += `</svg>`;
  box1.innerHTML = g;

  // η / ρ
  const W2 = box2.clientWidth || 520;
  const rho = hist.map(h => h.rho), eta = hist.map(h => h.eta);
  const X2 = i => pad.l + (outer.length < 2 ? 0 : (i / (outer.length - 1)) * (W2 - pad.l - pad.r));
  const mx = Math.max(...eta, 1), mr = Math.max(...rho, 1);
  const Y2 = (v, m) => pad.t + (1 - v / m) * (H - pad.t - pad.b);
  let g2 = `<svg viewBox="0 0 ${W2} ${H}">`;
  [0, .25, .5, .75, 1].forEach(t => {
    const y = pad.t + (1 - t) * (H - pad.t - pad.b);
    g2 += `<line x1="${pad.l}" y1="${y}" x2="${W2 - pad.r}" y2="${y}" stroke="#eef2f6"/>`;
  });
  g2 += `<path d="${rho.map((v, i) => `${i ? 'L' : 'M'}${X2(i).toFixed(1)},${Y2(v, mr).toFixed(1)}`).join(' ')}" fill="none" stroke="#e67e22" stroke-width="1.9"/>`;
  g2 += `<path d="${eta.map((v, i) => `${i ? 'L' : 'M'}${X2(i).toFixed(1)},${Y2(v, mx).toFixed(1)}`).join(' ')}" fill="none" stroke="#27ae60" stroke-width="1.9"/>`;
  g2 += `<text x="${pad.l - 6}" y="${pad.t + 8}" font-size="10" fill="#98a4b0" text-anchor="end">${mx.toFixed(1)}</text>`;
  g2 += `<text x="${pad.l - 6}" y="${H - pad.b}" font-size="10" fill="#98a4b0" text-anchor="end">0</text>`;
  g2 += `<line x1="${pad.l}" y1="${H - pad.b}" x2="${W2 - pad.r}" y2="${H - pad.b}" stroke="#d8dee6"/>`;
  g2 += `<rect x="${pad.l + 4}" y="${pad.t}" width="9" height="3" fill="#e67e22"/><text x="${pad.l + 17}" y="${pad.t + 5}" font-size="10.5" fill="#5b6b7c">ρ (max ${mr.toFixed(2)})</text>`;
  g2 += `<rect x="${pad.l + 130}" y="${pad.t}" width="9" height="3" fill="#27ae60"/><text x="${pad.l + 143}" y="${pad.t + 5}" font-size="10.5" fill="#5b6b7c">η (max ${mx.toFixed(2)})</text>`;
  g2 += `</svg>`;
  box2.innerHTML = g2;

  if (!live && S.result) {
    const last = hist[hist.length - 1];
    $('conv_note').innerHTML = `共 ${hist.length} 次外层迭代；终值 δ=${last.delta.toExponential(2)}，h=${last.h.toExponential(2)}。`
      + ` 绿线 h(W) 是 NOTEARS 的精确无环度量，蓝线 δ(W) 是 LEAST 的谱半径上界 —— 两者同步下降即验证了论文 §III 的 R1（一致性）要求。`
      + `<br><b>为什么 δ 停在 1e0 附近而不降到 1e-4？</b>δ 是 <b>k 步截断</b>的谱半径上界，天然偏松：`
      + `实测连"严格上三角的真 DAG"都会给出 δ≈6.8（真实谱半径为 0），一般比 ρ(W∘W) 大约 20 倍。`
      + `因此若只用 δ≤ε 作唯一判据，实测要跑满 300 次外层也只降到 8e-2，同时 ALM 的 η 从 1 涨到 119、`
      + `把 W 过度压缩，F1 由 0.92 掉到 0.53。论文 §V-A 为与 NOTEARS 对齐，实际用 <b>h(W)≤ε</b> 判停（默认"delta_or_h"），`
      + `所以这里看到 h 降到 1e-4 触发终止、而 δ 仍在 1e0 —— 这是上界的性质，不是实现偏差。`;
  }
}

/* ---------- 力导向 DAG（自绘 SVG，语义缩放：放大时节点视觉大小不变） ---------- */
function layout(nodes, edges, seedKey) {
  if (S.layouts[seedKey]) return S.layouts[seedKey];
  const d = nodes.length;
  const rng = mulberry(12345);
  // 初始位置：网格 + 抖动（纯随机初始会导致局部纠缠，归一化后成团）
  const cols = Math.ceil(Math.sqrt(d * 1.6)), rows = Math.ceil(d / cols);
  const P = nodes.map((_, i) => ({
    x: ((i % cols) + 0.5) / cols * 800 + (rng() - .5) * 50,
    y: (Math.floor(i / cols) + 0.5) / rows * 800 + (rng() - .5) * 50,
    vx: 0, vy: 0
  }));
  const REP = 26000 * Math.max(1, Math.pow(d / 60, 1.6)); // 全局斥力
  const REST = d > 200 ? 46 : d > 80 ? 62 : 95;           // 弹簧自然长度
  const ITER = Math.min(d > 250 ? 420 : 700, 350 + Math.round(2.5 * d));
  for (let it = 0; it < ITER; it++) {
    const cool = 1 - it / ITER;
    const step = 30 * cool + 2;
    for (let i = 0; i < d; i++) {
      let fx = 0, fy = 0;
      const xi = P[i].x, yi = P[i].y;
      for (let j = 0; j < d; j++) {
        if (i === j) continue;
        let dx = xi - P[j].x, dy = yi - P[j].y;
        let r2 = dx * dx + dy * dy;
        if (r2 < 1) { dx = rng() - .5; dy = rng() - .5; r2 = 1; }
        const r = Math.sqrt(r2);
        let f = REP / r2;
        if (r2 < 900) f += 9e5 / r2;    // 近距（<30px）强斥力，防重叠
        fx += dx / r * f; fy += dy / r * f;
      }
      P[i].vx = (P[i].vx + fx) * .8; P[i].vy = (P[i].vy + fy) * .8;
      const v = Math.hypot(P[i].vx, P[i].vy) || 1;
      const mv = Math.min(v, step);
      P[i].x += P[i].vx / v * mv; P[i].y += P[i].vy / v * mv;
    }
    edges.forEach(e => {
      const a = P[e.s], b = P[e.t];
      const dx = b.x - a.x, dy = b.y - a.y, r = Math.hypot(dx, dy) || 1;
      const f = (r - REST) * 0.06;
      const ux = dx / r * f, uy = dy / r * f;
      a.x += ux; a.y += uy; b.x -= ux; b.y -= uy;
    });
    // 中心引力
    let cx = 0, cy = 0; P.forEach(p => { cx += p.x; cy += p.y; }); cx /= d; cy /= d;
    P.forEach(p => { p.x += (400 - cx) * 0.02; p.y += (400 - cy) * 0.02; });
  }
  // 归一化到画布
  let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
  P.forEach(p => { minx = Math.min(minx, p.x); maxx = Math.max(maxx, p.x); miny = Math.min(miny, p.y); maxy = Math.max(maxy, p.y); });
  const W = GRAPH_W(), H = GRAPH_H();
  const sx = (W - 50) / Math.max(maxx - minx, 1), sy = (H - 42) / Math.max(maxy - miny, 1);
  const s = Math.min(sx, sy);
  P.forEach(p => { p.x = 25 + (p.x - minx) * s; p.y = 21 + (p.y - miny) * s; });
  // 关键：归一化**之后**在画布（像素）坐标系做防重叠 —— C 按节点直径取，直接保证最终视觉无重叠
  // （若在归一化前做，world 坐标的间距会被缩放系数压回去，前功尽弃）
  const C = 2 * nodeRadius(d) + 3, C2 = C * C;
  for (let pass = 0; pass < 60; pass++) {
    let conflict = false;
    for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) {
      const dx = P[j].x - P[i].x, dy = P[j].y - P[i].y;
      const r2 = dx * dx + dy * dy;
      if (r2 > C2) continue;
      conflict = true;
      if (r2 < 1e-9) { P[j].x += C; continue; }
      const r = Math.sqrt(r2), push = (C - r) / 2 / r;
      P[i].x -= dx * push; P[i].y -= dy * push;
      P[j].x += dx * push; P[j].y += dy * push;
    }
    if (!conflict) break;
  }
  P.forEach(p => {
    p.x = Math.max(12, Math.min(W - 12, p.x));
    p.y = Math.max(10, Math.min(H - 10, p.y));
  });
  S.layouts[seedKey] = P;
  return P;
}
function nodeRadius(d) { return d <= 30 ? 11 : d <= 60 ? 8.5 : d <= 120 ? 6 : d <= 250 ? 4.5 : 3.5; }
function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

function GRAPH_W() { const el = document.getElementById('svg_pred'); return Math.max(420, (el ? el.clientWidth : 520) - 8); }
function GRAPH_H() { return 520; }

function drawGraph(container, nodes, edges, fnEdges, seedKey, clickable) {
  if (!nodes || !nodes.length) { container.innerHTML = '<div class="empty">无数据</div>'; return; }
  const P = layout(nodes, edges.concat(fnEdges || []), seedKey);
  const d = nodes.length;
  const W = GRAPH_W(), H = GRAPH_H();
  const COL = { tp: '#27ae60', fp: '#e74c3c', rev: '#8e44ad', pred: '#2980b9', true: '#27ae60', fn: '#9aa5b1' };
  const deg = new Array(d).fill(0); edges.forEach(e => { deg[e.s]++; deg[e.t]++; });
  // 节点/字号随规模分档缩小（d=100 时必须用小节点，否则必然重叠）
  const R = d <= 30 ? 11 : d <= 60 ? 8.5 : d <= 120 ? 6 : d <= 250 ? 4.5 : 3.5;
  const FS = d <= 30 ? 10 : d <= 60 ? 9 : d <= 120 ? 7.5 : 6.5;
  const RMAX = R + (d > 120 ? 2.5 : 8);
  const EW = d > 120 ? 1.0 : 1.5;    // 基准线宽
  const geo = { P, nodes, edges, fnEdges, seedKey, d, W, H, R, FS, RMAX, EW, deg, clickable, COL };
  container.innerHTML =
    `<svg class="graph" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="touch-action:none;cursor:grab">
       <defs>
         <marker id="ar-${seedKey}" markerWidth="9" markerHeight="9" refX="8.4" refY="3" orient="auto" markerUnits="strokeWidth">
           <path d="M0,0 L0,6 L8,3 z" fill="#7d8b99"/></marker>
         <marker id="arf-${seedKey}" markerWidth="9" markerHeight="9" refX="8.4" refY="3" orient="auto" markerUnits="strokeWidth">
           <path d="M0,0 L0,6 L8,3 z" fill="#c8cfd6"/></marker>
       </defs>
       <g class="zl"></g>
     </svg><div class="graph-hint">滚轮缩放 · 拖拽平移 · 双击复位 · 悬停节点高亮其因果边</div>`;
  attachZoomPan(container.querySelector('svg'), geo);
}

/* 生成某一缩放级别下的图内容：所有尺寸乘 1/k（语义缩放 —— 放大后节点视觉大小不变，只有间隙变大） */
function graphInner(geo, k) {
  const inv = 1 / k;
  const { P, nodes, edges, fnEdges, seedKey, R, FS, RMAX, EW, deg, COL } = geo;
  let g = '';
  (fnEdges || []).forEach(e => {
    const a = P[e.s], b = P[e.t];
    g += `<line class="el" data-s="${e.s}" data-t="${e.t}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"
           stroke="${COL.fn}" stroke-width="${(EW * inv).toFixed(2)}" stroke-dasharray="4 4" opacity=".7"
           marker-end="url(#arf-${seedKey})"/>`;
  });
  edges.forEach(e => {
    const a = P[e.s], b = P[e.t];
    const c = COL[e.type] || COL.pred;
    const dx = b.x - a.x, dy = b.y - a.y, r = Math.hypot(dx, dy) || 1;
    const ux = dx / r, uy = dy / r;
    const rs = (R + 1) * inv, re = (R + 3) * inv;   // 端点缩进也按 1/k，视觉恒定
    const x1 = a.x + ux * rs, y1 = a.y + uy * rs, x2 = b.x - ux * re, y2 = b.y - uy * re;
    const off = 10 * inv;
    const mx = (x1 + x2) / 2 - uy * off, my = (y1 + y2) / 2 + ux * off;
    const w = Math.min(3.2, 0.8 + Math.abs(e.w) * 1.1) * inv;
    g += `<path class="el" data-s="${e.s}" data-t="${e.t}" d="M${x1.toFixed(1)},${y1.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}"
           fill="none" stroke="${c}" stroke-width="${w.toFixed(2)}" opacity=".8" marker-end="url(#ar-${seedKey})"/>`;
  });
  nodes.forEach((n, i) => {
    const p = P[i], r = (R + Math.min(RMAX - R, deg[i] * 0.8)) * inv;
    g += `<circle class="nd" data-i="${i}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${r.toFixed(2)}" fill="#3498db" stroke="#1a5490" stroke-width="${(1.2 * inv).toFixed(2)}"/>`;
    const showLabel = geo.d <= 40 || (geo.d <= 100 && deg[i] >= 3) || (geo.d <= 250 && deg[i] >= 5) || deg[i] >= 8;
    if (showLabel)
      g += `<text class="lb" data-i="${i}" x="${p.x.toFixed(1)}" y="${(p.y + r + FS * inv + 1).toFixed(1)}" font-size="${(FS * inv).toFixed(2)}" fill="#42596e"
             text-anchor="middle" style="pointer-events:none">${esc(n.label.length > 14 ? n.label.slice(0, 13) + '…' : n.label)}</text>`;
  });
  return g;
}

/* 语义缩放 + 平移 + 悬停高亮（事件委托，重渲染不丢监听） */
function attachZoomPan(svg, geo) {
  if (!svg) return;
  const g = svg.querySelector('g.zl');
  const Z = { k: 1, x: 0, y: 0 };
  const apply = () => {
    g.setAttribute('transform', `translate(${Z.x.toFixed(2)},${Z.y.toFixed(2)}) scale(${Z.k.toFixed(3)})`);
    g.innerHTML = graphInner(geo, Z.k);
    applyFocus();
  };
  const rootPt = ev => {
    const ctm = svg.getScreenCTM(); if (!ctm) return null;
    const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
    const q = p.matrixTransform(ctm.inverse());
    return { x: q.x, y: q.y };
  };
  svg.addEventListener('wheel', ev => {
    ev.preventDefault();
    const u = rootPt(ev); if (!u) return;
    const k2 = Math.min(40, Math.max(1, Z.k * (ev.deltaY > 0 ? 1 / 1.2 : 1.2)));
    const p = { x: (u.x - Z.x) / Z.k, y: (u.y - Z.y) / Z.k };  // 鼠标下的世界坐标（缩放锚点）
    Z.x = u.x - k2 * p.x; Z.y = u.y - k2 * p.y; Z.k = k2;
    if (Z.k === 1) { Z.x = 0; Z.y = 0; }
    apply();
  }, { passive: false });
  let drag = null;
  svg.addEventListener('pointerdown', ev => {
    if (ev.button !== 0) return;
    drag = { px: ev.clientX, py: ev.clientY };
    try { svg.setPointerCapture(ev.pointerId); } catch (e) {}
    svg.style.cursor = 'grabbing';
  });
  svg.addEventListener('pointermove', ev => {
    if (!drag) return;
    const ctm = svg.getScreenCTM(); if (!ctm) return;
    Z.x -= (ev.clientX - drag.px) / ctm.a;
    Z.y -= (ev.clientY - drag.py) / ctm.a;
    drag = { px: ev.clientX, py: ev.clientY };
    apply();
  });
  const end = () => { drag = null; svg.style.cursor = 'grab'; };
  svg.addEventListener('pointerup', end);
  svg.addEventListener('pointerleave', () => { end(); unfocus(svg); });
  svg.addEventListener('dblclick', () => { Z.k = 1; Z.x = 0; Z.y = 0; apply(); });

  /* 悬停高亮：与该节点相连的边 + 邻居保持醒目，其余淡出 */
  let cur = -1;
  function applyFocus() {
    if (cur < 0) return;
    setFocus(cur);
  }
  function setFocus(i) {
    const adj = new Set([i]);
    g.querySelectorAll('.el').forEach(el => {
      const s = +el.dataset.s, t = +el.dataset.t;
      const hit = (s === i || t === i);
      el.classList.toggle('hl', hit);
      if (hit) { adj.add(s); adj.add(t); }
    });
    g.querySelectorAll('.nd,.lb').forEach(el =>
      el.classList.toggle('hl', adj.has(+el.dataset.i)));
    svg.classList.add('focus');
  }
  function unfocusEl() {
    cur = -1;
    svg.classList.remove('focus');
    g.querySelectorAll('.hl').forEach(el => el.classList.remove('hl'));
  }
  svg.addEventListener('mouseover', ev => {
    const c = ev.target.closest('.nd');
    if (!c) return;
    const i = +c.dataset.i;
    if (i === cur) return;
    cur = i; setFocus(i);
    if (geo.clickable) showNodeInfo(i);
  });
  svg.addEventListener('mouseout', ev => {
    if (!ev.relatedTarget || !ev.relatedTarget.closest || !ev.relatedTarget.closest('.nd')) unfocusEl();
  });
  apply();   // 初始渲染（k=1）
}
function unfocus(svg) {
  svg.classList.remove('focus');
  svg.querySelectorAll('.hl').forEach(el => el.classList.remove('hl'));
}

function renderDag() {
  const r = S.result; if (!r) return;
  const nodes = r.graph.nodes;
  drawGraph($('svg_true'), nodes, (r.graph.true_edges || []).map(e => ({ ...e, type: 'true' })), [], 'true', false);
  const fn = $('showFN').checked ? (r.graph.fn_edges || []) : [];
  drawGraph($('svg_pred'), nodes, r.graph.pred_edges, fn, 'pred', true);
}
function relayout() { S.layouts = {}; renderDag(); }

function showNodeInfo(i) {
  const r = S.result; if (!r) return;
  const names = r.meta.node_names, E = r.graph.pred_edges;
  const par = E.filter(e => e.t === i).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
  const chi = E.filter(e => e.s === i).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
  const fmt = l => l.length ? l.slice(0, 6).map(e => `<b>${esc(names[e.s])}</b>→${esc(names[e.t])} (${e.w > 0 ? '+' : ''}${e.w.toFixed(3)})`).join('、') : '—';
  $('nodeinfo').innerHTML = `<b>${esc(names[i])}</b>　父节点（直接原因）：${fmt(par)}　｜　子节点（直接影响）：${fmt(chi)}`
    + `<br><span class="hint">若为推荐场景：父/子节点即"与该项目最相关的项目"，可用于"因为你喜欢 A，所以推荐 B"式解释。</span>`;
}

/* ---------- 权重矩阵热力图 ---------- */
function renderMatrix() {
  const r = S.result; if (!r || !r.matrix) { $('mat').innerHTML = '<div class="empty">矩阵过大或未返回</div>'; return; }
  const M = r.matrix, d = M.length;
  let mx = 0; M.forEach(row => row.forEach(v => mx = Math.max(mx, Math.abs(v))));
  if (mx === 0) mx = 1;
  const names = r.meta.node_names;
  const cell = Math.max(6, Math.min(20, Math.floor(560 / d)));
  const pad = 62, size = pad + d * cell;
  let g = `<svg viewBox="0 0 ${size} ${size}">`;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      const v = M[i][j], t = v / mx;
      const col = v >= 0 ? `rgba(192,57,43,${Math.min(.92, Math.abs(t) * 1.6)})`
        : `rgba(41,128,185,${Math.min(.92, Math.abs(t) * 1.6)})`;
      g += `<rect x="${pad + j * cell}" y="${pad + i * cell}" width="${cell}" height="${cell}"
             fill="${Math.abs(v) < 1e-9 ? '#f7f9fb' : col}"/>`;
    }
  }
  if (cell >= 11) {
    for (let i = 0; i < d; i++) {
      const lb = (names[i] || ('X' + (i + 1)));
      g += `<text x="${pad - 4}" y="${pad + i * cell + cell / 2 + 3.2}" font-size="8.5" fill="#5b6b7c" text-anchor="end">${esc(lb.length > 9 ? lb.slice(0, 8) + '…' : lb)}</text>`;
      g += `<text x="${pad + i * cell + cell / 2}" y="${pad - 4}" font-size="8.5" fill="#5b6b7c" text-anchor="middle"
             transform="rotate(-90 ${pad + i * cell + cell / 2} ${pad - 4})">${esc(lb.length > 9 ? lb.slice(0, 8) + '…' : lb)}</text>`;
    }
  }
  g += `</svg>`;
  $('mat').innerHTML = g + `<p class="hint">行=原因 i，列=结果 j；红色为正权重、蓝色为负权重（无 ground truth 时即学到的因果效应）。</p>`;
}

/* ---------- 边表 ---------- */
function renderEdges() {
  const r = S.result; if (!r) return;
  const names = r.meta.node_names;
  // 后端把 top_edges 放在结果顶层；graph.pred_edges 是完整列表，作为兜底
  const src = (r.top_edges && r.top_edges.length) ? r.top_edges
            : (r.graph && r.graph.pred_edges) ? r.graph.pred_edges : [];
  const E = src.slice().sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
  const total = (r.graph && r.graph.pred_edges) ? r.graph.pred_edges.length : E.length;
  if (!E.length) {
    $('edgetbl').innerHTML = `<thead><tr><th>#</th><th>原因 (i)</th><th>结果 (j)</th>
      <th>权重 W[i,j]</th><th>|W|</th><th>判定</th></tr></thead>
      <tbody><tr><td colspan="6" style="text-align:center;padding:18px;color:#8a6d3b">
      阈值 τ=${r.metrics.tau} 下没有任何 |W[i,j]| &gt; τ 的边。<br>
      请把左侧「出图阈值 τ」调小（如 0.05 / 0.02）后重新运行；
      若 τ 已是 0 仍无边，说明该轮训练得到的是全零解（可换 ε 或增大 T_i 重试）。
      </td></tr></tbody>`;
    return;
  }
  let h = `<tr><th>#</th><th>原因 (i)</th><th>结果 (j)</th><th>权重 W[i,j]</th><th>|W|</th><th>判定</th></tr>`;
  E.forEach((e, i) => {
    h += `<tr><td>${i + 1}</td><td>${esc(names[e.s])}</td><td>${esc(names[e.t])}</td>
      <td class="n">${e.w.toFixed(4)}</td><td class="n">${Math.abs(e.w).toFixed(4)}</td>
      <td><span class="tag ${e.type}">${e.type.toUpperCase()}</span></td></tr>`;
  });
  $('edgetbl').innerHTML = `<thead>${h}</thead>`;
  const noteEl = $('edgenote'); if (noteEl) noteEl.textContent =
    E.length < total ? `共 ${total} 条预测边，按 |W| 降序显示前 ${E.length} 条。` : '';
}

/* ---------- 根因分析 ---------- */
function initRca() {
  const r = S.result; if (!r) return;
  const sel = $('rca_target');
  const names = r.meta.node_names;
  const cur = sel.value;
  sel.innerHTML = names.map((n, i) => `<option value="${i}">${esc(n)}</option>`).join('');
  // 优先定位到内置数据集的目标节点
  let def = 0;
  ['err_payment', 'err_reserve', 'err_query_price', 'err_query_seat'].forEach(t => {
    const i = names.indexOf(t); if (i >= 0) def = i;
  });
  sel.value = (cur !== '' && cur !== null && +cur < names.length) ? cur : def;
  loadPaths();
}
function loadPaths() {
  const r = S.result; if (!r) return;
  const t = +$('rca_target').value, len = +$('rca_len').value || 4;
  $('rca_out').innerHTML = '<div class="empty">分析中…</div>';
  fetch(`/api/paths/${S.jobId}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target: t, max_len: len, top_k: 25 })
  }).then(r2 => r2.json()).then(d => {
    if (d.error) { $('rca_out').innerHTML = '<div class="empty">' + esc(d.error) + '</div>'; return; }
    let h = `<p class="hint" style="margin-bottom:9px">目标节点 <b>${esc(d.target_label)}</b> —— `
      + `下列因果链按 |权重乘积| 排序，链尾即最可能的<b>根本原因</b>（对应论文 §VI-A 的根因定位流程）。</p>`;
    if (d.parents && d.parents.length) {
      h += `<div style="margin-bottom:11px"><b>直接父节点（1 跳原因）：</b> `
        + d.parents.slice(0, 10).map(p => `<span class="path" style="display:inline-flex;margin:0 6px 6px 0">
             <span class="nd">${esc(p.label)}</span><span class="wt">w=${p.w.toFixed(3)}</span></span>`).join('') + `</div>`;
    } else h += `<p class="hint">该节点在学到的图中没有父节点（可能是根节点/外生变量）。</p>`;
    if (d.paths && d.paths.length) {
      h += `<b>多跳因果路径（长度 ≤ ${len}）：</b><div style="margin-top:7px">`;
      d.paths.forEach(p => {
        h += `<div class="path">` + p.labels.map((l, i) =>
          `<span class="nd${i === p.labels.length - 1 ? ' last' : ''}">${esc(l)}</span>${i < p.labels.length - 1 ? '<span class="ar">→</span>' : ''}`
        ).join('') + `<span class="wt">路径强度 ${p.weight.toFixed(4)} · 长度 ${p.len}</span></div>`;
      });
      h += `</div>`;
    } else h += `<p class="hint">未找到指向该节点的多跳路径。</p>`;
    $('rca_out').innerHTML = h;
  }).catch(e => { $('rca_out').innerHTML = '<div class="empty">' + esc(e) + '</div>'; });
}

/* ---------- 基准 ---------- */
let benchTimer = null;
function startBench() {
  const body = {
    d: $('b_d').value.split(',').map(s => +s.trim()).filter(v => !isNaN(v)),
    seeds: $('b_seeds').value.split(',').map(s => +s.trim()).filter(v => !isNaN(v)),
    graph: $('b_graph').value, noise: $('b_noise').value,
    with_notears: $('b_nt').checked,
  };
  $('bench_out').innerHTML = '<div class="empty">正在运行基准实验（需要数分钟）…</div>';
  fetch('/api/benchmark', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    .then(r => r.json()).then(() => { clearInterval(benchTimer); benchTimer = setInterval(pollBench, 1500); });
}
function pollBench() {
  fetch('/api/benchmark').then(r => r.json()).then(st => {
    if (st.status === 'idle') { $('bench_out').innerHTML = '<div class="empty">点击"开始"运行</div>'; return; }
    let h = '';
    if (st.status === 'running') {
      h += `<p class="hint">运行中… ${(st.progress * 100).toFixed(0)}%</p>`;
      h += `<pre class="logbox" style="max-height:180px">${esc(st.log.slice(-14).join('\n'))}</pre>`;
    }
    if (st.result) {
      const R = st.result;
      h += `<table><tr><th>d</th><th>n</th><th>LEAST F1</th><th>LEAST SHD</th>
            ${R.rows[0].nt_f1 ? '<th>NOTEARS F1</th><th>NOTEARS SHD</th><th title="NOTEARS λ网格总耗时 ÷ LEAST ε网格总耗时（对称口径）">加速比(网格)</th><th title="双方各自最佳配置的单次耗时之比（参考口径）">加速比(单次)</th>' : ''}
            <th title="LEAST 整个 ε 网格的总耗时">LEAST 网格用时</th><th>收敛</th></tr>`;
      R.rows.forEach(r2 => {
        h += `<tr><td>${r2.d}</td><td>${r2.n}</td>
          <td><b>${r2.f1[0].toFixed(3)}</b> ± ${r2.f1[1].toFixed(3)}</td>
          <td>${r2.shd[0].toFixed(1)} ± ${r2.shd[1].toFixed(1)}</td>
          ${r2.nt_f1 ? `<td>${r2.nt_f1[0].toFixed(3)} ± ${r2.nt_f1[1].toFixed(3)}</td>
            <td>${r2.nt_shd[0].toFixed(1)}</td>
            <td><b>${r2.speedup.toFixed(1)}×</b></td>
            <td>${r2.speedup_single ? r2.speedup_single.toFixed(1) + '×' : '—'}</td>` : ''}
          <td>${(r2.time_grid || r2.time).toFixed(1)}s</td><td>${r2.converged}/${r2.n_seed}</td></tr>`;
      });
      h += `</table><p class="hint" style="margin-top:9px">设置：${R.graph}-${R.degree}，噪声 ${R.noise}，n=10d；`
        + `每个 (d, seed) 在 ε∈{1e-1…1e-4} × τ∈{0.1…0.5} 网格上取最佳（论文 §V-A 协议）。`
        + `加速比(网格)为对称口径：NOTEARS λ 网格总耗时 ÷ LEAST ε 网格总耗时；加速比(单次)为双方最佳单次之比。</p>`;
      clearInterval(benchTimer);
    }
    if (st.status === 'error') h += `<p style="color:#c0392b">${esc(st.error)}</p>`;
    $('bench_out').innerHTML = h;
  });
}

/* ---------- 大规模可扩展性实测 ---------- */
let scalTimer = null;
function startScaling() {
  const nums = id => $(id).value.split(/[,，\s]+/).filter(Boolean).map(Number).filter(v => v > 0);
  const body = {
    d: nums('s_d').length ? nums('s_d') : [100, 200, 400, 800, 1600, 3200],
    e2e_d: nums('s_e2e'),
    nt_ref_d: parseInt($('s_ntref').value) || 64,
    with_notears: $('s_nt').checked,
    with_e2e: $('s_e2e_chk').checked
  };
  fetch('/api/scaling', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    .then(r => r.json()).then(() => { clearInterval(scalTimer); scalTimer = setInterval(pollScaling, 1200); });
  $('scal_out').innerHTML = '<p class="hint">启动中…</p>';
}
function fmtT(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return '—';
  if (sec < 1e-3) return (sec * 1e6).toFixed(0) + 'µs';
  if (sec < 1) return (sec * 1e3).toFixed(1) + 'ms';
  if (sec < 90) return sec.toFixed(1) + 's';
  if (sec < 5400) return (sec / 60).toFixed(1) + 'min';
  return (sec / 3600).toFixed(1) + 'h';
}
function pollScaling() {
  fetch('/api/scaling').then(r => r.json()).then(st => {
    if (st.status === 'idle') { return; }
    let h = '';
    if (st.status === 'running') {
      h += `<p class="hint">实测中… ${(st.progress * 100).toFixed(0)}%（大 d 的稠密计时与端到端实测需要一些时间）</p>`;
      h += `<pre class="logbox" style="max-height:150px">${esc(st.log.slice(-12).join('\n'))}</pre>`;
    }
    if (st.result) {
      const R = st.result, F = R.fit;
      h += `<p style="margin:6px 0"><b>结论：</b>稀疏 δ 实测复杂度 ≈ <b>O(d<sup>${F.sparse_order.toFixed(2)}</sup>)</b>（理论近线性），`
        + `稠密 δ ≈ <b>O(d<sup>${F.dense_order.toFixed(2)}</sup>)</b>（理论平方）`
        + (F.crossover_d ? `，稀疏引擎在 <b>d≈${Math.round(F.crossover_d)}</b> 之后稳定反超稠密。` : '。')
        + ` NOTEARS 的 h(W) 为 O(d³)，差距随 d 平方级拉大。</p>`;
      h += drawScaling(R.rows);
      h += `<table><tr><th>d</th><th>nnz</th><th>稀疏 δ fwd+bwd（实测）</th><th>稠密 δ fwd+bwd</th><th>算子加速</th></tr>`;
      R.rows.forEach(r => {
        const de = r.dense_measured ? fmtT(r.t_dense) : (r.t_dense_fit ? fmtT(r.t_dense_fit) + ' <span title="超出时间预算，按 log-log 拟合外推" style="color:#b26a00">†外推</span>' : '—');
        h += `<tr><td>${r.d}</td><td>${r.nnz}</td><td>${fmtT(r.t_sparse)}</td><td>${de}</td>
              <td><b>${r.speedup_op ? r.speedup_op.toFixed(1) + '×' : '—'}</b></td></tr>`;
      });
      h += `</table>`;
      if (R.notears) {
        const N = R.notears;
        h += `<p style="margin:8px 0 4px"><b>NOTEARS 对比（外推）：</b>现场实测 d=${N.d_ref} 时 NOTEARS 端到端单次 ${fmtT(N.t_ref)}、`
          + `仅 h(W) 单次 ${fmtT(N.t_h)}；按其 O(d³) 复杂度外推到更大规模<span style="color:#b26a00">†</span>：</p>`;
        h += `<table><tr><th>d</th><th>NOTEARS 端到端†</th><th>仅 h(W) 单次†</th><th>LEAST 稀疏 δ 单次（实测）</th></tr>`;
        N.extrap.forEach((e, i) => {
          h += `<tr><td>${e.d}</td><td>${fmtT(e.t_e2e)}</td><td>${fmtT(e.t_h)}</td><td>${fmtT(R.rows[i].t_sparse)}</td></tr>`;
        });
        h += `</table>`;
      }
      if (R.e2e && R.e2e.length) {
        h += `<p style="margin:8px 0 4px"><b>LEAST-SP 端到端实测</b>（ER-2，n=10d，ε∈{1e-1,1e-2} 取最佳，全部真实运行）：</p>`;
        h += `<table><tr><th>d</th><th>耗时</th><th>F1</th><th>SHD</th><th>外层</th><th>收敛</th><th>nnz 占比</th>`
          + (R.notears ? '<th>NOTEARS 端到端†</th><th>预计加速†</th>' : '') + `</tr>`;
        R.e2e.forEach(e => {
          h += `<tr><td>${e.d}</td><td>${fmtT(e.time)}</td><td><b>${e.f1.toFixed(3)}</b></td><td>${e.shd}</td>
                <td>${e.outer}</td><td>${e.converged ? '是' : '否'}</td><td>${(e.fill * 100).toFixed(1)}%</td>`
            + (R.notears ? `<td>${fmtT(e.nt_est)}</td><td><b>${e.speedup_est ? e.speedup_est.toFixed(1) + '×' : '—'}</b></td>` : '')
            + `</tr>`;
        });
        h += `</table>`;
      }
      h += `<p class="hint" style="margin-top:8px">† = 外推估计（基于实测参考点按理论复杂度放大），非直接测量；`
        + `未标 † 的数字均为本机现场实测。这正是论文的核心卖点：d 越大，LEAST 的 O(ks) 相对 NOTEARS 的 O(d³) 优势越大。</p>`;
      clearInterval(scalTimer);
    }
    if (st.status === 'error') { h += `<p style="color:#c0392b">${esc(st.error || '')}</p>`; clearInterval(scalTimer); }
    $('scal_out').innerHTML = h;
  });
}
/* log-log 缩放曲线：稀疏实测 / 稠密实测点 / 稠密外推虚线 */
function drawScaling(rows) {
  const W = 640, H = 240, pad = { l: 64, r: 16, t: 26, b: 40 };
  const ds = rows.map(r => r.d);
  const sp = rows.map(r => r.t_sparse);
  const deM = rows.filter(r => r.dense_measured).map(r => [r.d, r.t_dense]);
  const deF = rows.filter(r => !r.dense_measured && r.t_dense_fit).map(r => [r.d, r.t_dense_fit]);
  const allT = sp.concat(deM.map(p => p[1]), deF.map(p => p[1]));
  const lxv = [Math.min(...ds), Math.max(...ds)].map(Math.log10);
  const lyv = [Math.min(...allT), Math.max(...allT)].map(Math.log10);
  const X = d => pad.l + (Math.log10(d) - lxv[0]) / (lxv[1] - lxv[0]) * (W - pad.l - pad.r);
  const Y = t => H - pad.b - (Math.log10(t) - lyv[0]) / (lyv[1] - lyv[0]) * (H - pad.t - pad.b);
  const path = pts => pts.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1)).join(' ');
  const spPts = rows.map(r => [r.d, r.t_sparse]);
  // 稠密外推虚线：从最后一个实测点连到各外推点
  const deLine = deM.length && deF.length ? [deM[deM.length - 1]].concat(deF) : deF;
  let g = '';
  [1, 2, 3].forEach(i => {  // 横网格
    const y = pad.t + (H - pad.t - pad.b) * i / 4;
    g += `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="#e3e8ef"/>`;
  });
  ds.forEach(d => {  // x 刻度
    g += `<text x="${X(d)}" y="${H - pad.b + 16}" font-size="10" fill="#7a8aa0" text-anchor="middle">${d}</text>`;
  });
  [Math.pow(10, Math.ceil(lyv[0])), Math.pow(10, ((Math.ceil(lyv[0]) + Math.floor(lyv[1])) / 2) | 0), Math.pow(10, Math.floor(lyv[1]))].forEach(t => {
    if (t > Math.pow(10, lyv[0]) * 0.9 && t < Math.pow(10, lyv[1]) * 1.1)
      g += `<text x="${pad.l - 6}" y="${Y(t) + 3}" font-size="10" fill="#7a8aa0" text-anchor="end">${fmtT(t)}</text>`;
  });
  if (deLine.length > 1)
    g += `<path d="${path(deLine)}" fill="none" stroke="#c0392b" stroke-width="1.5" stroke-dasharray="5,4"/>`;
  g += `<path d="${path(spPts)}" fill="none" stroke="#2980b9" stroke-width="2"/>`;
  spPts.forEach(p => { g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3" fill="#2980b9"/>`; });
  deM.forEach(p => { g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3" fill="#c0392b"/>`; });
  deF.forEach(p => { g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3" fill="none" stroke="#c0392b"/>`; });
  g += `<text x="${pad.l + 8}" y="${pad.t - 10}" font-size="11" fill="#2980b9">— 稀疏 δ（实测）</text>`
    + `<text x="${pad.l + 130}" y="${pad.t - 10}" font-size="11" fill="#c0392b">— 稠密 δ（●实测 ┅外推）</text>`
    + `<text x="${W / 2}" y="${H - 6}" font-size="11" fill="#7a8aa0" text-anchor="middle">节点数 d（对数轴）</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:680px;background:#fbfcfe;border:1px solid #e3e8ef;border-radius:8px;margin:6px 0">${g}</svg>`;
}

window.addEventListener('resize', () => {
  if (S.result) { drawConv(S.result.history, false); renderDag(); }
});
