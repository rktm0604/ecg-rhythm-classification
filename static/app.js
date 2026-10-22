/* ═══════════════════════════════════════════════════════════════
   ECG Rhythm Classifier — Interactive Frontend
   ═══════════════════════════════════════════════════════════════ */

const API = window.location.origin;

// ── DOM refs ──
const $  = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const navBtns       = $$('.nav-btn');
const views         = $$('.view');
const statusDot     = $('.status-dot');
const statusText    = $('.status-text');
const featuresInput = $('#features-input');
const featureCount  = $('#feature-count');
const btnClassify   = $('#btn-classify');
const btnNormal     = $('#btn-load-normal');
const btnAbnormal   = $('#btn-load-abnormal');
const btnClear      = $('#btn-clear');
const resultCard    = $('#result-card');
const resultPH      = $('#result-placeholder');
const resultContent = $('#result-content');
const resultBadge   = $('#result-badge');
const confBarFill   = $('#conf-bar-fill');
const confValue     = $('#conf-value');
const resultDetails = $('#result-details');
const ecgCanvas     = $('#ecg-canvas');
const ecgCtx        = ecgCanvas.getContext('2d');

// ── Inject SVG gradient for KPI rings ──
(() => {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', '0');
  svg.setAttribute('height', '0');
  svg.style.position = 'absolute';
  svg.innerHTML = `<defs>
    <linearGradient id="kpi-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#6366f1"/>
      <stop offset="100%" stop-color="#06b6d4"/>
    </linearGradient>
  </defs>`;
  document.body.prepend(svg);
})();

// ═══════════════ NAVIGATION ═══════════════

navBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.view;
    navBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    views.forEach(v => {
      v.classList.toggle('active', v.id === `view-${target}`);
    });
  });
});

// ═══════════════ API HELPERS ═══════════════

async function apiFetch(endpoint, opts = {}) {
  const res = await fetch(`${API}${endpoint}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function checkApi() {
  try {
    const data = await apiFetch('/');
    statusDot.classList.add('online');
    statusDot.classList.remove('offline');
    statusText.textContent = `Online · ${data.expected_features} features`;
    return true;
  } catch {
    statusDot.classList.remove('online');
    statusDot.classList.add('offline');
    statusText.textContent = 'Offline';
    return false;
  }
}

// ═══════════════ DASHBOARD ═══════════════

function animateRing(id, value) {
  const card = $(`#${id}`);
  const ring = card.querySelector('.ring-fg');
  const label = card.querySelector('.kpi-value');
  const pct = Math.round(value * 100);
  const offset = 326.7 * (1 - value);
  // Trigger reflow for animation
  ring.style.strokeDashoffset = '326.7';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      ring.style.strokeDashoffset = offset;
    });
  });
  // Count-up animation
  let current = 0;
  const step = Math.ceil(pct / 30);
  const interval = setInterval(() => {
    current = Math.min(current + step, pct);
    label.textContent = current + '%';
    if (current >= pct) clearInterval(interval);
  }, 25);
}

function renderConfusionMatrix() {
  // Hard-coded from the README / metrics — these don't change without retraining
  const cm = [[5910, 279], [89, 1391]];
  const total = 7669;
  const labels = ['Normal', 'Abnormal'];
  const types = [['tp', 'fp'], ['fn', 'tn']];
  const grid = $('#cm-grid');
  grid.innerHTML = '';

  // Corner
  grid.appendChild(makeDiv('cm-header cm-corner', ''));
  // Col headers
  labels.forEach(l => grid.appendChild(makeDiv('cm-header', `Pred. ${l}`)));
  // Rows
  cm.forEach((row, i) => {
    grid.appendChild(makeDiv('cm-header', `True ${labels[i]}`));
    row.forEach((val, j) => {
      const cell = makeDiv(`cm-cell ${types[i][j]}`,
        `${val.toLocaleString()}<span class="cm-pct">${(val/total*100).toFixed(1)}%</span>`);
      grid.appendChild(cell);
    });
  });
}

function makeDiv(cls, html) {
  const d = document.createElement('div');
  d.className = cls;
  d.innerHTML = html;
  return d;
}

function renderClassTable(metrics) {
  const tbody = $('#class-table tbody');
  tbody.innerHTML = '';
  ['Normal', 'Abnormal'].forEach(name => {
    const m = metrics[name];
    if (!m) return;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${name}</td>
      <td>${(m.precision * 100).toFixed(1)}%</td>
      <td>${(m.recall * 100).toFixed(1)}%</td>
      <td>${(m['f1-score'] * 100).toFixed(1)}%</td>
      <td>${m.support.toLocaleString()}</td>`;
    tbody.appendChild(tr);
  });
}

function renderDatasetBars() {
  const data = [
    { split: 'Train', records: '101, 106, 119, 201, 203, 208', normal: 10647, abnormal: 3107, total: 13754 },
    { split: 'Test',  records: '200, 210, 221',                 normal: 6189,  abnormal: 1480, total: 7669  },
  ];
  const maxTotal = Math.max(...data.map(d => d.total));
  const container = $('#dataset-bars');
  container.innerHTML = '';

  data.forEach(d => {
    // Normal bar
    const rowN = document.createElement('div');
    rowN.className = 'bar-row';
    rowN.innerHTML = `
      <span class="bar-label">${d.split} (N)</span>
      <div class="bar-track">
        <div class="bar-fill normal" style="width: 0%;" data-w="${(d.normal / maxTotal * 100).toFixed(1)}">
          <span>${d.normal.toLocaleString()} normal</span>
        </div>
      </div>
      <span class="bar-count">${d.normal.toLocaleString()}</span>`;
    container.appendChild(rowN);

    // Abnormal bar
    const rowA = document.createElement('div');
    rowA.className = 'bar-row';
    rowA.innerHTML = `
      <span class="bar-label">${d.split} (A)</span>
      <div class="bar-track">
        <div class="bar-fill abnormal" style="width: 0%;" data-w="${(d.abnormal / maxTotal * 100).toFixed(1)}">
          <span>${d.abnormal.toLocaleString()} abnormal</span>
        </div>
      </div>
      <span class="bar-count">${d.abnormal.toLocaleString()}</span>`;
    container.appendChild(rowA);
  });

  // Animate bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      container.querySelectorAll('.bar-fill').forEach(el => {
        el.style.width = el.dataset.w + '%';
      });
    }, 200);
  });
}

async function initDashboard() {
  try {
    const metrics = await apiFetch('/metrics');
    animateRing('kpi-accuracy',  metrics.accuracy);
    animateRing('kpi-precision', metrics['Abnormal']?.precision ?? 0);
    animateRing('kpi-recall',    metrics['Abnormal']?.recall ?? 0);
    animateRing('kpi-f1',        metrics['Abnormal']?.['f1-score'] ?? 0);
    renderClassTable(metrics);
  } catch {
    // Use hard-coded fallback
    animateRing('kpi-accuracy',  0.952);
    animateRing('kpi-precision', 0.833);
    animateRing('kpi-recall',    0.940);
    animateRing('kpi-f1',        0.883);
  }
  renderConfusionMatrix();
  renderDatasetBars();
}

// ═══════════════ PREDICT VIEW ═══════════════

let sampleData = null;

async function loadSamples() {
  try {
    sampleData = await apiFetch('/sample');
  } catch {
    // Hard-coded fallback from sample_beats.json
    sampleData = {
      normal_example: { features: [262,202,1.168,0.901,-0.227,-0.299,-0.263,-0.047,0.078,-0.191,-0.173,-0.119,-0.101,-0.263,-0.245,-0.155,-0.155,0.006,1.950,4.272,1.320,-1.379,-0.695,-0.749,-0.479,-0.551,-0.443,-0.389,-0.353,-0.371,-0.281,-0.119,-0.011,0.024] },
      abnormal_example: { features: [202,275,0.876,1.193,-0.022,-0.037,-0.015,-0.030,-0.074,-0.111,-0.059,-0.126,-0.134,-0.096,-0.022,0.126,0.238,0.393,-1.522,-2.888,-2.286,-1.544,-0.891,-0.482,-0.089,0.371,0.586,0.787,0.965,1.106,1.277,1.358,1.269,1.106] }
    };
  }
}

function setFeatures(arr) {
  featuresInput.value = arr.map(v => typeof v === 'number' ? parseFloat(v.toFixed(6)) : v).join(', ');
  updateFeatureCount();
  drawECGWaveform(arr);
}

function parseFeatures() {
  const raw = featuresInput.value.trim();
  if (!raw) return [];
  return raw.split(/[\s,]+/).map(Number).filter(v => !isNaN(v));
}

function updateFeatureCount() {
  const feats = parseFeatures();
  const n = feats.length;
  featureCount.textContent = `${n} / 34 features`;
  featureCount.className = 'feature-count' + (n === 34 ? ' valid' : n > 0 ? ' invalid' : '');

  // Live waveform update
  if (n >= 8) drawECGWaveform(feats);
}

// ── ECG Waveform Drawing ──

function drawECGWaveform(features) {
  const canvas = ecgCanvas;
  const ctx = ecgCtx;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.scale(dpr, dpr);
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;

  // Waveform samples are features[4..33] (30 values)
  const waveform = features.slice(4, 34);
  if (waveform.length < 2) {
    ctx.clearRect(0, 0, W, H);
    return;
  }

  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(99, 102, 241, 0.06)';
  ctx.lineWidth = 1;
  for (let y = 0; y < H; y += 20) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  for (let x = 0; x < W; x += 20) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }

  // Zero line
  ctx.strokeStyle = 'rgba(99, 102, 241, 0.15)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
  ctx.setLineDash([]);

  // Calculate bounds for normalization
  const minV = Math.min(...waveform);
  const maxV = Math.max(...waveform);
  const range = maxV - minV || 1;
  const padY = 30;
  const plotH = H - padY * 2;
  const padX = 20;
  const plotW = W - padX * 2;

  const toX = (i) => padX + (i / (waveform.length - 1)) * plotW;
  const toY = (v) => padY + (1 - (v - minV) / range) * plotH;

  // Glow layer
  ctx.save();
  ctx.shadowColor = 'rgba(99, 102, 241, 0.5)';
  ctx.shadowBlur = 12;
  ctx.strokeStyle = 'rgba(99, 102, 241, 0.3)';
  ctx.lineWidth = 4;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  waveform.forEach((v, i) => {
    i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v));
  });
  ctx.stroke();
  ctx.restore();

  // Main line with gradient
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, '#6366f1');
  grad.addColorStop(0.5, '#818cf8');
  grad.addColorStop(1, '#06b6d4');
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  waveform.forEach((v, i) => {
    i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v));
  });
  ctx.stroke();

  // Data points
  waveform.forEach((v, i) => {
    ctx.beginPath();
    ctx.arc(toX(i), toY(v), 3, 0, Math.PI * 2);
    ctx.fillStyle = '#818cf8';
    ctx.fill();
    ctx.strokeStyle = 'rgba(6, 10, 19, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // Axis labels
  ctx.font = '10px Inter, sans-serif';
  ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
  ctx.textAlign = 'left';
  ctx.fillText(`max: ${maxV.toFixed(2)}`, padX, padY - 8);
  ctx.fillText(`min: ${minV.toFixed(2)}`, padX, H - padY + 16);
  ctx.textAlign = 'right';
  ctx.fillText('Waveform Samples (every 6th)', W - padX, padY - 8);
}

// Draw blank canvas on load
function drawBlankECG() {
  const dpr = window.devicePixelRatio || 1;
  ecgCanvas.width = ecgCanvas.clientWidth * dpr;
  ecgCanvas.height = ecgCanvas.clientHeight * dpr;
  ecgCtx.scale(dpr, dpr);
  const W = ecgCanvas.clientWidth;
  const H = ecgCanvas.clientHeight;
  ecgCtx.clearRect(0, 0, W, H);

  // Grid
  ecgCtx.strokeStyle = 'rgba(99, 102, 241, 0.06)';
  ecgCtx.lineWidth = 1;
  for (let y = 0; y < H; y += 20) {
    ecgCtx.beginPath(); ecgCtx.moveTo(0, y); ecgCtx.lineTo(W, y); ecgCtx.stroke();
  }
  for (let x = 0; x < W; x += 20) {
    ecgCtx.beginPath(); ecgCtx.moveTo(x, 0); ecgCtx.lineTo(x, H); ecgCtx.stroke();
  }

  // Flat line with animation
  ecgCtx.strokeStyle = 'rgba(99, 102, 241, 0.15)';
  ecgCtx.lineWidth = 1;
  ecgCtx.setLineDash([4, 4]);
  ecgCtx.beginPath();
  ecgCtx.moveTo(0, H / 2);
  ecgCtx.lineTo(W, H / 2);
  ecgCtx.stroke();
  ecgCtx.setLineDash([]);

  ecgCtx.font = '13px Inter, sans-serif';
  ecgCtx.fillStyle = 'rgba(148, 163, 184, 0.3)';
  ecgCtx.textAlign = 'center';
  ecgCtx.fillText('Load a sample beat to visualize the waveform', W / 2, H / 2 - 12);
}

// ── Classify ──

async function classify() {
  const feats = parseFeatures();
  if (feats.length !== 34) {
    alert(`Please enter exactly 34 features. You have ${feats.length}.`);
    return;
  }

  btnClassify.disabled = true;
  btnClassify.innerHTML = '<span class="spinner"></span> Classifying…';

  try {
    const result = await apiFetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features: feats }),
    });

    showResult(result);
  } catch (err) {
    alert('Prediction failed: ' + err.message);
  } finally {
    btnClassify.disabled = false;
    btnClassify.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Classify`;
  }
}

function showResult(result) {
  resultPH.style.display = 'none';
  resultContent.classList.remove('hidden');

  const isNormal = result.class === 'Normal';
  resultBadge.textContent = result.class;
  resultBadge.className = 'result-badge ' + (isNormal ? 'normal-badge' : 'abnormal-badge');

  const pct = Math.round(result.confidence * 100);
  confBarFill.style.width = '0%';
  confBarFill.className = 'conf-bar-fill' + (isNormal ? '' : ' abnormal-bar');
  requestAnimationFrame(() => {
    setTimeout(() => { confBarFill.style.width = pct + '%'; }, 50);
  });

  // Animate confidence value
  let current = 0;
  const step = Math.max(1, Math.ceil(pct / 25));
  const interval = setInterval(() => {
    current = Math.min(current + step, pct);
    confValue.textContent = current + '%';
    if (current >= pct) clearInterval(interval);
  }, 20);

  resultDetails.innerHTML = isNormal
    ? '✓ The model classifies this beat as a <strong>normal sinus rhythm</strong>.'
    : '⚠ The model flags this beat as <strong>abnormal</strong> — possible arrhythmia detected.';
}

// ── Event listeners ──

btnNormal.addEventListener('click', () => {
  if (sampleData?.normal_example) {
    setFeatures(sampleData.normal_example.features);
  }
});

btnAbnormal.addEventListener('click', () => {
  if (sampleData?.abnormal_example) {
    setFeatures(sampleData.abnormal_example.features);
  }
});

btnClear.addEventListener('click', () => {
  featuresInput.value = '';
  updateFeatureCount();
  resultPH.style.display = '';
  resultContent.classList.add('hidden');
  drawBlankECG();
});

featuresInput.addEventListener('input', updateFeatureCount);
btnClassify.addEventListener('click', classify);

// Allow Enter to classify (Ctrl+Enter in textarea)
featuresInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) {
    e.preventDefault();
    classify();
  }
});

// ═══════════════ INIT ═══════════════

(async function init() {
  await checkApi();
  initDashboard();
  loadSamples();
  drawBlankECG();

  // Periodic health check
  setInterval(checkApi, 15000);

  // Resize canvas on window resize
  window.addEventListener('resize', () => {
    const feats = parseFeatures();
    if (feats.length >= 8) drawECGWaveform(feats);
    else drawBlankECG();
  });
})();
