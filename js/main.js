/* main.js — application state, UI wiring and the training loop. */

const state = {
  features: { ta: true, rh: true, tw: true, tout: true, hour: true, doy: true, mov: true, vair: true },
  hidden: [6, 6],
  activation: 'relu',
  lr: 0.003,
  l2: 0,
  batch: 16,
  coverage: 'year',
  voteSource: 'sim',   // 'sim' — lived-in household; 'uniform' — theory sampling
  pref: 0,             // occupant preference: shifts the true PMV scale
  sigma: 0.15,         // vote inconsistency
  sensorNoise: 0.5,
  nex: 1200,
  balance: true,
  running: false,
  runEpochs: 0,        // how many epochs one press of play should run; 0 is "until I pause"
  stopAt: null,        // the epoch the current run stops at, or null for an open-ended run
  epoch: 0,
  probe: {},           // the current "moment in the room" — drives map, nodes and arithmetic
  mapX: 'ta',
  mapY: 'rh',
  showTruth: true,
  showVotes: true,
  allVotes: false,
  hover: null,
  selected: null,
};
FEATURES.forEach((f) => (state.probe[f.id] = f.dflt));

let model = null, train = null, test = null;
let layout = null, netCtx = null;
let hTrain = [], hTest = [];
let lastMetrics = null;
let frameNo = 0;

let grids = null;          // per-node response maps over the map plane
let gridsDirty = true;
let mapNet = null, mapNetDirty = true;
let mapTruth = null, mapTruthDirty = true;
let sliceVotes = null, sliceVotesDirty = true;
let mapDrawDirty = true;
let pf = null;             // forward pass at the probe (activations for the arithmetic)

const $ = (id) => document.getElementById(id);

function activeIds() { return FEATURES.map((f) => f.id).filter((id) => state.features[id]); }
function featX() { return FEATURES[FEAT_INDEX[state.mapX]]; }
function featY() { return FEATURES[FEAT_INDEX[state.mapY]]; }

function dataOpt() {
  return {
    coverage: state.coverage, pref: state.pref, sigma: state.sigma,
    sensorNoise: state.sensorNoise, balance: state.balance,
    insulation: +$('insul').value, pmax: +$('pmax').value, heater: $('heaterType').value,
  };
}
function houseCfg() {
  return {
    doy: +$('startMonth').value, insulation: +$('insul').value,
    pmax: +$('pmax').value, heater: $('heaterType').value, residents: 2,
  };
}

/** Everything that depends on the weights is stale after a training step. */
function markTrained() { gridsDirty = true; mapNetDirty = true; }
/** Everything that depends on the probe / axes / preference is stale. */
function markProbeDirty() {
  gridsDirty = true; mapNetDirty = true; mapTruthDirty = true;
  sliceVotesDirty = true; mapDrawDirty = true;
}

/* -------------------------------------------------------------- data */
function regenData() {
  const ids = activeIds();
  const opt = dataOpt();
  const make = state.voteSource === 'uniform' ? makeUniformDataset : makeDataset;
  train = make(state.nex, ids, opt);
  test = make(Math.round(state.nex * 0.4), ids, opt);
  sliceVotesDirty = true; mapDrawDirty = true;
  renderCounts();
}

function renderCounts() {
  const c = classCounts(train);
  $('countRow').title = 'How many votes of each kind ended up in the training set.';
  $('countRow').innerHTML = CLASSES.map((cl, i) =>
    '<span><span class="cchip" style="background:' + cl.color + '"></span>' + c[i] + '</span>'
  ).join('') + '<span>Σ ' + train.n + ' train · ' + test.n + ' test</span>';
}

/* ------------------------------------------------------------- model */
function rebuildModel() {
  const ids = activeIds();
  model = new MLP({
    inputDim: encodedDim(ids),
    hidden: state.hidden.slice(),
    activation: state.activation,
    nClasses: CLASSES.length,
  });
  state.epoch = 0; state.stopAt = null;
  hTrain = []; hTest = [];
  state.selected = null; state.hover = null;
  markTrained(); mapDrawDirty = true;
  $('layCount').textContent = state.hidden.length;
  $('paramCount').textContent = model.paramCount().toLocaleString('en-US') + ' parameters';
  buildLayerControls();
  $('qtResult').innerHTML = '';        // that score belonged to the old weights
  evaluate(); renderMetrics(); renderRunTarget(); renderMath(true);
}

/* ---------------------------------------------------------- training */
function trainOneBatch() {
  const B = state.batch;
  const idx = new Array(B);
  for (let i = 0; i < B; i++) idx[i] = Math.floor(Math.random() * train.n);
  model.trainBatch(train.xs, train.ys, idx, state.lr, state.l2);
  state.epoch += B / train.n;
}

function trainSlice(budgetMs) {
  const t0 = performance.now();
  let steps = 0;
  do {
    if (runFinished()) break;
    trainOneBatch(); steps++;
  } while (performance.now() - t0 < budgetMs);
  if (steps) markTrained();
  return steps;
}

/** True once a run with a fixed length has trained for all of its epochs. */
function runFinished() {
  return state.stopAt !== null && state.epoch >= state.stopAt;
}

/** Starts or stops the training loop and keeps the play button in step. */
function setRunning(on) {
  state.running = on;
  $('btnPlay').textContent = on ? '⏸' : '▶';
  $('btnPlay').classList.toggle('on', on);
}

/** Arms a run: a fixed number of epochs from here, or open-ended when 0. */
function armRun() {
  state.stopAt = state.runEpochs > 0 ? state.epoch + state.runEpochs : null;
  renderRunTarget();
}

/** Shows the epoch a fixed-length run is heading for, next to the counter. */
function renderRunTarget() {
  const el = $('epochTgt');
  if (el) el.textContent = state.stopAt === null ? '' : '/ ' + Math.round(state.stopAt);
}

function evaluate() {
  const a = model.evaluate(train, 3, 300);
  const b = model.evaluate(test, 3, 500);
  lastMetrics = { train: a, test: b };
  hTrain.push(a.loss); hTest.push(b.loss);
  if (hTrain.length > 320) { hTrain.shift(); hTest.shift(); }
  return lastMetrics;
}

/** F1 of the "comfortable" class from a confusion matrix — the zone score. */
function comfF1(conf) {
  const k = CLASSES.length, c = CLASS_INDEX.comf;
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const v = conf[i * k + j];
      if (i === c && j === c) tp += v;
      else if (j === c) fp += v;
      else if (i === c) fn += v;
    }
  }
  return (2 * tp) / Math.max(1, 2 * tp + fp + fn);
}

/* --------------------------------------------------------- main loop */
function mainLoop() {
  frameNo++;

  if (loop.on) {
    const steps = loopTick(model, activeIds(), loopOpt());
    if (steps > 0 && frameNo % 2 === 0) renderLoop();
  }

  if (state.running) {
    trainSlice(loop.on ? 6 : 11);
    if (runFinished()) {
      // the requested epochs are done — pause and leave the final numbers up
      setRunning(false);
      state.stopAt = null;
      evaluate(); renderMetrics(); renderRunTarget();
    } else if (frameNo % 20 === 0) {
      evaluate(); renderMetrics();
    }
    if (state.selected && frameNo % 10 === 0) renderMath(true);
  }

  if (frameNo % 2 === 0) renderNet();
  renderMapMaybe();
  requestAnimationFrame(mainLoop);
}

function loopOpt() {
  return { pref: state.pref, sigma: state.sigma, onVote: onLiveVote };
}

/** Online learning: a live vote becomes a training example on the spot. */
function onLiveVote(v) {
  const ids = activeIds();
  train.xs.push(encodeState(v.state, ids));
  train.ys.push(v.label);
  train.states.push(v.state);
  train.pmvs.push(v.pmv);
  if (train.xs.length > 4000) { train.xs.shift(); train.ys.shift(); train.states.shift(); train.pmvs.shift(); }
  train.n = train.xs.length;
  for (let k = 0; k < 6; k++) trainOneBatch();     // a short burst of learning
  markTrained();
  sliceVotesDirty = true;
  if (frameNo % 4 === 0) renderCounts();
}

/* -------------------------------------------------- node response maps */
/**
 * Evaluates the network over a GRES×GRES grid of the map plane and keeps every
 * node's value at every cell — the little heatmaps in the diagram.
 */
function computeGrids() {
  const ids = activeIds();
  const fx = featX(), fy = featY();
  const nIn = model.cfg.inputDim;
  const g = {
    inputs: [], hidden: model.cfg.hidden.map((u) => []), outputs: [],
  };
  for (let i = 0; i < nIn; i++) g.inputs.push(new Float32Array(GRES * GRES));
  model.cfg.hidden.forEach((units, li) => {
    for (let u = 0; u < units; u++) g.hidden[li].push(new Float32Array(GRES * GRES));
  });
  for (let c = 0; c < CLASSES.length; c++) g.outputs.push(new Float32Array(GRES * GRES));

  const s = Object.assign({}, state.probe);
  for (let gy = 0; gy < GRES; gy++) {
    s[fy.id] = axisValue(fy, gy / (GRES - 1));
    for (let gx = 0; gx < GRES; gx++) {
      s[fx.id] = axisValue(fx, gx / (GRES - 1));
      const x = encodeState(s, ids);
      const p = model.forward(x, true);
      const cell = gy * GRES + gx;
      for (let i = 0; i < nIn; i++) g.inputs[i][cell] = x[i];
      model.acts.forEach((a, li) => {
        for (let u = 0; u < a.length; u++) g.hidden[li][u][cell] = a[u];
      });
      for (let c = 0; c < CLASSES.length; c++) g.outputs[c][cell] = p[c];
    }
  }
  grids = g;
  gridsDirty = false;
}

/** Forward pass at the probe itself — activations for readout and arithmetic. */
function probeForward() {
  const x = encodeState(state.probe, activeIds());
  const p = model.forward(x, true);
  pf = { x, preacts: model.preacts, acts: model.acts, logits: model.logits.slice(), probs: p.slice() };
  return pf;
}

/* ----------------------------------------------------------- network */
function renderNet() {
  const wrap = $('netWrap');
  const cssW = Math.max(540, wrap.clientWidth - 2);
  const ids = activeIds();
  const names = encodedNames(ids);
  layout = layoutNetwork(model, names, cssW);
  netCtx = dpiSetup($('net'), Math.max(cssW, layout.width), layout.height);
  if (gridsDirty && frameNo % 3 === 0) computeGrids();
  probeForward();
  const fx = featX(), fy = featY();
  drawNetwork(netCtx, {
    model, layout, grids,
    inNames: names,
    inValues: pf.x,
    acts: pf.acts,
    probs: pf.probs,
    hover: state.hover,
    selected: state.selected,
    axisInfo: { fx: axisFrac(fx, state.probe[fx.id]), fy: axisFrac(fy, state.probe[fy.id]) },
  });
  positionLayerControls();
  renderReadout();
}

/* ----------------------------------------------------------- metrics */
function renderMetrics() {
  const v = Math.floor(state.epoch);
  const sPad = String(v % 1000).padStart(3, '0');
  $('epoch').textContent = String(Math.floor(v / 1000)).padStart(3, '0') + ',' + sPad;
  if (!lastMetrics) return;
  $('lossTrain').textContent = lastMetrics.train.loss.toFixed(3);
  $('lossTest').textContent = lastMetrics.test.loss.toFixed(3);
  $('accTest').textContent = (lastMetrics.test.acc * 100).toFixed(1) + '%';
  $('f1Comf').textContent = (comfF1(lastMetrics.test.conf) * 100).toFixed(1) + '%';
  const lossC = $('loss');
  drawLossChart(dpiSetup(lossC, lossC.clientWidth, 110), lossC.clientWidth, 110, hTrain, hTest);
  const confC = $('conf');
  drawConfusion(dpiSetup(confC, confC.clientWidth, 150), confC.clientWidth, 150, lastMetrics.test.conf, CLASSES);
}

/* ------------------------------------------------------- comfort map */
function renderMapMaybe() {
  const busy = state.running || loop.on;
  const slow = busy ? frameNo % 15 === 0 : true;
  let changed = false;
  if (mapNetDirty && slow) {
    mapNet = mapEvalNet(model, activeIds(), state.probe, featX(), featY());
    mapNetDirty = false; changed = true;
  }
  if (mapTruthDirty && slow) {
    mapTruth = state.showTruth ? mapEvalTruth(state.probe, featX(), featY(), state.pref) : null;
    mapTruthDirty = false; changed = true;
  }
  if (sliceVotesDirty && slow) {
    sliceVotes = state.showVotes
      ? votesNearSlice(train, activeIds(), state.probe, featX(), featY(), state.allVotes)
      : null;
    sliceVotesDirty = false; changed = true;
  }
  if (changed || mapDrawDirty) {
    mapDrawDirty = false;
    drawMap();
  }
}

function drawMap() {
  const c = $('map');
  const w = Math.max(380, c.parentElement.clientWidth - 2);
  const ctx = dpiSetup(c, w, 360);
  drawComfortMap(ctx, w, 360, {
    fx: featX(), fy: featY(),
    net: mapNet,
    truth: state.showTruth ? mapTruth : null,
    votes: state.showVotes ? sliceVotes : null,
    probe: state.probe,
  });
}

/* ----------------------------------------------------- probe & votes */
function renderReadout() {
  if (!pf) return;
  const truth = comfortTruth(state.probe, state.pref);
  const nn = argmax(pf.probs);
  let html = '';
  CLASSES.forEach((c, i) => {
    html += '<div class="bar"><span class="nm">' + c.name + '</span>' +
      '<span class="track"><span class="fill" style="width:' + (pf.probs[i] * 100).toFixed(1) +
      '%;background:' + c.color + '"></span></span>' +
      '<span class="pc">' + (pf.probs[i] * 100).toFixed(1) + '%</span></div>';
  });
  const agree = nn === truth.label;
  html += '<div class="verdict">Network: <b style="color:' + CLASSES[nn].color + '">' +
    CLASSES[nn].name + '</b> · true model: <b style="color:' + CLASSES[truth.label].color + '">' +
    CLASSES[truth.label].name + '</b> (PMV ' + (truth.pmv >= 0 ? '+' : '−') + Math.abs(truth.pmv).toFixed(2) +
    ', ' + truth.ppd.toFixed(0) + '% dissatisfied) — ' +
    (agree ? '<span class="ok">match</span>' : '<span class="bad">mismatch</span>') +
    '<br><span style="color:#7b8794">assumed ' + truth.met.toFixed(2) + ' met, ' +
    truth.clo.toFixed(2) + ' clo' + (sleepHours(state.probe.hour) && state.probe.mov < 0.12 ? ' (asleep, in bed)' : '') +
    '</span></div>';
  $('probeReadout').innerHTML = html;
}

function setProbe(patch, fromSliders) {
  Object.assign(state.probe, patch);
  markProbeDirty();
  if (!fromSliders) syncProbeSliders();
  renderNet();
  renderMath(true);
}

function syncProbeSliders() {
  FEATURES.forEach((f) => {
    const row = $('probe_' + f.id);
    if (!row) return;
    row.querySelector('input').value = state.probe[f.id];
    row.querySelector('b').textContent = fmtFeat(f, state.probe[f.id]);
  });
}

function fmtFeat(f, v) {
  if (f.id === 'hour') {
    const h = Math.floor(v), m = Math.round((v - h) * 60);
    return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0');
  }
  if (f.id === 'doy') {
    const M = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const d = Math.max(1, Math.round(v));
    return d + ' (' + M[Math.min(11, Math.floor((d - 1) / 30.42))] + ')';
  }
  return (+v).toFixed(f.fmt) + (f.unit ? ' ' + f.unit : '');
}

/* -------------------------------------------------------- quick test */
/**
 * Scores the network against the NOISELESS comfort model on fresh random
 * moments — the honest question: did it find the zone, not the noise?
 */
function quickTest() {
  const ids = activeIds();
  const N = 300;
  let ok = 0;
  const per = [[0, 0], [0, 0], [0, 0]];        // [correct, total] per true class
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < N; i++) {
    const s = randomState(state.coverage);
    const y = comfortTruth(s, state.pref).label;
    const a = argmax(model.forward(encodeState(s, ids), false));
    per[y][1]++;
    if (a === y) { ok++; per[y][0]++; }
    const c = CLASS_INDEX.comf;
    if (a === c && y === c) tp++;
    else if (a === c) fp++;
    else if (y === c) fn++;
  }
  const f1 = (2 * tp) / Math.max(1, 2 * tp + fp + fn);
  let html = '<div class="head"><span>vs the noiseless model · epoch ' +
    Math.floor(state.epoch) + '</span><b>' + (ok / N * 100).toFixed(1) + '%</b></div>';
  CLASSES.forEach((c, i) => {
    const r = per[i][1] ? per[i][0] / per[i][1] : 0;
    html += '<div class="bar"><span class="nm">' + c.name + '</span>' +
      '<span class="track"><span class="fill" style="width:' + (r * 100).toFixed(1) +
      '%;background:' + c.color + '"></span></span>' +
      '<span class="pc">' + (r * 100).toFixed(0) + '%</span></div>';
  });
  html += '<div class="note">Recall per true class over ' + N + ' fresh moments, ' +
    '"comfortable" F1 <b>' + (f1 * 100).toFixed(1) + '%</b>. ' +
    'Votes in the training set are noisy; this test is not.</div>';
  $('qtResult').innerHTML = html;
}

/* --------------------------------------------------------- live loop */
function renderLoop() {
  $('loopStatus').innerHTML = loopStatusHtml(loopOpt());
  const t = $('tchart');
  drawLoopTemps(dpiSetup(t, t.clientWidth, 170), t.clientWidth, 170);
  const r = $('lribbon');
  drawLoopRibbon(dpiSetup(r, r.clientWidth, 84), r.clientWidth, 84);
}

function setLoop(on) {
  if (on) loopEnsureSim(houseCfg());
  loop.on = on;
  const b = $('loopToggle');
  b.textContent = on ? '⏸ Pause loop' : '▶ Start loop';
  b.classList.toggle('on', on);
  if (on) renderLoop();
}

/* ------------------------------------------------------- UI builders */
function buildFeatureList() {
  const host = $('featList');
  host.innerHTML = '';
  FEATURES.forEach((f) => {
    const row = document.createElement('label');
    row.className = 'featrow' + (state.features[f.id] ? '' : ' off');
    row.title = f.tip;
    row.innerHTML = '<input type="checkbox"' + (state.features[f.id] ? ' checked' : '') + '>' +
      '<span class="nm">' + f.name + (f.unit ? ' <span style="color:#98a2ad">[' + f.unit + ']</span>' : '') + '</span>' +
      '<span class="enc">' + (f.cyc ? 'sin+cos' : '1×') + '</span>';
    row.querySelector('input').onchange = (e) => {
      // at least one input must remain
      if (!e.target.checked && activeIds().length === 1) { e.target.checked = true; return; }
      state.features[f.id] = e.target.checked;
      row.classList.toggle('off', !e.target.checked);
      regenData();
      rebuildModel();
      syncProbeDim();
    };
    host.appendChild(row);
  });
}

/** Grey out probe sliders of inactive features (they still pin the true model). */
function syncProbeDim() {
  FEATURES.forEach((f) => {
    const row = $('probe_' + f.id);
    if (row) row.classList.toggle('dim', !state.features[f.id]);
  });
}

function buildProbeSliders() {
  const host = $('probeSliders');
  host.innerHTML = '';
  FEATURES.forEach((f) => {
    const row = document.createElement('div');
    row.className = 'proberow';
    row.id = 'probe_' + f.id;
    row.innerHTML = '<span class="nm" title="' + f.tip + '">' + f.name + '</span>' +
      '<input type="range" min="' + f.min + '" max="' + f.max + '" step="' + f.step + '" value="' + state.probe[f.id] + '">' +
      '<b>' + fmtFeat(f, state.probe[f.id]) + '</b>';
    const inp = row.querySelector('input');
    inp.oninput = () => {
      state.probe[f.id] = +inp.value;
      row.querySelector('b').textContent = fmtFeat(f, +inp.value);
      markProbeDirty();
      renderNet(); renderMath(true);
    };
    host.appendChild(row);
  });
}

function buildAxisSelects() {
  ['mapX', 'mapY'].forEach((which) => {
    const sel = $(which);
    sel.innerHTML = '';
    FEATURES.forEach((f) => {
      const o = document.createElement('option');
      o.value = f.id; o.textContent = f.name;
      sel.appendChild(o);
    });
    sel.value = state[which];
    sel.onchange = () => {
      const other = which === 'mapX' ? 'mapY' : 'mapX';
      if (sel.value === state[other]) {           // the two axes must differ — swap
        $(other).value = state[which];
        state[other] = state[which];
      }
      state[which] = sel.value;
      markProbeDirty();
      renderNet();
    };
  });
}

function buildLayerControls() {
  const host = $('layerControls');
  host.innerHTML = '';
  state.hidden.forEach((units, li) => {
    const card = document.createElement('div');
    card.className = 'laycard';
    card.title = 'Units in this hidden layer. Each unit is one small learned detector; '
      + 'more units can carve a finer comfort boundary but need more data.';
    card.dataset.layer = li;
    card.innerHTML = '<div class="row"><button data-a="m">−</button><b>' + units +
      ' unit' + (units > 1 ? 's' : '') + '</b><button data-a="p">+</button></div>';
    card.querySelector('[data-a=m]').onclick = () => {
      if (state.hidden[li] > 1) { state.hidden[li]--; rebuildModel(); }
    };
    card.querySelector('[data-a=p]').onclick = () => {
      if (state.hidden[li] < 10) { state.hidden[li]++; rebuildModel(); }
    };
    host.appendChild(card);
  });
}

/** Puts each layer's −/+ card above its column in the diagram. */
function positionLayerControls() {
  if (!layout) return;
  const cards = $('layerControls').children;
  for (const card of cards) {
    const li = +card.dataset.layer;
    const col = layout.cols[1 + li];
    if (!col) continue;
    card.style.left = (col.x + col.nodes[0].w / 2 - card.offsetWidth / 2) + 'px';
  }
}

/* --------------------------------------------------------- arithmetic */
function n3(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(3); }
function wColor(v) { return v >= 0 ? '#c2760f' : '#0877bd'; }

/** Which feature (and sin/cos half) an encoded input index belongs to. */
function encMeta() {
  const out = [];
  activeIds().forEach((id) => {
    const f = FEATURES[FEAT_INDEX[id]];
    if (f.cyc) { out.push({ f, comp: 'sin' }); out.push({ f, comp: 'cos' }); }
    else out.push({ f, comp: null });
  });
  return out;
}

/** Display names of the inputs feeding hidden layer li. */
function layerInputNames(li) {
  if (li === 0) return encodedNames(activeIds());
  return state.hidden.slice(0, li).length
    ? Array.from({ length: state.hidden[li - 1] }, (_, k) => 'h' + li + '·' + (k + 1))
    : [];
}

function renderMath(force) {
  const host = $('mathBody');
  const sel = state.selected;
  if (!sel) {
    $('mathTitle').textContent = 'Arithmetic of the selected node';
    host.innerHTML = '<p class="empty">Click a node in the network diagram — every number of its ' +
      'computation appears here, live. Move the probe sliders and the numbers follow.</p>';
    return;
  }
  if (!pf) probeForward();
  if (sel.kind === 'in') renderInputMath(host, sel);
  else if (sel.kind === 'hid') renderHiddenMath(host, sel);
  else renderOutputMath(host, sel);
}

function renderInputMath(host, sel) {
  const meta = encMeta()[sel.unit];
  const f = meta.f;
  const raw = state.probe[f.id];
  const x = pf.x[sel.unit];
  $('mathTitle').textContent = 'Input · ' + (meta.comp ? meta.comp + ' ' : '') + f.name;
  let h = '<h4>Encoding — raw sensor value → network input</h4>';
  if (meta.comp) {
    h += '<div class="formula">x <span class="op">=</span> ' + meta.comp +
      '(2π · <b>' + fmtFeat(f, raw) + '</b> / ' + f.cyc + ') <span class="op">=</span> ' +
      '<span class="res">' + n3(x) + '</span></div>' +
      '<p class="hint">A clock and a calendar are circles: encoding the ' + f.name.toLowerCase() +
      ' as sin and cos puts 23:59 next to 00:01 and December next to January, ' +
      'which a single number cannot do.</p>';
  } else {
    h += '<div class="formula">x <span class="op">=</span> (<b>' + raw.toFixed(f.fmt) + '</b> − ' +
      f.mean + ') / ' + f.span + ' <span class="op">=</span> <span class="res">' + n3(x) + '</span></div>' +
      '<p class="hint">Standardised so every sensor speaks with a similar magnitude — the raw units ' +
      '(' + (f.unit || 'dimensionless') + ') would otherwise decide which input matters.</p>';
  }
  const d = model.denses[0];
  h += '<h4>Where it goes — weights into hidden layer 1</h4><div class="scrollx"><table class="mtab"><tr><th>unit</th><th>w</th><th>w · x</th></tr>';
  for (let j = 0; j < d.nout; j++) {
    const w = d.W[j * d.nin + sel.unit];
    h += '<tr><td class="ch">h1·' + (j + 1) + '</td><td style="color:' + wColor(w) + '">' + n3(w) +
      '</td><td>' + n3(w * x) + '</td></tr>';
  }
  h += '</table></div>';
  host.innerHTML = h;
}

function renderHiddenMath(host, sel) {
  const li = sel.layer, u = sel.unit;
  const d = model.denses[li];
  const inputs = li === 0 ? pf.x : pf.acts[li - 1];
  const names = layerInputNames(li);
  const z = pf.preacts[li][u];
  const a = pf.acts[li][u];
  $('mathTitle').textContent = 'Hidden unit h' + (li + 1) + '·' + (u + 1);

  let h = '<h4>Weighted sum over its inputs</h4>';
  h += '<div class="scrollx"><table class="mtab"><tr><th>input</th><th>value</th><th>w</th><th>w · value</th></tr>';
  for (let i = 0; i < d.nin; i++) {
    const w = d.W[u * d.nin + i];
    h += '<tr><td class="ch">' + names[i] + '</td><td>' + n3(inputs[i]) + '</td>' +
      '<td style="color:' + wColor(w) + '">' + n3(w) + '</td><td>' + n3(w * inputs[i]) + '</td></tr>';
  }
  h += '<tr><td class="ch">bias</td><td></td><td></td><td>' + n3(d.b[u]) + '</td></tr>';
  h += '<tr><td class="ch sum">z</td><td class="sum" colspan="3">' + n3(z) + '</td></tr>';
  h += '</table></div>';

  h += '<h4>Activation</h4><div class="formula">' + actExpr(z, a) + '</div>';

  const next = li + 1 < model.denses.length ? model.denses[li + 1] : model.out;
  const isOut = next === model.out;
  h += '<h4>Contribution downstream</h4><div class="scrollx"><table class="mtab"><tr><th>' +
    (isOut ? 'class' : 'unit') + '</th><th>w</th><th>w · a</th></tr>';
  for (let j = 0; j < next.nout; j++) {
    const w = next.W[j * next.nin + u];
    const nm = isOut
      ? '<span class="classline"><span class="chip" style="background:' + CLASSES[j].color + '"></span>' + CLASSES[j].name + '</span>'
      : 'h' + (li + 2) + '·' + (j + 1);
    h += '<tr><td class="ch">' + nm + '</td><td style="color:' + wColor(w) + '">' + n3(w) +
      '</td><td>' + n3(w * a) + '</td></tr>';
  }
  h += '</table></div>';
  host.innerHTML = h;
}

function actExpr(z, a) {
  const k = state.activation;
  if (k === 'relu') return 'a <span class="op">=</span> max(0, ' + n3(z) + ') <span class="op">=</span> <span class="res">' + n3(a) + '</span>';
  if (k === 'tanh') return 'a <span class="op">=</span> tanh(' + n3(z) + ') <span class="op">=</span> <span class="res">' + n3(a) + '</span>';
  if (k === 'leaky') return 'a <span class="op">=</span> ' + (z > 0 ? n3(z) : '0.1 · ' + n3(z)) + ' <span class="op">=</span> <span class="res">' + n3(a) + '</span>';
  return 'a <span class="op">=</span> 1/(1+e<sup>−z</sup>) <span class="op">=</span> <span class="res">' + n3(a) + '</span>';
}

function renderOutputMath(host, sel) {
  const c = sel.unit;
  const d = model.out;
  const li = model.denses.length - 1;
  const inputs = pf.acts[li];
  const z = pf.logits;
  $('mathTitle').textContent = 'Output · ' + CLASSES[c].name;

  let h = '<h4>Logit — weighted sum over the last hidden layer</h4>';
  h += '<div class="scrollx"><table class="mtab"><tr><th>input</th><th>a</th><th>w</th><th>w · a</th></tr>';
  for (let i = 0; i < d.nin; i++) {
    const w = d.W[c * d.nin + i];
    h += '<tr><td class="ch">h' + (li + 1) + '·' + (i + 1) + '</td><td>' + n3(inputs[i]) + '</td>' +
      '<td style="color:' + wColor(w) + '">' + n3(w) + '</td><td>' + n3(w * inputs[i]) + '</td></tr>';
  }
  h += '<tr><td class="ch">bias</td><td></td><td></td><td>' + n3(d.b[c]) + '</td></tr>';
  h += '<tr><td class="ch sum">z</td><td class="sum" colspan="3">' + n3(z[c]) + '</td></tr></table></div>';

  h += '<h4>Softmax over the three votes</h4>';
  h += '<div class="scrollx"><table class="mtab"><tr><th>class</th><th>z</th><th>e^z</th><th>p</th></tr>';
  let sum = 0;
  const mx = Math.max(z[0], z[1], z[2]);
  for (let j = 0; j < 3; j++) sum += Math.exp(z[j] - mx);
  for (let j = 0; j < 3; j++) {
    h += '<tr' + (j === c ? ' style="background:#f7faf8"' : '') + '><td class="ch"><span class="classline">' +
      '<span class="chip" style="background:' + CLASSES[j].color + '"></span>' + CLASSES[j].name + '</span></td>' +
      '<td>' + n3(z[j]) + '</td><td>' + n3(Math.exp(z[j] - mx)) + '</td>' +
      '<td><b>' + (pf.probs[j] * 100).toFixed(1) + '%</b></td></tr>';
  }
  h += '</table></div>';

  const truth = comfortTruth(state.probe, state.pref);
  const loss = -Math.log(Math.max(1e-9, pf.probs[truth.label]));
  h += '<h4>Against the true model at this moment</h4>' +
    '<div class="formula">true vote <span class="op">=</span> <b>' + CLASSES[truth.label].name +
    '</b> (PMV ' + (truth.pmv >= 0 ? '+' : '−') + Math.abs(truth.pmv).toFixed(2) + ')' +
    ' <span class="op">→</span> cross-entropy <span class="op">=</span> −ln p<sub>true</sub> ' +
    '<span class="op">=</span> <span class="res' + (loss > 1 ? ' warn' : '') + '">' + loss.toFixed(3) + '</span></div>';
  host.innerHTML = h;
}

/* ------------------------------------------------------------ wiring */
function bindUI() {
  $('btnPlay').onclick = () => {
    if (!state.running) armRun();
    setRunning(!state.running);
  };
  $('runEpochs').oninput = () => {
    const v = parseInt($('runEpochs').value, 10);
    state.runEpochs = isFinite(v) && v > 0 ? v : 0;
    if (state.running) armRun();
  };
  $('runEpochs').onchange = () => { $('runEpochs').blur(); };
  $('btnStep').onclick = () => {
    setRunning(false);
    const batches = Math.ceil(train.n / state.batch);
    for (let i = 0; i < batches; i++) trainOneBatch();
    markTrained();
    evaluate(); renderMetrics(); renderMath(true);
  };
  $('btnReset').onclick = () => {
    setRunning(false);
    rebuildModel();
  };
  $('lr').onchange = (e) => { state.lr = +e.target.value; };
  $('l2').onchange = (e) => { state.l2 = +e.target.value; };
  $('batch').onchange = (e) => { state.batch = +e.target.value; };
  $('act').onchange = (e) => { state.activation = e.target.value; rebuildModel(); };

  $('layPlus').onclick = () => {
    if (state.hidden.length < 4) { state.hidden.push(4); rebuildModel(); }
  };
  $('layMinus').onclick = () => {
    if (state.hidden.length > 1) { state.hidden.pop(); rebuildModel(); }
  };

  // data panel
  $('voteSource').onchange = (e) => { state.voteSource = e.target.value; regenData(); evaluate(); renderMetrics(); };
  $('coverage').onchange = (e) => { state.coverage = e.target.value; regenData(); evaluate(); renderMetrics(); };
  const pref = $('pref'), sigma = $('sigma'), snoise = $('snoise'), nex = $('nex');
  pref.oninput = () => {
    state.pref = +pref.value;
    $('prefVal').textContent = state.pref === 0 ? 'neutral'
      : (state.pref > 0 ? '+' : '−') + Math.abs(state.pref).toFixed(2) + (state.pref > 0 ? ' warmer' : ' cooler');
    mapTruthDirty = true; mapDrawDirty = true;
    renderReadout();
  };
  pref.onchange = () => { regenData(); evaluate(); renderMetrics(); };
  sigma.oninput = () => { state.sigma = +sigma.value; $('sigmaVal').textContent = state.sigma.toFixed(2); };
  sigma.onchange = () => { regenData(); evaluate(); renderMetrics(); };
  snoise.oninput = () => { state.sensorNoise = +snoise.value; $('snoiseVal').textContent = state.sensorNoise.toFixed(1); };
  snoise.onchange = () => { regenData(); evaluate(); renderMetrics(); };
  nex.oninput = () => { state.nex = +nex.value; $('nexVal').textContent = state.nex; };
  nex.onchange = () => { regenData(); evaluate(); renderMetrics(); };
  $('balance').onchange = (e) => { state.balance = e.target.checked; regenData(); evaluate(); renderMetrics(); };
  $('btnData').onclick = () => { regenData(); evaluate(); renderMetrics(); };

  // map panel
  $('showTruth').onchange = (e) => { state.showTruth = e.target.checked; mapTruthDirty = true; mapDrawDirty = true; };
  $('showVotes').onchange = (e) => { state.showVotes = e.target.checked; sliceVotesDirty = true; mapDrawDirty = true; };
  $('allVotes').onchange = (e) => { state.allVotes = e.target.checked; sliceVotesDirty = true; mapDrawDirty = true; };
  $('btnRandState').onclick = () => { setProbe(randomState(state.coverage)); };
  $('btnSimState').onclick = () => {
    if (loop.sim) setProbe(sensorState(loop.sim, 0));
  };
  $('btnQuick').onclick = quickTest;

  const mapC = $('map');
  mapC.onclick = (e) => {
    const r = mapC.getBoundingClientRect();
    const padL = 40, padB = 26, padT = 8, padR = 10;
    const pw = r.width - padL - padR, ph = r.height - padT - padB;
    const tx = clamp((e.clientX - r.left - padL) / pw, 0, 1);
    const ty = clamp(1 - (e.clientY - r.top - padT) / ph, 0, 1);
    const patch = {};
    patch[state.mapX] = Math.round(axisValue(featX(), tx) / featX().step) * featX().step;
    patch[state.mapY] = Math.round(axisValue(featY(), ty) / featY().step) * featY().step;
    setProbe(patch);
  };

  // network canvas
  const net = $('net');
  net.onmousemove = (e) => {
    const r = net.getBoundingClientRect();
    state.hover = layout ? hitTest(layout, e.clientX - r.left, e.clientY - r.top) : null;
    net.style.cursor = state.hover ? 'pointer' : 'crosshair';
  };
  net.onmouseleave = () => { state.hover = null; };
  net.onclick = () => {
    state.selected = state.hover ? Object.assign({}, state.hover) : null;
    renderMath(true);
  };
  $('btnClearSel').onclick = () => { state.selected = null; renderMath(true); };

  // control loop
  $('loopToggle').onclick = () => setLoop(!loop.on);
  $('loopSpeed').onchange = (e) => { loop.speed = +e.target.value; };
  $('loopMode').onchange = (e) => {
    loop.mode = e.target.value;
    $('setpointWrap').style.display = loop.mode === 'thermo' ? '' : 'none';
    $('manualWrap').style.display = loop.mode === 'manual' ? '' : 'none';
  };
  const setp = $('setpoint'), man = $('manual');
  setp.oninput = () => { loop.setpoint = +setp.value; $('setpointVal').textContent = loop.setpoint.toFixed(1) + '°'; };
  man.oninput = () => { loop.manual = +man.value; $('manualVal').textContent = Math.round(loop.manual * 100) + '%'; };
  $('heaterType').onchange = (e) => { if (loop.sim) loop.sim.cfg.heater = e.target.value; };
  $('pmax').onchange = (e) => { if (loop.sim) loop.sim.cfg.pmax = +e.target.value; };
  const insul = $('insul');
  insul.oninput = () => {
    $('insulVal').textContent = insul.value;
    if (loop.sim) loop.sim.cfg.insulation = +insul.value;
  };
  $('startMonth').onchange = () => { loopRestart(houseCfg()); renderLoop(); };
  $('btnWindow').onclick = () => {
    loopEnsureSim(houseCfg());
    loop.sim.window = !loop.sim.window;
    $('btnWindow').classList.toggle('on', loop.sim.window);
    renderLoop();
  };
  $('btnParty').onclick = () => {
    loopEnsureSim(houseCfg());
    loop.sim.guests += 4;
    loop.sim.guestUntil = loop.sim.minute + randInt(90, 150);
    renderLoop();
  };
  $('learnVotes').onchange = (e) => { loop.learn = e.target.checked; };

  window.addEventListener('resize', () => {
    mapDrawDirty = true;
    renderNet(); renderMetrics();
    if (loop.hist.length) renderLoop();
  });
}

/* -------------------------------------------------------------- init */
function init() {
  buildFeatureList();
  buildProbeSliders();
  buildAxisSelects();
  bindUI();
  regenData();
  rebuildModel();
  syncProbeDim();
  renderNet();
  renderMath(true);
  requestAnimationFrame(mainLoop);
}

init();
