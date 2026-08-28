/* main.js — application state, UI wiring and the training loop. */

const state = {
  // the default is the cleanest experiment: two inputs, controlled study
  features: { ta: true, rh: false, tw: false, tout: false, hour: true, doy: false, mov: false, vair: false, hvac: false },
  arch: 'cnn',         // 'cnn' — 1D convolution; 'rnn' — recurrence over the last hour
  cnnLayers: [{ filters: 4, kernel: 5 }, { filters: 4, kernel: 5 }],
  rnnLayers: [{ units: 6 }],
  cell: 'gru',         // rnn | gru | lstm
  readout: 'mean',     // RNN readout: mean | last | max
  head: 'gap',       // CNN output head: gap | gmp | flat
  sensorDropout: false,  // training augmentation: randomly blind one channel
  failSensor: '',    // live loop: this sensor's channels arrive zeroed
  activation: 'relu',
  lr: 0.003,
  l2: 0,
  batch: 16,
  coverage: 'year',
  voteSource: 'uniform',   // 'sim' — lived-in household; 'uniform' — controlled study
  studyDoy: 15,        // the study's season, when the season input is off
  studyHour: 19.5,     // the study's hour, when the time input is off
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
  mapX: 'hour',
  mapY: 'ta',
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

/**
 * Width available to a canvas, measured from its PARENT. Asking the canvas for
 * its own clientWidth after dpiSetup fixed an inline width creates a feedback
 * loop: at browser zoom levels other than 100% the rounding loses a pixel per
 * redraw and the chart slowly shrinks to nothing.
 */
function availWidth(el) {
  const par = el.parentElement;
  const cs = getComputedStyle(par);
  return Math.max(60, Math.round(
    par.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)));
}

function activeIds() { return FEATURES.map((f) => f.id).filter((id) => state.features[id]); }
/** Both architectures read the multi-channel window of the last hour. */
function dataX(ds) { return ds.ws; }
function encProbe(s) { return flatWindow(s, activeIds()); }
/** The live loop's encoder — with a simulated dead sensor when one is chosen. */
function encLive(s) {
  const ids = activeIds();
  const w = flatWindow(s, ids);
  return state.failSensor ? zeroFeatureChannels(w, ids, state.failSensor) : w;
}
function featX() { return FEATURES[FEAT_INDEX[state.mapX]]; }
function featY() { return FEATURES[FEAT_INDEX[state.mapY]]; }

function dataOpt() {
  return {
    coverage: state.coverage, pref: state.pref, sigma: state.sigma,
    sensorNoise: state.sensorNoise, balance: state.balance,
    pin: { ta: 21, hour: state.studyHour, doy: state.studyDoy },
    insulation: +$('insul').value, pmax: +$('pmax').value, heater: $('heaterType').value,
  };
}
function houseCfg() {
  return {
    doy: +$('startMonth').value, insulation: +$('insul').value,
    pmax: +$('pmax').value, coolMax: +$('coolMax').value,
    heater: $('heaterType').value, residents: 2,
  };
}

/** Everything that depends on the weights is stale after a training step. */
function markTrained() { gridsDirty = true; mapNetDirty = true; }
/** Everything that depends on the probe / axes / preference is stale. */
function markProbeDirty() {
  gridsDirty = true; mapNetDirty = true; mapTruthDirty = true;
  sliceVotesDirty = true; mapDrawDirty = true;
}

/**
 * The controlled study's deterministic world: inactive time anchors come from
 * the explicit Study season / Study hour settings in the Data panel — never
 * from the probe sliders, which only inspect — and every other unseen sensor
 * follows the typical household.
 */
function studyFill(s, keep) {
  const a = {};
  keep.forEach((id) => (a[id] = true));
  const o = Object.assign({}, s);
  if (!a.doy) o.doy = state.studyDoy;
  if (!a.hour) o.hour = state.studyHour;
  return typicalFill(o, keep);
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
  if (state.arch === 'rnn') {
    model = new RNNNet({
      cell: state.cell,
      channels: encodedDim(ids),
      T: WIN_T,
      layers: JSON.parse(JSON.stringify(state.rnnLayers)),
      readout: state.readout,
      nClasses: CLASSES.length,
    });
  } else {
    model = new ConvNet1D({
      channels: encodedDim(ids),
      T: WIN_T,
      layers: JSON.parse(JSON.stringify(state.cnnLayers)),
      activation: state.activation,
      head: state.head,
      nClasses: CLASSES.length,
    });
  }
  state.epoch = 0; state.stopAt = null;
  hTrain = []; hTest = [];
  state.selected = null; state.hover = null;
  markTrained(); mapDrawDirty = true;
  $('layerLbl').textContent = state.arch === 'rnn' ? 'RNN layers' : 'Conv layers';
  $('layCount').textContent = state.arch === 'rnn' ? state.rnnLayers.length : state.cnnLayers.length;
  $('paramCount').textContent = model.paramCount().toLocaleString('en-US') + ' parameters';
  buildLayerControls();
  $('qtResult').innerHTML = '';        // that score belonged to the old weights
  evaluate(); renderMetrics(); renderRunTarget(); renderMath(true);
}

function setArch(a) {
  state.arch = a;
  document.body.classList.toggle('arch-cnn', a === 'cnn');
  document.body.classList.toggle('arch-rnn', a === 'rnn');
  $('archCnn').classList.toggle('on', a === 'cnn');
  $('archRnn').classList.toggle('on', a === 'rnn');
  setRunning(false);
  rebuildModel();
}

/* ---------------------------------------------------------- training */
function trainOneBatch() {
  const B = state.batch;
  const idx = new Array(B);
  for (let i = 0; i < B; i++) idx[i] = Math.floor(Math.random() * train.n);
  if (!state.sensorDropout) {
    model.trainBatch(dataX(train), train.ys, idx, state.lr, state.l2);
  } else {
    // sensor dropout: ~a third of the examples lose one random channel, so the
    // network learns to answer from any subset — a dead sensor later degrades
    // the thermostat gracefully instead of breaking it
    const ids = activeIds();
    const X = dataX(train);
    const xs = [], ys = [];
    for (const i of idx) {
      xs.push(Math.random() < 0.35
        ? zeroFeatureChannels(X[i], ids, ids[randInt(0, ids.length)])
        : X[i]);
      ys.push(train.ys[i]);
    }
    model.trainBatch(xs, ys, xs.map((_, k) => k), state.lr, state.l2);
  }
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
  // a pass over 32 timepoints is expensive — sample fewer examples per curve point
  const la = 120, lb = 200;
  const a = model.evaluate(dataX(train), train.ys, 3, la);
  const b = model.evaluate(dataX(test), test.ys, 3, lb);
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
  loop.failName = state.failSensor ? FEATURES[FEAT_INDEX[state.failSensor]].name : '';
  return { pref: state.pref, sigma: state.sigma, onVote: onLiveVote, enc: encLive };
}

/** Online learning: a live vote becomes a training example on the spot. */
function onLiveVote(v) {
  const ids = activeIds();
  train.xs.push(encodeState(v.state, ids));
  train.ws.push(flatWindow(v.state, ids));
  train.ys.push(v.label);
  train.states.push(v.state);
  train.pmvs.push(v.pmv);
  if (train.xs.length > 4000) {
    train.xs.shift(); train.ws.shift(); train.ys.shift(); train.states.shift(); train.pmvs.shift();
  }
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
  // hidden units are drawn as state/output curves at the probe; only the class
  // nodes get response maps over the plane
  const g = { inputs: [], hidden: [], outputs: [] };
  for (let c = 0; c < CLASSES.length; c++) g.outputs.push(new Float32Array(GRES * GRES));
  const s = Object.assign({}, state.probe);
  for (let gy = 0; gy < GRES; gy++) {
    s[fy.id] = axisValue(fy, gy / (GRES - 1));
    for (let gx = 0; gx < GRES; gx++) {
      s[fx.id] = axisValue(fx, gx / (GRES - 1));
      const p = model.forward(flatWindow(s, ids), false);
      const cell = gy * GRES + gx;
      for (let c = 0; c < CLASSES.length; c++) g.outputs[c][cell] = p[c];
    }
  }
  grids = g;
  gridsDirty = false;
}

/** Forward pass at the probe itself — activations for readout and arithmetic. */
function probeForward() {
  const x = encProbe(state.probe);
  const p = model.forward(x, true);
  pf = {
    x, acts: model.acts, logits: model.logits.slice(), probs: p.slice(),
    preacts: model.preacts, embedding: model.embedding,
  };
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
  if (gridsDirty && frameNo % 15 === 0) computeGrids();
  probeForward();
  const fx = featX(), fy = featY();
  const inVals = Array.from({ length: encodedDim(activeIds()) }, (_, c) => pf.x[c * WIN_T + WIN_T - 1]);
  drawNetwork(netCtx, {
    model, layout, grids,
    window: pf.x,
    winT: WIN_T,
    inNames: names,
    inValues: inVals,
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
  const lossC = $('loss'), lw = availWidth(lossC);
  drawLossChart(dpiSetup(lossC, lw, 110), lw, 110, hTrain, hTest);
  const confC = $('conf'), cw = availWidth(confC);
  drawConfusion(dpiSetup(confC, cw, 150), cw, 150, lastMetrics.test.conf, CLASSES);
}

/* ------------------------------------------------------- comfort map */
function renderMapMaybe() {
  const busy = state.running || loop.on;
  const slow = busy ? frameNo % 15 === 0 : true;
  let changed = false;
  if (mapNetDirty && slow) {
    mapNet = mapEvalNetWin(model, activeIds(), state.probe, featX(), featY());
    mapNetDirty = false; changed = true;
  }
  if (mapTruthDirty && slow) {
    const fill = state.voteSource === 'uniform'
      ? (s) => studyFill(s, activeIds().concat([state.mapX, state.mapY]))
      : null;
    mapTruth = state.showTruth ? mapEvalTruth(state.probe, featX(), featY(), state.pref, fill) : null;
    mapTruthDirty = false; changed = true;
  }
  if (sliceVotesDirty && slow) {
    const ref = state.voteSource === 'uniform'
      ? studyFill(state.probe, activeIds().concat([state.mapX, state.mapY]))
      : state.probe;
    sliceVotes = state.showVotes
      ? votesNearSlice(train, activeIds(), ref, featX(), featY(), state.allVotes)
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
/** The state the TRUE model is asked about at the probe. In controlled-study
 * mode the unseen sensors follow the study's typical household — same as the
 * generated votes — so the verdict compares like with like. */
function probeTruthState() {
  return state.voteSource === 'uniform'
    ? studyFill(state.probe, activeIds())
    : state.probe;
}

function renderReadout() {
  if (!pf) return;
  const ts = probeTruthState();
  const truth = comfortTruth(ts, state.pref);
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
    truth.clo.toFixed(2) + ' clo' + (sleepHours(ts.hour) && ts.mov < 0.12 ? ' (asleep, in bed)' : '') +
    (state.voteSource === 'uniform' ? ' · unseen sensors at the study\'s typical values' : '') +
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
/** A fresh moment drawn from the SAME world the training votes came from:
 * household-plausible in simulation mode, the controlled study's sweep in
 * uniform mode. Testing on a different world would measure the wrong thing. */
function quickState(ids) {
  if (state.voteSource !== 'uniform') return randomState(state.coverage);
  const s = Object.assign({}, state.probe);
  ids.forEach((id) => {
    const f = FEATURES[FEAT_INDEX[id]];
    if (id === 'hour') s.hour = rand(0, 24);
    else if (id === 'doy') s.doy = pickDoy(state.coverage);
    else s[id] = rand(f.min, f.max);
  });
  return studyFill(s, ids);
}

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
    const s = quickState(ids);
    const y = comfortTruth(s, state.pref).label;
    const a = argmax(model.forward(encProbe(s), false));
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
  const draw = (id, h, fn) => {
    const c = $(id);
    const w = availWidth(c);
    const ctx = dpiSetup(c, w, h);
    try {
      fn(ctx, w, h);
    } catch (err) {
      // never a silently blank chart: name the problem where it happened
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#b0561d';
      ctx.font = '11px system-ui,sans-serif';
      ctx.fillText('chart error: ' + err.message, 8, 18);
    }
  };
  draw('tchart', 170, drawLoopTemps);
  draw('lribbon', 84, drawLoopRibbon);
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
  syncStudyControls();
}

/** The study's season/hour settings appear only when they are actually used:
 * controlled mode, with the corresponding input switched off. */
function syncStudyControls() {
  const u = state.voteSource === 'uniform';
  $('studyDayWrap').style.display = u && !state.features.doy ? '' : 'none';
  $('studyHourWrap').style.display = u && !state.features.hour ? '' : 'none';
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
  const cnn = state.arch === 'cnn';
  const list = cnn ? state.cnnLayers : state.rnnLayers;
  list.forEach((entry, li) => {
    const count = cnn ? entry.filters : entry.units;
    const card = document.createElement('div');
    card.className = 'laycard';
    card.title = cnn
      ? 'Filters in this convolutional layer. Each filter slides one small learned '
        + 'kernel along the hour of history, looking for a temporal pattern.'
      : 'Recurrent units in this layer. Each unit is one little memory, updated '
        + 'minute by minute as the hour is read.';
    card.dataset.layer = li;
    card.innerHTML = '<div class="row"><button data-a="m">−</button><b>' + count +
      (cnn ? ' filter' : ' unit') + (count > 1 ? 's' : '') + '</b><button data-a="p">+</button></div>';
    card.querySelector('[data-a=m]').onclick = () => {
      if (cnn) { if (entry.filters > 1) { entry.filters--; rebuildModel(); } }
      else if (entry.units > 1) { entry.units--; rebuildModel(); }
    };
    card.querySelector('[data-a=p]').onclick = () => {
      if (cnn) { if (entry.filters < 8) { entry.filters++; rebuildModel(); } }
      else if (entry.units < 10) { entry.units++; rebuildModel(); }
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
  if (sel.kind === 'in') renderChannelMath(host, sel);
  else if (sel.kind === 'hid') {
    if (state.arch === 'rnn') renderUnitMath(host, sel);
    else renderFilterMath(host, sel);
  } else renderOutputMath(host, sel);
}

/* ------------------------------------------------- arithmetic · RNN */
const GATE_NAMES = {
  gru: ['z (update)', 'r (reset)', 'n (candidate)'],
  lstm: ['i (input)', 'f (forget)', 'o (output)', 'g (candidate)'],
  rnn: ['h'],
};

function renderUnitMath(host, sel) {
  const li = sel.layer, u = sel.unit;
  const dir = model.layers[li];
  const inNames = li === 0
    ? encodedNames(activeIds())
    : Array.from({ length: model.cfg.layers[li - 1].units }, (_, k) => 'h' + li + '·' + (k + 1));
  const gnames = GATE_NAMES[model.cfg.cell];
  $('mathTitle').textContent = 'Unit h' + (li + 1) + '·' + (u + 1) + ' · ' + model.cfg.cell.toUpperCase();

  let h = '<h4>Input weights — one row of taps per gate</h4>' +
    '<div class="scrollx"><table class="mtab"><tr><th>gate</th>';
  inNames.forEach((nm) => { h += '<th>' + nm + '</th>'; });
  h += '<th>bias</th></tr>';
  for (let g = 0; g < dir.G; g++) {
    h += '<tr><td class="ch">' + gnames[g] + '</td>';
    for (let d = 0; d < dir.D; d++) {
      const w = dir.px.W[(g * dir.H + u) * dir.D + d];
      h += '<td style="color:' + wColor(w) + '">' + n3(w) + '</td>';
    }
    h += '<td>' + n3(dir.px.b[g * dir.H + u]) + '</td></tr>';
  }
  h += '</table></div>';

  h += '<h4>Recurrent weights — what it reads from its own layer\'s past</h4>' +
    '<div class="scrollx"><table class="mtab"><tr><th>gate</th>';
  for (let v = 0; v < dir.H; v++) h += '<th>h·' + (v + 1) + '</th>';
  h += '</tr>';
  for (let g = 0; g < dir.G; g++) {
    h += '<tr><td class="ch">' + gnames[g] + '</td>';
    for (let v = 0; v < dir.H; v++) {
      const w = dir.ph.W[(g * dir.H + u) * dir.H + v];
      h += '<td style="color:' + wColor(w) + '">' + n3(w) + '</td>';
    }
    h += '</tr>';
  }
  h += '</table></div>';

  const cell = model.cfg.cell;
  const upd = cell === 'gru' ? 'h = (1−z)·n + z·h<sub>prev</sub>'
    : cell === 'lstm' ? 'c = f·c<sub>prev</sub> + i·g &nbsp;·&nbsp; h = o·tanh(c)'
    : 'h = tanh(W<sub>x</sub>x + W<sub>h</sub>h<sub>prev</sub> + b)';
  const a = pf.acts[li];
  let mean = 0;
  for (let t = 0; t < WIN_T; t++) mean += a[u * WIN_T + t];
  mean /= WIN_T;
  h += '<h4>The state update, every minute of the hour</h4>' +
    '<div class="formula">' + upd + '</div>' +
    '<p class="hint">The curve in the unit\'s box is h(t) over the window for the probe. Final state ' +
    '<span class="res">' + n3(a[u * WIN_T + WIN_T - 1]) + '</span>, time-average ' +
    '<span class="res">' + n3(mean) + '</span> — the readout (' + model.cfg.readout +
    ') decides which of these the vote sees.</p>';
  host.innerHTML = h;
}

/* ------------------------------------------------ arithmetic · 1D CNN */
function renderChannelMath(host, sel) {
  const meta = encMeta()[sel.unit];
  const f = meta.f;
  const raw = state.probe[f.id];
  const xEnd = pf.x[sel.unit * WIN_T + WIN_T - 1];
  $('mathTitle').textContent = 'Channel · ' + (meta.comp ? meta.comp + ' ' : '') + f.name;
  let h = '<h4>One channel of the window</h4>' +
    '<p class="hint">This channel carries the last ' + Math.round(WIN_T * 2) +
    ' minutes of <b>' + f.name + '</b>, one encoded value every 2 minutes. ' +
    'The newest sample (right edge of the box) is:</p>';
  if (meta.comp) {
    h += '<div class="formula">x <span class="op">=</span> ' + meta.comp +
      '(2π · <b>' + fmtFeat(f, raw) + '</b> / ' + f.cyc + ') <span class="op">=</span> ' +
      '<span class="res">' + n3(xEnd) + '</span></div>';
  } else {
    h += '<div class="formula">x <span class="op">=</span> (<b>' + raw.toFixed(f.fmt) + '</b> − ' +
      f.mean + ') / ' + f.span + ' <span class="op">=</span> <span class="res">' + n3(xEnd) + '</span></div>';
  }
  if (state.arch === 'rnn') {
    const dir = model.layers[0];
    const gnames = GATE_NAMES[model.cfg.cell];
    h += '<h4>How the first RNN layer reads this channel — weight per gate</h4>' +
      '<div class="scrollx"><table class="mtab"><tr><th>unit</th>';
    for (let g = 0; g < dir.G; g++) h += '<th>' + gnames[g] + '</th>';
    h += '</tr>';
    for (let u = 0; u < dir.H; u++) {
      h += '<tr><td class="ch">h1·' + (u + 1) + '</td>';
      for (let g = 0; g < dir.G; g++) {
        const w = dir.px.W[(g * dir.H + u) * dir.D + sel.unit];
        h += '<td style="color:' + wColor(w) + '">' + n3(w) + '</td>';
      }
      h += '</tr>';
    }
    h += '</table></div><p class="hint">Every minute of the hour, each unit feeds this channel\'s ' +
      'value through these weights into its gates — deciding what to write into its memory ' +
      'and what to keep from the past.</p>';
    host.innerHTML = h;
    return;
  }
  const conv = model.convs[0];
  h += '<h4>Kernel taps reading this channel — Conv 1</h4>' +
    '<div class="scrollx"><table class="mtab"><tr><th>filter</th>';
  for (let j = 0; j < conv.k; j++) h += '<th>t' + (j - (conv.k >> 1)) + '</th>';
  h += '</tr>';
  for (let co = 0; co < conv.cout; co++) {
    h += '<tr><td class="ch">c1·' + (co + 1) + '</td>';
    const wb = (co * conv.cin + sel.unit) * conv.k;
    for (let j = 0; j < conv.k; j++) {
      const w = conv.W[wb + j];
      h += '<td style="color:' + wColor(w) + '">' + n3(w) + '</td>';
    }
    h += '</tr>';
  }
  h += '</table></div><p class="hint">Each filter slides these taps along the hour: a pattern like ' +
    '−,−,+,+ reads "was low, became high" — a rise the snapshot network cannot represent.</p>';
  host.innerHTML = h;
}

function renderFilterMath(host, sel) {
  const li = sel.layer, u = sel.unit;
  const conv = model.convs[li];
  const inNames = li === 0
    ? encodedNames(activeIds())
    : Array.from({ length: model.cfg.layers[li - 1].filters }, (_, k) => 'c' + li + '·' + (k + 1));
  $('mathTitle').textContent = 'Filter c' + (li + 1) + '·' + (u + 1);
  let h = '<h4>The learned kernel — ' + conv.k + ' taps over every input channel</h4>' +
    '<div class="scrollx"><table class="mtab"><tr><th>input channel</th>';
  for (let j = 0; j < conv.k; j++) h += '<th>t' + (j - (conv.k >> 1)) + '</th>';
  h += '</tr>';
  for (let ci = 0; ci < conv.cin; ci++) {
    h += '<tr><td class="ch">' + inNames[ci] + '</td>';
    const wb = (u * conv.cin + ci) * conv.k;
    for (let j = 0; j < conv.k; j++) {
      const w = conv.W[wb + j];
      h += '<td style="color:' + wColor(w) + '">' + n3(w) + '</td>';
    }
    h += '</tr>';
  }
  h += '<tr><td class="ch">bias</td><td colspan="' + conv.k + '">' + n3(conv.b[u]) + '</td></tr>';
  h += '</table></div>';
  let mean = 0;
  const map = pf.acts[li];
  for (let t = 0; t < WIN_T; t++) mean += map[u * WIN_T + t];
  mean /= WIN_T;
  h += '<p class="hint">At every position t of the window the filter computes ' +
    'Σ w·x over this little table centred at t, adds the bias and applies the activation — ' +
    'the curve in its box is that output. Its time-average is ' +
    '<span class="res">' + n3(mean) + '</span>' +
    (li === model.convs.length - 1
      ? (model.headKind === 'gap' ? ' — after global average pooling, the single number the vote layer sees from this filter.'
        : model.headKind === 'gmp' ? '; with the Global Max Pool head the vote instead sees the curve\'s PEAK.'
        : '; with the Flatten head the vote sees this whole curve, position by position.')
      : '.') + '</p>';
  host.innerHTML = h;
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
  const rnn = state.arch === 'rnn';
  const inputs = pf.embedding;
  const L = rnn ? model.layers.length : model.convs.length;
  const pre = rnn
    ? (model.headKind === 'last' ? 'last h' : model.headKind === 'max' ? 'max h' : 'avg h')
    : (model.headKind === 'gmp' ? 'max c' : 'avg c');
  const nameOf = !rnn && model.headKind === 'flat'
    ? (i) => 'c' + L + '·' + (Math.floor(i / WIN_T) + 1) + ' t' + (i % WIN_T)
    : (i) => pre + L + '·' + (i + 1);
  const z = pf.logits;
  $('mathTitle').textContent = 'Output · ' + CLASSES[c].name;

  let h = '<h4>Logit — weighted sum over ' +
    (rnn
      ? (model.headKind === 'last' ? 'each unit\'s final state (Last state)'
        : model.headKind === 'max' ? 'each unit\'s strongest state (Max over time)'
        : 'the time-averaged states (Mean over time)')
      : model.headKind === 'flat' ? 'every filter at every position (Flatten)'
      : model.headKind === 'gmp' ? 'each filter\'s loudest moment (Global Max Pool)'
      : 'the time-averaged filters (Global Avg Pool)') + '</h4>';
  h += '<div class="scrollx"><table class="mtab"><tr><th>input</th><th>a</th><th>w</th><th>w · a</th></tr>';
  for (let i = 0; i < d.nin; i++) {
    const w = d.W[c * d.nin + i];
    h += '<tr><td class="ch">' + nameOf(i) + '</td><td>' + n3(inputs[i]) + '</td>' +
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

  const truth = comfortTruth(probeTruthState(), state.pref);
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
  $('head').onchange = (e) => { state.head = e.target.value; rebuildModel(); };

  $('layPlus').onclick = () => {
    if (state.arch === 'rnn') {
      if (state.rnnLayers.length < 2) { state.rnnLayers.push({ units: 4 }); rebuildModel(); }
    } else if (state.cnnLayers.length < 3) { state.cnnLayers.push({ filters: 4, kernel: 5 }); rebuildModel(); }
  };
  $('layMinus').onclick = () => {
    if (state.arch === 'rnn') {
      if (state.rnnLayers.length > 1) { state.rnnLayers.pop(); rebuildModel(); }
    } else if (state.cnnLayers.length > 1) { state.cnnLayers.pop(); rebuildModel(); }
  };
  $('archCnn').onclick = () => { if (state.arch !== 'cnn') setArch('cnn'); };
  $('archRnn').onclick = () => { if (state.arch !== 'rnn') setArch('rnn'); };
  $('cell').onchange = (e) => { state.cell = e.target.value; rebuildModel(); };
  $('readout').onchange = (e) => { state.readout = e.target.value; rebuildModel(); };

  // data panel
  $('voteSource').onchange = (e) => {
    state.voteSource = e.target.value;
    syncStudyControls();
    regenData(); evaluate(); renderMetrics(); markProbeDirty();
  };
  $('studyDoy').onchange = (e) => {
    state.studyDoy = +e.target.value;
    regenData(); evaluate(); renderMetrics(); markProbeDirty();
  };
  $('studyHour').onchange = (e) => {
    state.studyHour = +e.target.value;
    regenData(); evaluate(); renderMetrics(); markProbeDirty();
  };
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
  $('coolMax').onchange = (e) => { if (loop.sim) loop.sim.cfg.coolMax = +e.target.value; };
  $('output').onchange = (e) => { loop.output = e.target.value; };
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
  $('sensorDrop').onchange = (e) => { state.sensorDropout = e.target.checked; };
  $('failSensor').onchange = (e) => { state.failSensor = e.target.value; if (loop.hist.length) renderLoop(); };

  window.addEventListener('resize', () => {
    mapDrawDirty = true;
    renderNet(); renderMetrics();
    if (loop.hist.length) renderLoop();
  });
}

/* -------------------------------------------------------------- init */
function buildFailSelect() {
  const sel = $('failSensor');
  sel.innerHTML = '<option value="">none — all sensors fine</option>';
  FEATURES.forEach((f) => {
    const o = document.createElement('option');
    o.value = f.id; o.textContent = f.name;
    sel.appendChild(o);
  });
}

function init() {
  buildFailSelect();
  document.body.classList.add('arch-' + state.arch);
  $('archCnn').classList.toggle('on', state.arch === 'cnn');
  $('archRnn').classList.toggle('on', state.arch === 'rnn');
  $('voteSource').value = state.voteSource;
  $('studyDoy').value = state.studyDoy;
  $('studyHour').value = state.studyHour;
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
