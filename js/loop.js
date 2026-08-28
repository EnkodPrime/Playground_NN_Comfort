/* loop.js — the closed control loop: the trained network runs the heater.
 *
 * Every simulated minute the controller scans the air-temperature axis through
 * the network — "at which temperature would these people, at this humidity,
 * draught, hour and season, say `comfortable`?" — takes the most comfortable
 * point as its setpoint T*, and drives the heater towards it with a small PI
 * controller. The comfort zone is not programmed anywhere: it exists only in
 * the learned weights, and the thermostat follows it around as conditions
 * change.
 */

const LOOP_VIEW = 480;               // minutes of history on the charts (8 h)

const loop = {
  on: false,
  sim: null,
  mode: 'nn',                        // 'nn' | 'thermo' | 'manual'
  speed: 30,                         // simulated minutes per real second
  acc: 0,                            // fractional minutes carried between frames
  setpoint: 21, manual: 0.35,
  uI: 0.3, u: 0,                     // PI integrator and last command
  band: null, bandAge: 999,
  learn: false,
  hist: [],                          // one entry per minute
  votes: [],                         // {min, ta, label, agree}
  minutes: 0,
  noVote: 0, agreeVote: 0,
};

function loopEnsureSim(cfg) {
  if (!loop.sim) loop.sim = simInit(cfg);
}

function loopRestart(cfg) {
  loop.sim = simInit(cfg);
  loop.hist.length = 0; loop.votes.length = 0;
  loop.minutes = 0; loop.acc = 0; loop.uI = 0.3; loop.band = null; loop.bandAge = 999;
  loop.noVote = 0; loop.agreeVote = 0;
}

/**
 * Scans Ta through the network at the otherwise-current state. The wall
 * follows the scan almost fully (with a radiant heater the walls sit close to
 * the air at any steady temperature), so the scanned states stay on the
 * manifold the votes came from. T* is the probability-weighted centre of the
 * comfortable stretch — the single best point of a softmax plateau is noise.
 */
function loopComputeBand(model, ids, sens, enc) {
  const s = Object.assign({}, sens);
  let run = null, best = null;
  for (let t = 12; t <= 30.001; t += 0.25) {
    s.ta = t;
    s.tw = sens.tw + (t - sens.ta) * 0.85;
    const p = model.forward(enc ? enc(s) : encodeState(s, ids), false);
    if (argmax(p) === CLASS_INDEX.comf) {
      const w = p[CLASS_INDEX.comf];
      if (!run) run = { lo: t, hi: t, n: 1, wsum: w, tsum: w * t, peak: w };
      else { run.hi = t; run.n++; run.wsum += w; run.tsum += w * t; run.peak = Math.max(run.peak, w); }
    } else if (run) {
      if (!best || run.wsum > best.wsum) best = run;
      run = null;
    }
  }
  if (run && (!best || run.wsum > best.wsum)) best = run;
  // a credible zone is a contiguous stretch of at least ~1 °C the network is
  // actually confident about — a few stray cells of extrapolation are not one
  if (!best || best.n < 5 || best.peak < 0.5) return { lo: null, hi: null, tstar: null };
  return { lo: best.lo, hi: best.hi, tstar: best.tsum / best.wsum };
}

/** One simulated minute: control decision → physics → sensing → charts. */
function loopMinute(model, ids, opt) {
  const s = loop.sim;

  // ---- decide the heater command
  let u;
  if (loop.mode === 'manual') {
    u = loop.manual;
  } else {
    let target;
    if (loop.mode === 'thermo') target = loop.setpoint;
    else {
      target = (loop.band && loop.band.tstar !== null) ? loop.band.tstar : 21;
      if (loop.empty) target -= 1.5;                // setback while nobody is home
    }
    const e = target - s.ta;
    loop.uI = clamp(loop.uI + 0.015 * e, 0, 1);       // slow integral part
    u = clamp(loop.uI + 0.35 * e, 0, 1);              // plus a proportional kick
  }
  loop.u = u;

  simStep(s, u);
  loop.minutes++;

  const sens = sensorState(s, 0.3);                    // live sensors are a bit noisy
  const probs = model.forward(opt.enc ? opt.enc(sens) : encodeState(sens, ids), false);

  // refresh the learned comfort band every few minutes (it moves slowly).
  // An empty room is a state nobody ever votes in, so the network knows
  // nothing there — the controller instead asks "would a typical seated
  // occupant be comfortable?" and holds a little below that while saving.
  loop.bandAge++;
  loop.empty = s.occ.n === 0;
  if (loop.mode === 'nn' && loop.bandAge >= 5) {
    const scanSens = loop.empty ? Object.assign({}, sens, { mov: 0.35 }) : sens;
    loop.band = loopComputeBand(model, ids, scanSens, opt.enc);
    loop.bandAge = 0;
  }

  // do the occupants have an opinion?
  const truth = comfortTruth(sens, opt.pref);
  const v = simMaybeVote(s, opt.pref, opt.sigma, sens);
  if (v) {
    const agree = v.label === argmax(probs);
    loop.votes.push({ min: loop.minutes, ta: s.ta, label: v.label, agree });
    if (loop.votes.length > 400) loop.votes.shift();
    loop.noVote++; if (agree) loop.agreeVote++;
    if (loop.learn && opt.onVote) opt.onVote(v);       // online learning hook
  }

  loop.hist.push({
    ta: s.ta, tw: s.tw, tout: s.tout,
    lo: loop.band ? loop.band.lo : null,
    hi: loop.band ? loop.band.hi : null,
    tstar: loop.band ? loop.band.tstar : null,
    u: s.power / s.cfg.pmax,
    probs: probs.slice(),
    occ: s.occ.n, asleep: s.occ.asleep,
    truthLabel: s.occ.n > 0 ? truth.label : -1,
    pmv: truth.pmv,
  });
  if (loop.hist.length > LOOP_VIEW) loop.hist.shift();
}

/** Advances the simulation by the frame's worth of minutes. */
function loopTick(model, ids, opt) {
  loop.acc += loop.speed / 60;
  let steps = Math.floor(loop.acc);
  loop.acc -= steps;
  if (steps > 240) steps = 240;                        // don't freeze the tab
  for (let i = 0; i < steps; i++) loopMinute(model, ids, opt);
  return steps;
}

/* ----------------------------------------------------------- drawing */
function finite(v) { return typeof v === 'number' && isFinite(v); }

/** Temperatures, the learned comfort band, T* and the votes. */
function drawLoopTemps(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  const H = loop.hist;
  if (!H.length) {
    ctx.fillStyle = '#98a2ad'; ctx.font = '10px system-ui,sans-serif';
    ctx.fillText('Start the loop to watch the network run the heater.', 6, h / 2);
    return;
  }
  let lo = 10, hi = 25;
  for (const e of H) {
    if (!finite(e.ta) || !finite(e.tout)) continue;   // a bad entry must not blank the chart
    lo = Math.min(lo, e.tout - 1, e.ta - 1);
    hi = Math.max(hi, e.ta + 1, e.tout + 1, (finite(e.hi) ? e.hi : 0) + 1);
  }
  if (!finite(lo) || !finite(hi) || hi - lo < 1) { lo = 10; hi = 25; }
  const X = (i) => (i * (w - 1)) / (LOOP_VIEW - 1);
  const Y = (t) => h - 12 - ((t - lo) / (hi - lo)) * (h - 20);

  // horizontal grid every 5 °C
  ctx.font = '9px system-ui,sans-serif';
  for (let t = Math.ceil(lo / 5) * 5; t <= hi; t += 5) {
    ctx.strokeStyle = GRID; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(24, Y(t)); ctx.lineTo(w, Y(t)); ctx.stroke();
    ctx.fillStyle = '#98a2ad'; ctx.fillText(t + '°', 4, Y(t) + 3);
  }

  // the comfort band the network believes in, as one shape: along the upper
  // edge, back along the lower edge — gaps (no band found) break the shape
  ctx.fillStyle = 'rgba(46,158,91,0.15)';
  let runStart = -1;
  const closeRun = (end) => {
    if (runStart < 0) return;
    for (let i = runStart; i < end; i++) if (!finite(H[i].lo) || !finite(H[i].hi)) { runStart = -1; return; }
    ctx.beginPath();
    for (let i = runStart; i < end; i++) ctx.lineTo(X(i), Y(H[i].hi));
    for (let i = end - 1; i >= runStart; i--) ctx.lineTo(X(i), Y(H[i].lo));
    ctx.closePath();
    ctx.fill();
    runStart = -1;
  };
  for (let i = 0; i < H.length; i++) {
    if (H[i].lo === null || H[i].hi === null) closeRun(i);
    else if (runStart < 0) runStart = i;
  }
  closeRun(H.length);

  const line = (get, color, width, dash) => {
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < H.length; i++) {
      const v = get(H[i]);
      if (!finite(v)) { started = false; continue; }
      if (!started) { ctx.moveTo(X(i), Y(v)); started = true; }
      else ctx.lineTo(X(i), Y(v));
    }
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.setLineDash(dash || []);
    ctx.stroke();
    ctx.setLineDash([]);
  };
  line((e) => e.tout, '#9aa5b1', 1.1);
  line((e) => e.tw, '#8b5cf6', 1.1);
  line((e) => e.tstar, '#1f7a45', 1, [4, 3]);
  line((e) => e.ta, '#24313d', 1.6);

  // votes as dots on the air-temperature line
  const first = loop.minutes - H.length;
  for (const v of loop.votes) {
    const i = v.min - first - 1;
    if (i < 0 || i >= H.length) continue;
    ctx.fillStyle = CLASSES[v.label].color;
    ctx.beginPath(); ctx.arc(X(i), Y(v.ta), 3.4, 0, 6.284); ctx.fill();
    ctx.strokeStyle = v.agree ? '#fff' : '#1d2b38';
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  ctx.fillStyle = '#7b8794'; ctx.font = '9px system-ui,sans-serif';
  ctx.fillText('← ' + (LOOP_VIEW / 60) + ' h of simulated time', 28, 10);
}

/** The network's live opinion, the heater power and the agreement strip. */
function drawLoopRibbon(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  const H = loop.hist;
  if (!H.length) return;
  const barH = h - 9;
  const cw = w / LOOP_VIEW;
  for (let i = 0; i < H.length; i++) {
    const e = H[i];
    const x = i * cw;
    let y = 0;
    for (let j = 0; j < 3; j++) {
      const hh = e.probs[j] * barH;
      if (!finite(hh)) continue;
      ctx.fillStyle = CLASSES[j].color;
      ctx.globalAlpha = e.occ > 0 ? 0.85 : 0.3;        // dim when nobody is home
      ctx.fillRect(x, y, Math.ceil(cw), hh);
      y += hh;
    }
    ctx.globalAlpha = 1;
    // agreement strip: network vote vs the true (noiseless) vote
    let c = '#d5dbe1';
    if (e.truthLabel >= 0) {
      let am = 0; for (let j = 1; j < 3; j++) if (e.probs[j] > e.probs[am]) am = j;
      c = am === e.truthLabel ? '#2e9e5b' : '#e0342b';
    }
    ctx.fillStyle = c;
    ctx.fillRect(x, h - 7, Math.ceil(cw), 7);
  }
  // heater power as a line over the ribbon
  ctx.beginPath();
  for (let i = 0; i < H.length; i++) {
    const y = finite(H[i].u) ? (1 - H[i].u) * (barH - 2) + 1 : barH - 1;
    if (i === 0) ctx.moveTo(i * cw, y); else ctx.lineTo(i * cw, y);
  }
  ctx.strokeStyle = 'rgba(20,30,40,0.85)';
  ctx.lineWidth = 1.4;
  ctx.stroke();
  ctx.fillStyle = '#fff'; ctx.font = '600 9px system-ui,sans-serif';
  ctx.fillText('heater power', 5, (1 - H[H.length - 1].u) * (barH - 2) - 3 < 10 ? 12 : (1 - H[H.length - 1].u) * (barH - 2) - 3);
}

/** One line of live numbers under the loop controls. */
function loopStatusHtml(opt) {
  const s = loop.sim;
  if (!s || !loop.hist.length) return 'Loop stopped. Press ▶ to simulate the room with the network in charge.';
  const e = loop.hist[loop.hist.length - 1];
  const M = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const hh = String(Math.floor(s.hour)).padStart(2, '0');
  const mm = String(Math.floor((s.hour % 1) * 60)).padStart(2, '0');
  const who = s.occ.n === 0 ? 'nobody home' : s.occ.n + (s.occ.asleep ? ' asleep' : ' at home');
  const band = loop.band && loop.band.lo !== null
    ? loop.band.lo.toFixed(1) + '–' + loop.band.hi.toFixed(1) + ' °C, T* = <b>' + loop.band.tstar.toFixed(1) + ' °C</b>' +
      (loop.empty ? ' <span style="color:#7b8794">(−1.5° setback, empty)</span>' : '')
    : '<span class="bad">no credible zone in reach — train the network (or it is genuinely too hot: a heater cannot cool)</span>';
  const agree = loop.noVote ? ' · votes matched <b>' + Math.round(100 * loop.agreeVote / loop.noVote) + '%</b>' : '';
  const dead = loop.failName
    ? ' · <span class="bad">' + loop.failName + ' sensor failed — its channels arrive zeroed</span>' : '';
  const broken = e.probs && !isFinite(e.probs[0])
    ? ' · <span class="bad">network output is NaN — training diverged, press ⟳ to reset the weights</span>' : '';
  return 'Day ' + s.doy + ' (' + M[s.dow] + ') <b>' + hh + ':' + mm + '</b> · ' + who +
    ' · T room <b>' + s.ta.toFixed(1) + '°</b> · walls <b>' + s.tw.toFixed(1) + '°</b> · outside <b>' + s.tout.toFixed(1) + '°</b>' +
    ' · RH <b>' + Math.round(s.rh) + '%</b> · heater <b>' + Math.round(s.power) + ' W</b>' +
    (s.window ? ' · <span class="bad">window open</span>' : '') +
    (s.guests ? ' · ' + s.guests + ' guests' : '') +
    ' · true PMV <b>' + (e.pmv >= 0 ? '+' : '−') + Math.abs(e.pmv).toFixed(2) + '</b>' +
    (loop.mode === 'nn' ? ' · learned zone: ' + band : '') + dead + agree + broken;
}
