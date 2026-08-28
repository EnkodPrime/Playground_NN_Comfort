/* map.js — the comfort map: a 2D slice through the learned comfort function.
 *
 * Two sensors span the plane, every other sensor is pinned at the probe value.
 * The colour is the network's vote at that point, its saturation the
 * confidence. On top of it: the TRUE comfort zone (the PMV = ±0.5 band of the
 * generating model) as a dotted boundary, and the training votes that fall
 * near this slice. Learning worked when the colours fill the dotted outline.
 */

const MAP_GX = 108, MAP_GY = 80;    // network evaluation grid
const TRUTH_GX = 108, TRUTH_GY = 80;

/** Fraction 0..1 → raw feature value across its plot range. */
function axisValue(f, t) { return f.min + t * (f.max - f.min); }
/** Raw feature value → fraction 0..1 of its plot range. */
function axisFrac(f, v) { return clamp((v - f.min) / (f.max - f.min), 0, 1); }

/**
 * Evaluates the network over the plane.
 * @returns {{cls:Uint8Array, p:Float32Array}} argmax class and its probability per cell
 */
function mapEvalNet(model, ids, probe, fx, fy) {
  const cls = new Uint8Array(MAP_GX * MAP_GY);
  const p = new Float32Array(MAP_GX * MAP_GY);
  const s = Object.assign({}, probe);
  for (let gy = 0; gy < MAP_GY; gy++) {
    s[fy.id] = axisValue(fy, gy / (MAP_GY - 1));
    for (let gx = 0; gx < MAP_GX; gx++) {
      s[fx.id] = axisValue(fx, gx / (MAP_GX - 1));
      const pr = model.forward(encodeState(s, ids), false);
      const a = argmax(pr);
      cls[gy * MAP_GX + gx] = a;
      p[gy * MAP_GX + gx] = pr[a];
    }
  }
  return { cls, p };
}

/** The true vote over the plane — same slice through the PMV model. In
 * uniform (controlled-study) mode `fill` replaces the unseen sensors with the
 * study's typical household, so the outline matches the generated votes. */
function mapEvalTruth(probe, fx, fy, pref, fill) {
  const cls = new Uint8Array(TRUTH_GX * TRUTH_GY);
  const s = Object.assign({}, probe);
  for (let gy = 0; gy < TRUTH_GY; gy++) {
    s[fy.id] = axisValue(fy, gy / (TRUTH_GY - 1));
    for (let gx = 0; gx < TRUTH_GX; gx++) {
      s[fx.id] = axisValue(fx, gx / (TRUTH_GX - 1));
      cls[gy * TRUTH_GX + gx] = comfortTruth(fill ? fill(s) : s, pref).label;
    }
  }
  return cls;
}

/**
 * Draws the map. `net` and `truth` come from the two functions above; either
 * may be null (not yet computed / switched off).
 */
function drawComfortMap(ctx, w, h, o) {
  const { fx, fy, net, truth, votes, probe } = o;
  const padL = 40, padB = 26, padT = 8, padR = 10;
  const pw = w - padL - padR, ph = h - padT - padB;
  ctx.clearRect(0, 0, w, h);

  // network regions
  if (net) {
    const cw = pw / MAP_GX, ch = ph / MAP_GY;
    for (let gy = 0; gy < MAP_GY; gy++) {
      for (let gx = 0; gx < MAP_GX; gx++) {
        const i = gy * MAP_GX + gx;
        const conf = (net.p[i] - 1 / 3) / (2 / 3);
        ctx.fillStyle = probColor(0.10 + 0.72 * conf, CLASSES[net.cls[i]].color);
        ctx.fillRect(padL + gx * cw, padT + (MAP_GY - 1 - gy) * ch, cw + 0.5, ch + 0.5);
      }
    }
  } else {
    ctx.fillStyle = '#f7f9fb';
    ctx.fillRect(padL, padT, pw, ph);
  }

  // the true comfort zone as a dotted boundary
  if (truth) {
    ctx.fillStyle = 'rgba(24,34,44,0.85)';
    const cw = pw / TRUTH_GX, ch = ph / TRUTH_GY;
    for (let gy = 0; gy < TRUTH_GY; gy++) {
      for (let gx = 0; gx < TRUTH_GX; gx++) {
        const c = truth[gy * TRUTH_GX + gx];
        const right = gx + 1 < TRUTH_GX && truth[gy * TRUTH_GX + gx + 1] !== c;
        const up = gy + 1 < TRUTH_GY && truth[(gy + 1) * TRUTH_GX + gx] !== c;
        if (!right && !up) continue;
        if ((gx + gy) % 2) continue;                        // dotted, not solid
        const x = padL + (gx + (right ? 1 : 0.5)) * cw;
        const y = padT + (TRUTH_GY - 1 - gy + (up ? 0 : 0.5)) * ch;
        ctx.fillRect(x - 1, y - 1, 2, 2);
      }
    }
  }

  // votes near this slice
  if (votes) {
    for (const v of votes) {
      ctx.fillStyle = CLASSES[v.y].color;
      ctx.globalAlpha = v.a;
      const x = padL + axisFrac(fx, v.s[fx.id]) * pw;
      const y = padT + (1 - axisFrac(fy, v.s[fy.id])) * ph;
      ctx.beginPath(); ctx.arc(x, y, 2.4, 0, 6.284); ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = 'rgba(255,255,255,0.7)';
      ctx.lineWidth = 0.7;
      ctx.stroke();
    }
  }

  // probe cross-hair
  if (probe) {
    const x = padL + axisFrac(fx, probe[fx.id]) * pw;
    const y = padT + (1 - axisFrac(fy, probe[fy.id])) * ph;
    ctx.strokeStyle = '#1d2b38';
    ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.arc(x, y, 5, 0, 6.284); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - 9, y); ctx.lineTo(x - 3, y); ctx.moveTo(x + 3, y); ctx.lineTo(x + 9, y);
    ctx.moveTo(x, y - 9); ctx.lineTo(x, y - 3); ctx.moveTo(x, y + 3); ctx.lineTo(x, y + 9);
    ctx.stroke();
  }

  // frame and axes
  ctx.strokeStyle = AXIS; ctx.lineWidth = 1;
  ctx.strokeRect(padL + 0.5, padT + 0.5, pw - 1, ph - 1);
  ctx.fillStyle = '#7b8794';
  ctx.font = '9.5px system-ui,sans-serif';
  const xt = axisTicks(fx), yt = axisTicks(fy);
  xt.forEach((v) => {
    const x = padL + axisFrac(fx, v) * pw;
    ctx.strokeStyle = 'rgba(185,192,200,0.5)';
    ctx.beginPath(); ctx.moveTo(x, padT + ph); ctx.lineTo(x, padT + ph + 3); ctx.stroke();
    const t = fmtTick(fx, v);
    ctx.fillText(t, x - ctx.measureText(t).width / 2, padT + ph + 13);
  });
  yt.forEach((v) => {
    const y = padT + (1 - axisFrac(fy, v)) * ph;
    ctx.strokeStyle = 'rgba(185,192,200,0.5)';
    ctx.beginPath(); ctx.moveTo(padL - 3, y); ctx.lineTo(padL, y); ctx.stroke();
    const t = fmtTick(fy, v);
    ctx.fillText(t, padL - 6 - ctx.measureText(t).width, y + 3);
  });
  ctx.font = '600 10px system-ui,sans-serif';
  ctx.fillStyle = '#5b6873';
  const xl = fx.name + (fx.unit ? ' [' + fx.unit + ']' : '');
  ctx.fillText(xl, padL + pw / 2 - ctx.measureText(xl).width / 2, h - 4);
  ctx.save();
  ctx.translate(10, padT + ph / 2);
  ctx.rotate(-Math.PI / 2);
  const yl = fy.name + (fy.unit ? ' [' + fy.unit + ']' : '');
  ctx.fillText(yl, -ctx.measureText(yl).width / 2, 0);
  ctx.restore();
}

function axisTicks(f) {
  const span = f.max - f.min;
  const step = span > 200 ? 90 : span > 40 ? 10 : span > 20 ? 5 : span > 3 ? 4 : 0.25;
  const out = [];
  for (let v = Math.ceil(f.min / step) * step; v <= f.max + 1e-9; v += step) out.push(v);
  return out;
}
function fmtTick(f, v) {
  if (f.id === 'doy') {
    const M = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return M[Math.min(11, Math.floor((v - 1) / 30.42))];
  }
  if (f.id === 'hour') return (v | 0) + 'h';
  return String(Math.round(v * 10) / 10);
}

/**
 * Which training votes lie near the current slice? Distance is measured in the
 * encoded space of the NON-axis features of the FULL state — including sensors
 * the network is not using: a vote from a summer night is not "the same slice"
 * as a winter evening just because the season input is switched off. Close
 * votes are opaque, far ones fade out (or hide unless "all votes" is ticked).
 */
function votesNearSlice(ds, ids, probe, fx, fy, showAll, limit) {
  const out = [];
  const others = FEATURES.map((f) => f.id).filter((id) => id !== fx.id && id !== fy.id);
  const pv = encodeState(probe, others);
  const n = Math.min(ds.n, limit || 700);
  for (let i = 0; i < n; i++) {
    const sv = encodeState(ds.states[i], others);
    let d2 = 0;
    for (let k = 0; k < pv.length; k++) { const d = sv[k] - pv[k]; d2 += d * d; }
    const dist = Math.sqrt(d2 / Math.max(1, pv.length));
    const a = showAll ? clamp(1 - dist * 0.55, 0.25, 1) : 1 - dist / 0.45;
    if (a > 0.25) out.push({ s: ds.states[i], y: ds.ys[i], a: clamp(a, 0, 0.95) });
  }
  return out;
}
