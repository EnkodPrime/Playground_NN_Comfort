/* viz.js — draws the network, the metrics and the small charts.
 *
 * Every node in the diagram is a little map: the unit's response over the two
 * features chosen as the comfort-map axes, with all other sensors held at the
 * probe values. Orange is positive, blue negative — exactly the TensorFlow
 * Playground convention, only the plane is (say) temperature × humidity
 * instead of x × y.
 */

const POS = '#f0921f';   // positive values (orange, as in TF Playground)
const NEG = '#0877bd';   // negative values (blue)
const GRID = '#e3e6ea';
const AXIS = '#b9c0c8';

const GRES = 20;         // node heatmap resolution (GRES × GRES)

function dpiSetup(canvas, cssW, cssH) {
  const r = window.devicePixelRatio || 1;
  const W = Math.max(1, Math.round(cssW * r));
  const H = Math.max(1, Math.round(cssH * r));
  const ctx = canvas.getContext('2d');
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W; canvas.height = H;
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
  }
  ctx.setTransform(r, 0, 0, r, 0, 0);
  return ctx;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/* Colour lookup tables: t ∈ [−1,1] → white-to-blue / white-to-orange. */
const _lutPos = [], _lutNeg = [];
(function buildLuts() {
  const mix = (hex, t) => {
    const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
    const R = Math.round(255 + (r - 255) * t), G = Math.round(255 + (g - 255) * t), B = Math.round(255 + (b - 255) * t);
    return 'rgb(' + R + ',' + G + ',' + B + ')';
  };
  for (let i = 0; i <= 48; i++) { _lutPos.push(mix(POS, i / 48)); _lutNeg.push(mix(NEG, i / 48)); }
})();

/** Signed value → colour, squashed through tanh so big activations saturate. */
function heatColor(v) {
  const t = Math.tanh(v);
  const i = Math.min(48, Math.round(Math.abs(t) * 48));
  return t >= 0 ? _lutPos[i] : _lutNeg[i];
}

/** Probability → tint of a class colour. */
function probColor(p, hex) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  const t = Math.max(0, Math.min(1, p));
  return 'rgb(' + Math.round(255 + (r - 255) * t) + ',' + Math.round(255 + (g - 255) * t) + ',' + Math.round(255 + (b - 255) * t) + ')';
}

/* ------------------------------------------------------------ Layout */
const IN_W = 44, IN_H = 34, HID_W = 48, HID_H = 44, OUT_W = 52, OUT_H = 44;
const NODE_VGAP = 11, LABEL_W = 96, OUT_LABEL_W = 96, COL_GAP = 86;

/**
 * Positions of every node. Columns: encoded inputs, hidden layers, outputs.
 * @param {MLP} model
 * @param {string[]} inNames encoded input names
 */
function layoutNetwork(model, inNames, cssW) {
  const cols = [];
  const nIn = model.cfg.inputDim;
  const inNodes = [];
  for (let i = 0; i < nIn; i++) inNodes.push({ w: IN_W, h: IN_H, unit: i });
  cols.push({ kind: 'in', nodes: inNodes });
  model.cfg.hidden.forEach((units, li) => {
    const nodes = [];
    for (let u = 0; u < units; u++) nodes.push({ w: HID_W, h: HID_H, layer: li, unit: u });
    cols.push({ kind: 'hid', layer: li, nodes });
  });
  const outNodes = [];
  for (let c = 0; c < CLASSES.length; c++) outNodes.push({ w: OUT_W, h: OUT_H, unit: c });
  cols.push({ kind: 'out', nodes: outNodes });

  let x = LABEL_W;
  let maxH = 0;
  cols.forEach((col) => {
    const n = col.nodes.length;
    const H = n * col.nodes[0].h + (n - 1) * NODE_VGAP;
    maxH = Math.max(maxH, H);
    col.x = x; col.colH = H;
    x += col.nodes[0].w + COL_GAP;
  });
  const width = Math.max(cssW, x - COL_GAP + OUT_LABEL_W + 8);
  const topPad = 26;
  cols.forEach((col) => {
    let y = topPad + (maxH - col.colH) / 2;
    col.nodes.forEach((nd) => { nd.x = col.x; nd.y = y; y += nd.h + NODE_VGAP; });
  });
  // centre horizontally when there is spare room
  const used = x - COL_GAP + OUT_LABEL_W;
  if (used < cssW) {
    const dx = (cssW - used) / 2;
    cols.forEach((col) => { col.x += dx; col.nodes.forEach((nd) => (nd.x += dx)); });
  }
  return { cols, width, height: topPad + maxH + 18 };
}

/** Largest |W| of a dense layer, for scaling link thickness. */
function denseMaxAbs(d) {
  let m = 1e-6;
  for (let i = 0; i < d.W.length; i++) m = Math.max(m, Math.abs(d.W[i]));
  return m;
}

/**
 * Draws the whole diagram.
 * @param {{model, layout, grids, inNames, inValues, acts, probs, hover, selected, axisInfo}} o
 *  grids: {inputs:Float32Array[], hidden:Float32Array[][], outputs:Float32Array[]}
 *  each grid is GRES×GRES of the node's value over the map plane.
 */
function drawNetwork(ctx, o) {
  const { model, layout } = o;
  const cols = layout.cols;
  ctx.clearRect(0, 0, layout.width, layout.height);

  const sel = o.selected, hov = o.hover;
  const emph = sel || hov;                          // node whose links stand out

  // ------------------------------------------------ links
  const denses = [...model.denses, model.out];
  denses.forEach((d, di) => {
    const from = cols[di], to = cols[di + 1];
    const mx = denseMaxAbs(d);
    for (let j = 0; j < to.nodes.length; j++) {
      for (let i = 0; i < from.nodes.length; i++) {
        const w = d.W[j * d.nin + i];
        const a = Math.abs(w) / mx;
        if (a < 0.02) continue;
        const hot = emph && ((emphMatches(emph, di, i, cols)) || (emphMatches(emph, di + 1, j, cols)));
        drawLink(ctx, from.nodes[i], to.nodes[j], a, w, hot, !!emph);
      }
    }
  });

  // ------------------------------------------------ column titles
  label(ctx, cols[0].x, 14, 'Sensors (' + model.cfg.inputDim + ' inputs)');
  cols.forEach((col) => {
    if (col.kind === 'hid') label(ctx, col.x - 6, 14, 'Hidden ' + (col.layer + 1));
  });
  label(ctx, cols[cols.length - 1].x - 4, 14, 'Vote');

  // ------------------------------------------------ input nodes
  cols[0].nodes.forEach((nd, i) => {
    const hot = isHover(hov, 'in', 0, i), selHot = isSel(sel, 'in', 0, i);
    drawNodeBox(ctx, nd, hot, selHot);
    if (o.grids && o.grids.inputs[i]) drawGrid(ctx, nd, o.grids.inputs[i], heatColor);
    // name and current encoded value to the left
    ctx.fillStyle = '#5b6873';
    ctx.font = '10px system-ui,sans-serif';
    const nm = o.inNames[i];
    ctx.fillText(nm, nd.x - 8 - ctx.measureText(nm).width, nd.y + nd.h / 2 - 1);
    ctx.fillStyle = '#98a2ad';
    ctx.font = '9px ui-monospace,Consolas,monospace';
    const v = fmt2(o.inValues[i]);
    ctx.fillText(v, nd.x - 8 - ctx.measureText(v).width, nd.y + nd.h / 2 + 10);
  });

  // ------------------------------------------------ hidden nodes
  cols.forEach((col) => {
    if (col.kind !== 'hid') return;
    col.nodes.forEach((nd, u) => {
      const hot = isHover(hov, 'hid', col.layer, u), selHot = isSel(sel, 'hid', col.layer, u);
      drawNodeBox(ctx, nd, hot, selHot);
      const g = o.grids && o.grids.hidden[col.layer] && o.grids.hidden[col.layer][u];
      if (g) {
        // each unit scaled by its own maximum — ReLU outputs would otherwise
        // saturate the palette and every box would look uniformly orange
        let mx = 1e-6;
        for (let k = 0; k < g.length; k++) mx = Math.max(mx, Math.abs(g[k]));
        drawGrid(ctx, nd, g, (v) => heatColor(1.5 * v / mx));
      }
      if (o.acts && o.acts[col.layer]) {
        markProbe(ctx, nd, o.axisInfo);
      }
    });
  });

  // ------------------------------------------------ output nodes
  const outCol = cols[cols.length - 1];
  outCol.nodes.forEach((nd, c) => {
    const hot = isHover(hov, 'out', 0, c), selHot = isSel(sel, 'out', 0, c);
    drawNodeBox(ctx, nd, hot, selHot);
    const g = o.grids && o.grids.outputs[c];
    if (g) drawGrid(ctx, nd, g, (v) => probColor(v, CLASSES[c].color));
    // class name and current probability
    const p = o.probs ? o.probs[c] : 0;
    ctx.fillStyle = CLASSES[c].color;
    ctx.fillRect(nd.x + nd.w + 8, nd.y + nd.h / 2 - 10, 8, 8);
    ctx.fillStyle = '#33414d';
    ctx.font = '600 10.5px system-ui,sans-serif';
    ctx.fillText(CLASSES[c].name, nd.x + nd.w + 20, nd.y + nd.h / 2 - 2);
    ctx.fillStyle = '#5b6873';
    ctx.font = '10px ui-monospace,Consolas,monospace';
    ctx.fillText((p * 100).toFixed(1) + '%', nd.x + nd.w + 20, nd.y + nd.h / 2 + 10);
  });
}

function emphMatches(emph, colIdx, unit, cols) {
  const col = cols[colIdx];
  if (col.kind === 'in') return emph.kind === 'in' && emph.unit === unit;
  if (col.kind === 'out') return emph.kind === 'out' && emph.unit === unit;
  return emph.kind === 'hid' && emph.layer === col.layer && emph.unit === unit;
}
function isHover(h, kind, layer, unit) {
  return !!h && h.kind === kind && (kind !== 'hid' || h.layer === layer) && h.unit === unit;
}
function isSel(s, kind, layer, unit) {
  return !!s && s.kind === kind && (kind !== 'hid' || s.layer === layer) && s.unit === unit;
}

/** The node's little response map. Row 0 of the grid is the BOTTOM (low Y value). */
function drawGrid(ctx, nd, grid, colorFn) {
  const cw = (nd.w - 2) / GRES, ch = (nd.h - 2) / GRES;
  for (let gy = 0; gy < GRES; gy++) {
    for (let gx = 0; gx < GRES; gx++) {
      ctx.fillStyle = colorFn(grid[gy * GRES + gx]);
      ctx.fillRect(nd.x + 1 + gx * cw, nd.y + 1 + (GRES - 1 - gy) * ch, cw + 0.5, ch + 0.5);
    }
  }
}

/** Cross-hair: where the probe sits on the node's map. */
function markProbe(ctx, nd, ax) {
  if (!ax) return;
  const px = nd.x + 1 + ax.fx * (nd.w - 2);
  const py = nd.y + 1 + (1 - ax.fy) * (nd.h - 2);
  ctx.strokeStyle = 'rgba(36,49,61,0.8)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(px, py, 2.3, 0, 6.284); ctx.stroke();
}

function drawLink(ctx, from, to, a, w, hot, hoverActive) {
  const x1 = from.x + from.w, y1 = from.y + from.h / 2;
  const x2 = to.x, y2 = to.y + to.h / 2;
  const dim = hoverActive && !hot ? 0.15 : 1;
  ctx.strokeStyle = (w > 0 ? POS : NEG);
  ctx.globalAlpha = Math.min(1, (0.12 + 0.8 * a) * dim);
  ctx.lineWidth = hot ? 1 + 3.4 * a : 0.5 + 2.6 * a;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  const mx = (x1 + x2) / 2;
  ctx.bezierCurveTo(mx, y1, mx, y2, x2, y2);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawNodeBox(ctx, nd, hot, selected) {
  ctx.save();
  ctx.shadowColor = 'rgba(20,40,60,0.10)';
  ctx.shadowBlur = hot || selected ? 10 : 4;
  ctx.shadowOffsetY = 1;
  roundRect(ctx, nd.x, nd.y, nd.w, nd.h, 5);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  ctx.restore();
  roundRect(ctx, nd.x - 0.5, nd.y - 0.5, nd.w + 1, nd.h + 1, 5);
  ctx.strokeStyle = selected ? '#1d4ed8' : hot ? '#2b6cb0' : '#cfd6de';
  ctx.lineWidth = selected ? 2.4 : hot ? 1.8 : 1;
  ctx.stroke();
}

function label(ctx, x, y, text) {
  ctx.fillStyle = '#7b8794';
  ctx.font = '600 10px system-ui, sans-serif';
  ctx.fillText(text, x, y);
}

function fmt2(v) { return (v < 0 ? '−' : '') + Math.abs(v).toFixed(2); }

/** Finds the node under the cursor. */
function hitTest(layout, mx, my) {
  const cols = layout.cols;
  for (let ci = 0; ci < cols.length; ci++) {
    const col = cols[ci];
    for (let i = 0; i < col.nodes.length; i++) {
      const nd = col.nodes[i];
      if (mx >= nd.x && mx <= nd.x + nd.w && my >= nd.y && my <= nd.y + nd.h) {
        if (col.kind === 'in') return { kind: 'in', unit: i };
        if (col.kind === 'out') return { kind: 'out', unit: i };
        return { kind: 'hid', layer: col.layer, unit: i };
      }
    }
  }
  return null;
}

/* ----------------------------------------------------------- Metrics */
function drawLossChart(ctx, w, h, hTrain, hTest) {
  ctx.clearRect(0, 0, w, h);
  const n = Math.max(hTrain.length, hTest.length);
  ctx.strokeStyle = GRID; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = 4 + (i * (h - 12)) / 4;
    ctx.beginPath(); ctx.moveTo(28, y); ctx.lineTo(w - 4, y); ctx.stroke();
  }
  if (n < 2) return;
  let mx = 0.05;
  for (const v of hTrain) mx = Math.max(mx, v);
  for (const v of hTest) mx = Math.max(mx, v);
  mx *= 1.08;
  ctx.fillStyle = '#98a2ad';
  ctx.font = '9px system-ui, sans-serif';
  ctx.fillText(mx.toFixed(2), 2, 11);
  ctx.fillText('0', 2, h - 6);

  const plot = (hist, color) => {
    ctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const X = 28 + (i * (w - 34)) / Math.max(1, n - 1);
      const Y = 4 + (h - 12) * (1 - hist[i] / mx);
      if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();
  };
  plot(hTrain, '#2b6cb0');
  plot(hTest, '#e0342b');
}

function drawConfusion(ctx, w, h, conf, classes) {
  ctx.clearRect(0, 0, w, h);
  const k = classes.length;
  const pad = 34;
  const cell = Math.min((w - pad - 6) / k, (h - pad - 6) / k);
  const totals = new Array(k).fill(0);
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) totals[i] += conf[i * k + j];
  ctx.font = '9px system-ui, sans-serif';
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const v = totals[i] ? conf[i * k + j] / totals[i] : 0;
      const x = pad + j * cell, y = pad + i * cell;
      ctx.fillStyle = i === j
        ? 'rgba(46,158,91,' + (0.12 + 0.8 * v) + ')'
        : 'rgba(224,52,43,' + (0.08 + 0.8 * v) + ')';
      ctx.fillRect(x, y, cell - 1.5, cell - 1.5);
      if (v > 0.03) {
        ctx.fillStyle = v > 0.55 ? '#fff' : '#33414d';
        const t = (v * 100).toFixed(0);
        ctx.fillText(t, x + cell / 2 - ctx.measureText(t).width / 2, y + cell / 2 + 3);
      }
    }
    ctx.fillStyle = classes[i].color;
    ctx.fillRect(2, pad + i * cell + cell / 2 - 3, 6, 6);
    ctx.fillStyle = '#5b6873';
    ctx.fillText(classes[i].short, 11, pad + i * cell + cell / 2 + 3);
    ctx.save();
    ctx.translate(pad + i * cell + cell / 2 + 3, pad - 4);
    ctx.rotate(-Math.PI / 3);
    ctx.fillText(classes[i].short, 0, 0);
    ctx.restore();
  }
}
