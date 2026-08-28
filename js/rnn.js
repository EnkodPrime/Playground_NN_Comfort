/* rnn.js — recurrent networks over the sensor window: forward, BPTT, Adam.
 *
 * Ported from Playground_NN_architectures and given several input channels:
 * there the sequence was one channel of mains voltage, here every sensor is a
 * channel and the recurrence walks through the last hour of the room's story.
 * Three cells share one scaffolding: simple tanh RNN, GRU and LSTM.
 * Sequences are channel-major Float32Arrays indexed as [channel * T + t].
 */

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

/** One trainable block: a weight matrix plus an optional bias, with Adam state. */
function makeParam(rows, cols, scale, nbias) {
  const W = new Float32Array(rows * cols);
  for (let i = 0; i < W.length; i++) W[i] = randn() * scale;
  const b = new Float32Array(nbias || 0);
  return {
    W, b, rows, cols,
    gW: new Float32Array(W.length), gb: new Float32Array(b.length),
    mW: new Float32Array(W.length), vW: new Float32Array(W.length),
    mb: new Float32Array(b.length), vb: new Float32Array(b.length),
  };
}

/* ---------------------------------------------------------- one direction */
/** A single recurrent pass over the window.
 * Gate rows are stacked: LSTM = [i, f, o, g], GRU = [z, r, n], RNN = [h]. */
class RNNDir {
  constructor(kind, D, H) {
    this.kind = kind; this.D = D; this.H = H;
    this.G = kind === 'lstm' ? 4 : kind === 'gru' ? 3 : 1;
    const R = this.G * H;
    this.px = makeParam(R, D, 1 / Math.sqrt(D), R);          // input weights + bias
    this.ph = makeParam(R, H, 1 / Math.sqrt(H), 0);          // recurrent weights
    if (kind === 'lstm') {                                   // forget-gate bias = 1
      for (let u = 0; u < H; u++) this.px.b[H + u] = 1;
    }
    this.params = [this.px, this.ph];
  }

  forward(x, L) {
    const { D, H, G, kind } = this;
    const R = G * H;
    const Wx = this.px.W, b = this.px.b, Wh = this.ph.W;

    const hs = new Float32Array((L + 1) * H);      // hs[(s+1)*H+u] after step s
    const gs = new Float32Array(L * R);            // post-activation gates per step
    const cs = kind === 'lstm' ? new Float32Array((L + 1) * H) : null;
    const tc = kind === 'lstm' ? new Float32Array(L * H) : null;   // tanh(c)
    const qs = kind === 'gru' ? new Float32Array(L * H) : null;    // Whn · h_prev
    const z = new Float32Array(R);

    for (let s = 0; s < L; s++) {
      const t = s;
      const hp = s * H;

      for (let r = 0; r < R; r++) {
        let acc = b[r];
        for (let d = 0; d < D; d++) acc += Wx[r * D + d] * x[d * L + t];
        z[r] = acc;
      }
      if (kind === 'gru') {
        // the reset gate multiplies only the recurrent part of the candidate
        for (let r = 0; r < 2 * H; r++) {
          let acc = 0;
          for (let v = 0; v < H; v++) acc += Wh[r * H + v] * hs[hp + v];
          z[r] += acc;
        }
        const zg = gs, off = s * R;
        for (let u = 0; u < H; u++) zg[off + u] = sigmoid(z[u]);
        for (let u = 0; u < H; u++) zg[off + H + u] = sigmoid(z[H + u]);
        for (let u = 0; u < H; u++) {
          let q = 0;
          for (let v = 0; v < H; v++) q += Wh[(2 * H + u) * H + v] * hs[hp + v];
          qs[s * H + u] = q;
          const npre = z[2 * H + u] + zg[off + H + u] * q;
          zg[off + 2 * H + u] = Math.tanh(npre);
        }
        for (let u = 0; u < H; u++) {
          const g = gs[off + u], n = gs[off + 2 * H + u];
          hs[(s + 1) * H + u] = (1 - g) * n + g * hs[hp + u];
        }
      } else {
        for (let r = 0; r < R; r++) {
          let acc = 0;
          for (let v = 0; v < H; v++) acc += Wh[r * H + v] * hs[hp + v];
          z[r] += acc;
        }
        const off = s * R;
        if (kind === 'lstm') {
          for (let u = 0; u < H; u++) {
            const i = sigmoid(z[u]), f = sigmoid(z[H + u]);
            const o = sigmoid(z[2 * H + u]), g = Math.tanh(z[3 * H + u]);
            gs[off + u] = i; gs[off + H + u] = f;
            gs[off + 2 * H + u] = o; gs[off + 3 * H + u] = g;
            const c = f * cs[hp + u] + i * g;
            cs[(s + 1) * H + u] = c;
            const th = Math.tanh(c);
            tc[s * H + u] = th;
            hs[(s + 1) * H + u] = o * th;
          }
        } else {
          for (let u = 0; u < H; u++) {
            const h = Math.tanh(z[u]);
            gs[off + u] = h;
            hs[(s + 1) * H + u] = h;
          }
        }
      }
    }

    this.x = x; this.L = L;
    this.hs = hs; this.gs = gs; this.cs = cs; this.tc = tc; this.qs = qs;

    const out = new Float32Array(H * L);
    for (let s = 0; s < L; s++) {
      for (let u = 0; u < H; u++) out[u * L + s] = hs[(s + 1) * H + u];
    }
    return out;
  }

  /** BPTT. dout is channel-major [H*L]; returns dx channel-major [D*L]. */
  backward(dout) {
    const { D, H, G, kind, L, hs, gs, cs, tc, qs, x } = this;
    const R = G * H;
    const Wx = this.px.W, Wh = this.ph.W;
    const gWx = this.px.gW, gb = this.px.gb, gWh = this.ph.gW;
    const dx = new Float32Array(D * L);
    const dh = new Float32Array(H);
    const dc = new Float32Array(H);
    const dz = new Float32Array(R);

    for (let s = L - 1; s >= 0; s--) {
      const t = s;
      const hp = s * H, off = s * R;
      for (let u = 0; u < H; u++) dh[u] += dout[u * L + t];
      dz.fill(0);
      const dhprev = new Float32Array(H);

      if (kind === 'lstm') {
        for (let u = 0; u < H; u++) {
          const i = gs[off + u], f = gs[off + H + u];
          const o = gs[off + 2 * H + u], g = gs[off + 3 * H + u];
          const th = tc[s * H + u];
          const dO = dh[u] * th;
          dc[u] += dh[u] * o * (1 - th * th);
          const dI = dc[u] * g, dG = dc[u] * i, dF = dc[u] * cs[hp + u];
          dz[u] = dI * i * (1 - i);
          dz[H + u] = dF * f * (1 - f);
          dz[2 * H + u] = dO * o * (1 - o);
          dz[3 * H + u] = dG * (1 - g * g);
          dc[u] = dc[u] * f;
        }
      } else if (kind === 'gru') {
        for (let u = 0; u < H; u++) {
          const zg = gs[off + u], r = gs[off + H + u], n = gs[off + 2 * H + u];
          const hprev = hs[hp + u];
          const dN = dh[u] * (1 - zg);
          const dZ = dh[u] * (hprev - n);
          dhprev[u] += dh[u] * zg;
          const dNpre = dN * (1 - n * n);
          dz[2 * H + u] = dNpre;
          const dR = dNpre * qs[s * H + u];
          dz[u] = dZ * zg * (1 - zg);
          dz[H + u] = dR * r * (1 - r);
        }
        for (let u = 0; u < H; u++) {
          const r = gs[off + H + u];
          const dq = dz[2 * H + u] * r;
          const row = (2 * H + u) * H;
          for (let v = 0; v < H; v++) {
            gWh[row + v] += dq * hs[hp + v];
            dhprev[v] += dq * Wh[row + v];
          }
        }
      } else {
        for (let u = 0; u < H; u++) {
          const h = gs[off + u];
          dz[u] = dh[u] * (1 - h * h);
        }
      }

      for (let r = 0; r < R; r++) {
        const g = dz[r];
        if (g === 0) continue;
        gb[r] += g;
        const rw = r * D;
        for (let d = 0; d < D; d++) {
          gWx[rw + d] += g * x[d * L + t];
          dx[d * L + t] += g * Wx[rw + d];
        }
      }
      const rEnd = kind === 'gru' ? 2 * H : R;
      for (let r = 0; r < rEnd; r++) {
        const g = dz[r];
        if (g === 0) continue;
        const row = r * H;
        for (let v = 0; v < H; v++) {
          gWh[row + v] += g * hs[hp + v];
          dhprev[v] += g * Wh[row + v];
        }
      }
      dh.set(dhprev);
    }
    return dx;
  }
}

/* ---------------------------------------------------------------- readout */
/** Reduces the state sequence to one vector: last state, mean or max in time. */
class Readout {
  constructor(kind) { this.type = 'readout'; this.kind = kind; }
  forward(x, L) {
    const C = x.length / L;
    this.C = C; this.L = L;
    const out = new Float32Array(C);
    if (this.kind === 'mean') {
      for (let c = 0; c < C; c++) {
        let s = 0;
        for (let t = 0; t < L; t++) s += x[c * L + t];
        out[c] = s / L;
      }
    } else if (this.kind === 'max') {
      this.arg = new Int32Array(C);
      for (let c = 0; c < C; c++) {
        let best = -Infinity, bi = 0;
        for (let t = 0; t < L; t++) { const v = x[c * L + t]; if (v > best) { best = v; bi = t; } }
        out[c] = best; this.arg[c] = bi;
      }
    } else {
      for (let c = 0; c < C; c++) out[c] = x[c * L + (L - 1)];
    }
    return out;
  }
  backward(dout) {
    const dx = new Float32Array(this.C * this.L);
    if (this.kind === 'mean') {
      for (let c = 0; c < this.C; c++) {
        const g = dout[c] / this.L;
        for (let t = 0; t < this.L; t++) dx[c * this.L + t] = g;
      }
    } else if (this.kind === 'max') {
      for (let c = 0; c < this.C; c++) dx[c * this.L + this.arg[c]] += dout[c];
    } else {
      for (let c = 0; c < this.C; c++) dx[c * this.L + (this.L - 1)] += dout[c];
    }
    return dx;
  }
}

/* ------------------------------------------------------------------ Model */
class RNNNet {
  /** @param {{cell:string, channels:number, T:number, layers:{units:number}[],
   *          readout:string, nClasses:number}} cfg */
  constructor(cfg) {
    this.kind = 'rnn';
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;
  }
  build() {
    const cfg = this.cfg;
    this.layers = [];
    this.params = [];
    let D = cfg.channels;
    cfg.layers.forEach((ls) => {
      const dir = new RNNDir(cfg.cell, D, ls.units);
      this.layers.push(dir);
      this.params = this.params.concat(dir.params);
      D = ls.units;
    });
    this.pool = new Readout(cfg.readout);
    this.headKind = cfg.readout;
    this.out = new Dense(D, cfg.nClasses);
    this.params.push(this.out);
  }
  /** With keepActs=true, this.acts[l] holds layer l's states, [units*T]. */
  forward(x, keepActs) {
    const T = this.cfg.T;
    if (keepActs) { this.acts = []; this.input = x; }
    let a = x;
    for (const dir of this.layers) {
      a = dir.forward(a, T);
      if (keepActs) this.acts.push(a);
    }
    this.embedding = this.pool.forward(a, T);
    this.logits = this.out.forward(this.embedding);
    return this.softmax(this.logits);
  }
  backward(probs, target) {
    const d = new Float32Array(probs.length);
    for (let i = 0; i < probs.length; i++) d[i] = probs[i];
    d[target] -= 1;
    let g = this.out.backward(d);
    g = this.pool.backward(g);
    for (let l = this.layers.length - 1; l >= 0; l--) g = this.layers[l].backward(g);
    return -Math.log(Math.max(1e-9, probs[target]));
  }
  /**
   * Adam with global gradient-norm clipping. Backpropagating through the whole
   * window produces occasional huge gradients; without clipping the loss
   * oscillates instead of converging.
   */
  step(lr, scale, l2) {
    const clip = 1.0;
    let sum = 0;
    for (const p of this.params) {
      for (let i = 0; i < p.gW.length; i++) { const g = p.gW[i] * scale; sum += g * g; }
      for (let i = 0; i < p.gb.length; i++) { const g = p.gb[i] * scale; sum += g * g; }
    }
    const norm = Math.sqrt(sum);
    const k = (norm > clip ? clip / norm : 1) * scale;
    for (const p of this.params) {
      for (let i = 0; i < p.gW.length; i++) p.gW[i] *= k;
      for (let i = 0; i < p.gb.length; i++) p.gb[i] *= k;
    }
    MLP.prototype.step.call(this, lr, 1, l2);
  }
}
RNNNet.prototype.softmax = MLP.prototype.softmax;
RNNNet.prototype.zeroGrads = MLP.prototype.zeroGrads;
RNNNet.prototype.trainBatch = MLP.prototype.trainBatch;
RNNNet.prototype.evaluate = MLP.prototype.evaluate;
RNNNet.prototype.paramCount = MLP.prototype.paramCount;
