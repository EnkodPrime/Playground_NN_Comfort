/* nn.js — a fully connected network written from scratch: forward, backprop,
 * Adam. No libraries. This is the TensorFlow-Playground network, only the
 * inputs are sensors of a room instead of x and y of a point.
 */

function heInit(arr, fanIn) {
  const std = Math.sqrt(2 / Math.max(1, fanIn));
  for (let i = 0; i < arr.length; i++) arr[i] = randn() * std;
}

/* ------------------------------------------------------------------ Dense */
class Dense {
  constructor(nin, nout) {
    this.type = 'dense';
    this.nin = nin; this.nout = nout;
    this.W = new Float32Array(nout * nin);   // W[j*nin + i]: input i → unit j
    this.b = new Float32Array(nout);
    heInit(this.W, nin);
    this.gW = new Float32Array(this.W.length);
    this.gb = new Float32Array(nout);
    this.mW = new Float32Array(this.W.length);
    this.vW = new Float32Array(this.W.length);
    this.mb = new Float32Array(nout);
    this.vb = new Float32Array(nout);
  }
  forward(x) {
    this.x = x;
    const out = new Float32Array(this.nout);
    for (let j = 0; j < this.nout; j++) {
      let s = this.b[j];
      const wb = j * this.nin;
      for (let i = 0; i < this.nin; i++) s += this.W[wb + i] * x[i];
      out[j] = s;
    }
    return out;
  }
  backward(dout) {
    const dx = new Float32Array(this.nin);
    for (let j = 0; j < this.nout; j++) {
      const d = dout[j];
      if (d === 0) continue;
      const wb = j * this.nin;
      this.gb[j] += d;
      for (let i = 0; i < this.nin; i++) {
        this.gW[wb + i] += d * this.x[i];
        dx[i] += d * this.W[wb + i];
      }
    }
    return dx;
  }
}

/* -------------------------------------------------------------- Activations */
class Activation {
  constructor(kind) { this.type = 'act'; this.kind = kind; }
  forward(x) {
    const out = new Float32Array(x.length);
    if (this.kind === 'relu') {
      for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
    } else if (this.kind === 'tanh') {
      for (let i = 0; i < x.length; i++) out[i] = Math.tanh(x[i]);
    } else if (this.kind === 'leaky') {
      for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0.1 * x[i];
    } else { // sigmoid
      for (let i = 0; i < x.length; i++) out[i] = 1 / (1 + Math.exp(-x[i]));
    }
    this.out = out;
    return out;
  }
  backward(dout) {
    const o = this.out;
    const dx = new Float32Array(o.length);
    if (this.kind === 'relu') {
      for (let i = 0; i < o.length; i++) dx[i] = o[i] > 0 ? dout[i] : 0;
    } else if (this.kind === 'tanh') {
      for (let i = 0; i < o.length; i++) dx[i] = dout[i] * (1 - o[i] * o[i]);
    } else if (this.kind === 'leaky') {
      for (let i = 0; i < o.length; i++) dx[i] = o[i] > 0 ? dout[i] : 0.1 * dout[i];
    } else {
      for (let i = 0; i < o.length; i++) dx[i] = dout[i] * o[i] * (1 - o[i]);
    }
    return dx;
  }
}

/* ------------------------------------------------------------------ Model */
class MLP {
  /** @param {{inputDim:number, hidden:number[], activation:string, nClasses:number}} cfg */
  constructor(cfg) {
    this.cfg = JSON.parse(JSON.stringify(cfg));
    this.build();
    this.t = 0;                 // Adam step counter
  }

  build() {
    const cfg = this.cfg;
    this.seq = [];
    this.denses = [];
    let nin = cfg.inputDim;
    cfg.hidden.forEach((units) => {
      const d = new Dense(nin, units);
      this.seq.push(d, new Activation(cfg.activation));
      this.denses.push(d);
      nin = units;
    });
    this.out = new Dense(nin, cfg.nClasses);
    this.seq.push(this.out);
    this.params = [...this.denses, this.out];
  }

  /**
   * Forward pass. With keepActs=true, this.acts holds the post-activation
   * vector of every hidden layer and this.preacts the pre-activation z —
   * the diagram and the arithmetic panel read those.
   */
  forward(x, keepActs) {
    if (keepActs) { this.acts = []; this.preacts = []; this.input = x; }
    let a = x;
    for (const layer of this.seq) {
      a = layer.forward(a);
      if (keepActs && layer.type === 'dense' && layer !== this.out) this.preacts.push(a);
      if (keepActs && layer.type === 'act') this.acts.push(a);
    }
    this.logits = a;
    return this.softmax(a);
  }

  softmax(z) {
    let m = -Infinity;
    for (let i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    const p = new Float32Array(z.length);
    let s = 0;
    for (let i = 0; i < z.length; i++) { p[i] = Math.exp(z[i] - m); s += p[i]; }
    for (let i = 0; i < z.length; i++) p[i] /= s;
    this.probs = p;
    return p;
  }

  /** Backward pass from cross-entropy. Returns the loss for this example. */
  backward(probs, target) {
    const d = new Float32Array(probs.length);
    for (let i = 0; i < probs.length; i++) d[i] = probs[i];
    d[target] -= 1;
    let g = d;
    for (let i = this.seq.length - 1; i >= 0; i--) g = this.seq[i].backward(g);
    return -Math.log(Math.max(1e-9, probs[target]));
  }

  zeroGrads() {
    for (const p of this.params) { p.gW.fill(0); p.gb.fill(0); }
  }

  /** Adam update. scale = 1/batchSize */
  step(lr, scale, l2) {
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (const p of this.params) {
      upd(p.W, p.gW, p.mW, p.vW);
      upd(p.b, p.gb, p.mb, p.vb, true);
    }
    function upd(W, g, m, v, isBias) {
      for (let i = 0; i < W.length; i++) {
        let gi = g[i] * scale;
        if (l2 > 0 && !isBias) gi += l2 * W[i];
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        W[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
      }
    }
  }

  /** One mini-batch update. Returns the mean loss. */
  trainBatch(xs, ys, idx, lr, l2) {
    this.zeroGrads();
    let loss = 0;
    for (const i of idx) {
      const p = this.forward(xs[i], false);
      loss += this.backward(p, ys[i]);
    }
    this.step(lr, 1 / idx.length, l2);
    return loss / idx.length;
  }

  /** Evaluates a dataset: loss, accuracy and the confusion matrix. */
  evaluate(ds, nClasses, limit) {
    const n = Math.min(ds.n, limit || ds.n);
    let loss = 0, correct = 0;
    const conf = new Int32Array(nClasses * nClasses);
    for (let i = 0; i < n; i++) {
      const p = this.forward(ds.xs[i], false);
      const y = ds.ys[i];
      loss += -Math.log(Math.max(1e-9, p[y]));
      let arg = 0;
      for (let c = 1; c < p.length; c++) if (p[c] > p[arg]) arg = c;
      if (arg === y) correct++;
      conf[y * nClasses + arg]++;
    }
    return { loss: loss / n, acc: correct / n, conf, n };
  }

  /** Number of trainable parameters. */
  paramCount() {
    let s = 0;
    for (const p of this.params) s += p.W.length + p.b.length;
    return s;
  }
}

function argmax(p) {
  let a = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[a]) a = i;
  return a;
}
