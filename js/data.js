/* data.js — the sensor features and the dataset.
 *
 * A training example is one moment: the sensor readings at the instant an
 * occupant voted, and the vote itself. The dataset is made by *living in the
 * simulator* under an exploring heater policy, so the room visits cold,
 * comfortable and overheated states across the year.
 */

/* Every input the network may see. `mean`/`span` standardise a raw value;
 * cyclic features (hour, season) become a sin/cos pair so that 23:59 and 00:01
 * end up next to each other. */
const FEATURES = [
  { id: 'ta',   name: 'Room temperature',    short: 'T room', unit: '°C',  min: 10,  max: 32,  mean: 21,   span: 8,    step: 0.1,  fmt: 1, dflt: 21,
    tip: "The air temperature a thermostat's own sensor reports — the most direct comfort signal."},
  { id: 'rh',   name: 'Room humidity',       short: 'RH',     unit: '%',   min: 15,  max: 95,  mean: 50,   span: 25,   step: 1,    fmt: 0, dflt: 45,
    tip: "Relative humidity. Damp air feels different: sweat evaporates more slowly, so warm-humid feels warmer."},
  { id: 'tw',   name: 'Wall temperature',    short: 'T wall', unit: '°C',  min: 8,   max: 32,  mean: 20,   span: 8,    step: 0.1,  fmt: 1, dflt: 20,
    tip: "Surface temperature of walls and floor. Half of comfort is radiation — cold walls make 21 °C air feel chilly."},
  { id: 'tout', name: 'Outdoor temperature', short: 'T out',  unit: '°C',  min: -15, max: 35,  mean: 8,    span: 13,   step: 0.5,  fmt: 1, dflt: 3,
    tip: "The weather outside. It never touches the skin indoors, but it decides how people dress season by season."},
  { id: 'hour', name: 'Time of day',         short: 'Hour',   unit: 'h',   min: 0,   max: 24,  cyc: 24,    step: 0.25, fmt: 2, dflt: 19.5,
    tip: "The clock, encoded as sin+cos so 23:59 and 00:01 are neighbours. Lets the network tell sleeping comfort from waking comfort."},
  { id: 'doy',  name: 'Day of year',         short: 'Season', unit: '',    min: 1,   max: 365, cyc: 365,   step: 1,    fmt: 0, dflt: 20,
    tip: "The season, also sin+cos so December meets January. People wear very different clothes in July and January."},
  { id: 'mov',  name: 'Movement of people',  short: 'Move',   unit: '',    min: 0,   max: 1,   mean: 0.3,  span: 0.35, step: 0.01, fmt: 2, dflt: 0.32,
    tip: "A motion-sensor reading: how much the people are moving. Active bodies make their own heat and prefer it cooler."},
  { id: 'vair', name: 'Air movement',        short: 'Air v',  unit: 'm/s', min: 0,   max: 1,   mean: 0.15, span: 0.2,  step: 0.01, fmt: 2, dflt: 0.07,
    tip: "Air speed — draughts. Moving air carries heat off the skin, so the same temperature feels colder."},
];
const FEAT_INDEX = {};
FEATURES.forEach((f, i) => (FEAT_INDEX[f.id] = i));

/** Encoded width of one feature: 2 for a sin/cos pair, 1 otherwise. */
function featWidth(f) { return f.cyc ? 2 : 1; }

/** Total encoded input dimension for the active feature ids. */
function encodedDim(ids) {
  let d = 0;
  ids.forEach((id) => { d += featWidth(FEATURES[FEAT_INDEX[id]]); });
  return d;
}

/** Names of the encoded inputs, in order — used by the diagram and arithmetic. */
function encodedNames(ids) {
  const out = [];
  ids.forEach((id) => {
    const f = FEATURES[FEAT_INDEX[id]];
    if (f.cyc) { out.push('sin ' + f.short.toLowerCase()); out.push('cos ' + f.short.toLowerCase()); }
    else out.push(f.short);
  });
  return out;
}

/**
 * Encodes one sensor state into the network input vector.
 * @param {{ta,rh,tw,tout,hour,doy,mov,vair}} s
 * @param {string[]} ids active feature ids
 */
function encodeState(s, ids) {
  const x = new Float32Array(encodedDim(ids));
  let j = 0;
  ids.forEach((id) => {
    const f = FEATURES[FEAT_INDEX[id]];
    const v = s[id];
    if (f.cyc) {
      const a = 2 * Math.PI * v / f.cyc;
      x[j++] = Math.sin(a); x[j++] = Math.cos(a);
    } else {
      x[j++] = (v - f.mean) / f.span;
    }
  });
  return x;
}

/* ------------------------------------------------- dataset generation */

/** A random day of the year within the chosen coverage. */
function pickDoy(coverage) {
  if (coverage === 'winter') return (randInt(0, 120) + 305) % 365 + 1;  // Nov–Feb
  if (coverage === 'summer') return randInt(152, 244);                  // Jun–Aug
  if (coverage === 'shoulder') return Math.random() < 0.5 ? randInt(60, 152) : randInt(244, 335);
  return randInt(1, 366);
}

/**
 * Generates a labelled dataset by simulation.
 *
 * The heater does NOT try to please anyone here — it explores. Each simulated
 * day it either chases a random setpoint (12…28 °C, bang-bang) or slowly
 * sweeps its power, so the votes cover the whole range from shivering to
 * overheated. Windows open now and then, guests arrive, seasons change.
 *
 * @param {number} n how many votes to collect
 * @param {{coverage:string, pref:number, sigma:number, sensorNoise:number,
 *          balance:boolean, insulation:number, pmax:number, heater:string}} opt
 */
function makeDataset(n, ids, opt) {
  const xs = [], ys = [], states = [], pmvs = [];
  const counts = [0, 0, 0];
  // votes of every kind are needed: a room that was never overheated cannot
  // teach the warm edge of the zone, so collection keeps simulating until each
  // class holds at least 18% of the target
  const enough = () => xs.length >= n && Math.min(...counts) >= 0.18 * n;
  let guard = 0;
  while (!enough() && guard++ < 6000) {
    const sim = simInit({
      doy: pickDoy(opt.coverage),
      insulation: opt.insulation, pmax: opt.pmax, heater: opt.heater,
    });
    // one exploring day per chunk. Exploration is curious about what it
    // lacks: short of "too warm" votes it overheats on purpose, short of
    // "too cold" ones it lets the room drop.
    const needWarm = xs.length > 0.3 * n && counts[2] < 0.18 * n;
    const needCold = xs.length > 0.3 * n && counts[0] < 0.18 * n;
    const mode = needWarm || needCold || Math.random() < 0.55 ? 'setpoint' : 'sweep';
    let setp = needWarm ? rand(26, 31) : needCold ? rand(11, 15) : rand(12, 31);
    let u = Math.random(), phase = rand(0, 6.28);
    const minutes = 1440;
    for (let m = 0; m < minutes && !enough(); m += 2) {
      if (mode === 'setpoint') {
        if (!needWarm && !needCold && Math.random() < 0.0015) setp = rand(12, 31); // new target now and then
        u = sim.ta < setp ? 1 : 0;
      } else {
        u = clamp(0.5 + 0.5 * Math.sin(phase + m / rand(180, 420)) + 0.25 * randn() * 0.1, 0, 1);
      }
      if (Math.random() < 0.0008) sim.window = !sim.window;   // airing the room
      if (sim.window && Math.random() < 0.02) sim.window = false;
      if (sim.guests === 0 && sim.hour > 17 && Math.random() < 0.0004) {
        sim.guests = randInt(2, 6); sim.guestUntil = sim.minute + randInt(90, 180);
      }
      simStep(sim, u); simStep(sim, u);                       // 2-minute stride
      const sens = sensorState(sim, opt.sensorNoise);
      const v = simMaybeVote(sim, opt.pref, opt.sigma, sens);
      if (v) {
        xs.push(encodeState(v.state, ids));
        ys.push(v.label);
        states.push(v.state);
        pmvs.push(v.pmv);
        counts[v.label]++;
      }
    }
  }

  return finishDataset(xs, ys, states, pmvs, opt.balance);
}

/** Shared tail of every dataset: optional class balancing and a shuffle. */
function finishDataset(xs, ys, states, pmvs, balance) {
  // class balancing: duplicate minority votes until the classes match
  if (balance) {
    const byClass = [[], [], []];
    ys.forEach((y, i) => byClass[y].push(i));
    const most = Math.max(...byClass.map((a) => a.length));
    byClass.forEach((idxs) => {
      if (!idxs.length) return;
      for (let k = idxs.length; k < most; k++) {
        const i = idxs[Math.floor(Math.random() * idxs.length)];
        xs.push(xs[i]); ys.push(ys[i]); states.push(states[i]); pmvs.push(pmvs[i]);
      }
    });
  }
  // shuffle
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
    [ys[i], ys[j]] = [ys[j], ys[i]];
    [states[i], states[j]] = [states[j], states[i]];
    [pmvs[i], pmvs[j]] = [pmvs[j], pmvs[i]];
  }
  return { xs, ys, states, pmvs, n: xs.length };
}

/**
 * The other way to get labels: a CONTROLLED STUDY instead of a lived-in home.
 * Only the sensors the network can actually see are swept — evenly across
 * their whole range, so coverage over those dimensions is uniform by
 * construction. Everything the network cannot see follows one typical
 * household instead of being randomised: the occupants sleep at night (with
 * day-to-day jitter), walls sit just below the air, humidity and weather take
 * physically ordinary values, and switched-off time inputs are pinned at the
 * probe sliders. Varying what the model sees while holding the rest typical
 * is what keeps the hidden dimensions from smearing the labels into noise.
 */
function makeUniformDataset(n, ids, opt) {
  const xs = [], ys = [], states = [], pmvs = [];
  const active = {};
  ids.forEach((id) => (active[id] = true));
  const F = {};
  FEATURES.forEach((f) => (F[f.id] = f));
  const pin = opt.pin || {};
  const k = opt.sensorNoise || 0;
  for (let i = 0; i < n; i++) {
    const doy = active.doy ? pickDoy(opt.coverage) : Math.round(pin.doy || 20);
    const hour = active.hour ? rand(0, 24) : (pin.hour != null ? pin.hour : 12);
    // the typical household behind the sensors the network cannot see
    const wake = 6.75 + rand(-0.5, 0.9), bed = 22.75 + rand(-0.6, 0.9);
    const asleep = hour >= bed || hour < wake;
    const ta = active.ta ? rand(F.ta.min, F.ta.max) : (pin.ta != null ? pin.ta : 21);
    const s = {
      ta, hour, doy,
      mov: active.mov ? rand(0, 1) : (asleep ? rand(0.01, 0.06) : rand(0.25, 0.5)),
      vair: active.vair ? rand(0, 1) : 0.04 + rand(0, 0.05),
      tout: active.tout ? rand(F.tout.min, F.tout.max) : outdoorTemp(doy, hour, 2.5 * randn()),
      tw: active.tw ? rand(F.tw.min, F.tw.max) : ta - rand(0.3, 1.4),
      rh: active.rh ? rand(F.rh.min, F.rh.max) : clamp(48 - 0.8 * (ta - 21) + 4 * randn(), 25, 70),
    };
    // the sensors report the moment imperfectly, same as in the simulation
    const m = {
      ta: s.ta + 0.15 * k * randn(),
      rh: clamp(s.rh + 2.0 * k * randn(), 10, 100),
      tw: s.tw + 0.2 * k * randn(),
      tout: s.tout + 0.3 * k * randn(),
      hour: s.hour, doy: s.doy,
      mov: clamp(s.mov + 0.02 * k * randn(), 0, 1),
      vair: Math.max(0, s.vair + 0.015 * k * randn()),
    };
    const v = voteSample(m, opt.pref, opt.sigma);
    xs.push(encodeState(m, ids));
    ys.push(v.label);
    states.push(m);
    pmvs.push(v.pmv);
  }
  return finishDataset(xs, ys, states, pmvs, opt.balance);
}

/**
 * The deterministic centre of that typical household — the comfort map draws
 * its true zone through this in uniform mode, so the dotted outline agrees
 * with the votes it generated. `keep` lists ids that must not be overwritten
 * (the active inputs plus the map axes).
 */
function typicalFill(s, keep) {
  const a = {};
  keep.forEach((id) => (a[id] = true));
  const out = Object.assign({}, s);
  const asleep = out.hour >= 22.75 || out.hour < 6.75;
  if (!a.mov) out.mov = asleep ? 0.03 : 0.35;
  if (!a.vair) out.vair = 0.06;
  if (!a.tout) out.tout = outdoorTemp(out.doy, out.hour, 0);
  if (!a.tw) out.tw = out.ta - 0.85;
  if (!a.rh) out.rh = clamp(48 - 0.8 * (out.ta - 21), 25, 70);
  return out;
}

/** Class counts of a dataset, for the data panel. */
function classCounts(ds) {
  const c = [0, 0, 0];
  for (let i = 0; i < ds.n; i++) c[ds.ys[i]]++;
  return c;
}

/**
 * A random but plausible VOTING moment — someone is present and behaving
 * normally for the hour. Used by the quick test and the probe, so it stays on
 * the manifold where votes actually happen; a state no one ever votes in
 * (an empty room) would only measure extrapolation.
 */
function randomState(coverage) {
  const doy = pickDoy(coverage || 'year');
  const hour = rand(0, 24);
  const tout = outdoorTemp(doy, hour, 4 * randn());
  // indoors is bounded by what heater and weather can do: a winter room tops
  // out around 27 °C, a summer room will not drop to 12 °C
  const ta = rand(Math.max(12, tout - 6), Math.min(30, Math.max(tout + 2, 20) + 7));
  const inBed = sleepHours(hour) ? (isNight(hour) ? 0.85 : 0.4) : 0;
  const mov = Math.random() < inBed ? rand(0.01, 0.08) : rand(0.1, 0.8);
  return {
    ta,
    rh: clamp(rand(28, 68) - 0.8 * (ta - 21), 18, 92),
    tw: ta + rand(-2.2, 1.2) + clamp((tout - ta) * 0.06, -1.5, 1.5),
    tout, hour, doy, mov,
    vair: Math.random() < 0.88 ? rand(0.03, 0.18) : rand(0.18, 0.6),
  };
}
