/* comfort.js — the ground truth: Fanger's PMV comfort model (ISO 7730).
 *
 * The network never sees any of this. It only gets raw sensor values and the
 * occupants' votes; PMV is what generates those votes in the simulator, and
 * what the comfort map draws as the "true" zone for comparison.
 */

const CLASSES = [
  { id: 'cold', name: 'Too cold',    short: 'Cold',  color: '#0877bd' },
  { id: 'comf', name: 'Comfortable', short: 'OK',    color: '#2e9e5b' },
  { id: 'warm', name: 'Too warm',    short: 'Warm',  color: '#e0342b' },
];
const CLASS_INDEX = {};
CLASSES.forEach((c, i) => (CLASS_INDEX[c.id] = i));

function rand(a, b) { return a + Math.random() * (b - a); }
function randInt(a, b) { return a + Math.floor(Math.random() * (b - a)); }
function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

let _spare = null;
function randn() {
  if (_spare !== null) { const s = _spare; _spare = null; return s; }
  let u = 0, v = 0, s = 0;
  do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s = u * u + v * v; }
  while (s === 0 || s >= 1);
  const m = Math.sqrt(-2 * Math.log(s) / s);
  _spare = v * m;
  return u * m;
}

/**
 * PMV / PPD after Fanger, as standardised in ISO 7730 (the BASIC listing of
 * annex D, translated). Heat balance of a clothed body: metabolic rate in,
 * six heat losses out; PMV is the imbalance scaled by thermal sensitivity.
 *
 * @param {number} ta  air temperature [°C]
 * @param {number} tr  mean radiant temperature [°C] — the walls
 * @param {number} vel air speed [m/s]
 * @param {number} rh  relative humidity [%]
 * @param {number} met metabolic rate [met], 1 met = 58.15 W/m²
 * @param {number} clo clothing insulation [clo], 1 clo = 0.155 m²K/W
 * @returns {{pmv:number, ppd:number}} vote scale −3…+3, % dissatisfied
 */
function pmvPPD(ta, tr, vel, rh, met, clo) {
  const pa = rh * 10 * Math.exp(16.6536 - 4030.183 / (ta + 235)); // vapour pressure [Pa]
  const icl = 0.155 * clo;             // clothing resistance [m²K/W]
  const m = met * 58.15;               // metabolic rate [W/m²], no external work
  const fcl = icl <= 0.078 ? 1 + 1.29 * icl : 1.05 + 0.645 * icl; // clothing area factor
  const hcf = 12.1 * Math.sqrt(Math.max(0.05, vel));              // forced convection
  const taa = ta + 273, tra = tr + 273;

  // clothing surface temperature by fixed-point iteration
  let tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1);
  const p1 = icl * fcl, p2 = p1 * 3.96, p3 = p1 * 100, p4 = p1 * taa;
  const p5 = 308.7 - 0.028 * m + p2 * Math.pow(tra / 100, 4);
  let xn = tcla / 100, xf = xn, hc = hcf;
  for (let i = 0; i < 150; i++) {
    xf = (xf + xn) / 2;
    const hcn = 2.38 * Math.pow(Math.abs(100 * xf - taa), 0.25); // natural convection
    hc = Math.max(hcf, hcn);
    xn = (p5 + p4 * hc - p2 * Math.pow(xf, 4)) / (100 + p3 * hc);
    if (Math.abs(xn - xf) <= 0.00015) break;
  }
  const tcl = 100 * xn - 273;

  const hl1 = 3.05 * 0.001 * (5733 - 6.99 * m - pa);       // skin diffusion
  const hl2 = m > 58.15 ? 0.42 * (m - 58.15) : 0;          // sweating
  const hl3 = 1.7 * 0.00001 * m * (5867 - pa);             // latent respiration
  const hl4 = 0.0014 * m * (34 - ta);                      // dry respiration
  const hl5 = 3.96 * fcl * (Math.pow(xn, 4) - Math.pow(tra / 100, 4)); // radiation
  const hl6 = fcl * hc * (tcl - ta);                       // convection

  const ts = 0.303 * Math.exp(-0.036 * m) + 0.028;         // thermal sensitivity
  const pmv = ts * (m - hl1 - hl2 - hl3 - hl4 - hl5 - hl6);
  const ppd = 100 - 95 * Math.exp(-0.03353 * Math.pow(pmv, 4) - 0.2179 * pmv * pmv);
  return { pmv, ppd };
}

/* ------------------------------------------------- occupant behaviour */

/** Seasonal daily-mean outdoor temperature [°C] — a Sofia-like climate. */
function climateMean(doy) {
  return 10.5 - 11.5 * Math.cos(2 * Math.PI * (doy - 15) / 365);
}

/** Is that hour normally spent asleep? Used with low movement to infer "in bed". */
function isNight(hour) { return hour >= 22.75 || hour < 6.75; }

/** Hours one could plausibly be in bed — households differ and drift day to
 * day, so the truth infers "asleep" from low movement anywhere in this window,
 * not from one fixed bedtime. */
function sleepHours(hour) { return hour >= 21.5 || hour < 9.5; }

/**
 * Clothing the occupants are wearing, derived from the state the network sees.
 * People dress by season, not by the instantaneous weather; in bed a duvet adds
 * a lot of insulation — the reason bedrooms are comfortable so much cooler.
 */
function cloOf(doy, hour, mov) {
  const asleep = sleepHours(hour) && mov < 0.12;
  let clo = clamp(0.9 - 0.02 * (climateMean(doy) - 5), 0.45, 1.1);
  if (asleep) {
    // bed and bedding: a thick duvet in January, little more than a sheet in
    // July — which is why winter bedrooms are comfortable at 17 °C and summer
    // nights at 24 °C are not
    const w = clamp((14 - climateMean(doy)) / 14, 0, 1);
    clo += 0.9 + 1.4 * w;
  }
  return clo;
}

/** Metabolic rate from the movement level the sensor reports. */
function metOf(hour, mov) {
  const asleep = sleepHours(hour) && mov < 0.12;
  if (asleep) return 0.75;
  return 0.9 + 1.0 * clamp(mov, 0, 1);
}

/**
 * The true comfort evaluation for a sensor state — PMV with met and clo filled
 * in from the occupants' routine. `pref` shifts the whole scale: some
 * households simply like it warmer (+) or cooler (−).
 * @param {{ta,rh,tw,tout,hour,doy,mov,vair}} s
 * @returns {{pmv:number, ppd:number, label:number}} label indexes CLASSES
 */
function comfortTruth(s, pref) {
  const met = metOf(s.hour, s.mov);
  const clo = cloOf(s.doy, s.hour, s.mov);
  const r = pmvPPD(s.ta, s.tw, Math.max(0.05, s.vair), s.rh, met, clo);
  const pmv = r.pmv + (pref || 0);
  const label = pmv < -0.5 ? CLASS_INDEX.cold : pmv > 0.5 ? CLASS_INDEX.warm : CLASS_INDEX.comf;
  return { pmv, ppd: r.ppd, label, met, clo };
}

/**
 * One noisy vote. Real people are not thermometers: the same person at the same
 * PMV answers differently on different days. `sigma` is that inconsistency.
 */
function voteSample(s, pref, sigma) {
  const t = comfortTruth(s, pref);
  const pmv = t.pmv + sigma * randn();
  const label = pmv < -0.5 ? CLASS_INDEX.cold : pmv > 0.5 ? CLASS_INDEX.warm : CLASS_INDEX.comf;
  return { label, pmv: t.pmv, ppd: t.ppd };
}
