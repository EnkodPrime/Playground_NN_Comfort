/* sim.js — a small physical world for the network to live in.
 *
 * One room with a heater: outdoor weather, a two-node thermal model (air and
 * wall), a moisture balance, an occupancy schedule and the heater itself.
 * Everything is stepped in one-minute increments; the same simulator generates
 * the training data and runs the closed control loop.
 */

const HOUSE = {
  volume: 45,          // room air volume [m³]
  cAir: 8e5,           // air + furniture heat capacity [J/K]
  cWall: 2.0e7,        // wall/floor mass heat capacity [J/K]
  hAirWall: 550,       // air ↔ wall surface conductance [W/K]
  achBase: 0.6,        // background air changes per hour
  achWindow: 7,        // with a window open
  perPerson: 95,       // sensible heat per person [W]
  moistPerPerson: 45,  // moisture per person [g/h]
  appliances: 60,      // lights & devices when someone is up [W]
};

/* A cooler is a different machine from a heater: a fan coil reacts in a couple
 * of minutes, blows all of its effect straight into the air (no radiant share),
 * stirs the room, and condenses moisture out of it. */
const COOLER = { tau: 2, stirs: true, dryGramsPerKWmin: 4 };

/* Heater types differ in inertia and in how much of the heat goes to the walls
 * (radiant fraction) rather than straight into the air. */
const HEATERS = {
  fan:      { name: 'Fan heater',     tau: 1.5,  radiant: 0.08, stirs: true },
  radiator: { name: 'Radiator',       tau: 9,    radiant: 0.30, stirs: false },
  stove:    { name: 'Masonry stove',  tau: 35,   radiant: 0.55, stirs: false },
};

/** Saturation vapour density [g/m³] at temperature t [°C] (Magnus formula). */
function satMoisture(t) {
  const psat = 6.112 * Math.exp(17.62 * t / (243.12 + t));   // [hPa]
  return 216.7 * psat / (t + 273.15);
}

/** Outdoor temperature: seasonal mean + weather front + day–night swing. */
function outdoorTemp(doy, hour, front) {
  return climateMean(doy) + front - 4.5 * Math.cos(2 * Math.PI * (hour - 14.5) / 24);
}

/**
 * Creates a fresh simulation.
 * @param {{doy:number, insulation:number, pmax:number, heater:string, residents:number}} cfg
 *   insulation: wall→out conductance [W/K] (40 good … 160 poor)
 */
function simInit(cfg) {
  const doy = cfg.doy || 15;
  const front = 4 * randn();
  const tout = outdoorTemp(doy, 0, front);
  const ta = rand(18, 22);
  const s = {
    cfg: Object.assign({ insulation: 80, pmax: 2000, coolMax: 0, heater: 'radiator', residents: 2 }, cfg),
    minute: 0,                    // minutes since simulation start
    doy, hour: 0, dow: (doy + 2) % 7,   // day 1 ≈ Wednesday; 5,6 are the weekend
    front,                        // slow weather deviation [K]
    tout,
    ta, tw: ta - rand(0.5, 1.5),  // walls slightly behind the air
    moist: satMoisture(ta) * 0.45,   // indoor absolute humidity [g/m³]
    power: 0,                     // actual heater output [W] (lags the command)
    cool: 0,                      // actual cooler output [W] (lags its own way)
    window: false,
    guests: 0, guestUntil: 0,
    shower: 0,                    // minutes of shower moisture left
    occ: { n: 0, act: 0, asleep: false },
    mov: 0, movEma: 0, vair: 0.05, rh: 50,
    lastVoteMin: -999, lastVotePmv: 0,
  };
  drawSleepTimes(s);
  stepOccupancy(s);
  return s;
}

/** Tonight's bedtime and tomorrow's alarm — different every day, later at
 * weekends. Fixed clock times would stamp the dataset with artificial vertical
 * stripes of votes at the exact same minute of every simulated day. */
function drawSleepTimes(s) {
  const weekend = s.dow >= 5;
  s.wake = 6.75 + rand(-0.5, 0.9) + (weekend ? rand(0.3, 1.2) : 0);
  s.bed = 22.75 + rand(-0.6, 0.9);
}

/** Occupancy and activity from the time of day. Rough but recognisable. */
function stepOccupancy(s) {
  const h = s.hour, weekend = s.dow >= 5;
  const res = s.cfg.residents;
  let n = res, act = 0.35, asleep = false;

  if (h >= s.bed || h < s.wake) { act = 0.03; asleep = true; }
  else if (h < s.wake + 1.5) { act = 0.55; }              // the morning rush
  else if (!weekend && h < 17.5) {
    if (s._dayHome === undefined) s._dayHome = Math.random() < 0.35;
    n = s._dayHome ? 1 : 0; act = 0.3;                    // at work, or a home-office day
  } else if (weekend && h < 17.5) {
    if (s._wkOut === undefined) s._wkOut = Math.random() < 0.5;
    n = (s._wkOut && h >= 15) ? 0 : res; act = 0.42;      // maybe an afternoon out
  } else {
    act = (h >= 18.5 && h < 19.25) ? 0.6 : 0.35;          // cooking, then the sofa
  }
  if (h < 0.12) { s._dayHome = undefined; s._wkOut = undefined; drawSleepTimes(s); }

  n += s.guests;
  if (n === 0) act = 0;
  s.occ = { n, act, asleep };
}

/**
 * Advances the world by one minute with a SIGNED command u ∈ [−1, +1]:
 * positive is a fraction of heating power, negative a fraction of cooling.
 * One number, two machines — never both at once, which is exactly what a real
 * HVAC controller must guarantee.
 * All the physics is explicit Euler at dt = 60 s — every term is a W/K
 * conductance times a temperature difference.
 */
function simStep(s, u) {
  const dt = 60;
  const cfg = s.cfg;

  // clock
  s.minute++;
  s.hour += 1 / 60;
  if (s.hour >= 24) { s.hour -= 24; s.doy = (s.doy % 365) + 1; s.dow = (s.dow + 1) % 7; }
  if (s.minute % 5 === 0) stepOccupancy(s);

  // weather: the front is a slow random walk pulled back to zero
  s.front += (-s.front / (2.5 * 1440)) + 0.10 * randn() / Math.sqrt(1440 / dt * 0.1);
  s.tout = outdoorTemp(s.doy, s.hour, s.front);

  // heater and cooler, each with its own first-order lag. Part of the heat is
  // radiant (into the wall); cooling is all air.
  const ht = HEATERS[cfg.heater] || HEATERS.radiator;
  const uHeat = Math.max(0, u), uCool = Math.max(0, -u);
  s.power += (uHeat * cfg.pmax - s.power) * (1 - Math.exp(-1 / ht.tau));
  s.cool += (uCool * (cfg.coolMax || 0) - s.cool) * (1 - Math.exp(-1 / COOLER.tau));
  const qWall = s.power * ht.radiant;
  const qAir = s.power - qWall - s.cool;

  // internal gains
  const qPeople = s.occ.n * HOUSE.perPerson;
  const qApp = s.occ.n > 0 && !s.occ.asleep ? HOUSE.appliances : 0;

  // ventilation
  const ach = s.window ? HOUSE.achWindow : HOUSE.achBase;
  const hVent = 0.34 * ach * HOUSE.volume;              // [W/K]

  // two-node thermal balance
  const dTa = (HOUSE.hAirWall * (s.tw - s.ta) + hVent * (s.tout - s.ta)
             + qAir + qPeople + qApp) * dt / HOUSE.cAir;
  const dTw = (HOUSE.hAirWall * (s.ta - s.tw) + cfg.insulation * (s.tout - s.tw)
             + qWall) * dt / HOUSE.cWall;
  s.ta += dTa; s.tw += dTw;

  // moisture balance: people and showers add, ventilation swaps with outdoors
  if (s.shower > 0) s.shower--;
  const morning = s.hour > 6.9 && s.hour < 8.4, evening = s.hour > 21.2 && s.hour < 22.7;
  if (s.occ.n > 0 && !s.occ.asleep && (morning || evening) && s.shower === 0 && Math.random() < 0.004) {
    s.shower = 15;                                       // someone takes a shower
  }
  const gen = s.occ.n * HOUSE.moistPerPerson / 60 + (s.shower > 0 ? 40 : 0)
    - COOLER.dryGramsPerKWmin * s.cool / 1000;      // the cooler wrings the air out
  const moistOut = satMoisture(s.tout) * clamp(0.75 - 0.012 * s.front, 0.35, 0.98);
  s.moist += gen / HOUSE.volume - (ach / 60) * (s.moist - moistOut);
  s.moist = Math.max(0.3, s.moist);
  s.rh = clamp(100 * s.moist / satMoisture(s.ta), 15, 98);

  // what the movement and draught sensors report
  const movRaw = s.occ.n === 0 ? 0
    : clamp(s.occ.act * (0.55 + 0.15 * Math.min(s.occ.n, 4)) + 0.04 * randn(), 0, 1);
  s.mov += (movRaw - s.mov) * 0.5;
  s.movEma += (s.mov - s.movEma) / 20;         // ~20-minute metabolic memory
  const stir = ((ht.stirs && s.power > 0.1 * cfg.pmax) ? 0.12 : 0)
    + (s.cool > 0.1 * (cfg.coolMax || 1) ? 0.16 : 0);
  s.vair = clamp(0.04 + 0.05 * s.occ.act + (s.window ? 0.4 : 0) + stir + 0.01 * randn(), 0, 1.2);

  // guests go home
  if (s.guests > 0 && s.minute >= s.guestUntil) s.guests = 0;
}

/** The eight sensor readings the network is allowed to see. */
function sensorState(s, noiseScale) {
  const k = noiseScale || 0;
  return {
    ta: s.ta + 0.15 * k * randn(),
    rh: clamp(s.rh + 2.0 * k * randn(), 10, 100),
    tw: s.tw + 0.2 * k * randn(),
    tout: s.tout + 0.3 * k * randn(),
    hour: s.hour,
    doy: s.doy,
    mov: clamp(s.mov + 0.02 * k * randn(), 0, 1),
    movEff: s.movEma,                    // what the body actually feels
    vair: Math.max(0, s.vair + 0.015 * k * randn()),
  };
}

/**
 * Do the occupants feel like saying something right now? People comment every
 * 20–50 minutes when they are up, rarely in their sleep — unless discomfort
 * changed a lot, which wakes an opinion (or the sleeper) immediately.
 */
function simMaybeVote(s, pref, sigma, sensors) {
  if (s.occ.n === 0) return null;
  const since = s.minute - s.lastVoteMin;
  const truth = comfortTruth(sensors, pref);
  const changed = Math.abs(truth.pmv - s.lastVotePmv) > 0.7 && since > 8;
  // A quiet night is continuous implicit feedback — nobody complained for
  // hours — so sleep is sampled almost as densely as waking hours, or the
  // night side of the comfort zone gets several times fewer examples.
  const period = s.occ.asleep ? 45 : 32;
  const due = since > period * rand(0.7, 1.5);
  if (!due && !changed) return null;
  // Sleepers too must produce COMFORTABLE examples, not only complaints: a
  // night with no discomfort is evidence, reported as the calm assessment of
  // someone stirring (or the morning-after "slept fine"). Without these the
  // night-time zone has no positive examples at all and cannot be learned.
  s.lastVoteMin = s.minute;
  s.lastVotePmv = truth.pmv;
  const v = voteSample(sensors, pref, sigma);
  return { label: v.label, pmv: v.pmv, state: sensors };
}
