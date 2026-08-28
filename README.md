# NN Comfort Playground

An interactive playground in the spirit of the
[TensorFlow Playground](https://playground.tensorflow.org/), but the two inputs are not x and y of
a point — they are eight sensors of a home. A small neural network learns a household's **comfort
zone** from nothing but occupant votes (*too cold / comfortable / too warm*), and then gets to run
the heating: a radiator, a fan heater or a masonry stove, in a simulated room with real weather,
walls, moisture and people.

The comfort zone is nowhere in the code the network sees. It emerges in the weights, and the
thermostat follows it around as the season, the draught and the people change.

Everything runs in the browser. No dependencies, no build step, no server.

**Live:** https://enkodprime.github.io/Playground_NN_Comfort/

A sibling project, [Playground_NN_architectures](https://github.com/EnkodPrime/Playground_NN_architectures),
does the same kind of thing for signal architectures on 50 Hz mains voltage.

## Running locally

Open `index.html` directly, or serve the folder:

```bash
python -m http.server 8765
```

## The task

Every training example is **one moment in the room**: the sensor readings at the instant someone
said something about the temperature, and what they said.

| Input | Encoding |
|---|---|
| Room air temperature [°C] | standardised |
| Room humidity [%] | standardised |
| Wall temperature [°C] | standardised — the radiant half of comfort |
| Outdoor temperature [°C] | standardised |
| Time of day | sin + cos — a clock is a circle |
| Day of year | sin + cos — so is a calendar |
| Movement of people | standardised — a PIR-style activity level |
| Air movement [m/s] | standardised — draughts |

Any input can be switched off in the Data panel to see what the network can no longer learn.
The output is a three-way softmax: <em>too cold / comfortable / too warm</em>.

## Where the votes come from

The generating truth is **Fanger's PMV model (ISO 7730)** — the heat balance of a clothed body,
implemented in `js/comfort.js` directly from the standard. PMV needs two things no sensor
measures, so they follow from the occupants' routine:

* **Metabolic rate** comes from the movement level: 0.75 met asleep, up to ~1.9 met for a busy
  household.
* **Clothing** follows the season (1.1 clo in January, 0.5 in July) — and in bed the bedding is
  added: a thick duvet in winter, little more than a sheet in summer. That is why a winter bedroom
  is comfortable at 17 °C, a summer night at 24 °C, and the evening sofa at 21.5 °C — three
  different answers the network has to reconcile through the time, season and movement inputs.

**PMV alone is not the whole story, and the standard says so.** It is a *whole-body,
steady-state* index — it cannot know that a room is falling, or that something is blowing on your
neck. ISO 7730 treats those separately, and so does the generating model here:

* **Draught rate** — the standard's own local-discomfort formula,
  `DR = (34 − ta)·(v − 0.05)^0.62·(0.37·v·Tu + 3.14)`, the percentage of people dissatisfied by
  moving air. Still air scores 3%, a working cooler 23%, an open window in winter 72% — which is
  −1.1 PMV on its own. This is the most common complaint about air conditioning, and the reason
  an open window is felt long before the thermometer moves.
* **The element itself** — a radiant heater warms the body directly, minutes before the wall mass
  follows, so its output raises the radiant temperature the skin sees. It is also an *input* the
  network may use (`Heating / cooling output`): a controller always knows what it is doing, and
  it matters. Adding that one input took a CNN from 89.0% to 92.4% on a controlled study, and
  from unlearnable to 92.5% on the moments when the element is working hard.
* **Transients** — a room falling at 5 K/h feels colder than the same room sitting still, because
  the skin answers in seconds while the core lags. The rate is nowhere in the inputs: it lives in
  the *history*, so only a model that reads the window can recover it.

A vote is PMV plus these, plus preference, plus noise, thresholded at ±0.5: real households simply like it
warmer or cooler (the *preference* slider shifts the truth), and real people are not thermometers
(the *inconsistency* slider is the standard deviation of their answers). The network never sees a
PMV value — only sensors and votes.

## The simulated home

`js/sim.js` steps a small physical world one minute at a time; the same simulator generates the
dataset and runs the live control loop.

* **Weather** — a Sofia-like seasonal curve, a day–night swing and a slowly wandering front.
* **Building** — a two-node RC model: room air (fast) and wall mass (slow), coupled by a surface
  conductance, losing heat through the envelope (the *insulation* slider) and through ventilation;
  opening a window multiplies the air exchange and the draught.
* **Moisture** — people and showers add vapour, ventilation swaps it with outdoors; relative
  humidity then falls as the room warms, which is a real coupling the network can pick up.
* **People** — a weekday/weekend schedule with mornings, work hours, evenings, sleep, random
  days at home, guests; each person adds heat, moisture and movement.
* **Heaters** — first-order lag and a radiant split: a fan heater is fast and all-air, a radiator
  slower and 30% radiant, a masonry stove very slow and mostly radiant. The slow heaters are why
  the wall-temperature input earns its place.

**Two sources of votes.** The *Votes come from* switch chooses between a lived-in
household — the simulation below, with its uneven, behaviour-shaped coverage — and *uniform
sampling*: moments drawn evenly across the whole sensor space and labelled straight from the
comfort theory, the way a lab study of ISO 7730 would. Comparing the two is the difference
between learning from life and learning from the textbook.

**Data collection is exploration, not comfort.** While collecting votes the heater deliberately
wanders — random setpoints, slow power sweeps — so the room visits shivering, comfort and
overheating across the whole year. The collector is curious about what it lacks: short of
"too warm" votes it overheats on purpose until every class holds a fair share.

## The networks

Two architectures share the data, the map and the control loop, switchable at the top of the
page — both read the same input: the last ~64 minutes of every sensor as a multi-channel window.

* **1D CNN · last hour** — the convolutional architecture of the sibling signal playground,
  given several input channels: 1–3 layers of sliding kernels read rises, falls and recent
  activity. The output head is selectable, as in the sibling project: Global Average Pool (each
  filter's verdict over the whole hour), Global Max Pool (its loudest moment) or Flatten (the
  vote sees every position — the only head that knows *when* something happened).
* **RNN · last hour** — a recurrence walking through the same hour step by step: simple tanh
  RNN, GRU or LSTM cells (1–2 layers of 1–10 units), trained by backpropagation through time
  with global gradient-norm clipping, read out by the last state, the mean or the max over time.
  Instead of sliding patterns it carries a state — a little memory updated every minute.

Architectures matter here because comfort has **memory**: the generating model gives the
body a ~20-minute metabolic inertia (what counts is the activity of the last quarter hour, not
this second), and the datasets contain moments where someone just sat down or the room is
mid-warm-up. A snapshot network cannot represent those — the window network can, and wins
exactly there.

Everything is written from scratch in `js/nn.js` and `js/rnn.js` — dense, 1D convolutional and
recurrent layers, forward, backprop/BPTT, Adam, softmax and cross-entropy over plain
`Float32Array`s; every gradient, gates included, verified against numeric differences.

Every node in the diagram is drawn the TensorFlow-Playground way: its response over the two
features chosen as the comfort-map axes, all other sensors pinned at the probe. Clicking a node
expands the entire computation in the arithmetic panel — every `w·x` with the sensor's name on the
row, the bias, the activation, and for outputs the softmax and the cross-entropy against the true
model, all live.

## The comfort map

The central plot is a 2D slice through the learned function — say temperature × humidity, or
temperature × draught — with three things on it:

* the network's vote across the plane, saturation showing its confidence;
* the **true zone** of the generating PMV model as a dotted outline;
* the training votes that fall near this slice.

Learning has worked when the colours fill the dotted outline. Drag the probe sliders and both the
learned and the true zone move together — or apart, where the network is extrapolating.

**Sensor dropout.** A toolbar switch trains with about a third of the examples missing one
random channel, so the network learns to answer from any sensor subset — and *Fail a sensor* in
the live loop delivers a chosen channel as zeros while the room's physics stays real. Together
they are the missing-sensor story of real hardware: a normally trained network stumbles when a
sensor dies; a dropout-trained one degrades gracefully, at almost no cost while everything works.

## From a vote to a control signal

The network answers *too cold / comfortable / too warm*. Turning that into wires takes one more
step, and the loop offers the two ways it is done in practice:

* **Comfort zone → setpoint → PI.** Scan the air-temperature axis through the network at the
  current conditions, take the most comfortable point as **T\***, and let a conventional PI
  controller chase it. The learned model supplies the *target*; a century of control engineering
  supplies the *tracking*. Interpretable — you can read T\* off the chart and argue with it.
* **Demand signal (direct).** No setpoint at all. The class probabilities already are a signed
  comfort error:

  ```
  demand = p(too warm) − p(too cold)     ∈ [−1, +1]
  ```

  −1 means "certainly too cold", +1 "certainly too warm", ≈0 inside the zone. Drive the plant
  against it — heat when negative, cool when positive — with a slow filter (comfort is a slow
  quantity, sensors are not) and a deadband. This is what ships on a device that has no
  thermostat logic in it at all.

Either way the command is **one signed number**: positive is a fraction of heating power,
negative a fraction of cooling, and the output stage decides what the wires do — **modulating**
(a valve or inverter unit anywhere between 0 and 100%) or **relay** (a plain contactor with
deadband, hysteresis and a minimum cycle time, because short-cycling destroys compressors).
The ribbon draws the command signed around zero, with a strip below it showing what actually
switched on: red heating, blue cooling, grey off.

## The control loop

Press ▶ on the live loop and the trained network is put in charge of the plant:

1. Every few simulated minutes the controller scans the air-temperature axis through the network
   at the current humidity, walls, weather, hour and movement: *"at which temperature would these
   people, right now, say comfortable?"*
2. The scan must produce a **credible zone** — one contiguous stretch the network is confident
   about; a few stray cells of extrapolation are ignored.
3. The probability-weighted centre of that stretch becomes the setpoint **T\***, and a small PI
   controller drives the plant towards it. Nobody home? The scan asks for a typical seated
   occupant, holds 1.5 °C below the answer and widens the deadband.

The chart shows the room, the walls, the weather, the learned band, T\*, the heater power and
every vote as it happens — with a strip that stays green while the network's opinion matches the
comfort model. A fixed thermostat and manual power are there for comparison, windows can be
opened, guests arrive, and with **keep learning from the votes** every live vote becomes a
training example on the spot — change the household's preference mid-run and watch the band
slide after it.

What falls out of the learned zone, with nobody programming it: a night setback (sleepers under a
duvet like it cooler), a draught raising the setpoint, a party lowering it, and the seasonal
drift of what "comfortable" means.

## Metrics

Test loss, accuracy and the confusion matrix are scored against the noisy votes, so even a
perfect network cannot reach 100% there. The **quick test** in the map panel asks the honest
question instead: 300 fresh plausible moments scored against the noiseless model. Errors
concentrate in the ±0.5 PMV boundary strip — the network averages the voters' noise away in the
core of the zone, which is the point of learning from many votes.

## Files

| File | Contents |
|---|---|
| `js/comfort.js` | PMV/PPD after ISO 7730, clothing & metabolism from the routine, the vote model |
| `js/sim.js` | weather, the RC building, moisture, occupancy, heaters |
| `js/data.js` | sensor encoding, curious exploration, dataset collection and balancing |
| `js/nn.js` | dense + 1D convolutional layers: forward, backprop, Adam, evaluation |
| `js/viz.js` | the network diagram with response-map nodes, loss and confusion charts |
| `js/map.js` | the comfort map: learned regions, true outline, votes near the slice |
| `js/rnn.js` | RNN / GRU / LSTM cells over the window, BPTT, gradient clipping, readouts |
| `js/loop.js` | the closed loop: zone scan, T\*, PI control, charts |
| `js/main.js` | state, UI, training loop, probe, arithmetic panel |

## Contributing

Issues and pull requests are welcome. The project is deliberately dependency-free — clone it,
open `index.html`, and everything is editable in place. Each piece lives in its own file;
comments explain the physics and the maths rather than the syntax.

## Licence

MIT — see [LICENSE](LICENSE).
