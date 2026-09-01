# SAGE-HAR

**Software Adaptive Gating Engine for Human Activity Recognition**

A pure-software, hardware-independent statistical gating layer that sits
between raw tri-axial sensor streams and downstream HAR deep learning
inference engines. It skips unnecessary model inferences during rest by
tracking a rest-state-filtered EWMA baseline of signal variance and applying
a dual-threshold Schmitt-trigger (hysteresis) decision rule.

## How it works

```
Raw Sensor Stream (accel x,y,z)
        │
        ▼
Signal Vector Magnitude:  SVM = sqrt(x² + y² + z²)
        │
        ▼
Window Variance:  σ²w = Var(SVM_window)
        │
        ▼
   Gate Active? ──No──▶ Update var_rest (EWMA) ──▶ σ²w > eps_high? ──▶ ACTIVATE
        │                                                  │
       Yes                                                 └─▶ else: skip inference
        │
        ▼
  Freeze var_rest ──▶ σ²w < eps_low? ──▶ DEACTIVATE / else: keep running
```

- `var_rest` only updates while the gate is **inactive**, preventing sustained
  exercise from inflating the resting baseline ("baseline contamination").
- `eps_high = k_high * var_rest`, `eps_low = k_low * var_rest` (with
  `k_high > k_low`) form the hysteresis band, so the gate doesn't chatter at
  the boundary.
- **Warm start**: on the very first window, `var_rest` is snapped directly to
  that window's variance instead of blending in from a guessed constant.
  (An earlier version seeded `var_rest` near zero, which made window 0 look
  like a huge spike and falsely activated the gate immediately — fixed by
  warm-starting from real data instead of a guess.)

## Project layout

Flat, no package structure — every file can be read top to bottom in a
single sitting:

```
sage-har/
├── sage_gate.py       # the gate engine — zero dependencies (stdlib only)
├── evaluate.py        # synthetic benchmark — fake classifier, gate tuning only
├── har_cnn.py         # PyTorch 1D-CNN architecture (the model the gate protects)
├── train.py           # actually trains har_cnn.py on real data (UCI-HAR)
├── evaluate_real.py   # real end-to-end: real data + real gate + real trained model
├── evaluate_cross_dataset.py  # does the model generalize to a different dataset/device?
├── export_onnx.py     # exports har_cnn_uci.pt -> ONNX, for the dashboard's real mode
├── generate_real_data_js.py  # builds web_real_data.js (see Dashboard below)
├── index.html         # dashboard — no build step; three data sources, see Dashboard below
├── web_real_data.js   # generated: real sensor data + trained model, for index.html
├── requirements.txt   # torch/numpy/onnx (har_cnn.py, train.py, export_onnx.py) + pytest (tests/)
├── results.csv        # output of the last evaluate.py run
├── results_real.csv   # output of the last evaluate_real.py run
├── results_cross_dataset.csv  # output of the last evaluate_cross_dataset.py run
├── .github/workflows/tests.yml  # CI — see Testing below
├── tests/             # pytest suite — see Testing below
└── data/
    ├── raw/uci_har/       # real UCI-HAR inertial signals (see Training below)
    ├── raw/motionsense/   # real MotionSense signals (see Cross-dataset below)
    └── processed/
        └── har_cnn_uci.pt # trained weights (see Training below)
```

## Quickstart

### Gate engine — zero dependencies

```bash
python sage_gate.py
```

Runs a small built-in demo: 20 resting windows, a burst of motion, then 20
more resting windows, printing the gate's variance/baseline/status each step
so you can watch it activate and deactivate.

### Benchmark harness

```bash
python evaluate.py
```

Generates synthetic sensor streams for 3 subject profiles (Calm, Tremor /
Parkinsonian, Restless), runs each through the gate, and prints a summary
table of skip rate, estimated power saved, and macro-F1 (gated vs.
always-on). Full results are written to `results.csv`.

### 1D-CNN classifier (requires `pip install -r requirements.txt`)

```bash
pip install -r requirements.txt
python har_cnn.py
```

Runs a shape-check smoke test only: builds the model with random weights,
pushes a batch of shape `(8, 3, 128)` through it, confirms the output shape.
Does **not** train anything.

### Training (real data, real gradient descent)

```bash
python train.py
```

Trains on [UCI-HAR](https://archive.ics.uci.edu/dataset/240) — 30 subjects,
6 activities, already windowed into 128-sample chunks at 50Hz. Uses
`total_acc_{x,y,z}` (raw accelerometer incl. gravity, matching what
`sage_gate.py`'s SVM expects downstream). 15 epochs, Adam, cross-entropy;
keeps the checkpoint with the best validation accuracy and reports final
accuracy/macro-F1 on UCI-HAR's `test/` split, which is held out completely
until the very last step.

Last run:

```
Test accuracy: 92.60%
Test macro-F1: 0.9260

class                precision   recall      f1  support
WALKING                 91.33%   99.80%   0.954      496
WALKING_UPSTAIRS        94.46%   86.84%   0.905      471
WALKING_DOWNSTAIRS      96.55%  100.00%   0.982      420
SITTING                 85.15%   82.89%   0.840      491
STANDING                88.31%   86.65%   0.875      532
LAYING                 100.00%  100.00%   1.000      537
```

The `Inertial Signals/` + label files needed to reproduce this are checked
into `data/raw/uci_har/` (61MB — the pre-engineered 561-feature files from
the original zip are dropped since `train.py` only needs the raw signals).
Trained weights are saved to `data/processed/har_cnn_uci.pt`.

### Real end-to-end evaluation (real data + real gate + real trained model)

```bash
python train.py            # produces data/processed/har_cnn_uci.pt, if not already run
python evaluate_real.py
```

This is different from `evaluate.py`, which benchmarks the gate in
isolation on synthetic data against a fake 97%-accuracy stand-in classifier
(fast, zero extra dependencies, good for tuning gate parameters). This
script instead runs the actual pipeline: `sage_gate.py`'s gate decisions,
window by window, over UCI-HAR's real held-out test subjects, calling the
*actual trained CNN* only on windows the gate marks ACTIVE.

**First attempt, no safety net, was bad and we're not hiding that:**
running the gate exactly as specified (skip inference whenever
INACTIVE, carry the last prediction forward) cost **25.5 accuracy points**
(92.6% → 67.1%). The cause, confirmed by instrumenting the gate directly:
UCI-HAR's SITTING / STANDING / LAYING are three different classes that all
look like near-zero accelerometer variance, so the gate can go quiet for
up to **79 windows (202 seconds) in a row** — and if the one classification
made at the start of that quiet stretch was wrong, that mistake rides
along unchanged for over three minutes.

**Fix:** a heartbeat — force a reclassification every `N` windows
regardless of gate state, bounding how stale a carried-forward prediction
can get. This is standard practice in real duty-cycled embedded sensing,
not a workaround specific to this project. The value was chosen by
sweeping, not guessed:

| heartbeat (windows) | skip rate | power saved | accuracy | macro-F1 |
|---|---|---|---|---|
| disabled | 49.9% | 48.3% | 67.1% | 0.669 |
| 100 | 49.8% | 48.3% | 67.3% | 0.671 |
| 50 | 49.4% | 47.9% | 71.4% | 0.715 |
| 30 | 48.8% | 47.3% | 76.9% | 0.769 |
| 20 | 48.3% | 46.7% | 81.5% | 0.818 |
| 15 | 47.3% | 45.7% | 84.3% | 0.845 |
| **10 (default)** | **45.6%** | **44.1%** | **87.0%** | **0.872** |
| 5 | 40.7% | 39.1% | 89.7% | 0.898 |

10 windows (~25.6s) is the sweet spot used by default: keeps most of the
skip rate while cutting the accuracy cost of gating from **-25.5pp down to
-5.6pp** relative to always-on (92.60% → 87.00%). Override with
`--heartbeat N` (or `--heartbeat 0` to disable it and reproduce the -25.5pp
number above).

*(A second, subtler bug lived here too, caught and fixed after this table
was first generated: the very first window of each subject's stream used
to seed its "last prediction" with that window's own true label, as a
supposedly harmless placeholder. It wasn't harmless — with the gate and
heartbeat both quiet past window 0, that seed kept being reused for a few
more windows, leaking a small amount of ground truth into the reported
accuracy (the default config's number was inflated from 87.0% to a
false 88.2%). Fixed by always classifying window 0 for real instead of
seeding it from the label being predicted; see `evaluate_real.py` and
`tests/test_evaluate_real.py::TestFirstWindowNeverLeaksGroundTruth`. All
numbers in this README reflect the corrected, leak-free run.)*

**Gate parameters were also swept against this same real dataset**, not
just reused from the synthetic benchmark's defaults (`alpha=0.05,
k_high=3.0, k_low=1.5`, heartbeat fixed at 10):

| alpha | k_high | k_low | skip rate | power saved | accuracy | macro-F1 |
|---|---|---|---|---|---|---|
| 0.10 | 2.0 | 1.2 | 43.1% | 41.5% | 88.6% | 0.887 |
| 0.10 | 2.0 | 1.5 | 43.8% | 42.2% | 88.3% | 0.884 |
| 0.05 | 2.0 | 1.2 | 44.6% | 43.0% | 88.0% | 0.881 |
| 0.10 | 3.0 | 1.2 | 44.8% | 43.3% | 88.1% | 0.881 |
| **0.05 (default), 3.0, 1.5** | | | **45.6%** | **44.1%** | **87.0%** | **0.872** |

`alpha=0.10, k_high=2.0, k_low=1.2` is a small but real improvement (+1.6pp
accuracy at -2.5pp skip rate) — run it with `python evaluate_real.py
--tuned`, or set `--alpha`/`--k-high`/`--k-low` individually. It's
deliberately **not** the new global `GateConfig` default in `sage_gate.py`,
since that default also drives `evaluate.py`'s synthetic benchmark and
conflating "tuned for this one real dataset" with "reasonable
general-purpose default" would be its own kind of dishonesty.

Full per-subject breakdown is written to `results_real.csv`.

### Cross-dataset validation: does this generalize at all?

Everything above trains and tests on UCI-HAR only — same 30 subjects' data
pool, same phone-on-waist mounting, just split into train/val/test. That
never actually answers "does this work on a different phone, worn
somewhere else, on people it's never seen?" This does:

```bash
python evaluate_cross_dataset.py
```

Evaluates the UCI-HAR-trained model, completely unmodified, against
[MotionSense](https://github.com/mmalekzadeh/motion-sense) — a different
dataset: iPhone 6s (vs. UCI-HAR's Android devices), carried in the front
trouser pocket (vs. UCI-HAR's waist belt clip), 24 different subjects with
zero overlap. Evaluated on the 5 activity classes both datasets share
(`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`,
`STANDING`) — MotionSense has no `LAYING` and UCI-HAR's training data has
no `JOGGING` equivalent, so both are excluded rather than forced into a
misleading mapping.

**Result: it does not generalize.** 2.45% accuracy — worse than random
guessing among 6 classes (~16.7%). Diagnosed the cause rather than just
reporting the number: comparing per-channel signal statistics between the
two datasets showed UCI-HAR's gravity component (~1g) sits on channel 0
(`[0.80, 0.03, 0.09]` mean per channel) while MotionSense's sits on channel
1 (`[0.04, 0.78, -0.11]`) — a different phone orientation. The CNN, trained
on raw (x, y, z) channels with no orientation-invariance built in, learned
"gravity is on channel 0" as an implicit shortcut. Swapping channels 0 and
1 to test that hypothesis directly:

| | accuracy |
|---|---|
| As MotionSense provides it | 2.45% |
| After swapping channels 0↔1 to match UCI-HAR's gravity axis | 26.43% |

Axis orientation alone accounts for ~24 percentage points of the gap — but
26.43% is still far below the 92.60% same-dataset test accuracy. The
remaining gap is genuine distribution shift (different phone hardware,
different subjects, pocket-vs-waist motion dynamics), not just an axis
convention. **This model has only been shown to work within UCI-HAR's own
device/placement/subject pool.** Shipping it against a different phone or
mounting position would need retraining on data from that setup, or an
explicitly orientation-invariant feature representation (e.g. classifying
on SVM-derived features rather than raw per-axis channels) — neither of
which this project does.

Per-class precision/recall/F1 is written to `results_cross_dataset.csv`.

### Dashboard

No build step. Either:

```bash
python -m http.server 8000
# then open http://localhost:8000
```

or just double-click `index.html` — it works as a plain `file://` page too.

The dashboard has three data sources, switchable live via a segmented
control:

- **Synthetic Sim** (default): a JS-simulated sensor stream + gate logic
  only. Fully offline, no external files needed.
- **Real Model Replay**: replays one real UCI-HAR test subject's window
  data through the *actual trained CNN*, running in-browser via
  [onnxruntime-web](https://github.com/microsoft/onnxruntime), with the
  same gate + heartbeat logic as `evaluate_real.py`. Shows the model's
  real prediction against the real ground-truth label for every window,
  plus a live running accuracy in the metrics bar.
- **Live Sensor**: real phone accelerometer via the browser's
  [DeviceMotion API](https://developer.mozilla.org/en-US/docs/Web/API/Window/devicemotion_event)
  (`accelerationIncludingGravity`), through the same gate + real CNN. Click
  `[ Request Sensor Access ]` — on iOS 13+ this triggers an explicit
  permission prompt (required to be called from a user gesture, which this
  is); Android/desktop-with-a-sensor just starts listening. There is no
  ground truth for a live stream, so this shows only the model's
  prediction, not an accuracy figure, and the window length is however
  long 128 real sensor events take to arrive — there's no dataset-recording
  guarantee of a fixed 50Hz rate the way UCI-HAR/MotionSense were captured.
  **Needs a real device with a motion sensor** — most desktop browsers
  (and this project's own automated test environment) have none, and will
  correctly land on `SENSOR_STATE: NO_DATA_RECEIVED` after a 3s timeout
  rather than hang or crash. Verified in this environment: the UI wires up
  correctly, permission/no-sensor states are handled without errors, no
  dangling event listener after switching away — but the actual "reads a
  real accelerometer" path has only been code-reviewed against the
  DeviceMotion spec, not exercised on physical hardware, since none was
  available while building this.

This needs two generated files that aren't hand-written — `web_real_data.js`
(real sensor data + the trained model as base64 ONNX, ~980KB) and the ONNX
export it's built from:

```bash
python train.py                  # if data/processed/har_cnn_uci.pt doesn't exist yet
python export_onnx.py            # -> data/processed/har_cnn_uci.onnx
python generate_real_data_js.py  # -> web_real_data.js
```

Both are already checked into the repo, so this is only needed if you
retrain the model. `web_real_data.js` is loaded via a plain `<script src>`
tag rather than `fetch()`, specifically so **Real Model Replay still works
under a double-clicked `file://` page** — `fetch()` of local files is
blocked by the browser's same-origin policy, a `<script src>` tag isn't.
The one piece that genuinely needs internet access either way is
onnxruntime-web itself, loaded from a CDN; **Synthetic Sim needs neither
file and works fully offline.**

*(One real bug from wiring this up, since fixed: the original synthetic
tick loop only ever checked whether playback was `running`, never which
data source was selected. It kept advancing invisibly in the background
even while Real Model Replay was active, and its faster 80ms cadence beat
the real loop's 220ms updates almost every time — silently overwriting
the real mode's MODEL_ACCURACY readout with `--` on every other frame.
Fixed with an explicit data-source check in that loop.)*

### Testing

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

48 tests across 8 files. Everything that's been fixed in this project as a
result of an actual bug has a regression test guarding it specifically —
not generic coverage, tests written *because* something broke:

- `test_sage_gate.py` — the warm-start fix (window 0 must never falsely
  activate the gate), hysteresis, baseline freezing while ACTIVE
- `test_evaluate.py` — the carry-forward-defaults-to-REST fix, and
  reproducibility (runs `evaluate.py` twice as subprocesses and diffs
  the output, guarding against the `hash()`-seeding bug)
- `test_evaluate_real.py` — the window-0 ground-truth-leak fix
- `test_evaluate_cross_dataset.py` — the MotionSense label-mapping logic
  (correct UCI-HAR indices, JOGGING/LAYING correctly excluded)
- `test_har_cnn.py` — shape/gradient plumbing (fast, no dataset needed)
- `test_train.py` — metric math + UCI-HAR file parsing; the tests that
  need the real 61MB dataset auto-skip if `data/raw/uci_har/` isn't
  present, so the suite still runs clean in a fresh checkout
- `test_export_onnx.py` — the ONNX export actually matches the PyTorch
  checkpoint (guards the dashboard's Real Model Replay mode against
  silently going stale after a retrain)
- `test_web_real_data.py` — the deployed `web_real_data.js`'s embedded
  model actually matches the current ONNX export (the other half of that
  same staleness guard — verified this one specifically catches a real
  mismatch, not just trivially passes, by testing it against a
  deliberately corrupted copy before trusting it)

**CI**: [`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs
this full suite on every push/PR via GitHub Actions (`ubuntu-latest`,
Python 3.10). Since `data/raw/uci_har/` and the trained checkpoint are
committed to the repo, none of the tests above need to skip in CI the way
they would in a checkout that hasn't downloaded/trained anything yet.

## Tuning

| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | 0.05 | EWMA smoothing factor for `var_rest` |
| `k_high` | 3.0 | Activation threshold multiplier |
| `k_low` | 1.5 | Deactivation threshold multiplier |
| `eps_floor` | 1e-6 | Numerical floor on `var_rest` before scaling |

The dashboard exposes all three via paired sliders + numeric inputs, plus
`[ Inject Spike ]` (forces a burst of motion) and `[ Reset Baseline ]`
(re-warm-starts `var_rest`) for hands-on testing.

## Deploying the dashboard

The deployable artifact is exactly **two files**: `index.html` and
`web_real_data.js`. Everything else in this repo (`sage_gate.py`,
`train.py`, `evaluate*.py`, `tests/`, `data/raw/`, CI config) is dev/training
tooling that produced those two files — none of it needs to ship.

Verified this is actually true, not assumed: copied only those two files
into an empty directory, served *that* directory in isolation (no access
to anything else in the repo), and confirmed Synthetic Sim, Real Model
Replay, and the model-loading path all work identically to the full repo,
with zero console errors and zero failed network requests. This is what a
static host (GitHub Pages, Netlify, Vercel, S3+CloudFront, etc.) would
actually serve.

Steps:
1. Copy `index.html` and `web_real_data.js` to your static host's publish
   directory (same directory, flat — the relative `<script src="web_real_data.js">`
   depends on that).
2. No build step, no server-side code, no environment variables.
3. The page needs outbound HTTPS access to two CDNs at runtime:
   `fonts.googleapis.com` (typography) and `cdn.jsdelivr.net`
   (onnxruntime-web, only needed for the Real Model Replay / Live Sensor
   modes — Synthetic Sim works with neither).
4. `index.html`'s Live Sensor mode needs a secure context (HTTPS, or
   `localhost` for local testing) for the browser to grant DeviceMotion
   permission at all — plan for HTTPS on whatever host you pick, not
   plain HTTP.

If you retrain the model or change the gate later, regenerate the deployed
artifact before pushing it live:

```bash
python train.py
python export_onnx.py
python generate_real_data_js.py   # -> web_real_data.js (re-embeds the new model)
python -m pytest tests/           # confirm nothing regressed
```

`tests/test_export_onnx.py` guards the `.pt` → `.onnx` step, and
`tests/test_web_real_data.py` guards the `.onnx` → `web_real_data.js` step
— together they fail if either generation step was skipped after a
retrain, so a stale deployed model can't pass CI silently.
