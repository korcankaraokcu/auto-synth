# Contributing

Everything a contributor needs: how to build it, how it is put together, what
has been measured, and what is left to do.

---

## 1. Installation

### Prerequisites

| | |
|---|---|
| **OS** | Windows 10 or later. Linux and macOS are on the roadmap; nothing here is deliberately Windows-only, but neither is tested. |
| **Compiler** | Visual Studio 2019 or later, "Desktop development with C++". The standalone Build Tools are enough. |
| **CMake** | 3.22 or later. `pip install cmake` works if you would rather not install it system-wide. |

JUCE 8.0.15 and Catch2 v3.7.1 are fetched at configure time and pinned in
`plugin/CMakeLists.txt`. Nothing is vendored, so a fresh clone is small and the
first configure is slow.

> JUCE needs a **C** compiler as well as a C++ one and will refuse to configure
> without it, with a message that does not obviously say so. If the C++ workload
> is installed you already have it.

### Quick start

```powershell
.\scripts\bootstrap.ps1            # configure, build, test
.\scripts\bootstrap.ps1 -Install   # ...and install the VST3
.\scripts\bootstrap.ps1 -Config Debug -SkipTests
```

The script checks the toolchain first, because the interesting failures in this
build are not build errors — see [Known build traps](#known-build-traps).

### Manual build

```powershell
cmake -S plugin -B plugin/build
cmake --build plugin/build --config Release --target AutoSynth_VST3 AutoSynth_Standalone
cmake --build plugin/build --config Release --target autosynth_tests
ctest --test-dir plugin/build -C Release --output-on-failure
```

### Targets

| Target | What it is |
|---|---|
| `AutoSynth_VST3` / `AutoSynth_Standalone` | The plugin |
| `autosynth_dsp` | Static library: engine, analysis and fitting. Everything else links it. |
| `autosynth_tests` | The test suite |
| `autosynth_render` | Headless renderer — patch JSON in, WAV out |
| `autosynth_probe` | Analysis probe — WAV in, every intermediate stage out as JSON |
| `autosynth_diff` | A/B diagnosis — two WAVs in, the difference on named axes out |
| `autosynth_eval` | Ground-truth recovery harness — how good is the fitter? |
| `autosynth_vital` | Renders, fits and evaluates through the installed Vital VST3 |
| `install_plugin` | Copies the VST3 to the per-user plug-in folder |

`autosynth_vital` needs Vital installed and links none of its code; it hosts
whatever is at `C:\Program Files\Common Files\VST3\Vital.vst3` unless
`--plugin` says otherwise. That is deliberate — it renders the version the
presets will actually be opened in. The export gap is then an ordinary diff:

```
autosynth_render patch.json ours.wav --note 440
autosynth_vital  patch.json theirs.wav
autosynth_diff   ours.wav   theirs.wav
```

It has two further modes, both of which put Vital where this engine usually
sits: `--fit target.wav` runs the fitter with Vital rendering every candidate,
and `--eval` runs the recovery harness entirely inside Vital -- random patches
rendered by Vital as the targets, and the control and every candidate rendered
there too.

```
autosynth_vital fitted.json out.wav --fit samples/clarinet.wav --dur 4 --gate 3
autosynth_vital --eval --trials 12 --seed 0
autosynth_eval  --trials 12 --seed 0        # the same question, this engine
```

The last two are *not* comparable by their absolute scores: the targets are
different populations, because one set was rendered by Vital and the other by
this engine. What compares is how far each run beats the control it ran with.

### Known build traps

- **`COPY_PLUGIN_AFTER_BUILD` is off deliberately.** JUCE's auto-copy targets
  `C:\Program Files\Common Files\VST3`, which needs elevation. The build
  compiles and links successfully and then fails at the very last step with a
  permission error, which reads like a build failure but is not one. Use
  `install_plugin`, which writes to the per-user folder instead.
- **Build the configuration you intend to test.** Building `RelWithDebInfo`
  while running a `Release` binary means a fixed bug keeps reproducing. This
  cost an afternoon once.
- **Rebuild *every* tool you are about to measure with.** The same trap in a
  second costume: after changing the envelope fitter, `autosynth_probe` and
  `autosynth_tests` were rebuilt but `autosynth_eval` was not, so a harness
  "baseline" was recorded from a binary predating the change. An hour then went
  into bisecting a regression that was really a comparison against the wrong
  build. `.\scripts\bootstrap.ps1` builds all of them, which is the reason to
  prefer it over hand-picked targets.
- **Keep the checkout path short on Windows.** JUCE's repository contains very
  deep paths of its own (iOS demo assets nested a dozen levels down), and with
  `MAX_PATH` at 260 characters the *fetch* fails before anything is compiled --
  a wall of "Filename too long" followed by `Failed to checkout tag: '8.0.15'`,
  which reads like a network or CMake problem and is neither. A checkout around
  40 characters deep leaves plenty of room; one nested under a long temp path
  does not. Either shorten the path or enable long paths:

  ```powershell
  git config --global core.longpaths true
  ```

---

## 2. Architecture

```
sample → analysis → structure decisions → parameter mapping → refinement → patch
         (DSP)      (grouping, roles)     (IR)                (CMA-ES)
```

Three decisions shape everything else.

**The IR is the deliverable, not the engine.** `src/ir/Patch.h` defines a
synth-agnostic patch. Analysis produces one, the engine renders one, exporters
translate one. "Should we support Vital?" is therefore not an architecture
question — it is a later exporter.

**We own the synthesizer.** Targeting someone else's plugin means you cannot
render inside an optimisation loop cheaply, cannot model the forward pass, and
face a parameter space far larger than analysis can constrain.

**Analysis first, optimiser second, ML last (or never).** DSP analysis gives an
interpretable intermediate representation you can debug and show a user. A
derivative-free optimiser polishes from an already-good initialisation.
[Instrumental (arXiv 2603.15905)](https://arxiv.org/html/2603.15905) found
CMA-ES beats gradient descent on this landscape, and that spectral-analysis
initialisation beats random starts.

### The structure / precision split

This is the load-bearing rule and it is worth stating plainly:

> **Analysis owns discrete and structural decisions. Refinement owns continuous
> values. Refinement never searches a discrete parameter.**

`Refine::scopeFor` builds its search space from the patch, and excludes
waveform, oscillator count, filter type and every other discrete choice. It
also excludes parameters belonging to disabled oscillators and unrouted LFOs,
because CMA-ES sample efficiency falls off with dimension and every dead
parameter is paid for in convergence speed.

One consequence, which surprised a test into being rewritten: refinement *can*
retire an oscillator, by driving a level it legitimately owns down to zero. It
cannot create one, because an oscillator at zero level is not in scope and
nothing can bring it back. Analysis proposes; refinement can only prune.

### Layout

```
plugin/
  src/ir/          Patch.h/.cpp — the contract, plus JSON read and write
  src/dsp/         Tables, Envelope, Svf, Delay, Reverb, Voice — the engine
  src/analysis/    Stft, Yin, Partials, Grouping, Roles
  src/fit/         WaveformFit, EnvelopeFit, FilterFit, Modulation, EffectsFit,
                   PartialFit, Nnls, Nmf, CmaEs, Refine
  src/eval/        Recovery — the ground-truth harness
  src/Parameters*  host-automatable parameters
  src/Plugin*      processor and editor
  src/ParamKnob.h  shared rotary knob; yields the wheel to the scroll panel
  src/ControlGroup.h  a titled box of related global controls
  tools/           render_main.cpp, probe_main.cpp, eval_main.cpp
  tests/           the suite, and golden/ — frozen reference data
scripts/
  bootstrap.ps1
```

### Signal flow

Oscillators (each with its own filter and envelope) → noise → amplitude
envelope and amp LFO → global filter → delay → **+ reverb return** → master.

The reverb is one shared unit fed by **per-oscillator sends**, not three
instances. Each oscillator controls how much reverb it gets; the size, damping
and return are common to all three. A send gives the usual creative result —
one layer drenched, another dry — while keeping the patch in a single acoustic
space, which is how a real instrument behaves.

`level` is a return gain rather than a dry/wet balance, and that is what makes
the per-oscillator send meaningful: the dry path is already whatever the sends
did not take.

Note that the cost argument for this is weaker than it might appear, and is not
the reason. The reverb lives on the `Engine`, not the `Voice`, so three of them
would be three instances in total rather than three per voice — polyphony does
not multiply it. Per-oscillator reverb *character* is therefore a reasonable
thing to add later if a sound calls for it; it was left out as a scope and
interface decision, not a performance one.

---

## 3. Testing

```powershell
cmake --build plugin/build --config Release --target autosynth_tests
ctest --test-dir plugin/build -C Release --output-on-failure

# or the binary directly, for tags and filters
.\plugin\build\autosynth_tests_artefacts\Release\autosynth_tests.exe "[engine]"
.\plugin\build\autosynth_tests_artefacts\Release\autosynth_tests.exe "[.report]"
```

118 cases. Tags: `[ir]`, `[engine]`, `[analysis]`, `[fit]`, `[capabilities]`,
`[recovery]`, `[unison]`, `[golden]`.

Tests tagged `[.slow]` are hidden by default and run only when asked for by
name — `autosynth_tests "[.slow]"`. There is one: the assertion that the fitter
beats the control, which has to run real trials.

### The golden fixtures

This project was developed against a Python reference implementation. Every
stage of the C++ engine and analysis chain was validated against it
element-wise, and that reference was then removed once the port was complete.

`plugin/tests/golden/` is what remains: the reference implementation is gone,
but the evidence it produced is not.

- `golden/engine/` — a patch and the audio the reference rendered from it.
- `golden/analysis/` — an input signal and the full analysis report the
  reference produced, mirroring the schema `autosynth_probe` emits, so the two
  can be compared field by field.

Analysis inputs are stored as 16-bit and read back *before* analysis, so both
implementations saw exactly the same samples and quantisation is not a
difference between them.

### Blessing fixtures

Regenerating a golden fixture from the current engine is a deliberate act that
belongs in a commit message, not a way to make a red test green.

A conformance bound that gets loosened to accommodate a defect has stopped
being a measurement. That is not hypothetical here: the all-filters bound was
once widened from 3.33 dB to 6.0 dB to "account for per-oscillator filters",
when in fact the C++ voice was silently dropping per-oscillator filtering
entirely. Once the filters actually ran, the figure fell to 2.33 dB. The bound
had been fitted to the bug.

If a change alters engine output on purpose, say which fixtures moved and why.

### Writing tests that can fail

Three bugs once sat in `Voice::render` simultaneously — per-oscillator filters
never applied, the solo/mute mask never applied, the reverb send never
accumulated — and the entire suite passed. Each was a *no-op*, and every test
compared audio against audio, so a control that did nothing still matched a
reference captured after it stopped working.

The lesson is a rule for new features:

> **A feature needs a test that fails when the feature does nothing.**

In practice that means comparing a feature against its own absence, in the same
patch:

- `per-oscillator filters shape oscillators independently` compares a filtered
  oscillator against an open one *within one patch* — a global filter cannot
  produce that signal.
- `the monitor mask silences oscillators` asserts that muting everything yields
  actual silence — with the mask unapplied this returned full level.
- Every "is an exact no-op" case asserts sample equality when a control is off,
  which catches leakage in the other direction.

### Measured conformance

Run `autosynth_tests "[.report]"` to reproduce the distribution.

| | Loudness | Brightness |
|---|---|---|
| Filter-free cases (mean of 25) | **0.057 dB** | — |
| Filtered cases (mean of 15) | **2.89 dB** | 0.44 oct |

The filter-free figure is the one to watch. With no filter running, the two
implementations shared oscillators, envelopes, LFOs, delay and reverb outright
and had no licence to differ at all, so anything above the noise there is a
genuine defect. The filtered figure has to absorb up to four substitutions of
an STFT magnitude response for a state-variable filter per patch.

Two per-case outliers are known and documented in `test_golden_engine.cpp`:

- **`filter_bandpass`, 10.43 dB.** A bandpass has steep skirts on both sides,
  so a magnitude-response approximation diverges most there. It is the worst
  case of the seam.
- **`wave_pulse`, 0.79 dB** against under 0.02 for every other waveform. Small
  in absolute terms but forty times its neighbours, and it points at
  pulse-width table construction rather than at the filter seam. **Unexplained
  — worth investigating.**

### A known limit of cross-implementation comparison

Level calibration ends in a non-negative least squares over near-collinear
columns. On such a system tiny input differences move the answer, and the
reference implementation was itself inconsistent at that scale: it solved one
marginal oscillator to exactly zero and kept another of the same magnitude
(~0.005, some 46 dB down) at a non-zero level.

So the golden analysis test compares oscillators that carry audible weight
(above 1% of the loudest) exactly, and allows the total count to differ by the
one marginal oscillator. Demanding more would be demanding that two
implementations share a rounding pattern, not that they agree about the sound.

---

## 4. Measurements worth knowing

These drove design decisions and are cheaper to read than to rediscover.

### Measuring the fitter

```
autosynth_eval --trials 24 --seed 0            # with refinement
autosynth_eval --trials 24 --seed 0 --no-refine
autosynth_eval --trials 24 --json
```

24 trials, seed 0, current IR. Both columns face the *same* targets, so this is
a paired comparison and only refinement differs. Control — an untouched default
patch against those same targets — is spectral 2.272, loudness 21.406, centroid
1.574.

| | `PartialFit` | `PartialFit` +CMA |
|---|---|---|
| spectral | 0.946 | **0.583** |
| loudness_db | 10.083 | **8.938** |
| centroid_oct | 0.819 | **0.254** |
| oscillator count exact | 62.5% | 58.3% |
| root within a semitone | 79.2% | 79.2% |

Refinement earns its keep on brightness above all — centroid error falls by
about two thirds — and cuts spectral distance by roughly a third. Loudness
moves least, at around 11%.

Two things in that table are worth not over-reading:

- **The oscillator count drops slightly with refinement.** That is one trial out
  of twenty-four, which is noise at this sample size, not a finding. The
  mechanism is real, though, and worth remembering: refinement can retire an
  oscillator by driving a level to zero, so it can occasionally remove a
  *correct* one. It can never add one back.
- **Root accuracy is identical.** It has to be — `root_hz` is not in refinement
  scope, so pitch is settled entirely by analysis. Seeing the two columns agree
  exactly is a useful check that the structure/precision split is holding.

**These numbers are not comparable to any taken before the reverb parameters
were added.** Random patches are sampled from the IR, so changing the IR changes
the target population — the control column moved from 12.3 to 21.4 loudness on
the same nominal settings, which is the distribution shifting, not a
regression. Always read a fitted score against the control from its own run.

### Two corrections the harness needed before it could be believed

Both were found by using it, and both had been silently flattening every
number it produced.

**It scored parameters that were switched off.** The first version compared
every continuous parameter on every trial, including a detune on an oscillator
with one unison voice, a cutoff with the filter bypassed, an LFO rate with no
destination. None of those affect the audio, so the fitter cannot recover them
and has no reason to. Averaging them in produced errors near 0.5 — which is
exactly the expected distance between two independent uniform draws, and
indistinguishable from a real failure. The whole worst-recovered list read as
"the fitter recovers nothing". `affectsAudio` now scores a parameter only where
the target's own settings give it an audible effect.

**It compared oscillator 0 against oscillator 0.** The fitter emits sources in
salience order; a random target's oscillators are in no order at all. Slot *n*
on one side has no reason to be the same voice as slot *n* on the other, so most
per-oscillator comparisons were between unrelated oscillators. `matchOscillators`
now pairs them by sounding frequency, loudest first, and refuses a pairing more
than half a semitone apart.

Together these moved `cents` from invisible to a clear 0.18–0.25 — pitch was
being recovered well the whole time and the measurement could not see it.

The general lesson, since it will happen again: **a measurement that averages
over unidentifiable parameters converges on the number you would get from a
coin.** If a metric looks uniformly bad, check that every term in it could have
been known.

### What the worst-recovered list is telling you

The harness ranks parameters by mean error, normalised by each parameter's own
range. The top of that list currently sits near **0.5** — `reverb.level`,
`lfos.*.phase`, `lfos.*.delay`, the various `env.curve` entries.

That number is a signature, not noise: a fitter that returns a *constant*
against a uniformly random target scores almost exactly 0.5. Every parameter up
there is one nothing currently estimates. The list is a to-do list, and it is
reporting honestly.

The shipped pipeline is `PartialFit` plus refinement.

### What the recovery harness cannot see

It generates targets by rendering **this engine**, so it only ever tests sounds
this engine can already make. It measures how well we solve the *inverse*
problem and is structurally blind to the *modelling gap* — everything a real
recording contains that the parameterisation cannot express.

This is not hypothetical. The envelope was originally linear-only. Every
harness trial scored well, because targets and fits were both linear and agreed
perfectly. The first real sample mis-stated a plucked decay by 5–7 dB and then
dropped to digital silence while the real tail was still audible — a 54 dB
error the harness had rated as fine. Adding `Adsr::curve` took it to 2.8 dB.

> **A real sample tells you *that* something is wrong; the harness tells you
> *why*.** Use both, always.

### Why oscillator counting is hard

The original approach factorised a harmonic amplitude matrix `H[k, t]` and read
the rank. On synthetic matrices of known rank it recovers the rank every time.
On real ones it sat at chance — and that is a limit of sampling `H` on a fixed
harmonic grid, not a tuning problem:

- Two oscillators an **octave** apart put energy on overlapping harmonic
  indices. Sharing an amplitude envelope, they are not merely hard to separate,
  they are *mathematically identical* to one oscillator with a different
  waveform.
- Two oscillators a **fifth** apart are worse: 3:2 puts partials on half-integer
  multiples of the fundamental, so they land between grid points and are simply
  *absent* from `H`. No factorisation recovers what was never sampled.

Partial tracking (McAulay–Quatieri) plus greedy multi-f0 grouping replaces the
grid with measured partials, which is why `PartialFit` is the shipped path.

### Unison by beating

Spectral clustering can only count unison voices once they resolve into
separate peaks, which needs a detune wide enough to beat the analysis
resolution. Below that they merge into one partial, the count came back as one,
and narrow unison was systematically missed.

Two voices *d* cents apart do not stop existing when they stop resolving — they
amplitude-modulate at their difference frequency. At harmonic *k* of a group
with fundamental *f₀*:

```
beat rate = k · f₀ · (2^(d/1200) − 1)
```

The factor of *k* is the entire discriminator. A beat rate **grows in
proportion to harmonic number**, because the frequency gap between two detuned
partials widens up the series. Nothing else in this chain does that — tremolo
and vibrato modulate every harmonic at the *same* rate, and a decaying envelope
is not periodic at all. So the test is not "are the harmonic envelopes
periodic", which would fire on every LFO in the test set; it is "does the rate
rise proportionally with *k*", checked against a constant-rate model that
tremolo fits perfectly and unison does not.

Measured against known detunes, on two voices:

| true detune | recovered |
|---|---|
| 8 cents | 8.01 |
| 12 cents | 11.77 |
| 25 cents | 24.46 |
| 40 cents | 40.45 |

On the recovery harness, mean `unison_detune` error falls from 0.352 to 0.306
— smaller than the isolated figures suggest, because random targets often have
more than two voices, where the beat measures adjacent spacing rather than the
full spread.

One implementation note worth keeping. The first version took the **tallest**
autocorrelation peak and reported roughly half the true beat rate often enough
to wreck the proportional fit — the classic octave error, since autocorrelation
peaks at every multiple of the period. Taking the *first* peak within 85% of the
tallest fixed it outright: 12 cents went from reading 6.34 to reading 11.77.

Least squares was also tried for the proportional fit and is the wrong tool
here. A few harmonics whose period detection goes astray drag the line badly,
and R² then condemns a set of rates that are mostly right. Counting inliers
around a *median* slope is unbothered by a minority of bad harmonics, which is
the normal case — high harmonics are weak and their envelopes noisy.

### Exporting to Vital

The first exporter, and the reason the IR is described as the deliverable rather
than the engine: nothing in `VitalExport` touches synthesis. It translates one
description into another, and a second target would be another file this size.

Vital was the right first choice because its oscillators are wavetables and ours
now are too, so the part that would otherwise be a lossy approximation is a
direct copy. Sixteen harmonic amplitudes inverse-transform into the 2048 samples
per frame Vital expects, and a frame nobody has drawn on is its own waveform's
series {d} so an analog patch exports as the shape it says it is rather than as a
sixteen-harmonic impression of one. Rebuilt from the harmonics rather than
resampled from our own 4096-point tables: resampling needs a filter and lands a
sample short at the seam, which is where a wavetable is least forgiving, because
the loop point is heard on every cycle.

**Written from assumption, then corrected against evidence.** The first version
guessed the numeric mappings and got most of them wrong. Checking a machine with
Vital installed turned up 299 real presets, and comparing against 274 of them
found this:

| | assumed | measured |
|---|---|---|
| envelope times | seconds | the **fourth root** of seconds |
| LFO rate | hertz | **log2** of hertz, and only when not tempo-synced |
| oscillator tune | cents | a **semitone** fraction, so cents over 100 |
| master volume | 0 to 1 | a unit spanning 1755 to 7399 |
| modulation amount | inside the routing entry | a separate `modulation_N_amount` |
| wavetable position | 0 to 255 | 0 to **256** |
| wavetable object | had a `remapper` key | has `full_normalize` and `remove_all_dc` |

The envelope one is worth showing the reasoning for, because it is the kind of
thing that is invisible until measured. Stored decay across the collection spans
0.27 to 2.37 and clusters hard on 1.0. As seconds that would mean no preset in
299 decays faster than a quarter second, which is plainly false. As a fourth
root it means 0.006 s to 31.4 s, the maximum lands on Vital's documented ceiling
of 32, and the default sits at exactly one second. Everything fits at once.

The modulation one was the worst, because it would have failed *silently*:
`modulations` entries carry only source and destination, and the depth lives in
a numbered parameter block. An amount written inside the entry is ignored, so
every routing would have connected at zero and the preset would have loaded
looking correct and sounding static.

**Still assumed:** unison detune. It is a 0-to-10 control whose unit is not
recoverable from presets {d} 124 of them sit on 4.4721, the square root of
twenty, which suggests a power curve rather than a width in cents. Ours are
placed linearly on that range.

**And then it did not load.** Vital called the first corrected file corrupted,
which found the last mistake and the most instructive one. A `Wave Source`
component carries exactly four keys: `type`, `keyframes`, `interpolation`,
`interpolation_style`. The exporter was writing thirteen, because the survey
that produced the list had counted keys across *all* component types at once
and handed back the union {d} `audio_file`, `num_points`, `window_size` and the
rest belong to `Audio File Source` and `Line Source`. Counting what a field is
*named* across a corpus is not the same as knowing which object it belongs to,
and a schema derived that way is a superset that no parser accepts.

Checked properly this time, by key *set* rather than by key name: all 510 Wave
Source components across 299 presets carry that one four-key set and nothing
else, and all 2063 of their keyframes carry `position` and `wave_data`. Every
object the exporter now writes {d} preset, wavetable, group, component,
keyframe, modulation {d} matches a real one exactly, with no extra key and none
missing.

**And it still would not load, because a preset is not a bag of settings.**
Vital's engine is open source, and reading `LoadSave::jsonToState` answered in a
minute what a week of file comparison could not. It reaches into
`settings["lfos"]` and `settings["sample"]` *directly*, without checking they
exist, so a preset lacking either fails to parse and is reported as corrupted.
Every parameter we wrote was correct; the file was missing two blocks that no
amount of comparing parameters would have revealed, because they are not
parameters. Eight drawn LFO shapes {d} the canvas the `LfoShape` choice is drawn
on, not the choice itself {d} and the sampler's sample, which is silent here and
still has to decode.

The same function also explains the very first failure: it rejects any preset
whose feature version is *newer* than the running Vital. The original export
declared 1.5.5, a number invented from memory, against an installed 1.0.7. It
would have been refused before a single parameter was read.

The lesson is the one this project keeps relearning in new costumes. Four
rounds went into inferring a format from its outputs — key names, value
ranges, key sets, then gain staging — and each round found something real and
still left it wrong. Reading the *reader* was available the whole time, and once
it was opened, `LoadSave::jsonToState` explained the corruption and
`synth_parameters.cpp` explained the level in a single sitting, with every
mapping stated as a range and a curve rather than inferred from a histogram.

Measuring artefacts was not useless — it is what produced the envelope quartic
and the log2 LFO rate, neither of which is written down anywhere obvious. But it
cannot tell you about a field nobody in the sample happened to use, and it
cannot tell a linear control from a quadratic one when every preset you have was
written *through* that curve. When a thing has a parser, the parser is a
specification in disguise, and it should be the first stop rather than the last.

**It loaded, and came out roughly 20 dB quiet.** Two mistakes compounding, both
of which the source settles outright:

`volume` is a **square-root** control displayed in decibels with an offset of
-80, so the reading is `sqrt(stored) - 80`: silence is 0, unity is 6400, and the
ceiling of 7399.44 is +6 dB. Leaving it at the default was leaving the patch at
-6 dB. And `osc_N_level` is **quadratic** {d} the amplitude is the *square* of
the stored value. Writing a linear level there halved it in decibels a second
time, so folding the master gain in beforehand, as this exporter did, squared an
already-reduced number.

The derivation confirms itself from a direction it never used: a gain of one
half maps to 5472.95, and Vital's own default volume is 5473.04. The default is
exactly -6 dB.

Unison detune fell out of the same page {d} quadratic on 0 to 10, displayed as a
percentage, which is why 124 presets sit on 4.4721: the square root of twenty,
for a default of 20%. That was the one mapping still marked as assumed, and it
was assumed wrongly.

**The declaration is now a fixture, and the exporter is checked against it.**
`tests/golden/vital/parameters.json` holds Vital's own `ValueDetails` for every
parameter this writes {d} range, default, curve, offset {d} copied from
`synth_parameters.cpp` with its provenance and a note on how to refresh it. Two
tests use it. One exports a plain patch, an extreme one and a noise one, and
asserts every numeric value lands inside its declared range and that indexed
controls are whole numbers. The other puts each mapping *back* through the
declared curve and checks it returns what went in: seconds through the quartic,
amplitude through the quadratic, hertz through the exponential, decibels through
the square root.

The second is the one that matters, because range alone would not have caught
the two bugs that actually shipped. A linear oscillator level and a default
master volume are both perfectly inside their ranges and both wrong by 6 dB.

Confirmed to fail on demand, since a test nobody has seen fail is a test nobody
has: each of the four historical bugs was reintroduced in turn and the suite
caught every one {d} a linear level (2 cases), envelope times as seconds (3),
volume left at its default (1), LFO rate in hertz (2). That is the cheap 80% of
a real integration test. It cannot hear anything, but every mistake this
exporter has actually made was a legal-looking number that sat outside or askew
of a declaration, and all of them would now fail without a build of Vital, a
listener, or a round trip through a person.

Every settings key the exporter writes appears in real 1.0.7 presets {d} it
invents nothing {d} and `test_vital_export.cpp` decodes the base64 back to
samples and projects them onto the sines that went in: a frame asking for a
fundamental and a half-amplitude third harmonic comes back with exactly that and
nothing on the second, and an undrawn square comes back as odd harmonics falling
as 1/k.

### Rendering through Vital, and the three things it found

Everything above checks the preset on *this* side of the boundary. None of it
could answer whether the preset sounds like the fit, because answering that
needs the other synth {d} and the gap is structural rather than incidental:
refinement optimises against this engine and the file is then played by a
different one.

`autosynth_vital` hosts the installed VST3 and renders the exported preset, so
the gap becomes an ordinary `autosynth_diff`. It links no Vital code and ships
none. It renders whichever build is installed, which is also the build the
presets will be opened in {d} a submodule would render the public source drop,
which is not necessarily the same engine.

**The handover was wrong first, and reported success.** Vital's *plugin* state
chunk is the preset JSON {d} `getStateInformation` calls the same
`LoadSave::stateToJson` that writes a `.vital` file {d} so the exported text was
handed straight to `setStateInformation`. A *hosted* plugin's chunk is not that:
JUCE's VST3 host wraps the component and controller states as base64 inside an
XML document, and its `setStateInformation` does nothing at all when handed
anything else. `getXmlFromBinary` returns null and there is no error path. Vital
kept its init patch and rendered happily.

What caught it was the diff, not the tool: the harmonic profile came back 0.0,
-5.8, -9.5, -12.1, -14.0, -15.5, -16.7, -18.1 against the fundamental, which is
a mathematically exact sawtooth, with an instant attack and no modulation. That
is Vital's init patch and nothing else. A preset that fails to load and a preset
that loads and sounds wrong are indistinguishable from the audio alone unless
you recognise the default. `--dump-state` now reads the state back out, so the
question is answerable directly rather than by recognition.

Once it loaded, two real defects surfaced immediately, and both had been
invisible to every check that reads the preset, because both produce a perfectly
legal file:

**The filter envelope was never connected.** `env_2` was written with the
fitted shape and no routing mentioned it. Vital rendered a static cutoff while
the engine swept it 1.92 octaves, which read as harmonics two to eight sitting 3
to 13 dB low and the brightness 0.41 octaves dull. A dangling modulator is
silent, not wrong-looking. `test_vital_export.cpp` now asserts that an envelope
past `env_1` appears as a modulation source {d} `env_1` drives amplitude whether
or not anything routes it, every other envelope does nothing until the matrix
names it.

**The noise bed had nowhere to go.** Vital's oscillators are wavetables and have
no noise mode, so noise had simply been dropped: the violin measured 0.171 in
this engine and 0.048 in Vital. The sampler is the only continuous broadband
source Vital owns, and the block was already there as a silent stub for the
loader's benefit. It now carries one second of uniform full-scale noise, looped,
from a fixed seed so the export stays reproducible, routed to filter 1 rather
than Vital's default of the effects bus {d} this engine adds noise into the mix
*before* the filter, so an unfiltered bed is a different sound. Uniform rather
than Gaussian to match the engine's own generator, so the amplitude asked for is
the amplitude that arrives.

**And the room was never written.** The most audible of the three and the last
found, because the first renders stopped at two seconds with note-off at one and
a half, which is not long enough for a tail to be conspicuous by its absence.
Heard rather than measured: the exported preset went to digital silence 50 ms
after note-off. The fitted amplitude release is 5 ms, so the dry signal *should*
stop dead {d} everything after it is reverb, and the exporter wrote no reverb at
all. Both library patches carry one at a return gain near 0.1 and an RT60 near a
second.

Vital states a tail as a decay time in seconds where this engine states a room
size, so the size goes back through the reverb's own RT60 relation rather than
being copied across as though the two controls meant the same thing. That
relation now lives on `Reverb` rather than in the fitter, because two callers
need it from opposite directions: the fitter picks a size from a measured tail,
and the exporter has to restate that tail in another synth's units.

**Two numbers had to be measured rather than translated**, and getting the first
of them wrong is the part worth recording.

Our reverb `level` is a return gain added to the dry; Vital's `reverb_dry_wet`
is a crossfade. Writing one in as the other left a tail 14 dB under the source,
which is present in the file and inaudible in the room {d} and that is exactly
how it was reported: not "the reverb is quiet" but "there is silence after the
first second". A return gain of 0.1 is nothing like a mix of 0.1, because the
return is applied *after* a comb bank with a large gain of its own. The two are
now related by a measured factor of eight.

That factor was found, discarded, and reinstated. It was discarded because a
check against the recording appeared to show this engine's tail sitting three
times louder than the source, which would have made the engine the outlier and
the quiet export nearly right. That check was wrong: it measured the envelope in
167 ms slices, which straddles the release and smears the very transition being
measured. Redone at 20 ms with the release located first, the recordings sit at
0.218 and 0.211 of their sustain a tenth of a second after release, this engine
at 0.236 and 0.133, and the unscaled export at 0.042 and 0.039. The engine
agrees with the recordings; the export was the outlier all along.

The lesson is not about reverb. A measurement fine enough to see the thing you
are asking about is a precondition for the answer meaning anything, and an
envelope window wider than the event will always report the average of before
and after with total confidence. It briefly produced a *documented* wrong
conclusion here, which is worse than no conclusion.

The second number is the decay time. Vital rings about twice as long as the time
it is given: asked for 0.61 s, 0.89 s and 1.52 s it renders 1.19 s, 1.75 s and
3.42 s, a ratio of 1.96, 1.97 and 2.24 across a factor of three in the input.
Stable enough to be a convention rather than a coincidence, so the exporter
halves the fitted time on the way out.

A smaller question settled in passing: Vital's dry/wet reduces the dry far more
slowly than `1 - wet`, so compensating for the crossfade as though it were
linear puts the whole patch several decibels over. At these mix levels the loss
is under a decibel and is left alone.

All three fixes were verified by measurement rather than by reasoning, which is
the point of having the loop at all. Against this engine's render of the same
patch:

| | before | after |
|---|---|---|
| violin noisiness | 0.048 vs 0.171 | 0.178 vs 0.171 |
| violin brightness | 0.29 oct dull | ok |
| violin harmonics 2-8 | within 3.3 dB | within 3.3 dB |
| clarinet brightness | 0.41 oct dull | ok |
| clarinet harmonics 2-8 | up to 12.7 dB low | up to 7.6 dB low |
| clarinet tail 100 ms after release | digital silence | 0.170 of sustain, against 0.218 in the recording |
| violin tail 100 ms after release | digital silence | 0.151 of sustain, against 0.211 in the recording |

Both instruments now land inside tolerance on pitch, noisiness, attack,
note-off, vibrato and level; the violin on every axis the diff reports.
When rendering to listen, hold the note as long as the source does and give the
tail room: `--dur 4 --gate 3` for these two, whose recordings release at 3.00 s.
The defaults gate at 1.5 s in 2 s, which is right for a diff and wrong for an
ear {d} a one-second tail does not fit in it, and half a second of held note is
not long enough to hear whether the sustain sits where the recording's does.

What remains is *not* a transfer bug {d} every setting written survives the round
trip byte-exact, and the wavetables arrive with the right keyframe counts:

- **Amplitude modulation is about 1 dB shallow.** `osc_N_level` is quadratic and
  the modulation clips at 1.0, so a level of 0.947 with an amount of 0.297 has
  no room to swing upward and loses half its excursion. Giving it headroom means
  moving gain to the master, which is a decision about staging rather than a
  bug fix.
- **Delay is still not exported.** Neither library patch enables it, so it has
  never been exercised; `delay_*` maps as straightforwardly as the reverb did.

**On the dependency.** Vital's source is GPLv3 and this project is
GPL-3.0-or-later, so there is no licence obstacle to linking it; an earlier note
here suggesting otherwise was simply wrong. What is *not* GPL is the factory
preset library, which is separately licensed and must not be redistributed, and
the name, which GPLv3 §7(e) explicitly allows an author to withhold {d} GPL
builds are conventionally called Vitalium. Upstream is frozen (last code change
2022, README edits since) and takes no pull requests, so the fork at
`korcankaraokcu/vital`, pinned at `636ca0e`, is the insurance policy. Hosting
the installed plugin needs none of it: loading a plugin at runtime is not
distribution, and no Vital code enters this build.

**On versions.** The preset declares 1.0.7, and the format has been additive
across every version on hand: comparing 0.6, 0.8, 0.9 and 1.0 presets, no
settings key present in an older one is missing from a newer one, and 1.0 has
776 keys against 0.8's 775. A preset written to the 1.0 vocabulary should
therefore load in later versions with the newer parameters at their defaults.
That is an argument from four versions of evidence, not a guarantee about a
version nobody here has.

### Two axes the diagnostic did not have

Every axis in `autosynth_diff` was shape-relative: each signal normalised by its
own peak or its own profile before being compared. So all of them could read in
tolerance while a preset lost two thirds of its level after the attack, which is
exactly what happened -- a listener heard the peak stand out as a surge and the
whole report said the patch was fine.

**Sustain against peak**, in decibels, is one half of the hole. **Onset at fifty
milliseconds**, as a fraction of the peak, is the other: attack *time* is the
moment a threshold is crossed and says nothing about the shape of the rise, so
an onset that fades in and one that arrives measure the same. That is how an
envelope came to be described as too slow while its attack read `ok`.

Both were reported by ear before either existed, which is the argument for
adding them: the ear found two faults the eleven existing axes could not see.

They immediately said something new. Against the recordings, both presets sat at
**0.02** of their peak fifty milliseconds in, where the clarinet is at 0.45 and
the violin at 0.25. Two causes, in series: this engine's own fit reaches only
0.20 there, and Vital squares the envelope on top of that.

The export half is the same correction as the sustain, applied to a shape rather
than a number -- the curve written has to be the square root of the one fitted,
and the square root of an exponential-approach shape is close to the same shape
with a constant added to its rate. The fitting half is now in the objective, and
`amp_env.attack_curve` becomes searchable to go with it: `fitAttackCurve` picks
it from the contour and picks too gently, and before there was an onset term
there was nothing for a search to improve.

The violin's onset goes from 0.01 to 0.16 against a target of 0.25, which is in
tolerance, and its sustain-to-peak comes into range. The clarinet's onset
improves from 0.02 to 0.08 against 0.45 -- better and not fixed -- and its
sustain-to-peak comes into range. Timbre drift gives way on both, which is the
arithmetic of a weighted sum: a seventh term is a seventh of the weight, and
drift is what it came out of.

### Vital's amplitude envelope is squared, and its power sign was backwards

Reported as a sudden volume increase. Measured, it is the reverse: the note
peaks and then falls much further than it should, so the peak stands out as a
surge. Rendering the same patch both ways, this engine settles at 0.70 of its
peak and Vital at 0.33, with a written sustain of 0.508.

0.508 squared is 0.258, and that is the answer. Vital's voice multiplies by the
amplitude envelope and by a squared amplitude control, so a sustain written
linearly arrives squared. Writing its square root instead brings Vital to 0.65
against this engine's 0.70 -- the same shape, from the same numbers.

The same squaring explains an earlier complaint that the attack was too slow: a
squared envelope rises more slowly than the one that was fitted, and Vital's
onset read 0.14 where this engine's read 0.41.

**And the segment powers had their sign backwards.** The comment claimed
negative was the exponential direction; sweeping the attack power through -5,
-2.5, 0, +2 and +4 and rendering each shows the segment getting slower as it
goes negative and faster as it goes positive. It was assumed rather than
measured, and it had been wrong since the exporter was written. Corrected for
the attack, which is where it was verified; the decay and release keep their
sign, because flipping those as well pushed the peak from a quarter of a second
out to a full second and overshot.

Contours through Vital against this engine, peak-normalised:

| | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 | 2.00 | 2.50 |
|---|---|---|---|---|---|---|---|---|
| this engine | 0.41 | 1.00 | 0.92 | 0.73 | 0.70 | 0.70 | 0.69 | 0.67 |
| Vital before | 0.14 | 1.00 | 0.90 | 0.53 | 0.41 | 0.34 | 0.33 | 0.32 |
| Vital after | 0.36 | 1.00 | 0.92 | 0.74 | 0.69 | 0.65 | 0.64 | 0.62 |

Worth noting what found this: not a metric. Every axis the diagnostic reports
was inside tolerance while the preset was losing two thirds of its level after
the attack, because all of them are shape-relative and none of them compares the
*sustained* level to the peak. A listener heard it immediately.

### The filter sweep is measured over the sounding note

Reported by ear, and precisely: the *filter* envelope's attack is absurdly slow
against the recording. The numbers agree and say why. Before refinement touches
it the clarinet fits an attack of 2.00 s -- pinned at its maximum -- sweeping
3.13 octaves from a base cutoff of **295 Hz**, which is below its own
fundamental at 441 Hz. The filter starts almost shut and crawls open across two
thirds of the note.

The base is contaminated. A cutoff estimate after the note stops is not an
estimate of anything: with nothing left to measure it falls to its floor, the
fundamental, and on a four second render of a three second note that floor is
the last quarter of the trajectory. A quarter is enough to put the tenth
percentile inside the silence, so the sweep was measured from the floor of a
dead note up to a live one.

Confined to the sounding note the clarinet's base becomes 1034 Hz -- above its
fundamental, where a lowpass on a clarinet belongs -- and the sweep 1.68
octaves instead of 4.00.

**It costs the clarinet what the exaggeration was buying.** Brightness goes from
`ok` to 0.28 octaves dull, timbre drift from `ok` to 1.87 out, wobble from `ok`
to 0.99. The violin is unmoved, at seven of eight axes. This was tried once
before and reverted for exactly those regressions; it is kept this time because
the earlier revert was a choice to keep a number rather than a sound, and
because a base cutoff below the fundamental is not defensible whatever it
scores.

That is the fourth time in one session an error has turned out to be propping up
another, and the remedy is not a better filter: today's own measurement says the
clarinet's brightening is *source* movement, since its second harmonic sits
below the estimated cutoff throughout and still rises five decibels. The drift
has to come back from the wavetable frames, which is where it belongs.

### An envelope's full level is what the note holds, not its loudest frame

Reported by ear and long-standing: both envelopes rise too slowly and then drop
to a level that sounds unlike the recording. The violin fitted a 0.61 s
amplitude attack followed by an 8 ms collapse to 0.60, and a *filter* envelope
that swept two seconds up to fifteen kilohertz and then slammed back to four in
29 ms.

The step is structural rather than a measurement error. An ADSR attacks to one
and then decays, so if one is normalised to the loudest single frame -- and on
this violin the loudest frame is a vibrato crest 1.17 s into the note, not an
attack transient -- the fit has to spend a decay getting back down to the level
the note actually sustains. An earlier fix took the same view of the decay and
measured it on a smoothed contour; the step survived, because the peak it decays
*from* was still a crest.

Smoothing over about a vibrato period first and normalising by that maximum
makes one mean the level the note reaches and keeps. Crests sit a little above
one, which nothing downstream minds, and a note that rises and holds fits an
attack and a sustain near one with nothing in between. The absolute scale this
gives up is not information: oscillator levels are solved against the target
afterwards, so the envelope only carries the shape.

The filter envelope needed the same treatment for a different reason. Its shape
comes from the cutoff trajectory, which is an estimate on top of a wobble and
much the noisier of the two, so an ADSR fitted to it chases whichever spike is
highest and falls off it -- the violin's sustain came back at 0.02. Smoothing
the shape over a third of a second before fitting takes that to 0.47, and the
clarinet's to 0.99. Only the shape: the base cutoff and sweep depth are settled
by `trajectoryToEnv`, which has its own defence against outliers.

**And then the filter envelope still built slowly and dropped**, which was
heard before it was measured. Two more causes behind it.

The index where the attack ends was found on the raw curve while the decay was
judged on the smoothed contour, so on a noisy shape the raw curve could fail to
cross full level until some late spike -- by which point the contour had already
fallen past what the decay measures to, the decay loop broke on its first step,
and the envelope became a long climb and a cliff. Full level is now reached when
*either* says so, whichever comes first, which keeps a genuinely fast attack
because its raw crossing is early by definition.

And a trajectory that is itself an estimate does not get to assert a step. A
loudness contour is measured; a cutoff trajectory is inferred frame by frame
from a deconvolution with no unique answer, and a one-frame cliff in it says
more about the estimator than the instrument. A fall there now has a floor
proportional to its size, a quarter second for the full range. Measured
envelopes keep their own answer, so nothing stops a real one decaying fast.

Set from what it is worth rather than by taste: at half a second per unit the
violin loses its brightness and a third of its noise, at 0.15 it loses more, and
at 0.25 the clarinet reads `ok` on all eight axes and the violin on seven.

**The clarinet now reads `ok` on every axis**, which it has never done, and the
violin on every axis but one. Its noisiness is 0.281
against the recording's 0.287, which is the oldest complaint in this file and
the first time it has been in tolerance; brightness, wobble, vibrato and
note-off are all in range, and the timbre drift is 1.01 dB out against a
tolerance of 1.00. The clarinet keeps its wobble and drift and gives up some
brightness and 0.12 s of attack.

`lfo_amp.golden.json` was re-blessed: its amplitude sustain moves from 0.492 to
0.664, which is the same fixture measured under the new convention rather than a
change in what the analysis sees. It is an amplitude-modulated case, which is
exactly where a crest and a held level differ.

### Trajectory terms in the objective

Everything that resisted fitting today resisted in the same way. Refinement
moved the clarinet's attack *away* from its target while improving its own loss.
The wavetable ladder rejects the frames that carry movement because it minimises
average profile error. The reverb matched the tail and got the balance during the
note wrong. Four attempts at timbre drift, and the answer each time was that the
thing being minimised was not the thing wanted.

Spectral, loudness and centroid distances are all averages over frames. A model
can match every one of them and still get the *shape* of the note wrong.
Meanwhile `autosynth_diff` measures eleven named axes whose verdicts have agreed
with a listener repeatedly, and the objective was using none of them.

So three of them are now inside it: the attack in seconds, the harmonic movement
in decibels, and the wobble depth in decibels, each as the distance between a
measurement of the candidate and the same measurement of the target. The
adaptive weighting already normalised each term by its value at the starting
patch so that all of them contribute equally and the optimiser is asked to
improve them proportionally; six terms extend that rule rather than change it.

The drift is read off the spectrogram at multiples of the known fundamental
rather than from tracked partials -- the diagnostic tracks partials, which is far
too expensive to do a hundred and ninety-two times, and the harmonic bins are the
same measurement to within the vibrato both signals share.

**On the two recordings it fixes the two things that would not move.**

| | before | after | target |
|---|---|---|---|
| clarinet attack | 0.256 s | **0.299 s** | 0.337 s |
| clarinet timbre drift | 1.7 dB | **4.9 dB** | 4.3 dB |
| violin attack | 0.299 s | **0.405 s** | 0.401 s |
| violin timbre drift | 0.3 dB | **1.2 dB** | 1.6 dB |

Attack and drift both read `ok` on both samples, where four attempts at the
wavetable frames and two at the filter had left them 0.08 s and 2.6 dB out.

**And it costs brightness.** The clarinet goes from 0.17 octaves dull to 0.38,
the violin from `ok` to 0.21. That is not sample-specific: over twelve harness
trials the centroid distance goes 0.533 to 0.676 and the loudness distance 3.27
to 4.47 against an unchanged control. Asking for six things instead of three
halves what each of the original three can claim, and brightness is what gave
way.

**The harness cannot see the win**, which is worth stating plainly: it scores
spectral, loudness and centroid, so a change that trades those for trajectory
accuracy reads there as a pure regression. Its scores got worse while the axes a
listener has objected to got better. Teaching it to score the attack, the drift
and the wobble is the obvious next thing, and until it does the two real
recordings are the only place this change can be judged.

### Refining against Vital, and what it cost to try

The reason to render through Vital at all was never the diagnostics. It was to
put Vital *inside* the refinement loop, so that the optimiser measures the synth
the preset will be played by and every difference between the two engines stops
needing to be found and hand-corrected in the exporter.

`Refine::Options::renderer` is how: a callback that turns a candidate patch into
mono samples, empty by default and meaning "the engine in this repository".
Nothing in `autosynth_dsp` learns about plug-in hosting; the tool that owns a
synth owns the renderer, and `autosynth_vital --fit target.wav` supplies one
backed by Vital.

**It costs about 90 s per fit** against 33 s through this engine, at 192
evaluations. Per evaluation that is 300 ms, split roughly 150 ms of preset load
and 150 ms of render for two seconds of audio, and the plug-in itself loads once
rather than once per candidate.

Two things had to be fixed before the number meant anything, and both were found
by rendering one patch repeatedly and comparing the results, which is a cheaper
question than "is the fit any good" and answers it first.

**Vital randomises each note's starting phase.** Six renders of one patch
differed by 0.391 at the sample, on a signal peaking at 0.22 -- more noise than
signal, and an optimiser reading it would have been scoring the phase lottery
rather than the parameters. This engine starts every oscillator at zero, so
`osc_N_random_phase` is now exported off: a faithful translation that happens to
make fitting possible.

**Something still alternates.** With phase fixed the renders settle into two
states, flipping every other render, 0.035 apart. Not a decaying tail -- a tail
would fade rather than toggle -- and not the noise bed, which makes no
difference when removed, and not cured by `reset()`. It is eleven times smaller
than the phase problem and is left standing, named rather than guessed at.

**And the result is honestly mixed.** Fitted through Vital and played by Vital,
against the recording: amplitude wobble improves from 1.23 dB off to 0.89, and
attack from 0.10 s off to 0.06. Timbre drift goes the other way, from 1.39 dB
off to 2.69. Pitch and brightness are a wash. The final loss is 0.660 either
way, which is the more telling number -- it was 0.6603 with the phase lottery
running and 0.6600 without, so the noise that looked disqualifying was not what
was limiting the fit.

So: viable, affordable, and not yet demonstrated to be *better*. One sample is
not a decision, and the honest next step is the recovery harness over many
patches rather than another listen to the clarinet. What the experiment does
settle is that the mechanism works and what it costs, which is what it was for.

### The harness in Vital's world, and the gap it found

`Recovery::Options::renderer` takes the same shape, and `autosynth_vital --eval`
supplies it. When set it replaces *every* render -- target, fitted, control and
each refinement candidate -- so random patches rendered by Vital become the
targets. That is all-or-nothing on purpose: rendering the target in one synth
and the control in the other would measure the difference between two synths and
report it as a fitting error.

Twelve trials, seed 0, refinement on. Absolute scores are not comparable across
the two runs because the targets are different populations; what compares is how
far each run beats the control it ran with.

| | this engine | through Vital |
|---|---|---|
| spectral | 0.799 vs 2.312 control (2.9x) | 0.510 vs 1.279 (2.5x) |
| loudness dB | 3.704 vs 21.463 (5.8x) | 9.128 vs 25.075 (2.7x) |
| centroid oct | 0.384 vs 1.688 (4.4x) | 0.226 vs 1.290 (5.7x) |
| oscillator count exact | 83.3% | **58.3%** |
| root within a semitone | 91.7% | **75.0%** |

**The control tells the real story.** Its spectral distance nearly halves, from
2.312 to 1.279. The control is a *default patch* scored against the target, so a
smaller number means the targets are closer to the default -- Vital is rendering
a narrower range of sounds from the same random patches than this engine does.

That is not Vital being less capable. It is the exporter dropping most of the
IR. `Patch` carries a filter, an envelope and a reverb send **per oscillator**,
and `VitalExport` writes none of them: one global `filter_1`, and `env_1` to
`env_3`. So a random patch whose three oscillators have three different filters
exports as three oscillators sharing one, and the variety collapses.

Everything else follows from that. Those exact parameters dominate the
worst-recovered list in the Vital run -- `oscs.0.filter.resonance`,
`oscs.2.filter.env.curve`, `oscs.0.filter.env.attack`, `oscs.2.reverb_send` --
because they are unrecoverable by construction: nothing in the rendered audio
depends on them. Counting and root pitch fall because analysis is reading audio
whose sources have been flattened together. And refinement was searching
dimensions the renderer ignores, spending its budget on a flat objective, which
is the likeliest reason the clarinet fit came out no better than before.

**So the renderer swap was blocked on the exporter, not on the renderer.** The
measurement that looked like "is Vital a good enough optimiser target" turned
out to be "the preset does not yet say everything the patch does".

### Closing it, and the parameters that turned out not to exist

The guard first, because it is the part worth keeping: `test_vital_export.cpp`
now walks `Refine::scopeFor` on a patch with everything switched on, moves each
parameter to the far end of its declared range, and asserts the exported preset
changes. If the fitter is allowed to search it, the preset has to carry it.
Anything else is an optimiser working on a flat objective and a preset that
quietly means something else. It listed forty-five parameters on the first run.

Fifteen of them were the per-oscillator amplitude envelopes, and those are
straightforwardly expressible: Vital has six envelopes against this engine's
need for three, so env_4 to env_6 now drive `osc_N_level`. The level is written
as zero and the envelope's *amount* carries it, because modulation adds rather
than multiplies and an envelope that scales an oscillator has to be able to
silence it. The shape is not identical -- Vital's level control is quadratic, so
the amplitude follows the square of the curve -- and that matters less than it
sounds, because what the export owes the fitter is that the parameter has an
*effect*. Three more were the delay, dropped as silently as the reverb had been.

The remaining twenty-seven were per-oscillator filters and reverb sends, and the
plan for them was to trim the IR: two shared filters plus a routing choice,
matching Vital exactly. Sizing the change first is what stopped it. Sixteen
files, the golden engine fixtures, the editor -- and then the reason not to
bother at all: **`PartialFit` never enables a per-oscillator filter.** It leaves
`filterEnabled` false on every path. The only patches that ever had one came
from the recovery harness's own random sampler and from a hand switch in the
editor.

So the export was not dropping something real. The harness was *generating*
something unreachable, scoring the fitter on its failure to recover a feature no
fit can emit, and -- because those were exactly the parameters the exporter
dropped -- that is most of why the run through Vital looked so much worse than
the run through this engine. The fix is two small deletions rather than a
refactor: `scopeFor` no longer offers them, and `randomPatch` no longer
generates them.

Reverb send goes for the plainer reason. Vital sends an oscillator to a filter,
to the effects bus, or straight out; there is no per-oscillator send *level*, so
no value searched here could ever reach the preset.

The rule both follow, and the one worth carrying forward: **a parameter the
deliverable cannot express is one the fitter must not search.** Before the
renderer is Vital it is merely misleading; after, it is a flat direction the
optimiser spends its budget on. The harness stops *scoring* them for the same
reason -- a parameter nothing is permitted to move reports the distance between
two random draws, and reads in the worst-recovered list as a fitter failure
while crowding out the real ones.

**And the fresh baselines, which is where the warning at the top of this section
earns its keep.** Removing one `rng.nextBool()` per oscillator shifts every
draw after it, so the target population is not the one the earlier numbers were
measured on. It happens to be much harder: three-oscillator targets went from
one of twelve to six of twelve, and oscillator counting reads 33.3% where the
old population read 83.3%. That is not a regression, it is a different exam, and
the two must not be diffed. Both runs below share one population.

| | this engine | through Vital |
|---|---|---|
| spectral | 0.966 vs 2.004 control (2.1x) | 0.573 vs 1.022 (1.8x) |
| loudness dB | 3.266 vs 22.220 (6.8x) | 5.328 vs 15.197 (2.9x) |
| centroid oct | 0.533 vs 1.293 (2.4x) | 0.397 vs 0.757 (1.9x) |
| oscillator count exact | 33.3% | 33.3% |
| root within a semitone | 33.3% | **58.3%** |

Counting now lands identically, where before the trim the Vital run was
twenty-five points behind -- which is the clearest evidence that the gap was the
unreachable filters rather than anything about Vital. Root pitch is better
through Vital than through this engine, which was not predicted and is not yet
explained. This engine still beats its control by more on all three distances,
which is what you would expect of an analysis stage tuned against it for the
whole life of the project.

The control distances stay lower through Vital -- 2.004 against 1.022 on the
spectral term -- so Vital still renders a narrower range of sounds from the same
patches than this engine does. Some of that is Vital's own character and some is
export that has not been found yet. It is the number to watch as the exporter
grows.

### Timbre drift: the trajectory arches, and the frames are built from its ends

Both fits are too static -- the clarinet renders 2.0 dB of drift against the
recording's 4.3, the violin 0.3 against 1.6 -- and the same numbers come out of
both engines, so this is the fitter rather than the export.

Measuring the recording's harmonic profile in windows across the note says why.
The second harmonic runs -8.2 dB, -4.3, **0.0**, -4.8, -8.5: it climbs about
eight decibels and falls back. The trajectory *arches*, and its two ends are
within 0.3 dB of each other.

Two consequences, and the wavetable ladder is blind to both. The three frames
are the energy-weighted mean over three equal thirds of the note, which averages
the peak away -- the middle third reads -7.8 where its own middle reads 0.0 --
so the fitted frames span 1.7 dB of an eight decibel movement. And the rung that
decides whether to spend frames at all measures `driftDb` between the *first and
last* frames, which for an arch is close to nothing.

**Sampling the frames at the trajectory's extreme instead was built and
reverted.** Taking the ends as points and the middle as the point furthest from
the line between them is the obvious repair, and it made both fits worse: the
clarinet fell to 1.1 dB of rendered drift and, on inspection, to a single frame.
The scoring interpolates the three frames against a position that ramps linearly
across the note, so it assumes frame one sits at the temporal midpoint. Move the
frame and the model stops matching the data it is scored against, `frameError`
stops beating the static table by its twenty percent margin, and the ladder
throws the whole thing out.

Two further attempts, and then the actual reason.

**Sampling the frames where the sweep visits them** -- the ends and the middle,
as instants rather than averages -- keeps the model aligned with the position
ramp and still avoids the flattening. It fails too, and printing the ladder's
own numbers says why rather than leaving it to inference:

| clarinet | averaged over thirds | sampled at instants |
|---|---|---|
| static table error | 0.0248 | 0.0248 |
| three-frame error | **0.0188** | **0.0486** |
| verdict | beats the static table | twice as bad as it |

Averaging is not a mistake in the scoring, it is what makes the model *score*.
A mean over a third sits near the overall mean everywhere, so its
energy-weighted error is low; instants are extreme at the knots and wrong
between them, and a linear interpolation between three extremes is a triangle
drawn through a curve.

**Deciding on the averaged frames and drawing the sampled ones** is the obvious
way to have both, and it is defensible -- whether the movement is worth
thirty-two numbers and what the movement *is* are separate questions. It moved
the clarinet's rendered drift from 2.0 dB to 2.1 against a target of 4.3. Not
worth two frame models for a tenth of a decibel.

So the obstruction is not how the frames are sampled. **The ladder's criterion
is an energy-weighted average profile error, and an average is exactly what a
model reproducing *movement* does not minimise.** A static table is the best
possible answer to "what one profile is closest to all of them", and it will
keep winning against any three-frame model on that question no matter where the
three come from. The same shape of problem as the refinement objective moving
the attack away from its target while improving its own loss: the search is
fine, the thing being minimised is not what is wanted.

A drift term in the ladder was the next thing to try and turned out to address
neither recording. Printing the gates per sample says why: the clarinet passes
both and already spends its three frames, so a rule about *whether* to spend
them changes nothing there, and the violin was measured long ago as landing in
the same place with frames forced on. Building it anyway would have been adding
a criterion on faith.

**Where the drift actually lives.** `autosynth_diff` now breaks the number down
by harmonic, because a single figure saying the tone moves says nothing about
what moves, and four attempts had been aimed at a quantity nobody could point
to. On the clarinet, late third minus early third:

| harmonic | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| recording | +5.0 | +4.5 | +6.8 | +1.4 | +8.1 | +16.7 | +6.6 |
| fit | +2.5 | +0.1 | +3.5 | -2.8 | +3.0 | +9.8 | -1.2 |

Every harmonic rises against the fundamental, by more the higher it is. The note
brightens *monotonically* across its whole length. An earlier reading of the
same trajectory as an arch was an artefact of measuring fixed frequencies with a
plain transform instead of the tracked partials the diagnostic uses -- worth
recording as a caution, since it sent three attempts at the wavetable frames
after a shape that was not there.

Monotonic brightening is what a filter envelope is *for*, and the fitted one is
working against it: attack 1.41 s on a three second note, and a sustain of 0.51,
so the filter opens inside the first third and then closes halfway again.
Stretching the attack to 3 s and holding the sustain at 1.0 takes the rendered
drift from 2.0 dB to 2.6 against the target's 4.3, with brightness still in
tolerance -- and overshoots the top harmonics while undershooting the second and
third, which is a lowpass tilting the whole spectrum where the recording lifts
its low harmonics more than a tilt would.

So the wavetable ladder was never the place. The question became why the filter
envelope fits a fast attack and a half sustain against a note whose harmonics
rise throughout, and the answer is that **the cutoff trajectory it is fitted
from is not good enough to derive a sweep from, and the pipeline currently works
partly by accident.**

The estimate on the clarinet, frame by frame, runs 1180, 1292, 1731, 1505, 2498,
3108, 4124, 2128, 3975, 1076, 2866 Hz and spikes to 19.8 kHz. Then at note-off
it falls to 441.6 Hz -- the fundamental, its floor, because with nothing left to
measure there is nothing to estimate -- and sits there for the last quarter of a
four second render. A quarter is enough to put the tenth percentile *inside the
silence*, so the sweep is measured from the floor of a dead note up to a live
one and reads 3.4 octaves, which clips against the 4 octave limit.

Three repairs, each of which made both samples worse:

| clarinet | timbre drift | brightness |
|---|---|---|
| as it stands | 2.0 dB | ok |
| range taken from the sounding note only | 1.4 dB | dull by 0.31 oct |
| and the trajectory smoothed to 0.3 s | 0.4 dB | ok, but `envAmount` went *negative* |

The first says the 3.4 octaves was wrong *and* load-bearing: confined to the
sounding note the tenth-to-ninetieth range is 0.59 octaves, which is the width
of the noise band rather than any trend, and too small to produce a brightening
of five to sixteen decibels. The inflated figure was accidentally supplying a
sweep the honest one cannot. The second moved the base cutoff from 1515 Hz to
4338 Hz, leaving the filter wide open, after which refinement found it better to
*close* the envelope over the note than open it.

That is the third time this session an error turned out to be cancelling another
one -- the oscillator level clipping against the reverb proportion, the attack
solved against the wrong signal, and now this. It is worth naming as a pattern:
a pipeline tuned end to end against two recordings will find these balances, and
each one has to be unpicked with the other held still.

**The trend fit was then built**, since percentiles of a noisy estimate describe
its spread and an envelope needs its trend. Median-filtered over a twelfth of
the note to kill the spikes, with the sweep read as the difference between
medians of the first and last sixth: on the clarinet that gives 2.74 octaves
from a base of 1470 Hz, and brightness lands at 1171 Hz against the recording's
1143. It is the right way to summarise the trajectory and it made the drift
worse -- 2.0 dB to 1.4 -- so it went the way of the other three.

**And measuring why settled the whole question.** The estimated cutoff, as
medians by third of the sounding note, is 1424 Hz, 2224 Hz, 2128 Hz: it rises
about 0.58 of an octave and then stops. The recording's second harmonic sits at
883 Hz, *below that cutoff for the entire note*, and rises 5.0 dB from the early
third to the late one.

**A lowpass cannot raise a harmonic that is already in its passband.** So the
clarinet's brightening is not a filter sweep at all -- it is the source spectrum
moving, which is what the wavetable frames exist for. And the frames cannot see
it, because the deconvolution that hands them their input divides out a cutoff
trajectory that has absorbed part of the movement and invented the rest. The
1.77 dB the ladder measures is what survives that.

Which is the degeneracy this file warns about at the top of `WavetableFit`,
arriving from the direction nobody was watching: a free-form spectrum and a
filter explain the same signal, the order of fitting was chosen to stop the
frames stealing the filter's job, and the filter is quietly doing the reverse.

So the work is in the filter/source split rather than in either side of it. The
useful lever is that the per-harmonic early-to-late tilt is *robust* -- it comes
from tracked partials and it agrees with itself across measurements, unlike the
ALS trajectory -- so it can say how much of the movement any candidate filter is
allowed to claim. A sweep that would need to lift a harmonic below its own cutoff
is claiming too much.

Four attempts, four reverts, and the line of work redirected rather than
advanced. Recorded at this length because the next person to look at timbre
drift will otherwise start where the first attempt started.

### Wander stops costing a slot

The roadmap said an LFO is the wrong model for human wander, that it is
audible, and that the fitter lacked a slot to spend on it. The second half was
the real blocker, and it was this engine's shape rather than the target's:
`kNumLfo` is two, both are taken on both library samples, and the wander was
expressed as a *second LFO pointed at the first one's rate*. So the feature was
measured, built, and then never once applied to either recording it was written
for -- the gate `out.size() < maxCount` was never open.

Vital's four random LFOs sit outside its eight ordinary ones and cost none of
them. Expressing wander as two fields on the LFO that drifts -- how far the rate
moves, in octaves, and how fast it moves -- costs nothing here either, and the
gate is gone. Measured, not searched, on the same rule as every other
measurement: `detectWander` reports both numbers and refinement leaves them
alone.

The violin's amplitude LFO now carries 0.811 octaves of drift at 2.78 Hz, which
is what the detector reads off the recording. Rendered, this engine reaches 0.61
octaves against the recording's 0.81, up from 0.47 without it.

**What it does not reproduce is the drift being *periodic*.** The recording's
period wander is detected at a coherent 2.78 Hz; a random LFO is aperiodic by
construction, so the render measures more drift but still reports no drift
*rate*. Whether the recording's 2.78 Hz is a real periodicity in the playing or
an artefact of tracking a period that moves nearly as fast as the carrier is not
something the numbers here settle. If it is real, the faithful shape is an
ordinary LFO on the rate -- which Vital has seven spare of, and which the IR
could carry the same way.

The measured amplitude wobble now *overshoots*: 3.2 dB against the recording's
2.2, where before it was 1.8. A drifting rate spreads the wobble, and the metric
is an rms deviation that cannot tell depth from movement. This is the point
where the number stops being the arbiter.

### Solving the level and its modulation together

Vital's oscillator level is quadratic and its modulation *adds* to the stored
value. Writing a fitted tremolo depth in as the modulation amount gets both
wrong at once: the swing lands on the square of the parameter rather than on the
amplitude, and the top of it clips against the control's ceiling of one. A depth
of 0.30 on a level of 0.95 swung the stored value from 0.65 to 1.24, of which
everything above 1.0 was thrown away.

The arithmetic is not a matter of taste. For an amplitude ratio of (1+d)/(1-d)
between peak and trough the stored values need a ratio of r = sqrt((1+d)/(1-d)),
which fixes the amount as a fraction (r-1)/(r+1) of the level; the level follows
from wanting the mean amplitude to stay where it was. Where level plus
modulation would still pass one, every oscillator is scaled down together and
the master is given the difference back -- scaling one alone would change the
balance between them, and the master cannot fix that per oscillator.

**And the measured wobble got *smaller*.** The clipping had been exaggerating
the swing: throwing away the top of the stored range while leaving the bottom
alone widened the amplitude ratio from the intended 1.85 to 2.37, which read as
more tremolo, which was closer to the target. Correcting the mapping removed an
error that had been cancelling another one.

The other one is the reverb. Rendering the same patch through both engines with
the reverb switched off, the wobble reads 2.0 dB here and 1.7 dB in Vital, which
is agreement; with it on, 3.4 dB against 1.9 dB. Vital's reverb is
proportionally far louder in the mix, so it fills the dips.

That is the crossfade-against-send mismatch again, seen from the other side.
Matching the tail after the release needs a high dry/wet, because after the
release the tail is all there is; matching the balance *during* the note needs a
low one, because here the reverb is a send that leaves the dry alone. One
control cannot do both, and the tail is the half that was calibrated against the
recordings and listened to. The level mapping is now right and the wobble
shortfall is the reverb's, which is worth knowing before anyone tunes the
tremolo depth to compensate for it.

### Every searchable parameter is audible in Vital

The unit test guarding the export can only ask whether a parameter changes the
preset *text*. That catches a parameter nobody writes and misses the worse case:
one written into a control that turns out to be inert, because it is scaled to
nothing, or routed somewhere the signal never reaches, or sits on a page Vital
ignores. Both look identical from this side of the boundary, and only one of
them can be heard.

`autosynth_vital --sweep` moves each parameter in `Refine::scopeFor` to the far
end of its declared range, plays the result through Vital, and reports the
spectral distance from the unmoved patch. The answer is that nothing is inert.
The weakest, an oscillator's envelope curve, still moves the output by 0.118,
and the list runs smoothly up from there to 3.11 for the master level -- no
cliff towards zero, which is what an unreachable control would look like.

That closes the question the harness raised and rules out the obvious
explanation for the remaining gap. Vital still renders a narrower range of
sounds from the same random patches than this engine does -- control spectral
distance 1.022 against 2.004 -- and it is *not* because parameters are failing
to arrive. Something else is compressing the range, and the next place to look
is how far each parameter moves each synth, rather than whether it moves them at
all.

### The attack is a fitter problem, and the measurement is not solid enough to fix it

Both library samples export with an attack about 0.10 s faster than the
recording, which looked like an export fault until the same patch was rendered
by *this* engine and came out 0.10 s and 0.08 s fast as well. So the preset
carries a too-fast attack faithfully; the attack is fitted short.

The cause is a mismatch of signals rather than of numbers. `attackMeasuring`
solves the attack parameter whose *envelope* measures back as the target's
attack, which is the right idea against the wrong signal: what a listener hears,
and what `autosynth_diff` measures, is the loudness contour of the finished
audio, and that also carries the filter envelope, the reverb and the
oscillator's own onset. On the clarinet the filter envelope opens over 1.4
seconds, so the rendered rise is slower than the amplitude envelope alone
predicts -- the fit aims at 0.337 s and renders 0.453.

Refinement then moves it the other way. It is allowed to search the decay and
the sustain, and moving those changes the rendered rise without touching the
attack at all: the parameter goes 0.507 to 0.462 while the rendered attack goes
0.453 to 0.267. Deriving the attack holds a crossing of the *envelope* fixed,
and that is not the quantity that ends up being heard.

**Closing the loop against the render was built, measured and reverted.** The
obvious fix is the one the noise level already uses -- render, measure, correct,
repeat -- and it worked on the clarinet, moving the rendered attack from 0.267
to 0.315 against a target of 0.337, stable at both 44.1 and 48 kHz. A damped
proportional step left the violin untouched, because its response has a plateau
that reads as a small gradient; bisection fixed that, the response being
monotonic even where it is not smooth.

Then the violin went unstable. The same patch measured 0.499 s at 44.1 kHz and
0.341 s at 48 kHz -- a 46% swing from a 9% change of rate, which is not a fact
about the sound. Probing either side of both solutions explains it: a ten
percent move in the attack parameter swings the measurement by 75% on the
clarinet and 41% on the violin. Both solutions sit next to a cliff. The
clarinet's good result was luck about which side of one it landed on.

`attackSeconds` is the time to nine tenths of the level the note *holds*, and
when the decay is near-instant and the sustain low -- the violin fits a decay of
0.005 s and a sustain of 0.27 -- that crossing sits on a nearly flat stretch of
the contour, where which frame crosses first stops being a fact about the sound.

So the loop is reverted rather than shipped with a threshold tuned to make two
samples pass. Solving against an ill-conditioned measurement is the failure this
project keeps writing down in new costumes, and the order of work is the other
way round: **a measurement worth solving against first, then the loop.** A rise
time between two fractions of the peak, or a time to peak, would not depend on
an estimate of the held level at all. That change moves every attack number
this project has recorded, which is a thing to do deliberately.

### Noise as a waveform

Two attempts to give the noise bed an envelope failed by bolting one on beside
it: a separate ADSR competed with the amplitude envelope, both real samples got
worse, and the whole thing was reverted. `Waveform::noise` is the same idea with
the plumbing already built. Selecting it makes an oscillator a noise generator,
and it inherits everything an oscillator has — its own level, envelope, filter,
reverb send — from machinery that is already fitted.

The immediate payoff is the branch for unpitched material. It used to set a bare
global noise level and switch the filter off, which describes a cymbal, a snare
and a breath identically: flat, static, and lasting exactly as long as the
amplitude envelope says. As an oscillator it gets a measured attack and a
measured brightness instead of neither. The `noise` fixture now fits as one noise
oscillator rather than as zero oscillators plus a level.

Three details that are not obvious:

- **The waveform matcher must not be able to choose it.** Noise's harmonic
  profile is flat, and a flat profile is close enough to a badly-measured one
  that the matcher would reach for it whenever the analysis was struggling —
  describing a pitched note as noise, which is the least useful thing a patch can
  say. It is excluded from the candidate list; only the branch that has already
  decided a sound is unpitched selects it.
- **It still needs a harmonic profile.** The fitter reads profiles through
  `harmonicAmplitudes`, so a case that fell through the switch would silently
  return silence rather than an error. Flat is both the honest answer and the
  safe one.
- **Silence is not unpitched material.** A noise oscillator at any level is a
  worse description of an empty file than an empty patch, so a peak below a
  thousandth takes the old path and proposes nothing.

Two noise oscillators are two generators, seeded separately. Sharing one would
make them the same signal at double level, which is a chorus of one —
`test_engine.cpp` checks they add in power rather than in amplitude.

### The noise level is measured, not solved

A listener reported the violin had lost its background noise, and the cause was
a parameter three steps away from it. Noise is one column in a least-squares
solve against every oscillator, so *anything* that changes how the oscillators
render changes how much noise is left for it: fitting an attack curve — which
has nothing to do with noise — dropped a violin from 0.09 to 0.015 and audibly
stripped the bow off it.

`calibrateNoise` closes the loop instead: render, measure the energy between the
harmonics, scale, repeat. Two or three passes, against the same quantity
`autosynth_diff` prints as noisiness, so the number is stable under changes that
have nothing to do with it.

Two guards, both of which cost measurements to find:

- **Refinement may sharpen it but not replace it.** Left at its full range,
  CMA-ES undid the calibration immediately — broadband energy lowers a
  log-spectral error wherever the harmonic fit is imperfect, and the optimiser
  always finds that trade. Removing it from the search entirely was tried and
  costs *more* than it saves: CMA-ES loses a dimension and lands elsewhere,
  putting the harness's brightness error up by a third on patches whose noise
  was already zero. Half to double, like the measured attack before it.
- **A margin, because this measurement cannot tell hiss from smearing.** A
  vibrato'd oscillator spreads energy between its own harmonics, so a perfectly
  clean synthetic target still measures inter-harmonic energy, and matching that
  number means adding hiss to imitate a wobble.

The trade is real and is recorded rather than summarised away. Counting reached
its best figure, **75.0%**, and invented noise on noise-free targets fell from
0.030 to 0.015, audible in seven of twenty-four rather than twelve. Against that,
spectral distance went from 0.576 to 0.668 and brightness from 0.286 to 0.396 —
on a metric that has repeatedly been shown to *prefer* hiss, since filling empty
bins lowers a log-spectral error. The violin's noise came back, which is what the
change was for.

### Wander: measuring how mechanical an LFO is

A listener kept reporting more vibrato in the fits than in the recordings, while
every metric said the fits wobbled *less*. Both were true. What separates them is
regularity, and it is now measured: `Modulation::detectWander` reads the wobble's
period cycle by cycle and reports how far that period drifts, in octaves.

| violin | target | fit |
|---|---|---|
| pitch LFO period drift | 0.68 octaves over 15 cycles | **0.03** over 14 |
| amp LFO period drift | 0.81 octaves over 22 cycles | **0.08** over 10 |

A player's wobble period wanders by most of an octave. Ours is a metronome, by a
factor of ten to twenty. `autosynth_diff` reports it next to every detected
modulation, and the measurement is only trustworthy where the modulation is
strong {d} on the clarinet, whose modulation is weak, the same estimator reads
0.4 to 0.8 octaves on a signal we know to be a pure sine, which is noise.

The engine can now reproduce it: `LfoDest::lfoRate` and `lfoDepth` point an LFO
at the other slot, which is how Vital's matrix does it. Rate modulation warps
*time* rather than the rate, because the phase here is a closed form of elapsed
time and changing the rate directly would jump the phase every time the
modulator moved; at a scale of one it is arithmetically identical to what it was,
so nothing unmodulated changes by a bit.

**What the fitter can and cannot do with it.** A modulator only takes a slot
nobody else wanted. With two slots it otherwise costs a whole modulation, and
losing a violin's tremolo to smear its vibrato is not obviously a trade worth
making, so it is not made. Neither real sample has a spare slot, so neither gets
one.

A third slot was tried and reverted, and the reason is worth keeping. The
recovery harness randomises every continuous parameter, so a third LFO changes
how many random draws a target consumes and therefore *which targets get
generated*: the control distance moved from 2.28 to 2.46 on patches that were
supposed to be unchanged, which silently invalidates every number this project
has recorded. Widening the IR is not free even when nothing uses the new width.

### The attack gets its own curve

The third attempt at the same problem, and the one that worked.

A linear attack is a straight line in *amplitude*, which is a very quiet first
third: with the envelope reaching nine tenths at exactly the right moment, a
violin fit was still 16 dB below its target at note-on and converged only by
0.35 s. The note emerged from silence rather than starting.

Sharing the decay's `curve` was tried first and made both samples worse, because
one number cannot shape an attack, a decay and a release at once. `attackCurve`
is a separate field, defaulting to linear so nothing already written moves.

**And the fitting criterion mattered more than the parameter.** Fitted by
amplitude error over the rise, it chose linear every time {d} a linear ramp's
error is concentrated where it is 10 to 16 dB down but only a few percent away
in amplitude, so an amplitude comparison cannot see the fault it is looking for.
In decibels it picks a real curve, and the order matters too: the curve is fitted
*before* the attack length is solved from the measured crossing, because the
crossing depends on the curve.

| violin onset, dB below target | before | after |
|---|---|---|
| at 0.09 s | 9.3 | **6.3** |
| converged by | 0.35 s | **0.11 s** |

**But only for amplitude contours.** `fitAdsr` also fits the *filter* envelope,
whose input is a cutoff trajectory measured in octaves — already a logarithm.
Fitting its attack curve in decibels takes the logarithm twice and bends it far
too hard: a clarinet's filter came back with an attack curve of 2.0, opening most
of two octaves in the first fraction of its attack, and a listener heard a burst
of noise at the start of the note that then dissolved.

That one is worth dwelling on, because every measurement available said the fit
was *fine*. Its inter-harmonic noise was lower than the target's at every point
in the note (0.035 to 0.050 against 0.055 to 0.065), so `autosynth_diff`
reported "cleaner" and nothing else. What it did show, once read in
quarter-second windows, was brightness overshooting from 0.3 s onward — 1052 Hz
against the recording's 861, growing to 1189 against 917. An over-opened filter
on a reedy spectrum reads as hiss, and a noisiness metric that counts energy
*between* harmonics cannot see it, because the energy is still on them.
Isolating it took three rounds of A/B renders and a listener: reverb off, frames
collapsed, envelope flattened, and finally the filter envelope removed, which was
the one that did it.

Fitted in its own domain the same clarinet gets an attack curve of 0.50, the
violin 0.00, and the violin's overall brightness lands at 3120 Hz against a
target of 3118.

The harness agrees, which the shared-curve version never managed: spectral
distance 0.582 to 0.576, loudness envelope 4.23 to 4.19 dB, counting unchanged
at 70.8%.

### Rank within a group, and the four guards it needed

Grouping asks which partials share a fundamental and stops there. Two sources an
octave apart share every harmonic index the lower one has, so they arrive as one
group and the count comes back one short — which was every remaining
under-count on the harness.

Inside a group they are still separate *components*: one oscillator makes a
rank-one matrix, a fixed spectral profile times one envelope, and two make it
rank two whatever their interval. `nmf::selectRank` reads that rank per group.
On synthetic matrices of known rank it is exact, including the octave pair and
including the case that must *not* split — two spectra sharing one envelope,
which really are one oscillator with a different waveform.

On real material it was a disaster, four times over, and each guard is a
separate lesson.

**The filter has to come out first.** A filter sweep tilts the spectrum over
time, which is genuinely not rank one, so a single oscillator behind a moving
filter factorises as two. The rank is now read from the *deconvolved* matrix,
the same ordering the waveform fit already uses. The components are then applied
as a soft mask to the original matrix rather than as a reconstruction of it, so
everything downstream still sees measured data and the filter is divided out
once, where it always was.

**A ratio is not a parsimony rule.** Taken greedily, rank two split eight of
twelve *single*-oscillator targets in two, because a real oscillator's harmonics
are never exactly rank one and a second component always buys something.

**Nor is one absolute threshold.** Two were tried and each fails a case the
other handles. "Rank one must be visibly bad" refuses a *perfectly clean*
two-source matrix whenever rank one happens to score tolerably — the synthetic
octave pair reconstructs to 2e-7 at rank two and was rejected because rank one
reached 0.20. "Rank two must explain the matrix outright" refuses two real
instruments, where neither rank reconstructs cleanly. Either is now sufficient,
because they are evidence of different kinds.

**And a split has to be into different fundamentals.** Two components sharing one
fundamental are not two oscillators, they are one oscillator whose timbre moves
— which already has a model, the wavetable frames, describing it in sixteen
numbers rather than a whole second oscillator. Without this the two models
compete for the same evidence and the expensive one wins by default: a clarinet
with 4.3 dB of measured drift came back as two oscillators at the same pitch and
a violin as three. Same discipline as the release and the reverb.

A component's own fundamental comes from *which* harmonics it owns, by greatest
common divisor — energy only on even harmonics means it is really at 2*f0. It
has to be "owns" rather than "has energy at", because the factorisation leaks
worst onto the low harmonics where the other component is loudest: a source a
twelfth up came back with 1.00 on harmonic three and 0.13 on harmonic one, and
gcd({1, 3}) is 1, so a correct split was thrown away.

| | before | after |
|---|---|---|
| counting exact | 66.7% | **70.8%** |
| truth 3 recovered | never | once |
| spectral | 0.566 | 0.582 |
| clarinet, violin | 1 osc each | 1 osc each |

The octave case the whole thing was built for — `7 and 19` — is still
missed, and the honest reading is that this fixed the *general* machinery
without yet fixing that instance. What it did fix is the three-source targets,
which had never once been recovered.

### Counting: two bugs the confusion matrix found

"50% exact" says nothing useful. Splitting it by which way the miss went, and
printing the interval and level of every source in every trial that missed,
turned one number into a to-do list — and it named two separate bugs.

```
  truth \ fitted    1    2    3            truth \ fitted    1    2    3
              1    8    2    2                          1   11    1    0
              2    6    4    0                          2    6    4    0
              3    1    1    0                          3    1    1    0
        before: 50.0%                              after: 62.5%
```

**Cents are relative and harmonic slots are not.** A partial was claimed for
harmonic *k* if it sat within 50 cents of *k·f₀*. But the gap between harmonic
*k* and *k+1* is about 1731/*k* cents, so that window is a fifth of the gap at
harmonic 2 and *wider than the whole gap* by harmonic 35. From roughly the
seventeenth harmonic upward the windows of adjacent slots overlap and every
partial in that region matches some harmonic of whatever fundamental is being
tested. The first group therefore claimed the entire top of the spectrum
regardless of which source the partials came from, the second source was
stripped of its upper partials, dropped under the energy floor, and no second
group formed. Capping the window at a fraction of the slot spacing took the
under-counts from eight to five and, independently, improved the audio
distances.

**A source has a bottom.** The tighter window leaves ambiguous high partials in
the pool, and on a heavily vibrato'd note there are many of them — its upper
partials wander further than a harmonic slot is wide. Left alone they assemble
into a group with an invented fundamental and no first or second harmonic under
it. Requiring every group after the first to carry at least 5% of its energy in
harmonics 1 and 2 took the over-counts from four to one.

The two are independent, and both are worth having:

| | exact | spectral | centroid (oct) |
|---|---|---|---|
| before | 50.0% | 0.623 | 0.285 |
| slot-spacing window only | 54.2% | 0.581 | 0.267 |
| low-harmonic rule only | 58.3% | 0.624 | 0.314 |
| both | 62.5% | 0.598 | 0.282 |

Nine golden analysis fixtures moved and all nine moved the same way: a
*single-source* signal that had been grouped as two or three sources, and in
five cases fitted as two oscillators, is now grouped and fitted as one. The
fixtures had been pinning the reference's over-counting. `two_sources_fifth` and
`two_sources_octave` did **not** move — the change suppresses invented sources
without losing real ones, which is the whole claim, and those two fixtures are
what makes it checkable.

What is left is under-counting, six trials of it, and the intervals say why:
`7 and 19` is an octave apart, `-8 and 17` is two octaves and a tone. Sources an
octave apart sharing an envelope are not merely hard to separate, they are
*mathematically identical* to one oscillator with a different waveform. That one
needs the rank-within-a-group work, not another tolerance.

### Roles before counting, and the two thirds of it that did not pay

The plan was to classify partials by role — harmonic body, noise, transient —
and group only the body. Two thirds of that turned out to be wrong, and the
measurements are worth keeping because it is an obvious idea to have again.

**Filtering the grouping pool bought nothing.** Splitting the tracked partials
into a steady body and unstable fragments and handing grouping only the body
scored 62.5% either way at the loosest threshold that kept counting intact, and
*hurt* once a duration test was added — 54.2%, because duration correlates with
loudness and the partials it discarded belonged to the quiet second source the
count was already missing. Grouping's own guards were already doing this job:
salience charges for predicted harmonics that are absent, and a later group has
to show its own bottom.

**And the split is not a measurement.** At a 2048-point window a violin read 72%
of its tracked energy in unstable partials against a clarinet's 0.9%, which
looks like exactly the discrimination wanted. But the fitter picks its window
from the fundamental, and at the 1024 points it actually uses, both samples read
0.03%. A number that moves three orders of magnitude with an analysis window is
measuring the window.

**What did pay is measuring the noise in the spectrum.** `Roles::noiseShare`
takes the energy sitting *between* the harmonics of the fitted fundamental. That
is the distinction spectral flatness cannot make: broadband noise lifts the
floor between partials, while a partial wandering with vibrato stays near its
own harmonic. It reads 0.29 on the violin and 0.06 on the clarinet, at any
window, and `autosynth_diff` reports the same quantity through the same function
so the diagnostic and the fitter cannot drift apart.

That number now sets the noise ceiling, which used to be a constant. A fixed
ceiling says "every pitched sound may hiss this much", and the level solve takes
whatever it is offered, because broadband energy lowers a log-spectral error
wherever the harmonic fit is imperfect. Against 24 noise-free harness targets:

| | invented noise | audible in | count exact | spectral |
|---|---|---|---|---|
| fixed ceiling | 0.064 | 16 of 24 | 62.5% | 0.571 |
| measured | 0.038 | 14 of 24 | 62.5% | 0.579 |

Every one of those is an error — the harness generates its targets noise-free —
and it is an error no distance metric objects to, which is why the harness now
reports it on its own line rather than letting it hide inside the averages.

The threshold was chosen against both, and the disagreement is worth recording.
At 0.25 the harness count fell to 58.3% on a single knife-edge trial while the
violin came out `ok` on all nine diagnostics; at 0.50 the count held and the
invented noise halved again, but the violin lost nearly all its noise (0.077
measured against a target of 0.287) and no longer sounded like a bowed string.
0.35 keeps the count, halves the invented noise, and leaves the violin 0.06
clean.

### The transient role, and why the noise does not get its own envelope

This one was built, measured and taken out again, and the measurements are the
reason it is written down rather than quietly deleted.

The premise was sound. Measured frame by frame, the energy between the
harmonics does not follow the energy on them: on both real samples the noise
leads the note in by 6 to 8 dB and outlives it by 14 to 24. An amplitude
envelope fitted to the *total* loudness is therefore fitted to a mixture of two
shapes that disagree, and lands between them — which looked like a good
explanation for both fits still reaching full level about 0.1 s early.

So the noise got its own ADSR, fitted to the inter-harmonic contour up to
note-off, and its own reverb send; the amplitude envelope was refitted to the
harmonic contour alone. Every part of that is defensible on paper. All of it
made things worse:

| | spectral | centroid (oct) | count | invented noise |
|---|---|---|---|---|
| before | 0.579 | 0.283 | 62.5% | 0.038 |
| with the split | 0.595 | 0.320 | 62.5% | 0.050 |

and on the real samples, which is where it was supposed to help, the violin's
attack went from 0.10 s short of its target to 0.29 s short and its noise
collapsed from 0.224 to 0.081 against a target of 0.287.

Three things went wrong, and two of them are worth remembering:

- **The split is defined against one fundamental.** With two sources present the
  second one's partials sit away from the first one's harmonics, so they land on
  the noise side of the line and both contours become fiction. Guarding the
  whole thing to single-source fits changed nothing measurable, which says the
  damage was elsewhere too.
- **An ADSR cannot express "do nothing".** A release of zero still cuts the
  tail, so simply adding a second envelope to the signal path changed the sound
  of every patch that had never fitted one. That needed an explicit
  `noiseEnvEnabled` gate, in the way the per-oscillator envelopes already have
  one — which is the general lesson: a new modulator in the engine is a change
  to existing patches unless it is switched off by name.
- **The two envelopes then competed.** With the noise on its own release and its
  own send, the level solve, the noise ceiling and the reverb were all fitting
  the same tail, and the result moved further from the target in every direction
  at once.

What survives is the measurement itself. A transient role is still the most
likely explanation for the attack error, but it needs somewhere to *go* — the
IR has no transient generator, and bolting the job onto the noise source, which
is flat and static, only spreads the error around. The next attempt should add
the generator first and fit it second.

### Three things a listener heard that no metric did

A musician listening to the fits reported two faults in one sentence: the
clarinet had lost its vibrato, and the violin "vacuums back, as if you rewind an
old tape" in the first half second. Both were real, both were traced, and none
of the three underlying bugs would have been found by the recovery harness,
because all three depend on a *played* note rather than a synthesised one.

**The tremolo was fitted with a 1.78 second delay.** `fadeIn` measured how long
the modulation took to reach 70% of its **peak** excursion. A real tremolo is
not uniform, so some cycle is always the widest, and on this clarinet the widest
one happened to land 1.78 s into a three-second note. Two thirds of the note
therefore rendered perfectly steady. Referenced to the **median** excursion
instead, the same signal reports 0.07 s. A synthesised LFO has no widest cycle,
which is exactly why the harness never saw it.

**The decay was measured on a wobbling curve.** The fall from the attack peak
to the sustain was found by walking the *raw* amplitude contour until it dropped
below target — and on a note with vibrato, the first trough after the peak
crosses it immediately. A violin came back with a 5 ms fall from full level to
46%: a 3.7 dB cliff, which no bow makes. Judged on the same smoothed contour
that `fullLevel` already uses, it becomes a real measurement, and the harness
agrees: mean `amp_env.decay` error 0.442 to 0.431.

Two nearby ideas were tried and rejected by ground truth, which is worth
recording because both sound right: starting the decay search from the
*contour's* peak rather than the raw one (decay error 0.431 to 0.444, counting
66.7% to 62.5%), and flooring the decay at what the smoothing can resolve
(0.431 to 0.445). Both would have removed the violin's cliff. Neither is a
better measurement.

**And the attack is a straight line, which is a real gap — but curving it is
not the fix.** The `curve` parameter shapes the decay and the release; the
attack segment is `t / attack`, a linear ramp in *amplitude*. Linear amplitude
is a very quiet first third: with the envelope reaching nine tenths at exactly
the right moment, the fit was still 16 dB below the target at note-on and 5 to 9
dB below it for the first quarter second, so the note emerged out of
near-silence instead of starting.

Applying the same curve to the attack halved that gap — the fit caught the
target by 0.19 s instead of 0.35 — and it was reverted anyway. Three reasons,
in the order they became clear. The harness disagreed in part: loudness envelope
improved from 4.30 to 3.90 dB but spectral distance went from 0.566 to 0.614. It
was the only change in the project to make the engine diverge from the frozen
reference render, which had to be re-blessed. And a listener judged both samples
worse, which settled it.

The diagnosis of *why* is the part worth keeping. One curve cannot describe an
attack, a decay and a release at once. Asked to, the fit lands on a compromise
that suits none of them: the clarinet's came back at 5.5 where it had been 2.5,
and everything downstream of the envelope shifted with it. A curved attack needs
its own parameter, or it needs to stay out.

**Two further hypotheses about "it sounds noisy", both wrong.** Worth recording
because each was plausible and each took a measurement to kill. First, that the
tremolo rate was an octave out — `dominantRate` takes the tallest spectral
peak, and an asymmetric oscillation puts energy on its own second harmonic, the
same trap already documented in the unison estimator. It is not happening here:
the detector reports an oscillation ratio of 1.21, which is what a correctly
identified rate looks like. Second, that the 0.30 s detrend was high-passing a
slow tremolo away before the rate search — it does sit awkwardly against a
`kMinRateHz` of 0.3, but lengthening it to 0.6 s or 1.0 s does not reveal a
slower rate and does destroy detection elsewhere (spectral 0.566 to 0.585 and
0.612; the violin loses its vibrato entirely).

**A fourth — and this one is a modelling gap, not a bug.** A listener still
reported the clarinet as having "more vibrato than the original" while every
metric said the opposite: our wobble measured *shallower* than the target's, 2.6
dB against 3.3. Both were right. Taking the spectrum of the loudness envelope
directly, outside our own code:

| | spectrum of the amplitude envelope, 0.4 to 2.9 s |
|---|---|
| clarinet target | 2.00 Hz 1.00, 0.80 Hz 0.82, 1.60 Hz 0.82, 3.61 Hz 0.77, 0.40 Hz 0.70 |
| clarinet fit | 3.21 Hz 1.00, 3.61 Hz 0.72, 2.80 Hz 0.32, then nothing |

The target's amplitude does not oscillate. It *wanders*, with energy spread from
0.4 to 3.6 Hz and no peak worth the name — breath and room, not tremolo. The
fit replaces that with one clean sinusoid, and a clean sinusoid is audibly
vibrato however shallow it is. Depth was never the problem; regularity was.

Nothing available separates the two cases. `concentration` is the obvious
candidate and it points the wrong way: the clarinet's diffuse wander scores 0.23
and the violin's genuine pitch vibrato 0.08, so any threshold that rejected the
clarinet would take the violin's real vibrato with it. The rate is not wrong
either, by our own detector's oscillation ratio, and three separate attempts to
find it wrong — an octave error, a too-short detrend, a double high-pass —
were each killed by measurement. This wants either an LFO whose rate and depth
drift, or a periodicity statistic better than the concentration of a
periodogram. It is on the roadmap as its own item.

What *was* wrong was the diagnostic. `autosynth_diff` had its own wobble-rate
estimator — smooth, then count sign changes — and the smoothing that makes
crossing-counting usable is also a low-pass, so it read a pure 3.31 Hz sine as
2.8 Hz. It now calls `Modulation::dominantRateHz`, the same estimator the fitter
acts on. The rule this is the third instance of: a diagnostic that measures a
quantity differently from the code it is judging will eventually accuse it of a
bug it does not have.

### Attack: a parameter is not a measurement

Both real samples were fitted with visibly wrong attacks, and there were two
different bugs stacked on top of each other.

**`fitAdsr` returned a measurement in a parameter's field.** Attack was measured
as the time the contour took to reach nine tenths of the level it holds, which
is the right measurement — but it was then written straight into the ADSR's
`attack`, *and* the fit went on to choose a curve. With a curve of 3 the
envelope is already at nine tenths a third of the way through its attack
segment, so a violin measured at 0.401 s rendered as 0.277. The analysis had not
made an error; the write-out had. `EnvelopeFit::attackSeconds` is now the
measurement, `fitAdsr` returns the parameter that *renders as* that measurement,
and `test_fit.cpp` renders the fitted envelope and measures it back to keep the
two in step.

**Refinement was re-deciding it.** Given the full range, CMA-ES moved the
violin's attack to 0.123 s and the clarinet's to 0.501 — opposite directions,
both away from the truth, because a spectral distance is nearly blind to the
first tenth of a second and will spend it buying accuracy elsewhere. That is
precision overruling structure, which is the split this project keeps.

Bounding the *parameter* to half-to-double the analysed value was the first
attempt, and it is not enough, because the rendered attack depends on the attack
and the curve **together** and the curve is legitimately searchable. Held inside
those bounds a violin whose envelope should have measured 0.40 s still rendered
at 0.29, because refinement had flattened the curve underneath it.

So the attack is now *derived rather than searched*: what refinement holds fixed
is the crossing time, and the attack parameter is re-solved against whatever
curve each candidate chose. CMA-ES keeps the curve; the envelope keeps its shape
in time.

**And the solve itself had the same bug in miniature.** The first version
assumed the crossing was a fixed fraction of the attack segment, which is only
true when the decay does not pull the level down underneath it —
`attackSeconds` measures nine tenths of the level the note *holds*, not of the
attack's own peak, and with a sustain of 0.45 those are far apart. Bisecting
against `Envelope::evaluate` instead is exact and costs nothing offline.
`test_fit.cpp` pins it across three curves and two sustains.

| | session start | parameter bounds | derived crossing |
|---|---|---|---|
| clarinet (target 0.337 s) | 0.501 | 0.256 | **0.331** |
| violin (target 0.401 s) | 0.123 | 0.277 | 0.299 |

The harness improved with it, which is the part worth noticing: oscillator
counting went from 62.5% to **66.7%** and spectral distance from 0.579 to 0.566.
An envelope that keeps its measured shape is not just more faithful, it leaves
the optimiser with fewer ways to spend its budget on nothing.

### What two real samples exposed

A solo violin and a solo clarinet — four seconds each, sustained, with vibrato
and pre-engineered reverb — found five analysis bugs in an afternoon. None of
them could have been found by the recovery harness, because every one depends
on something synthetic targets do not have: room tone, bowing, or a sustain
that is not perfectly flat.

**The subharmonic guard was defeated by density.** Both samples had their
fundamental identified an octave low — the violin at 439 Hz where YIN said 877
with a rock-steady track, the clarinet at 226 against 441. The guard tested
whether the predicted harmonics were *present*, and with a few hundred tracked
partials there is always something within 50 cents of any predicted frequency,
so nothing ever looked missing. What separates a real fundamental from a
subharmonic is where the energy *is*: measured on the violin, the 439 Hz
candidate carried 0.5% of its energy on odd harmonics, and the clarinet's 226 Hz
candidate 0.1%, against 58.6% and 71.4% for the true fundamentals. The guard now
takes the greatest common divisor of the harmonics that carry real weight, which
generalises past octaves and is safe for a clarinet, whose odd-harmonic
dominance gives a divisor of 1.

**Track fragments were being counted as unison voices.** Both samples came back
as three oscillators fifty cents apart. Harmonic 1 of the violin had 189
partials assigned to it, of which exactly one lasted longer than a tenth of the
note — the rest were room noise two octaves away, plus fragments left behind
when vibrato walked a partial until the tracker gave up. Unison voices are
*simultaneous*; fragments are *sequential*, so requiring real duration
separates them and costs nothing on synthesised unison.

**The fundamental was never refined after selection.** Candidates come from a
coarse grid and matching tolerates 50 cents, so a candidate tens of cents off
claims exactly the right partials and wins. The clarinet was described as 430.9
Hz — flat by 42 cents, audibly out of tune. Fitting the fundamental to the
partials it claimed fixed it, and as a side effect every synthetic fixture
snapped to its exact generated frequency: 219.8 → 220.0, 439.7 → 440.0,
109.6 → 110.0.

**A played note is not a flat note.** Gate detection looked for a sustain
holding within 2.5 dB. The violin's sustain swings about 14 dB at vibrato rate,
so no plateau was ever found, a four-second bowed note was classified one-shot,
and it fitted a 1.2 second attack with a tail forty times too loud. Flatness is
now judged on a copy smoothed over roughly one vibrato period — which does not
weaken the pluck test, because smoothing flattens an oscillation but leaves a
monotonic decay's slope intact.

**Note-off was taken from the wrong place.** The gate was the end of the longest
flat run, which on a wobbling sustain finishes wherever the wobble happened to
be briefly calm: the clarinet, whose note plainly stops at 3.0 s, was gated at
1.38 s. Walking back from the end for the last frame within 6 dB of the held
level puts both samples within 20 ms of the truth, and improved the synthetic
`sustained` fixture from a 110 ms error to 33 ms.

### Fitting the room

With the five analysis bugs fixed, the error split cleanly in two: a note body
around 4 dB and a tail around 27 dB. Everything left was the room, so the room
became worth fitting.

`EffectsFit::detectReverb` answers the narrow question that becomes tractable
once note-off is known accurately — after the note stops, is there a diffuse
tail, how long does it take to die, and how loud is it? The decay is treated as
**two segments**: the direct sound's release owns the fast knee at the start,
and the reverb owns the slow exponential that follows. Each is measured on its
own segment, so neither can absorb the other. That is the degeneracy settled by
structure rather than left to the optimiser.

Three things had to be right for it to work, and each was wrong first:

- **The noise floor has to be measured, not assumed.** A fixed −60 dB threshold
  read a quiet room's hiss as part of the tail, which on the violin put two
  thirds of the fitted region inside the noise floor and returned a two-second
  RT60 for a room that had none.
- **`level` is not the wet/dry ratio.** A comb with feedback *f* settles at a
  gain of 1/(1−*f*), which at realistic room sizes is a factor of eight or nine.
  Setting the level to the measured tail-to-note ratio made it about 28 dB too
  loud.
- **Shortening the release and *enabling* the reverb are different
  conditions.** `detectReverb` can find a decay that `fitReverb` then declines
  to model because the return is too quiet to be worth switching on. Cutting the
  release on the first condition threw the tail away with nothing to replace it.

| | note body | tail | whole |
|---|---|---|---|
| violin | 3.51 dB | **2.92 dB** | **3.37 dB** (was 10.64) |
| clarinet | 3.45 dB | 9.69 dB | **4.98 dB** |

### Vibrato was being fitted as a delay

The violin's tail turned out not to be a reverb problem at all. It had a delay
fitted at 0.22 s with 0.84 feedback — ringing at under 7 dB per second, which
left the patch droning through the whole tail. 0.22 s is the period of 4.5 Hz.
It was the vibrato.

Delay detection correlates the loudness envelope, and vibrato makes that
envelope every bit as periodic as an echo does. The discriminator is structural:
**a delay repeats the signal, so its echoes outlive the note; modulation stops
when the note stops.** Looking for the same period *after* note-off separates
them, because a played note's decay is smooth and has no interior peak there.

This was the single largest error in the violin fit, and the reverb work had
been masking it.

### A note on the release/reverb degeneracy, caught in the act

Before the note-off fix, the clarinet scored *better* overall — 5.86 dB against
9.34 — with a gate 1.6 seconds early and a long release. That release was
imitating the reverb tail. A wrong analysis outscored a right one because the
patch had no other way to account for a room, which is exactly why the two must
be separated structurally and never left to a distance metric to arbitrate.

### Wavetables, and the ladder that keeps them honest

Five fixed shapes and a blend between two of them is a coarse net. Measured
against the clarinet, the best classic fit was still 6 dB out on the second
harmonic and 15 dB out on the sixth, because no combination of sine, triangle,
saw, square and pulse puts a peak on one harmonic and a cliff after it. Sixteen
harmonic amplitudes can.

**Every oscillator is a wavetable, and there is no switch.** A saw *is* a
one-frame table whose frame nobody has drawn on — the same model Vital uses,
where picking an analog shape gives you a wave source rather than a special
case. The oscillator's `waveform`, `waveform_b`, `wave_morph` and `pulse_width`
are that frame's *generator*.

A frame stays generated until it is edited, and that is not a detail. A
generated frame is built from its waveform's full Fourier series and
band-limited per octave, so a saw at 220 Hz keeps all seventy-five harmonics
that fit under Nyquist. Storing every shape as sixteen numbers instead — the
obvious way to make "always a wavetable" true — cuts it at the sixteenth, which
measures 1.7 octaves of brightness lost and would make every classic preset
sound muffled for nothing. Drawn frames get private band-limited mipmaps;
generated ones get none, because they already have one, so the memory and the
FFTs scale with how much of the table has actually been drawn on rather than
with `kMaxFrames`.

Those FFTs are also why editing is not simply "mutate the patch and hand it
over". A rebuild with three drawn frames measured 5 ms, and the audio callback
try-locks the patch and outputs silence when it loses — so dragging a bar
dropped several blocks a second. Two things fixed it. Every octave low enough to
fit all sixteen harmonics produces the *same* table, so eleven builds became
five and the cost fell to 1.5 ms. And `Engine::buildFrameTables` now runs
*outside* the lock, with `adoptFrameTables` handing the result over as a vector
move, so what the audio thread is kept out of is a pointer swap.

The remaining risk is that a drawn table is the least legible thing a patch can
contain. "Saw, morphed 40% toward a narrow pulse" is a sentence; forty-eight
numbers are a spectrum dump with a play button, which is the thing this project
exists not to produce. So `WavetableFit` is a ladder of three models, each rung
taken only if it beats the one below by a margin:

| rung | parameters | what it fixes |
|---|---|---|
| the generated frame, a blend of two shapes | 2 | nothing new — the existing behaviour |
| one drawn sixteen-harmonic frame | 16 | the *shape* |
| three drawn frames and a swept position | 48 + envelope | the shape *moving* |

The format carries sixteen frames of sixteen harmonics, matching what Vital
gives you to edit. The *fitter* uses at most three (`WavetableFit::kFittedFrames`):
a played note's timbre trajectory is smooth, and past a start, a middle and an
end the extra frames fit the analysis noise that the smoothing below exists to
remove. The other thirteen are for a person, not for the optimiser.

Measured on the two real samples, all three scored the same way:

| | blend | one table | three frames | measured drift |
|---|---|---|---|---|
| clarinet | 0.134 | 0.025 | 0.019 | 1.8 dB |
| violin | 0.116 | 0.017 | 0.013 | 1.4 dB |

Both take a table; only the clarinet takes three frames.

**The scoring has to happen on the slow profile.** The first version compared
each model against every analysis frame's raw harmonic profile, and the two
rivals came out within a percent of each other however far the tone actually
travelled. The reason is in the data: frame to frame the measured balance jumps
by 10 dB and more, because vibrato slides every partial across the analysis
bins. That is modulation, the LFOs already carry it, and it is common to both
models — left in, it drowns the thing being measured. Smoothing over a sixth of
a second either side separated them cleanly: 0.025 against 0.019 rather than
0.0684 against 0.0627.

**The third rung needs an absolute floor, and it is stated in decibels.** It is
the same number `autosynth_diff` prints as timbre drift, computed the same way,
because a diagnostic that says a fit is too static and a fitter that decides
whether to fix it must be measuring the same quantity. The threshold sits at
1.5 dB on evidence rather than taste: forced on regardless, the clarinet's
rendered drift went from 2.9 dB short of its target to 1.0 dB over it and its
harmonic profile improved slightly, while the violin landed in the same place
either way and would have paid thirty-two numbers for nothing.

**A ratio alone is not a parsimony rule.** The same lesson the waveform blend
had already learned, re-learned here at some cost: a clean saw scores about a
thousandth on this error, and a table beat that too, by fitting the third
decimal place of a profile that was already right. Every golden tone came back
as a wavetable — and worse, several changed *oscillator count*, because a
differently-rendered oscillator gets a different gain out of the level solve. A
model has to be bad enough to be worth fixing before a better one is considered.

**Only a source loud enough to believe gets one.** A quiet layer's harmonic
profile is the least reliable measurement in the analysis: it is read on top of
a louder sound, and partial tracking hands a shared partial to whichever source
is stronger. Below a quarter of the loudest source's energy, an oscillator keeps
its classic waveform.

**The frames are not searched, and they are not automatable.** Refinement never
sees them, for the same reason it never sees the LFO rate: they are a
measurement, and re-deciding them with a distance metric is the structure /
precision split going the wrong way. They are also absent from the parameter
layout — sixteen amplitudes times three frames are not knobs, and exposing them
would let a host automate an FFT onto the audio thread. `FrameTables::matches`
is what makes that safe in the other direction: `setPatch` can be called from a
parameter callback, so the rebuild is skipped unless the frame data genuinely
differs, which happens only on patch load and analysis.

**Both engines have to agree on what a frame is.** Frames are built in sine
phase, like every fixed shape except the pulse, so a crossfade of two tables is
exactly a crossfade of their harmonic amplitudes — the fitter's model of the
oscillator is the oscillator, not an approximation of it. There is one
definition of what "saw" means as sixteen numbers —
`WaveTables::blendedHarmonics` — and the oscillator, the fitter and the editor
all call it. `test_wavetable.cpp` pins both ends of that: where Nyquist rather
than the frame length limits them, a drawn frame and the fixed shape are the
same signal, and where it does not, an undrawn saw is more than twice as bright
as the same saw written out as sixteen harmonics.

**Editing.** The wavetable row of each oscillator strip draws the selected
frame's harmonics as sixteen bars and lets you drag them; the other active
frames stay visible as faint ticks, because the point of a multi-frame table is
the movement between frames and editing one blind to the others hides exactly
that. The first drag on a generated frame seeds it from the shape it was
generated from, so you move one bar rather than replacing the sound with a
single sine. Which frame is being edited is *not* a parameter: it is a view,
like solo and mute, and a saved patch must not remember which frame someone
happened to have open.

**One scale factor for the whole set.** Each table is *not* normalised on its
own. Peak-normalising per frame would flatten the differences between frames,
and it would do it by crest factor rather than by loudness — a frame with more
harmonics has a taller peak at the same energy, so it would come out quieter and
moving the position would sound like a volume change instead of a timbre change.

### Traps found along the way

- **Absolute cutoff is not identifiable.** Source spectrum times filter
  response is a blind deconvolution — "bright source, closed filter" and "dull
  source, open filter" are the same signal. Alternating estimation recovers how
  the cutoff *moves* but not where it *sits*. Anchoring to a joint
  waveform/cutoff search breaks the tie.
- **Broadband noise can absorb an entire fit.** In a magnitude-domain least
  squares the noise column has energy in every bin. Scaling levels by the max
  gain *including* noise drove every oscillator to zero and returned patches
  with no oscillators at all for a quarter of inputs. Scale by the loudest
  *oscillator*.
- **Shorter analysis windows are not free.** Adapting the window to f0 at three
  bins per harmonic measurably degraded waveform recovery: a Hann main lobe is
  ~4 bins wide, so peak-picking started catching the neighbour's leakage. Six
  bins is the working figure.
- **Pruning weak partials made grouping worse** (53% → 11%). Weak partials are
  the high harmonics that pin down a fundamental.
- **A purely spectral CMA-ES objective trades away the envelope.** It closed a
  filter from 3177 Hz to 639 Hz and took loudness error from 2.76 dB to 7.6 dB.
  Fixed with adaptive weights, each term normalised by its own starting value.
- **False-positive LFOs are audible.** A pitch track that steps as one source
  fades correlates with a square wave, and a permissive threshold turned that
  into a phantom vibrato in the plugin. Real vibrato measured 0.83
  concentration, the artefact 0.23; the threshold is now 0.45.
- **A port is not the place to improve the model.** The C++ port originally
  spread unison starting phases, which sounds better in isolation. The
  reference started them coherent and the difference was several dB. Reverted.

---

## 5. Roadmap

In priority order, which is the project owner's call and not a technical
ranking: rank-within-a-group, then LFO wander, then the attack curve, then
formants, then a Vital exporter. Linux and macOS are explicitly *not* wanted for
now — Windows is enough — and stereo comes after the exporter.

### Restore what the Python removal cost

- ~~Port the ground-truth recovery harness.~~ **Done** — `autosynth_eval`, see
  [Measuring the fitter](#measuring-the-fitter).
- ~~Port NMF rank selection.~~ **Done** — `src/fit/Nmf.cpp`, and unlike the
  Python original it *is* on the shipped path: run per harmonic group rather
  than globally, which is the placement that makes it answerable.

### Oscillator counting — the real weakness

At ~33% exact, this is the ceiling on everything else.

- ~~Split the accuracy figure by direction.~~ **Done** — `autosynth_eval`
  prints a truth-against-fitted confusion matrix and lists every miscount with
  its intervals and levels. See
  [Counting: two bugs the confusion matrix found](#counting-two-bugs-the-confusion-matrix-found).
- ~~Unison by beating.~~ **Done** — `Grouping::detectUnisonBeating`. See
  [Unison by beating](#unison-by-beating) below. What remains is recovering the
  voice *count*: the estimator reports two, which is the minimum that explains a
  beat, and the detune it recovers is the gap between *adjacent* voices rather
  than the full spread. For three voices spread over 20 cents it returns two
  voices 10 cents apart — a faithful description of the beating, and not the
  patch that produced it.
- ~~Roles before counting.~~ **Partly done, and partly disproved** — see
  [Roles before counting](#roles-before-counting-and-the-two-thirds-of-it-that-did-not-pay).
  The noise floor is now measured and bounds the noise level. Grouping the
  harmonic body alone was measured and does not help. Separating the transient
  onto its own envelope was built and measured and made things worse in every
  direction — see
  [The transient role](#the-transient-role-and-why-the-noise-does-not-get-its-own-envelope).
  It is still the most likely explanation for the attack error, but it needs a
  generator of its own in the IR before it can be fitted to anything.
- ~~Rank within a group.~~ **Done** — `nmf::selectRank` and
  `splitByRank`, see
  [Rank within a group](#rank-within-a-group-and-the-four-guards-it-needed).
  Counting 66.7% to 70.8%, and three-source targets recovered for the first
  time. The remaining misses are still octave-related, so the machinery is in
  place but that instance is not solved.

### Modulation and effects

- ~~Measure how mechanical an LFO is.~~ **Done** — `Modulation::detectWander`,
  reported by `autosynth_diff`, see
  [Wander](#wander-measuring-how-mechanical-an-lfo-is). The engine can chain
  LFOs; what the fitter lacks is a slot to spend on one. **Next:** either a
  third slot with the harness re-baselined deliberately, or a rule for when a
  modulator is worth more than the modulation it displaces.
- **An LFO is the wrong model for human wander, and it is audible.** Measured on
  a clarinet, the amplitude envelope's spectrum is spread from 0.4 to 3.6 Hz
  with no dominant peak; the fit gives it a single 3.2 Hz sinusoid, and a
  listener hears mechanical vibrato where the recording has breath. See
  [Three things a listener heard](#three-things-a-listener-heard-that-no-metric-did).
  Three possible directions. **An LFO modulating another LFO**, which is how
  Vital's mod matrix does it and which builds wander out of periodic parts
  rather than out of noise {d} a slow LFO on a fast one's rate produces exactly
  the smeared spectrum measured above, and it stays a *readable* patch, which a
  random walk does not. Or give the LFO an internal drift. Or find a periodicity
  statistic that separates "oscillates" from "wanders", which `concentration`
  demonstrably does not (0.23 for diffuse clarinet wander against 0.08 for real
  violin vibrato).

- **Mod matrix, and MSEG escalation.** Two LFO slots removed the worst
  limitation (vibrato and tremolo no longer compete), but routing is still
  one destination per slot rather than a matrix. Separately, the slow half of
  the envelope should escalate ADSR → multi-segment only when the residual
  justifies the extra parameters.
- ~~Reverb detection.~~ **Done** — `EffectsFit::detectReverb`, see
  [Fitting the room](#fitting-the-room). The original entry is kept below
  because the trap it describes is still the reason the code is shaped as it is.
  Measured on two real samples,
  the un-fitted reverb tail is 27 dB of error against a 4 dB note body: it is
  the entire remaining gap, and any library sample will have it. Fitting one is
  blind dereverberation and realistically gets a heuristic — detect a long
  diffuse tail, shorten the release, push it to a send. The trap is documented
  above and has now been observed: a reverb tail and a long release are
  near-degenerate, and a wrong gate with a long release scored *better* than a
  correct one because the release was imitating the reverb.

### Platform and reach

Deferred by decision, not by difficulty: Windows is enough for now, and stereo
waits until after the exporter.

- **Linux support.** Nothing here is deliberately Windows-only: JUCE is
  cross-platform, the DSP is standard C++17, and CMake already selects the
  right per-user VST3 folder for Linux and macOS. What is missing is a tested
  build — a `bootstrap.sh` alongside the PowerShell script, a CI job, and
  confirmation that the golden fixtures compare identically under GCC and Clang.
  That last point is the only real unknown: the bounds in
  `test_golden_engine.cpp` were measured with MSVC, and floating-point
  differences across compilers may need the tolerances revisited on evidence
  rather than by loosening them.
- **macOS support.** Same shape as Linux, plus AU packaging and code signing.
- ~~Exporters.~~ **Working, and checked against Vital's own declarations** — `VitalExport`, plus
  `autosynth_probe --vital out.vital` and an *Export Vital* button in the
  editor. See [Exporting to Vital](#exporting-to-vital). It needs one load in
  real Vital to confirm the numeric skews; the structure and the wavetable are
  covered by tests here.
- **Stereo.** Rendering is easy. *Fitting* is a separate project: the analysis
  chain is mono by construction, and the IR has no pan or unison stereo spread,
  so there is nothing to widen yet. Treat it as its own phase.

### Timbre and tone

- ~~Wavetable oscillators.~~ **Done** — `WavetableFit`, see
  [Wavetables, and the ladder that keeps them honest](#wavetables-and-the-ladder-that-keeps-them-honest).
- ~~Attack.~~ **Mostly done** — see
  [Attack: a parameter is not a measurement](#attack-a-parameter-is-not-a-measurement).
  The clarinet now lands on its target; the violin is still 0.10 s short and the
  cause is *not* the envelope, which measures back correctly on its own. Ruled
  out by measurement: the amp LFO, the filter envelope and the reverb each move
  it by less than 10 ms.
- ~~Give the attack its own curve.~~ **Done** — see
  [The attack gets its own curve](#the-attack-gets-its-own-curve). The violin's
  onset gap at 0.09 s fell from 9.3 dB to 6.3 and it now converges by 0.11 s
  instead of 0.35.
- ~~Formants.~~ **Subsumed, not built.** The idea was a sine oscillator tuned to
  a harmonic multiple — three parameters for a resonance, against a whole
  table. It was written down before the wavetable existed, and a sixteen-harmonic
  frame already puts a peak wherever it likes. Checked before dropping it:
  every profile error on both real samples falls in harmonics 1 to 8, inside the
  frame's reach, so a formant oscillator would be adding parameters that
  duplicate ones already present. The clarinet's remaining 6 dB at harmonics 5
  to 7 is a filter-against-table interaction, not a missing resonance — its
  frames *are* drawn, and they were fitted to the deconvolved profile while the
  error is measured after the filter.
- ~~Editing a fitted table.~~ **Done** — sixteen draggable harmonic bars per
  frame. Still missing: a way to seed a frame from a shape other than the
  oscillator's own, an undo, and a Vital wavetable exporter to take the frames
  somewhere else.

### Timbre and tone, continued

- ~~Noise as an oscillator, not a global level.~~ **Done** — `Waveform::noise`,
  see [Noise as a waveform](#noise-as-a-waveform). Unpitched material now gets a
  fitted envelope and filter instead of a flat bed. **What is left:** a *pitched*
  sound whose noise is shaped — a bowed string, a breathy flute — could carry
  a noise oscillator alongside its harmonic one, which would give bow noise its
  own attack and its own brightness. That costs an oscillator slot and has to
  beat the global bed on merit, so it needs the same kind of parsimony rule as
  the wavetable ladder.

- **The violin's reverb decay is fitted too long.** A tenth of a second after
  release the recording's tail is at 0.211 of its sustain and three tenths later
  it is at 0.011 {d} a drop of 26 dB in 0.2 s, an RT60 near half a second. The
  fit chose 1.45 s, and the engine duly renders 0.133 falling only to 0.037. The
  clarinet is fine on the same measurement (1.16 s fitted against 1.21 s
  measured), so this is not a broken relation, it is `detectReverb` reading a
  fast-decaying tail as a slow one on material that is itself noisy. Noticed
  while calibrating the Vital exporter against the recordings.

### Known limitations, all currently unhandled

Polyphonic input, FM and inharmonic timbres, reverb tails baked into the
release envelope, and very short percussive samples where time-frequency
resolution fights back.
