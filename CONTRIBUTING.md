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
| `autosynth_eval` | Ground-truth recovery harness — how good is the fitter? |
| `install_plugin` | Copies the VST3 to the per-user plug-in folder |

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
  src/analysis/    Stft, Yin, Partials, Grouping
  src/fit/         WaveformFit, EnvelopeFit, FilterFit, Modulation, EffectsFit,
                   PartialFit, Nnls, CmaEs, Refine
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

Roughly in priority order.

### Restore what the Python removal cost

- ~~Port the ground-truth recovery harness.~~ **Done** — `autosynth_eval`, see
  [Measuring the fitter](#measuring-the-fitter).
- **Port NMF rank selection** (`fit/rank.py`). Not on the shipped path, but
  needed for "rank within a group" below.

### Oscillator counting — the real weakness

At ~33% exact, this is the ceiling on everything else.

- ~~Unison by beating.~~ **Done** — `Grouping::detectUnisonBeating`. See
  [Unison by beating](#unison-by-beating) below. What remains is recovering the
  voice *count*: the estimator reports two, which is the minimum that explains a
  beat, and the detune it recovers is the gap between *adjacent* voices rather
  than the full spread. For three voices spread over 20 cents it returns two
  voices 10 cents apart — a faithful description of the beating, and not the
  patch that produced it.
- **Roles before counting.** Detect sub-octave energy, noise floor and transient
  separately; group only the harmonic body. More stable, and the resulting patch
  is more legible.
- **Rank within a group.** `PartialFit` assigns one oscillator per source, but
  several oscillators can share a fundamental and differ only in waveform and
  envelope. Needs the NMF port above, run per group rather than globally.

### Modulation and effects

- **Mod matrix, and MSEG escalation.** Two LFO slots removed the worst
  limitation (vibrato and tremolo no longer compete), but routing is still
  one destination per slot rather than a matrix. Separately, the slow half of
  the envelope should escalate ADSR → multi-segment only when the residual
  justifies the extra parameters.
- **Reverb detection — now the top priority.** Measured on two real samples,
  the un-fitted reverb tail is 27 dB of error against a 4 dB note body: it is
  the entire remaining gap, and any library sample will have it. Fitting one is
  blind dereverberation and realistically gets a heuristic — detect a long
  diffuse tail, shorten the release, push it to a send. The trap is documented
  above and has now been observed: a reverb tail and a long release are
  near-degenerate, and a wrong gate with a long release scored *better* than a
  correct one because the release was imitating the reverb.

### Platform and reach

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
- **Exporters.** Vital's preset format is JSON and the natural first target.
  Validate by exporting, rendering in real Vital, and comparing against this
  engine's render.
- **Stereo.** Rendering is easy. *Fitting* is a separate project: the analysis
  chain is mono by construction, and the IR has no pan or unison stereo spread,
  so there is nothing to widen yet. Treat it as its own phase.

### Known limitations, all currently unhandled

Polyphonic input, FM and inharmonic timbres, reverb tails baked into the
release envelope, and very short percussive samples where time-frequency
resolution fights back.
