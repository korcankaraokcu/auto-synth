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
  src/Parameters*  host-automatable parameters
  src/Plugin*      processor and editor
  tools/           render_main.cpp, probe_main.cpp
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

100 cases. Tags: `[ir]`, `[engine]`, `[analysis]`, `[fit]`, `[capabilities]`,
`[golden]`.

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

### Fitter comparison

From the ground-truth recovery harness (24 trials, seed 0), measured before the
Python reference was removed. **The harness has not yet been ported — see the
roadmap.** Control (an untouched default patch): spectral 4.271, loudness
12.326, centroid 2.756.

| | `baseline` | `baseline` +CMA | `partial` | `partial` +CMA |
|---|---|---|---|---|
| spectral | 1.018 | **0.561** | 1.118 | 0.600 |
| loudness_db | 3.379 | **2.106** | 5.201 | 2.886 |
| centroid_oct | 0.675 | 0.286 | 0.828 | **0.279** |
| oscillator count exact | 12% | 12% | **33%** | **33%** |

Read the columns together, because the conclusion only appears when you do.
Unrefined, the simplest fitter wins on audio while making worse structural
decisions. Refined, both roughly halve their audio distance and converge — at
which point the tiebreaker is structural, and 33% against 12% on oscillator
count is not a close call for a tool whose purpose is a *multi-oscillator*
patch.

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

- **Port the ground-truth recovery harness.** `eval/recovery.py` generated
  random patches, fitted them back and scored parameter recovery. It had no C++
  counterpart and went with the rest of Python. The golden fixtures cover
  *conformance* — that behaviour has not changed — but nothing currently
  measures *how good the fitter is*, so the table above cannot be reproduced or
  improved against. This is the highest-value gap.
- **Port NMF rank selection** (`fit/rank.py`). Not on the shipped path, but
  needed for "rank within a group" below.

### Oscillator counting — the real weakness

At ~33% exact, this is the ceiling on everything else.

- **Unison by beating.** Unison estimation under-counts because narrow detune
  does not resolve spectrally. Two voices *d* cents apart amplitude-modulate at
  their difference frequency, which is measurable long before they separate —
  and `Modulation` already has the periodicity machinery. Best
  value-per-effort item on this list.
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
- **Reverb detection.** The reverb exists and is editable; *fitting* one is
  blind dereverberation and realistically gets a heuristic — detect a long
  diffuse tail, shorten the release, push it to a send. The trap: a reverb tail
  and a long release are near-degenerate, so a fitter handed both without
  explicit scope will trade them against each other and get both wrong.

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
