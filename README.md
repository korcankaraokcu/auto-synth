# auto-synth

Turn a sample into an **editable** multi-oscillator synth patch.

Not a wavetable dump and not a black box: the output is a small set of
oscillators, envelopes, filters and modulation that a human can open and
change. That constraint is the whole project. The objective is not "minimise
reconstruction error" — a sampler already scores zero on that — it is
**minimise error subject to a parameter budget**.

## What it does

Give it a `.wav` and it writes a `.vital` preset. The sample is analysed into
partials, split into sources, fitted to a patch, and the patch is polished by
CMA-ES against the recording. Every control in the result is a real Vital knob
that a person can open and change; the only thing that is not is the harmonic
content of a fitted wavetable, which is drawn rather than dialled.

The intermediate patch is the deliverable in all but name -- a small, readable
description that a second exporter could translate somewhere else. Vital is the
first target because its oscillators are wavetables and so are ours.

- **3 oscillators**, each with its own tuning, unison and envelope
- **Every oscillator is a wavetable**, up to 16 frames of 16 harmonics. A
  classic shape is a one-frame table nobody has drawn on yet, so it keeps its
  full bandwidth; frames get drawn only when no shape can describe the sound
- **2 LFO slots**, so vibrato and tremolo can coexist rather than compete
- Filter with envelope, delay, and a reverb
- **Export to Vital** — the fitted patch, wavetables and all, as a preset

## Status

Working end to end: a recording goes in, a Vital preset comes out, and every
step in between is measured *through Vital itself*. Analysis, fitting and CMA-ES
refinement all run natively -- there is no Python at runtime and none in this
repository.

This is a converter, not a synth. The plug-in, the editor and the internal
engine are all gone: Vital renders every refinement candidate, every calibration
and every test that needs sound. Keeping two engines in step was itself a source
of bugs -- three in one week where an error in one cancelled an error in the
other -- and pointing the test suite at Vital immediately found four export
defects that had been shipping, including a preset driven 12 dB into Vital's own
limiter.

How good is the result? `autosynth diff` reports eleven named axes, and on two
instrument recordings a fit lands four to eight of them inside tolerance
depending on the draw -- the search is stochastic, and the same recording fitted
from two seeds is not the same preset. Pitch, attack, note-off and level are
reliable. Tremolo depth and timbre movement are the weakest, because a single
LFO is the wrong model for the way a player's tone actually moves. Oscillator
counting, at about 70% exact, is the weakest structural step.

Those recordings are library material and are not in this repository, so the
measurements throughout [CONTRIBUTING.md](CONTRIBUTING.md) can be read but not
re-run against the same audio. Nothing in the build or the test suite depends on
them.

Not started: exporters to other synths, and stereo.

## Use

One file: `autosynth.exe` from the releases tab. Nothing else to install, and no
runtime to go with it.

```powershell
autosynth fit your-recording.wav
```

Any mono `.wav` of a single sustained note will do. That writes
`your-recording.vital` next to it; open it in Vital and start turning knobs.
`--preset`, `--patch` and `--render` name the preset, the intermediate patch and
an audio rendering of the fit if you want them somewhere else.

[Vital](https://vital.audio/) must be installed -- the free version is enough --
because it *is* the synth here: the fitter renders every candidate through it.
No Vital code is compiled or shipped in this repository; the tool hosts whatever
VST3 it finds in the platform's plug-in folders, so the preset is fitted against
the version it will be opened in.

The other verbs are there for looking at what happened. `autosynth` on its own
lists them:

| Verb | |
|---|---|
| `fit` | recording in, preset out. The one you want |
| `diff` | how far two recordings are apart, on eleven named axes |
| `render` | play a fitted patch through Vital and write the audio |
| `probe` | every analysis stage as JSON |
| `score` | what the fitting objective makes of a patch, term by term |
| `eval` | how good is the fitter, against patches whose answers are known |
| `selftest` | does rendering repeat, and which parameters move the sound |

Windows only for now -- Linux and macOS are not supported yet, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Build

Windows, with the Visual Studio C++ build tools and CMake:

```powershell
.\scripts\bootstrap.ps1
```

That configures, builds the tool and runs the tests. There is a recording in the
repository to try it on, a two-second tremolo the test suite uses:

```powershell
autosynth fit plugin/tests/golden/analysis/lfo_amp.wav --preset out.vital
```

## Documentation

[CONTRIBUTING.md](CONTRIBUTING.md) covers the architecture, the measurements
behind the design decisions, how the test suite works, and the roadmap.

## License

GPL-3.0-or-later. Chosen deliberately so the project can draw on the GPL
research corpus in this area — Loris, sms-tools, DawDreamer, Vital's own
released source — rather than having to reimplement around it.
