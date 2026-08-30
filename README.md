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

Against the two library recordings the presets are within tolerance on most of
the axes the diagnostic reports -- pitch, brightness, noisiness, vibrato,
note-off and level -- and short on the clarinet's timbre movement and on both
onsets. Oscillator counting remains the weakest structural step.

Not started: exporters to other synths, and stereo.

## Build

Windows, with the Visual Studio C++ build tools and CMake:

```powershell
.\scripts\bootstrap.ps1
```

That configures, builds the tools and runs the tests.
[Vital](https://vital.audio/) must be installed -- the free version is enough --
because it is the synth: it renders the fit, and the test cases that need sound
are skipped without it.

```powershell
autosynth_vital fitted.json out.wav --fit samples/violin.wav --preset out.vital
```

No Vital code is compiled or shipped here; the tool hosts whatever VST3 it
finds in the platform's plug-in folders.

Linux and macOS are not supported yet -- see [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

[CONTRIBUTING.md](CONTRIBUTING.md) covers the architecture, the measurements
behind the design decisions, how the test suite works, and the roadmap.

## License

GPL-3.0-or-later. Chosen deliberately so the project can draw on the GPL
research corpus in this area — Loris, sms-tools, DawDreamer, Vital's own
released source — rather than having to reimplement around it.
