# auto-synth

Turn a sample into an **editable** multi-oscillator synth patch.

Not a wavetable dump and not a black box: the output is a small set of
oscillators, envelopes, filters and modulation that a human can open and
change. That constraint is the whole project. The objective is not "minimise
reconstruction error" — a sampler already scores zero on that — it is
**minimise error subject to a parameter budget**.

## What it does

Drop a `.wav` onto the plugin. It analyses the sample in process, builds a
patch, and plays it. Every control is a real knob and is host-automatable; the
only thing that is not is the harmonic content of a fitted wavetable, which is
drawn rather than dialled.

- **3 oscillators**, each with its own tuning, unison, envelope, filter and
  reverb send
- **Every oscillator is a wavetable**, up to 16 frames of 16 harmonics, drawn
  and edited as bars. A classic shape is a one-frame table nobody has drawn on
  yet, so it keeps its full bandwidth; frames get drawn only when no shape can
  describe the sound
- **2 LFO slots**, so vibrato and tremolo can coexist rather than compete
- Global filter with envelope, delay, and a shared reverb
- **Export to Vital** — the fitted patch, wavetables and all, as a 
  preset
- Solo/mute per oscillator, A/B against the original, and a spectrum overlay
  showing source against fit

## Status

Working end to end as a VST3 and a standalone application. Analysis, synthesis
and CMA-ES refinement all run natively — there is no Python at runtime and no
Python in this repository.

The weakest part is oscillator counting, at 70.8% exact against a ground-truth
harness. Everything else is downstream of that, and it is where the work is
going next: what is left is almost entirely *under*-counting, and the intervals
say why — two oscillators an octave apart sharing an envelope are
mathematically identical to one oscillator with a different waveform.

Not started: exporters to other synths, stereo, and automatic reverb
*detection* — the reverb exists and is editable, but recovering one from a
dry-unknown sample is blind dereverberation, which is a separate problem.

## Build

Windows, with the Visual Studio C++ build tools and CMake:

```powershell
.\scripts\bootstrap.ps1 -Install
```

That configures, builds, runs the tests, and installs the VST3 to your per-user
plug-in folder. The standalone application is left at
`plugin/build/AutoSynth_artefacts/Release/Standalone/auto-synth.exe`.

Linux and macOS are not supported yet — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

[CONTRIBUTING.md](CONTRIBUTING.md) covers the architecture, the measurements
behind the design decisions, how the test suite works, and the roadmap.

## License

GPL-3.0-or-later. Chosen deliberately so the project can draw on the GPL
research corpus in this area — Loris, sms-tools, DawDreamer, Vital's own
released source — rather than having to reimplement around it.
