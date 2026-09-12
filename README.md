# auto-synth

Turn a recording of a single note into an **editable** Vital preset.

Not a wavetable dump and not a black box: the output is a small set of
oscillators, envelopes, filters and modulation that a person can open and
change. The objective is not "minimise reconstruction error" — a sampler already
scores zero on that — it is **minimise error subject to a parameter budget**.

The recording is analysed into partials, split into sources and fitted to a
patch, and the patch is then refined by CMA-ES against Vital itself: every
candidate is rendered by the synth the preset will be opened in.

- **3 oscillators**, each with its own tuning, unison and envelope
- **Every oscillator is a wavetable**, up to 16 frames of 16 harmonics
- **2 LFO slots**, so vibrato and tremolo can coexist rather than compete
- Filter with envelope, delay and reverb

## Use

One file: `autosynth.exe` from the releases tab. [Vital](https://vital.audio/)
must be installed — the free version is enough — because it *is* the synth here.

### As a GUI

Double-click it, or drop a `.wav` straight onto it. It writes the preset next to
the recording and shows how close the fit came: the two loudness contours over
each other, and eleven named axes with a verdict on each.

![the UI](https://github.com/user-attachments/assets/03b6a46b-4c72-4d1c-b840-9c2e49ff2e8b)

### As a CLI

Run it with arguments and no window opens.

```powershell
autosynth fit your-recording.wav
```

Any mono `.wav` of a single sustained note will do. That writes
`your-recording.vital` next to it; open it in Vital and start turning knobs.
`--preset`, `--patch` and `--render` put the preset, the intermediate patch and
an audio rendering of the fit somewhere else.

The other verbs report on what happened. `autosynth` on its own lists them:

| Verb | |
|---|---|
| `fit` | recording in, preset out. The main one |
| `diff` | how far two recordings are apart, on eleven named axes |
| `render` | play a fitted patch through Vital and write the audio |
| `probe` | every analysis stage as JSON |
| `score` | what the fitting objective makes of a patch, term by term |
| `eval` | how good is the fitter, against patches whose answers are known |
| `selftest` | does rendering repeat, and which parameters move the sound |
| `gui` | open the UI from a terminal |

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
