#pragma once

#include "dsp/Voice.h"
#include "ir/Patch.h"
#include <vector>

namespace autosynth
{

// Port of autosynth/fit/partialfit.py -- the recommended fitter.
//
//   audio
//     -> track sinusoidal partials
//     -> group them into harmonic sources (greedy multi-f0)
//     -> one oscillator per source, tuned to that source's own fundamental
//     -> estimate the filter from the dominant source, split into base + env
//     -> solve levels by non-negative least squares
//
// The ordering is the substance. Estimating and removing the filter *before*
// reading waveforms is what stops a saw behind a closed filter being read as a
// triangle; anchoring the cutoff to a joint waveform/cutoff search is what
// makes its absolute value identifiable at all.
class PartialFit
{
public:
    // Most noise a *pitched* patch may carry. Our noise source is flat and
    // static, so it stands in badly for the shaped, note-correlated content a
    // real recording has; left unbounded the level solve and then refinement
    // drove a violin to 41% noise and the patch hissed. Unpitched material
    // takes a different branch and is not bounded by this.
    static constexpr float kMaxPitchedNoise = 0.12f;

    // The share of tracked energy that has to be non-harmonic before the full
    // ceiling above is allowed. Below it the ceiling scales down in proportion.
    //
    // A ceiling fixed in advance says "every pitched sound may hiss this much",
    // which is wrong in the one direction that matters: the level solve takes
    // whatever it is offered, because broadband energy lowers a log-spectral
    // error wherever the harmonic fit is imperfect. Measured between the
    // harmonics, a violin reads 0.29 and a clarinet 0.06, so the violin gets
    // the full ceiling and the clarinet a fifth of it.
    static constexpr double kFullNoiseShare = 0.35;

    struct Options
    {
        double noteHz = 0.0;      // 0 means "use the detected f0"
        double gateSeconds = -1.0; // negative means "detect it"
        int hop = 256;
        int maxOscillators = kNumOsc;
        double tolCents = 50.0;
    };

    static Patch fit (const float* samples, int numSamples, double sampleRate,
                      const Options& options = {});

    // Solve oscillator and noise levels against the target.
    //
    // The factorisation gives each source's shape reliably, but its scale is
    // split arbitrarily and does not map one-to-one onto `level` -- the
    // waveform tables are peak-normalised in time, not by harmonic amplitude.
    static Patch calibrateLevels (Patch patch, const float* target, int numSamples,
                                  double sampleRate, double gateSeconds,
                                  float noiseCeiling = kMaxPitchedNoise);

    // Sets the noise level so the *rendered* result carries as much
    // inter-harmonic energy as the target does.
    //
    // The level solve decides this today, and it decides it badly: noise is one
    // column in a least squares against everything else, so anything that
    // changes how the oscillators render changes how much noise they leave for
    // it. Fitting an attack curve -- which has nothing to do with noise --
    // dropped a violin from 0.09 to 0.015 and audibly stripped the bow off it.
    //
    // Measured and closed-loop instead: render, measure, scale, repeat. Two
    // passes are enough because the relationship is close to proportional, and
    // it is the same measurement `autosynth_diff` prints as noisiness.
    static Patch calibrateNoise (Patch patch, const float* target, int numSamples,
                                 double sampleRate, double gateSeconds, float ceiling);
};

} // namespace autosynth
