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
                                  double sampleRate, double gateSeconds);
};

} // namespace autosynth
