#pragma once

#include "ir/Patch.h"
#include <vector>

namespace autosynth
{

// Port of autosynth/fit/effects.py -- delay detection.
//
// Delay only. Discrete repeats are literal copies of the signal displaced in
// time, so they show up as clean peaks in the autocorrelation of the loudness
// envelope. Reverb is a dense diffuse tail with no discrete repeats to
// correlate against, and separating it from an instrument's own decay is blind
// dereverberation.
//
// There is a second reason to stop at delay: a reverb tail and a long release
// produce nearly identical loudness curves, so a fitter given both with no
// structure to separate them trades one against the other and gets both wrong.
// Delay has no such twin.
class EffectsFit
{
public:
    static constexpr double kMinDelaySeconds = 0.02;
    static constexpr double kMaxDelaySeconds = 1.0;
    static constexpr double kMinPeakRatio = 0.15;

    struct DelayEstimate
    {
        bool found = false;
        double time = 0.0;
        double feedback = 0.0;
        double mix = 0.0;
        double strength = 0.0;
    };

    static DelayEstimate detectDelay (const float* samples, int numSamples, double sampleRate,
                                      int hop = 256, double smoothSeconds = 0.01);

    // Detected delay as IR parameters, disabled when nothing was found.
    static DelayParams fitDelay (const float* samples, int numSamples, double sampleRate,
                                 int hop = 256);
};

} // namespace autosynth
