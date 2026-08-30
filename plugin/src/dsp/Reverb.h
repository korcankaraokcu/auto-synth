#pragma once

#include "ir/Patch.h"
#include <array>
#include <cmath>

namespace autosynth
{

// What a `ReverbParams` room size means as a decay time.
//
// Not a reverb: nothing here processes audio. The patch states a room size
// because that is what the fitter measures a tail into, and two callers need
// the relation from opposite directions -- the fitter picks a size from a
// measured tail, and the Vital exporter has to state the same tail in another
// synth's units, where the control is a decay time in seconds.
//
// The numbers are Freeverb's, by way of engine/reverb.py, so a size fitted here
// means what it meant in the reference implementation the fixtures came from.
class Reverb
{
public:
    // Freeverb's tunings at 44100 Hz. Mutually prime, which is what stops the
    // comb resonances lining up into a metallic ring.
    static constexpr std::array<int, 8> kCombDelays {
        1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617 };
    static constexpr double kReferenceSampleRate = 44100.0;

    // Mean comb delay in seconds.
    static constexpr double meanCombSeconds() noexcept
    {
        double sum = 0.0;
        for (const auto d : kCombDelays)
            sum += d;
        return sum / kCombDelays.size() / kReferenceSampleRate;
    }

    static double rt60ForSize (double size) noexcept
    {
        const auto feedback = juce::jlimit (0.0, 0.98, 0.7 + 0.28 * size);
        if (feedback <= 0.0 || feedback >= 1.0)
            return 0.0;   // caller decides what an unbounded tail means

        // Each pass round a comb multiplies by `feedback` and takes one mean
        // comb delay, so the tail falls -20*log10(feedback) dB every pass.
        const auto dbPerPass = -20.0 * std::log10 (feedback);
        if (dbPerPass <= 1.0e-9)
            return 0.0;
        return 60.0 * meanCombSeconds() / dbPerPass;
    }

    static double sizeForRt60 (double rt60Seconds) noexcept
    {
        if (rt60Seconds <= 0.0)
            return 0.0;
        // Inverse of the above: f = 10^(-3D/RT60).
        const auto feedback = std::pow (10.0, -3.0 * meanCombSeconds() / rt60Seconds);
        return juce::jlimit (0.0, 1.0, (feedback - 0.7) / 0.28);
    }
};

} // namespace autosynth
