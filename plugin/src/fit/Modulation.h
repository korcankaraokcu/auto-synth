#pragma once

#include "ir/Patch.h"
#include <vector>

namespace autosynth
{

// Port of autosynth/fit/modulation.py -- trajectory to LFO.
//
// A trajectory is split into a slow part and an oscillating part; the
// oscillating part becomes an LFO if it is periodic enough to be worth one.
// Three trajectories compete, one per destination the engine supports, and the
// IR carries a single LFO -- so a sound with both vibrato and tremolo can keep
// only one. That is a real limitation, and the point at which a mod matrix
// stops being optional.
class Modulation
{
public:
    static constexpr double kMinRateHz = 0.3;
    static constexpr double kMaxRateHz = 18.0;
    // A single step can fake 2.5 cycles of a square wave; it cannot fake four.
    static constexpr double kMinCycles = 3.5;
    // Measured: genuine 5 Hz vibrato scores 0.83, while the pitch-track step
    // produced by two decaying sources scores 0.23. The old 0.10 admitted the
    // artefact, which rendered as a note jumping a semitone and back.
    static constexpr double kMinConcentration = 0.45;
    static constexpr double kMinCorrelation = 0.45;
    static constexpr double kMinRelativeAmplitude = 0.05;
    static constexpr double kDetrendSeconds = 0.5;

    struct Detected
    {
        bool found = false;
        LfoDest dest = LfoDest::none;
        double rateHz = 0.0;
        double amplitude = 0.0; // in the trajectory's own units
        LfoShape shape = LfoShape::sine;
        double phase = 0.0;
        double delay = 0.0;
        double concentration = 0.0;
    };

    static Detected analyseTrajectory (const std::vector<float>& trajectory, double dt,
                                       LfoDest dest);

    static Lfo toLfo (const Detected& detected);

    // The three trajectories, each expressed in the unit its destination is
    // modulated in -- cents, linear gain, octaves -- so a detected amplitude
    // converts to `depth` without a second scaling step.
    struct Trajectories
    {
        std::vector<float> pitchCents;
        std::vector<float> ampRelative;
        std::vector<float> centroidOctaves;
        double dt = 0.0;
        bool hasPitch = false;
    };

    static Trajectories extract (const float* samples, int numSamples, double sampleRate,
                                 int hop = 256);

    // Returns an LFO with dest = none when nothing is periodic enough to be
    // worth the parameters.
    static Lfo best (const Trajectories& trajectories);

    // The most convincing modulations, at most one per destination. One per
    // destination on purpose: two LFOs both fitted to pitch would be the
    // detector finding the same wobble twice, not two modulations.
    static std::vector<Lfo> bestSeveral (const Trajectories& trajectories,
                                         int maxCount = kNumLfo);
};

} // namespace autosynth
