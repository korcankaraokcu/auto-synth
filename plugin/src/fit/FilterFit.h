#pragma once

#include "ir/Patch.h"
#include <vector>

namespace autosynth
{

// Port of autosynth/fit/filterfit.py.
//
// The filter has to be estimated and divided out *before* oscillator counting.
// An oscillator with a fixed waveform and an amplitude envelope is a rank-1
// component of the harmonic matrix; a filter sweep is not, because it changes
// the spectrum's shape over time. Factorise without removing it and a single
// swept oscillator comes back as four or five static ones.
class FilterFit
{
public:
    static constexpr float kDefaultQ = 0.707f;
    static constexpr int kNumCutoffGrid = 40;
    static constexpr int kNumIterations = 4;

    struct Trajectory
    {
        std::vector<float> cutoffHz;  // per frame
        std::vector<float> source;    // peak-normalised harmonic profile
    };

    // Alternates between "given the filter, what is the source spectrum?" and
    // "given the source, what is the filter?" -- neither is knowable first,
    // which is what alternating least squares exists for.
    //
    // Only the *relative* motion of the result is trustworthy. Source times
    // filter is a blind deconvolution, so the absolute level is not
    // identifiable from this alone; `trajectoryToEnv` re-anchors it.
    static Trajectory estimateCutoffTrajectory (const std::vector<float>& H,
                                                int numHarmonics, int numFrames,
                                                double f0Hz, double sampleRate,
                                                float q = kDefaultQ,
                                                FilterType type = FilterType::lowpass,
                                                int numIterations = kNumIterations,
                                                int smoothFrames = 9);

    // Divide the estimated response out of the harmonic matrix. The result is
    // what the oscillators produced before the filter.
    static std::vector<float> deconvolve (const std::vector<float>& H,
                                          int numHarmonics, int numFrames,
                                          const std::vector<float>& cutoffHz, double f0Hz,
                                          float q = kDefaultQ,
                                          FilterType type = FilterType::lowpass);

    struct EnvSplit
    {
        float baseCutoffHz = 0.0f;
        float envAmountOctaves = 0.0f;
        std::vector<float> shape; // unit range, for fitAdsr
    };

    // Splits a trajectory into base cutoff + normalised envelope.
    //
    // Uses only relative motion, re-anchored to `anchorHz`. Taking the base from
    // the trajectory's own floor -- as an earlier version did -- inherits the
    // unidentifiable offset and lands the filter octaves away from where it
    // belongs.
    static EnvSplit trajectoryToEnv (const std::vector<float>& cutoffHz, double anchorHz,
                                     float minOctaves = 0.25f);
};

} // namespace autosynth
