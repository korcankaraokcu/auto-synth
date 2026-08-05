#pragma once

#include "ir/Patch.h"
#include <string>
#include <vector>

namespace autosynth
{

// Port of autosynth/fit/refine.py -- CMA-ES polish against the target audio.
//
// Analysis decides *what* the parameters are; this lands the values. Three
// rounds of better analysis improved oscillator counts and source pitches
// without improving audio distance at all, and this is the stage that closed
// that gap -- roughly halving it on every fitter.
class Refine
{
public:
    struct Options
    {
        int maxEvaluations = 192;
        int populationSize = 16;
        double sigma = 0.12;
        unsigned seed = 1;
        double gateSeconds = -1.0;
    };

    struct Result
    {
        Patch patch;
        double initialLoss = 0.0;
        double finalLoss = 0.0;
        int evaluations = 0;
        bool improved = false;
    };

    static Result run (const Patch& patch, const float* target, int numSamples,
                       double sampleRate, const Options& options = {});

    // Which continuous parameters are worth searching for this patch.
    //
    // Only continuous ones. Waveform, oscillator count, filter type and LFO
    // destination stay exactly as analysis left them: a Gaussian search over a
    // normalised index is noise, and worse, it lets the optimiser destroy a
    // correct structural decision to buy a small local gain.
    //
    // Parameters of disabled oscillators and unrouted LFOs are dropped too --
    // CMA-ES sample efficiency falls off with dimension, so every dead
    // parameter is paid for in convergence speed.
    static std::vector<std::string> scopeFor (const Patch& patch);

    // The three loss terms, reported separately.
    struct Loss
    {
        double spectral = 0.0;
        double loudness = 0.0;
        double centroid = 0.0;
    };
};

} // namespace autosynth
