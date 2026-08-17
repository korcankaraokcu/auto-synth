#pragma once

#include "ir/Patch.h"
#include <functional>
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

        // How a candidate patch becomes audio. Left empty, this uses the engine
        // in this repository.
        //
        // It is a callback rather than a build-time choice so that nothing here
        // has to know about plug-in hosting: the caller that owns a synth owns
        // the renderer. That is what lets refinement optimise against the synth
        // the preset will actually be played by, which removes a whole class of
        // problem rather than solving it -- while the optimiser measures one
        // engine and the file is played by another, every difference between
        // them has to be found and hand-corrected in the exporter.
        //
        // Returns mono samples at the same rate as the target.
        std::function<std::vector<float> (const Patch& candidate,
                                          double durationSeconds,
                                          double gateSeconds)> renderer;
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

    // The searchable range of one continuous parameter.
    //
    // Exposed because the recovery harness needs exactly the same ranges: it
    // samples random patches from them and reports per-parameter error
    // normalised by them. A second copy of the table would drift, and the
    // harness would then be scoring the fitter against a parameter space the
    // fitter does not actually search.
    struct ParamSpec
    {
        std::string path;
        double lo = 0.0;
        double hi = 1.0;
        bool logarithmic = false;
    };

    // Every continuous parameter in the IR, whether or not a given patch would
    // put it in scope.
    static std::vector<ParamSpec> continuousSpecs();

    // Read and write a parameter by IR path. Unknown paths read as 0 and are
    // ignored on write.
    static double parameterValue (const Patch& patch, const std::string& path);
    static void setParameterValue (Patch& patch, const std::string& path, double value);
};

} // namespace autosynth
