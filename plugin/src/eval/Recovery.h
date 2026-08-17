#pragma once

#include "ir/Patch.h"

#include <juce_core/juce_core.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace autosynth
{

// Ground-truth recovery: render a random patch from our own engine, feed the
// audio to the fitter, and check whether it gets the parameters back.
//
// This is the highest-leverage measurement in the project. Exact labels, no
// annotation, unlimited data -- and unlike listening tests it says *which*
// parameter was missed, which is what turns "the fit sounds wrong" into a
// to-do list.
//
// Two things make the numbers mean anything:
//
//   * **A control.** Every run also scores an untouched default patch. A fitter
//     that cannot beat the control has learned nothing, and an absolute
//     distance on its own does not say whether it did.
//   * **Same-run comparison only.** Random patches are sampled from the
//     parameter ranges, so *adding a parameter to the IR changes the targets*.
//     Scores from before and after an IR change are measuring different
//     populations and must not be diffed. Read a fitted score against the
//     control from its own run.
//
// What it cannot see: it generates targets by rendering this engine, so it only
// ever tests sounds this engine can already make. It measures how well we solve
// the *inverse* problem and is structurally blind to the modelling gap --
// everything a real recording contains that the parameterisation cannot
// express. A real sample tells you *that* something is wrong; this tells you
// *why*. Both, always.
class Recovery
{
public:
    // Turns a patch into mono audio at Options::sampleRate. See Options below.
    using Renderer = std::function<std::vector<float> (const Patch& patch,
                                                       double durationSeconds,
                                                       double gateSeconds)>;

    struct Options
    {
        int trials = 24;
        unsigned seed = 0;
        double sampleRate = 48000.0;
        double duration = 1.0;
        double gate = 0.7;
        bool refine = true;
        int refineEvaluations = 192;
        // Random patches can come out inaudible -- three disabled oscillators,
        // or a master level near zero. Scoring those measures nothing, so they
        // are resampled rather than counted.
        float minPeak = 0.05f;

        // Which synth the harness is measuring. Left empty this is the engine
        // in this repository, and the note above about the modelling gap
        // applies to that engine.
        //
        // Supplied, it replaces *every* render -- target, fitted, control and
        // each refinement candidate -- so the whole question moves into the
        // other synth's world: can the fitter recover the parameters of a patch
        // that synth rendered? For a project whose deliverable is a preset for
        // someone else's synth, that is the question that matters, and it is
        // not the same one as recovering our own.
        //
        // It must be all or nothing. Rendering the target in one synth and the
        // control in the other would score the difference between two synths
        // and report it as a fitting error.
        Renderer renderer;
    };

    // Distances between a rendered patch and the target, all lower-is-better.
    struct Scores
    {
        double spectral = 0.0;
        double loudnessDb = 0.0;
        double centroidOctaves = 0.0;
    };

    struct Trial
    {
        Patch truth;
        Patch fitted;
        Scores fit;      // fitted patch against the target
        Scores control;  // an untouched default patch against the same target
        bool oscCountExact = false;
        int truthOscCount = 0;
        int fittedOscCount = 0;
        bool rootWithinSemitone = false;
        bool truthReverb = false;
        bool fittedReverb = false;
        bool truthDelay = false;
        bool fittedDelay = false;
    };

    // How often a yes/no decision was right, split by which way it went.
    //
    // Averaged distances hide structural mistakes: switching an effect on that
    // was never there costs a little on every frame and looks like general
    // imprecision, which is exactly how it goes unnoticed.
    struct Decision
    {
        int truePositive = 0;
        int falsePositive = 0;
        int trueNegative = 0;
        int falseNegative = 0;

        int total() const { return truePositive + falsePositive + trueNegative + falseNegative; }
        double accuracy() const
        {
            const auto n = total();
            return n > 0 ? static_cast<double> (truePositive + trueNegative) / n : 0.0;
        }
    };

    struct Summary
    {
        int trials = 0;
        Scores fit;
        Scores control;
        double oscCountAccuracy = 0.0;   // fraction exact
        double rootAccuracy = 0.0;       // fraction within a semitone
        Decision reverb;
        Decision delay;
        // Mean absolute error per continuous parameter, normalised by that
        // parameter's own range so the entries are comparable. Only parameters
        // belonging to oscillators both patches agree are active are counted --
        // comparing the cutoff of an oscillator the fitter never proposed is
        // not a measurement of anything.
        std::map<std::string, double> parameterError;
        std::vector<Trial> byTrial;
    };

    static Summary run (const Options& options = {});

    // A uniformly random patch, discrete parameters included. This is the
    // target distribution; see the note above about comparing across IR
    // changes.
    static Patch randomPatch (juce::Random& rng);

    static Scores score (const std::vector<float>& candidate,
                         const std::vector<float>& target,
                         double sampleRate);

    static juce::String toJson (const Summary& summary);
    static juce::String toText (const Summary& summary);
};

} // namespace autosynth
