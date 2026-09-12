#pragma once

// Turning a recording into a preset. One implementation, two callers.
//
// The `fit` verb and the UI run the same fit. Written twice they could differ
// by one of them missing a step, which has happened here and is invisible from
// the outside: a fit without its renderer comes back looking complete --
// oscillators, envelope, filter, all populated -- and is simply wrong. One
// function means one order of operations.
//
// It reports progress and accepts cancellation because the UI needs both:
// refinement is a couple of hundred renders through Vital, and whoever dropped
// the file is watching. The command line passes neither and pays nothing.

#include "eval/Diagnose.h"
#include "fit/PartialFit.h"
#include "fit/Refine.h"
#include "ir/Patch.h"
#include "vital/VitalHost.h"

#include <juce_core/juce_core.h>

#include <atomic>
#include <cmath>
#include <functional>
#include <vector>

namespace autosynth::job
{

struct Options
{
    // Negative means "take it from the recording", which is what the UI wants:
    // it has no flag to correct one with, and an assumed note-off is wrong on
    // anything but a two-second sample.
    double gateSeconds = -1.0;
    int refineEvaluations = 192;
    unsigned seed = 1;
};

struct Fitted
{
    bool cancelled = false;

    Patch patch;

    // The note-off refinement actually used, which is not always the one asked
    // for: a negative `gateSeconds` means "take it from the recording", and only
    // the fitter knows how to read that. Rendering needs a real number of
    // seconds -- given the sentinel, Vital releases the note before it starts
    // and returns silence.
    double gateSeconds = 0.0;

    double initialLoss = 0.0;
    double finalLoss = 0.0;
    int evaluations = 0;
    double seconds = 0.0;
};

// What stage the fit is in, and how far through. `fraction` is over the whole
// job rather than the stage, because a progress bar that restarts is worse
// than no progress bar.
using Progress = std::function<void (double fraction, const juce::String& stage)>;

// Analysis, then refinement against the synth that will play the result.
//
// `host` must already be open, and nothing else may touch it while this runs --
// it is one plug-in instance and one voice.
//
// On cancellation the renderer returns silence instead of calling Vital. The
// search then finishes its remaining evaluations in milliseconds against a
// constant and the caller throws the answer away, which is far simpler and far
// safer than unwinding an optimiser from the inside.
inline Fitted fit (VitalHost& host,
                   const std::vector<float>& target,
                   double sampleRate,
                   const Options& options,
                   const Progress& progress = {},
                   const std::atomic<bool>* cancelled = nullptr)
{
    Fitted result;

    const auto stopped = [cancelled] { return cancelled != nullptr && cancelled->load(); };
    const auto report = [&progress] (double fraction, const char* stage)
    {
        if (progress)
            progress (fraction, stage);
    };

    // Refinement is all but one of the renders, so the bar is its evaluation
    // count with the fixed stages given the ends.
    auto evaluations = 0;
    const auto budget = juce::jmax (1, options.refineEvaluations);

    // One renderer object handed to every stage rather than one built per call
    // site: the stages that take it are closed loops, and a loop that quietly
    // becomes no loop reports success.
    const Renderer renderer = [&] (const Patch& candidate, double dur, double gate)
    {
        if (stopped())
            return std::vector<float> ((size_t) juce::jmax (0.0, std::round (dur * sampleRate)),
                                       0.0f);

        report (juce::jlimit (0.0, 1.0, 0.10 + 0.85 * ++evaluations / (double) budget),
                "fitting");
        return host.render (candidate, candidate.rootHz, dur, gate);
    };

    report (0.0, "analysing");

    PartialFit::Options fitOptions;
    fitOptions.gateSeconds = options.gateSeconds;
    fitOptions.renderer = renderer;
    result.patch = PartialFit::fit (target.data(), (int) target.size(), sampleRate, fitOptions);

    if (stopped())
    {
        result.cancelled = true;
        return result;
    }

    Refine::Options refineOptions;
    refineOptions.maxEvaluations = options.refineEvaluations;
    refineOptions.gateSeconds = options.gateSeconds;
    refineOptions.renderer = renderer;

    // Exposed because one fit is one sample of a search, not the answer. The
    // objective has several terms and CMA-ES settles on a different trade
    // between them from a different draw.
    refineOptions.seed = options.seed;

    const auto started = juce::Time::getMillisecondCounterHiRes();
    const auto refined = Refine::run (result.patch, target.data(), (int) target.size(),
                                      sampleRate, refineOptions);
    result.seconds = (juce::Time::getMillisecondCounterHiRes() - started) / 1000.0;

    if (stopped())
    {
        result.cancelled = true;
        return result;
    }

    result.patch = refined.patch;
    result.gateSeconds = refined.gateSeconds;
    result.initialLoss = refined.initialLoss;
    result.finalLoss = refined.finalLoss;
    result.evaluations = refined.evaluations;
    return result;
}

struct Comparison
{
    std::vector<float> rendered;
    Diagnose::Measurements targetMeasured, fitMeasured;
    std::vector<Diagnose::Axis> axes;
};

// Play the answer and measure it against the recording.
//
// Separate from the fit because the command line renders itself -- it has a
// `--note` to honour and a file to write -- and rendering twice to reach the
// same samples wastes a second for nothing.
inline Comparison audition (VitalHost& host,
                            const Patch& patch,
                            const std::vector<float>& target,
                            double sampleRate,
                            double durationSeconds,
                            double gateSeconds)
{
    // A negative gate is the fitter's sentinel for "take it from the recording",
    // not a length of time. Rendering one releases the note before it starts,
    // and the resulting silence measures as a valid but completely wrong fit:
    // every axis reports a number, and the numbers are 0.0 Hz at 120 dB down.
    // Caught here rather than diagnosed from the output, which is how it was
    // found the first time.
    jassert (gateSeconds >= 0.0);
    const auto gate = gateSeconds >= 0.0 ? gateSeconds : durationSeconds;

    Comparison out;
    out.rendered = host.render (patch, patch.rootHz, durationSeconds, gate);
    out.targetMeasured = Diagnose::measure (target, sampleRate);
    out.fitMeasured = Diagnose::measure (out.rendered, sampleRate);
    out.axes = Diagnose::compare (out.targetMeasured, out.fitMeasured);
    return out;
}

} // namespace autosynth::job
