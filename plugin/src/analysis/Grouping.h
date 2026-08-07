#pragma once

#include "analysis/Partials.h"
#include <map>
#include <vector>

namespace autosynth
{

// Port of autosynth/analysis/grouping.py -- greedy multi-f0 extraction.
//
// Partial tracking says what frequencies are present; it says nothing about
// which belong together. This makes that call, and the answer is the oscillator
// count, arrived at from measured evidence rather than from factorising a grid
// that may never have sampled the sources.
//
// The subtlety that makes or breaks it: every harmonic of f0 is also a harmonic
// of f0/2, so a naive "how much energy does this fundamental explain" score is
// maximised by guessing absurdly low fundamentals, which then swallow every
// source at once. The fix is to charge for what a candidate *predicts but does
// not find*.
struct HarmonicGroup
{
    double f0 = 0.0;
    std::vector<float> H;          // [numHarmonics * numFrames], row-major by harmonic
    int numHarmonics = 0;
    int numFrames = 0;
    std::vector<int> harmonicIndices;
    double salience = 0.0;
    std::vector<Partial> partials;

    // Frames per second of `H`. Carried on the group because the unison
    // estimator reads beat *rates* out of the harmonic envelopes, and a rate
    // is meaningless without knowing how fast the frames go by.
    double frameRateHz = 0.0;

    const float* harmonic (int k) const noexcept
    {
        return H.data() + static_cast<size_t> (k) * static_cast<size_t> (numFrames);
    }
    float energy() const noexcept;
};

class Grouping
{
public:
    static constexpr double kMinF0 = 30.0;
    static constexpr double kMaxF0 = 2000.0;
    static constexpr int kMaxHarmonic = 32;
    static constexpr double kTolCents = 50.0;

    static std::vector<HarmonicGroup> group (const PartialSet& partials,
                                             int maxGroups = 3,
                                             double tolCents = kTolCents,
                                             double minEnergyFraction = 0.05);

    // Cents deviations bucketed by harmonic number. A single voice puts one
    // partial at each harmonic; N detuned unison voices put N of them, spread
    // by a constant number of cents at *every* harmonic.
    static std::map<int, std::vector<float>> harmonicClusters (const HarmonicGroup& group);

    // Under-counts, and the cause is resolution rather than the algorithm: at a
    // 2048-point window a 20-cent spread on a 220 Hz source does not separate
    // until roughly the eighteenth harmonic. The spread is recovered well
    // whenever anything resolves at all; only the count is short.
    // Evidence that a group's harmonics are beating, which is what narrow
    // unison looks like once the voices stop resolving into separate peaks.
    //
    // The test is not merely "are the harmonic envelopes periodic" -- tremolo
    // and vibrato are periodic too, and every LFO in the test set would trip
    // it. The test is whether the rate rises *in proportion to harmonic
    // number*, which only a frequency gap between detuned partials does.
    struct UnisonBeating
    {
        bool found = false;
        double detuneCents = 0.0;
        double beatRateAtFundamental = 0.0;
        double proportionalFit = 0.0;   // 1 is a perfect rate ∝ k relationship
        int harmonicsAgreeing = 0;
    };

    static UnisonBeating detectUnisonBeating (const HarmonicGroup& group, int minHarmonic = 2);

    static void estimateUnison (const HarmonicGroup& group, int& voicesOut, float& detuneOut,
                                int maxVoices = 7, int minHarmonic = 2);

    static double detuneCents (const HarmonicGroup& group);

    // Collapse deviations within `tol` cents, returning cluster means. A
    // partial that drops below threshold and is re-acquired becomes two
    // objects at nearly the same frequency; counting objects rather than
    // distinct frequencies reported seven voices for every unison patch.
    static std::vector<float> mergeClose (std::vector<float> cents, float tol = 6.0f);
};

} // namespace autosynth
