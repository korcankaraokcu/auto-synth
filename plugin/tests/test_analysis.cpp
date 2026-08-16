// Analysis behaviour on signals whose answer is known by construction.
//
// The golden fixtures check that this chain still produces the numbers the
// reference produced. These check that those numbers are *right* -- that a
// 220 Hz saw reads as 220 Hz, that noise is refused rather than guessed at,
// and that two sources a fifth apart come back as two. A frozen fixture would
// happily preserve a wrong answer.

#include "Helpers.h"

#include "analysis/Grouping.h"
#include "analysis/Roles.h"
#include "analysis/Partials.h"
#include "analysis/Yin.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>

using namespace autotest;
using namespace autosynth;

namespace
{

constexpr int kHop = 256;
constexpr int kFft = 2048;

std::vector<float> tone (Waveform waveform = Waveform::saw, double f0 = 220.0,
                         double duration = 1.0)
{
    auto patch = simplePatch (waveform);
    patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
    patch.rootHz = (float) f0;
    auto out = render (patch, f0, duration, duration);
    for (auto& v : out)
        v *= 0.8f;
    return out;
}

// Additive saws, so the number of sources is known exactly rather than being
// whatever the engine's grouping happens to produce.
std::vector<float> sawSum (const std::vector<std::pair<double, double>>& sources,
                           double duration = 1.0, int numHarmonics = 12)
{
    const auto n = (size_t) (kSampleRate * duration);
    std::vector<float> out (n, 0.0f);
    for (const auto& [f0, amp] : sources)
        for (int k = 1; k <= numHarmonics; ++k)
            for (size_t i = 0; i < n; ++i)
                out[i] += (float) (amp * std::sin (2.0 * juce::MathConstants<double>::pi
                                                   * f0 * k * (double) i / kSampleRate) / k);

    const auto peak = peakOf (out);
    if (peak > 0.0f)
        for (auto& v : out)
            v *= 0.8f / peak;
    return out;
}

PartialSet trackOf (const std::vector<float>& x)
{
    PartialTracker::Options options;
    options.fftSize = kFft;
    options.hop = kHop;
    return PartialTracker::track (x.data(), (int) x.size(), kSampleRate, options);
}

} // namespace

// --- STFT ------------------------------------------------------------------

TEST_CASE ("the STFT window is periodic, not symmetric", "[analysis]")
{
    // scipy's `hann(N, sym=False)` and JUCE's default differ in the last
    // sample. The difference is one bin of leakage, which is invisible until
    // it is compared against a reference that made the other choice.
    const auto w = Stft::periodicHann (8);
    REQUIRE (w.size() == 8);
    CHECK (w[0] == Catch::Approx (0.0f).margin (1.0e-6));
    // Symmetric would put a zero at the end too; periodic does not.
    CHECK (w[7] > 0.1f);
}

TEST_CASE ("frame count follows the documented formula", "[analysis]")
{
    for (int n : { 4096, 12000, 48000 })
    {
        const auto expected = Stft::numFramesFor (n, kFft, kHop);
        const std::vector<float> x ((size_t) n, 0.1f);
        const auto spec = Stft::magnitudeSpectrogram (x.data(), n, kFft, kHop, kSampleRate);
        INFO ("n = " << n);
        CHECK (spec.numFrames == expected);
        CHECK (spec.numBins == kFft / 2 + 1);
    }
}

TEST_CASE ("spectral centroid rises with brightness", "[analysis]")
{
    CHECK (meanCentroidHz (tone (Waveform::sine)) < meanCentroidHz (tone (Waveform::saw)));
}

TEST_CASE ("the loudness envelope follows the amplitude", "[analysis]")
{
    auto x = tone();
    const auto half = x.size() / 2;
    for (size_t i = half; i < x.size(); ++i)
        x[i] *= 0.25f;

    const auto loud = Stft::loudnessEnvelope (x.data(), (int) x.size(), kHop);
    REQUIRE (loud.size() > 8);
    const auto early = loud[loud.size() / 4];
    const auto late = loud[3 * loud.size() / 4];
    CHECK (late < 0.5f * early);
}

// --- f0 --------------------------------------------------------------------

TEST_CASE ("f0 is recovered across the range", "[analysis]")
{
    for (double f0 : { 55.0, 110.0, 220.0, 440.0, 880.0 })
    {
        double estimate = 0.0, confidence = 0.0;
        const auto x = tone (Waveform::saw, f0);
        Yin::estimate (x.data(), (int) x.size(), kSampleRate, estimate, confidence, kHop);
        INFO ("expected " << f0 << " got " << estimate);
        CHECK (estimate == Catch::Approx (f0).epsilon (0.01));
        CHECK (confidence > 0.5);
    }
}

TEST_CASE ("f0 is recovered across waveforms", "[analysis]")
{
    for (auto wf : { Waveform::sine, Waveform::triangle, Waveform::saw, Waveform::square })
    {
        double estimate = 0.0, confidence = 0.0;
        const auto x = tone (wf, 220.0);
        Yin::estimate (x.data(), (int) x.size(), kSampleRate, estimate, confidence, kHop);
        INFO ("waveform " << (int) wf << " got " << estimate);
        CHECK (estimate == Catch::Approx (220.0).epsilon (0.01));
    }
}

TEST_CASE ("noise reports low confidence", "[analysis]")
{
    std::mt19937 rng (7);
    std::normal_distribution<float> dist (0.0f, 0.3f);
    std::vector<float> x ((size_t) kSampleRate, 0.0f);
    for (auto& v : x)
        v = dist (rng);

    double f0 = 0.0, confidence = 0.0;
    Yin::estimate (x.data(), (int) x.size(), kSampleRate, f0, confidence, kHop);
    // Refusing to answer is the correct behaviour; a confident wrong f0 would
    // poison every stage downstream.
    CHECK (confidence < 0.5);
}

TEST_CASE ("silence reports no pitch", "[analysis]")
{
    const std::vector<float> x ((size_t) kSampleRate, 0.0f);
    double f0 = 0.0, confidence = 0.0;
    Yin::estimate (x.data(), (int) x.size(), kSampleRate, f0, confidence, kHop);
    CHECK (f0 <= 0.0);
}

TEST_CASE ("note naming matches the pitch", "[analysis]")
{
    struct Case { double hz; const char* name; };
    for (auto c : { Case { 440.0, "A4" }, Case { 220.0, "A3" }, Case { 261.63, "C4" } })
    {
        double cents = 0.0;
        INFO (c.hz);
        CHECK (Yin::noteName (c.hz, cents) == juce::String (c.name));
        CHECK (std::abs (cents) < 5.0);
    }
}

// --- partials --------------------------------------------------------------

TEST_CASE ("a sine yields a single partial", "[analysis]")
{
    const auto set = trackOf (tone (Waveform::sine, 220.0));
    REQUIRE (set.partials.size() == 1);
    CHECK (set.partials[0].meanFreq() == Catch::Approx (220.0f).epsilon (0.02));
}

TEST_CASE ("a saw yields many partials at harmonic spacing", "[analysis]")
{
    const auto set = trackOf (tone (Waveform::saw, 220.0));
    CHECK (set.partials.size() > 8);

    // The lowest tracked partial should be the fundamental.
    float lowest = 1.0e9f;
    for (const auto& p : set.partials)
        lowest = juce::jmin (lowest, p.meanFreq());
    CHECK (lowest == Catch::Approx (220.0f).epsilon (0.05));
}

TEST_CASE ("silence yields no partials", "[analysis]")
{
    const std::vector<float> x ((size_t) kSampleRate, 0.0f);
    CHECK (trackOf (x).partials.empty());
}

// --- grouping --------------------------------------------------------------

TEST_CASE ("a single source groups to one", "[analysis]")
{
    const auto groups = Grouping::group (trackOf (sawSum ({ { 220.0, 1.0 } })), 3);
    REQUIRE_FALSE (groups.empty());
    CHECK (groups[0].f0 == Catch::Approx (220.0).epsilon (0.05));
}

TEST_CASE ("two sources a fifth apart separate", "[analysis]")
{
    // The case a fixed harmonic grid cannot see. 330 is not a harmonic of 220,
    // so a grid anchored on 220 has nowhere to put it -- which is why partial
    // tracking plus greedy multi-f0 grouping replaced the grid.
    const auto groups = Grouping::group (
        trackOf (sawSum ({ { 220.0, 1.0 }, { 330.0, 0.8 } })), 3);

    REQUIRE (groups.size() >= 2);
    bool found220 = false, found330 = false;
    for (const auto& g : groups)
    {
        if (std::abs (g.f0 - 220.0) / 220.0 < 0.05) found220 = true;
        if (std::abs (g.f0 - 330.0) / 330.0 < 0.05) found330 = true;
    }
    CHECK (found220);
    CHECK (found330);
}

TEST_CASE ("the subharmonic trap is avoided", "[analysis]")
{
    // Every harmonic of 220 is also a harmonic of 110, so 110 explains the
    // data equally well and salience alone would happily pick it.
    const auto groups = Grouping::group (trackOf (sawSum ({ { 220.0, 1.0 } })), 3);
    REQUIRE_FALSE (groups.empty());
    CHECK (groups[0].f0 > 150.0);
}

TEST_CASE ("silence yields no groups", "[analysis]")
{
    const std::vector<float> x ((size_t) kSampleRate, 0.0f);
    CHECK (Grouping::group (trackOf (x), 3).empty());
}

TEST_CASE ("groups are ordered by salience", "[analysis]")
{
    const auto groups = Grouping::group (
        trackOf (sawSum ({ { 220.0, 1.0 }, { 330.0, 0.3 } })), 3);
    for (size_t i = 1; i < groups.size(); ++i)
        CHECK (groups[i - 1].salience >= groups[i].salience);
}

// Noise measured between the harmonics, not from the tracker.
//
// The tracker-based version of this reported a violin at 72% non-harmonic
// energy and a clarinet at 0.9% when given a 2048-point window, and 0.03% for
// both at the 1024 the fitter actually uses. A measurement that moves three
// orders of magnitude with an analysis window is not measuring the sound, which
// is why this one works on the spectrum instead.
TEST_CASE ("noise share separates a tone from a tone plus hiss", "[analysis]")
{
    constexpr double f0 = 220.0;
    const auto samples = static_cast<int> (kSampleRate);

    std::vector<float> clean (static_cast<size_t> (samples), 0.0f);
    for (int i = 0; i < samples; ++i)
    {
        auto v = 0.0;
        for (int k = 1; k <= 8; ++k)
            v += std::sin (2.0 * juce::MathConstants<double>::pi * k * f0 * i / kSampleRate) / k;
        clean[static_cast<size_t> (i)] = static_cast<float> (v * 0.3);
    }

    juce::Random rng (1234);
    auto hissy = clean;
    for (auto& v : hissy)
        v += (rng.nextFloat() * 2.0f - 1.0f) * 0.05f;

    const auto quiet = autosynth::Roles::noiseShare (clean.data(), samples, kSampleRate, f0);
    const auto loud = autosynth::Roles::noiseShare (hissy.data(), samples, kSampleRate, f0);

    INFO ("clean " << quiet << "  hissy " << loud);
    CHECK (quiet < 0.2);
    CHECK (loud > quiet * 2.0);

    // No pitch to measure against is not an excuse to invent a number.
    CHECK (autosynth::Roles::noiseShare (clean.data(), samples, kSampleRate, 0.0) == 0.0);
}
