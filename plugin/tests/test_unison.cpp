// Unison detection by beating.
//
// Spectral clustering only sees unison once the voices resolve into separate
// peaks. Below that they merge into one partial and the count came back as one
// -- which is why narrow unison was systematically under-counted, and why
// `unison_detune` sat near the top of the worst-recovered list.
//
// Two voices d cents apart still beat at their difference frequency, and at
// harmonic k that rate is k * f0 * (2^(d/1200) - 1). The factor of k is the
// whole discriminator: a beat rate *grows with harmonic number*, and tremolo,
// vibrato and a decaying envelope do not. Half of what follows is checking that
// it fires on unison; the other half is checking that it stays silent on
// everything else, because a detector that answers yes to vibrato is worse than
// no detector at all.

#include "Helpers.h"

#include "analysis/Grouping.h"
#include "analysis/Partials.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autotest;
using namespace autosynth;

namespace
{

constexpr int kHop = 256;
constexpr int kFft = 2048;

std::vector<HarmonicGroup> groupsOf (const std::vector<float>& x)
{
    PartialTracker::Options options;
    options.fftSize = kFft;
    options.hop = kHop;
    const auto set = PartialTracker::track (x.data(), (int) x.size(), kSampleRate, options);
    return Grouping::group (set, 3);
}

// A sustained saw, long enough that a slow beat completes several cycles.
Patch unisonPatch (int voices, float detuneCents, double duration = 2.0)
{
    juce::ignoreUnused (duration);
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
    patch.oscs[0].unisonVoices = voices;
    patch.oscs[0].unisonDetune = detuneCents;
    return patch;
}

} // namespace

// Hidden; run with `autosynth_tests "[.report]"`. Prints what the detector
// actually saw, which is the only way to set thresholds from evidence rather
// than by adjusting them until the suite goes green.
TEST_CASE ("report unison beating", "[.report]")
{
    struct Case { const char* name; int voices; float detune; };
    for (auto c : { Case { "single", 1, 0.0f },
                    Case { "detune 8", 2, 8.0f },
                    Case { "detune 12", 2, 12.0f },
                    Case { "detune 25", 2, 25.0f },
                    Case { "detune 40", 2, 40.0f },
                    Case { "3 voices 20", 3, 20.0f } })
    {
        const auto groups = groupsOf (render (unisonPatch (c.voices, c.detune), 220.0, 2.0, 2.0));
        if (groups.empty())
            continue;

        const auto b = Grouping::detectUnisonBeating (groups[0]);
        int voices = 1;
        float detune = 0.0f;
        Grouping::estimateUnison (groups[0], voices, detune);

        std::printf ("%-14s found=%d detune=%6.2f (true %5.1f) fit=%7.3f "
                     "rate0=%6.3f harmonics=%3d  -> voices=%d detune=%5.1f\n",
                     c.name, (int) b.found, b.detuneCents, c.detune, b.proportionalFit,
                     b.beatRateAtFundamental, b.harmonicsAgreeing, voices, detune);
    }
}

TEST_CASE ("a group carries its own frame rate", "[unison]")
{
    // The estimator reads rates out of the harmonic envelopes, and a rate is
    // meaningless without knowing how fast the frames go by.
    const auto groups = groupsOf (render (unisonPatch (1, 0.0f), 220.0, 2.0, 2.0));
    REQUIRE_FALSE (groups.empty());
    CHECK (groups[0].frameRateHz == Catch::Approx (kSampleRate / kHop));
}

TEST_CASE ("narrow unison is detected by beating", "[unison]")
{
    // 12 cents at 220 Hz beats at 1.5 Hz on the fundamental and 15 Hz by the
    // tenth harmonic -- far too narrow to resolve as two peaks, and plainly
    // periodic in the envelopes.
    const auto audio = render (unisonPatch (2, 12.0f), 220.0, 2.0, 2.0);
    const auto groups = groupsOf (audio);
    REQUIRE_FALSE (groups.empty());

    const auto beating = Grouping::detectUnisonBeating (groups[0]);
    INFO ("detune " << beating.detuneCents << " fit " << beating.proportionalFit
          << " harmonics " << beating.harmonicsAgreeing);
    CHECK (beating.found);
    CHECK (beating.detuneCents == Catch::Approx (12.0).margin (6.0));
}

TEST_CASE ("beating detection reports more voices than clustering alone", "[unison]")
{
    const auto audio = render (unisonPatch (2, 12.0f), 220.0, 2.0, 2.0);
    const auto groups = groupsOf (audio);
    REQUIRE_FALSE (groups.empty());

    int voices = 1;
    float detune = 0.0f;
    Grouping::estimateUnison (groups[0], voices, detune);

    INFO ("voices " << voices << " detune " << detune);
    CHECK (voices >= 2);
    CHECK (detune > 3.0f);
}

TEST_CASE ("a single voice does not beat", "[unison]")
{
    const auto groups = groupsOf (render (unisonPatch (1, 0.0f), 220.0, 2.0, 2.0));
    REQUIRE_FALSE (groups.empty());
    CHECK_FALSE (Grouping::detectUnisonBeating (groups[0]).found);

    int voices = 1;
    float detune = 0.0f;
    Grouping::estimateUnison (groups[0], voices, detune);
    CHECK (voices == 1);
}

TEST_CASE ("tremolo is not mistaken for unison", "[unison]")
{
    // The false positive that matters. An amplitude LFO modulates every
    // harmonic at the *same* rate, so the rates fit a horizontal line, not a
    // proportional one. Without the k-proportionality test this signal would
    // read as unison, and the golden fixtures contain one just like it.
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
    patch.lfos[0] = { LfoShape::sine, LfoDest::amp, 5.0f, 0.8f, 0.0f, 0.0f };

    const auto groups = groupsOf (render (patch, 220.0, 2.0, 2.0));
    REQUIRE_FALSE (groups.empty());

    const auto beating = Grouping::detectUnisonBeating (groups[0]);
    INFO ("fit " << beating.proportionalFit << " detune " << beating.detuneCents);
    CHECK_FALSE (beating.found);
}

TEST_CASE ("vibrato is not mistaken for unison", "[unison]")
{
    // Pitch modulation wobbles the harmonic amplitudes as partials move through
    // the analysis bins -- periodic, and again at one rate for all harmonics.
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
    patch.lfos[0] = { LfoShape::sine, LfoDest::pitch, 5.0f, 0.6f, 0.0f, 0.0f };

    const auto groups = groupsOf (render (patch, 220.0, 2.0, 2.0));
    REQUIRE_FALSE (groups.empty());
    CHECK_FALSE (Grouping::detectUnisonBeating (groups[0]).found);
}

TEST_CASE ("a plucked decay is not mistaken for unison", "[unison]")
{
    // Not periodic at all, but it is a large amplitude change, and dividing by
    // a smoothed envelope rather than subtracting is what keeps it from
    // dominating the fluctuation signal.
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.001f, 0.6f, 0.0f, 0.1f, 1.0f };

    const auto groups = groupsOf (render (patch, 220.0, 2.0, 1.8));
    REQUIRE_FALSE (groups.empty());
    CHECK_FALSE (Grouping::detectUnisonBeating (groups[0]).found);
}

TEST_CASE ("wider detune gives a higher beat rate", "[unison]")
{
    // The relationship the estimate is inverted from. If this ordering ever
    // broke, the detune numbers would be arbitrary even when detection worked.
    const auto rateFor = [] (float cents)
    {
        const auto groups = groupsOf (render (unisonPatch (2, cents), 220.0, 2.0, 2.0));
        REQUIRE_FALSE (groups.empty());
        return Grouping::detectUnisonBeating (groups[0]).beatRateAtFundamental;
    };

    const auto narrow = rateFor (8.0f);
    const auto wide = rateFor (25.0f);
    INFO ("narrow " << narrow << " wide " << wide);
    CHECK (wide > narrow);
}
