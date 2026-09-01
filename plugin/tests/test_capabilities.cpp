// What the patch format can express and the preset can carry: a second LFO
// slot, wavetable morph, and the scope refinement is allowed to search.
//
// Each is checked the same way -- against the patch with that one feature
// switched off -- because the failure mode these guard against is not a wrong
// number but a control that does nothing at all. Three of them were, in fact,
// silently doing nothing, and every test in the suite still passed.
//
// The per-oscillator filter and reverb send used to be checked here too. They
// are gone from the patch format: a Vital preset has two filters and a routing
// choice rather than one filter per oscillator, and no per-oscillator send
// level at all, so both were fields that could be set and never heard.

#include "Helpers.h"

#include "fit/Refine.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autotest;
using namespace autosynth;

// --- second LFO slot -------------------------------------------------------

TEST_CASE ("two LFO slots reach two destinations at once", "[capabilities]")
{
    auto base = simplePatch (Waveform::saw);
    base.filter.type = FilterType::lowpass;
    base.filter.cutoffHz = 1500.0f;

    auto one = base;
    one.lfos[0] = { LfoShape::sine, LfoDest::pitch, 5.0f, 0.5f, 0.0f, 0.0f };

    auto two = one;
    two.lfos[1] = { LfoShape::sine, LfoDest::cutoff, 2.0f, 0.6f, 0.0f, 0.0f };

    CHECK (centroidDistanceOctaves (render (one), render (two)) > 0.02);
}

TEST_CASE ("both LFO slots can target the same destination", "[capabilities]")
{
    // Summing rather than replacing is what makes the second slot general:
    // two rates on pitch give a compound wobble one slot cannot express.
    auto one = simplePatch (Waveform::saw);
    one.lfos[0] = { LfoShape::sine, LfoDest::amp, 5.0f, 0.4f, 0.0f, 0.0f };

    auto two = one;
    two.lfos[1] = { LfoShape::sine, LfoDest::amp, 1.7f, 0.4f, 0.0f, 0.0f };

    CHECK (loudnessDistanceDb (render (one), render (two)) > 0.5);
}

TEST_CASE ("an LFO with zero depth is an exact no-op", "[capabilities]")
{
    auto patch = simplePatch (Waveform::saw);
    auto configured = patch;
    configured.lfos[1] = { LfoShape::square, LfoDest::pitch, 7.0f, 0.0f, 0.0f, 0.0f };

    const auto a = render (patch);
    const auto b = render (configured);
    REQUIRE (a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        REQUIRE (a[i] == Catch::Approx (b[i]).margin (kRenderFloor));
}

// --- wavetable morph -------------------------------------------------------

TEST_CASE ("morph at zero is exactly the first waveform", "[capabilities]")
{
    auto plain = simplePatch (Waveform::saw);
    auto configured = plain;
    configured.oscs[0].waveformB = Waveform::square;
    configured.oscs[0].waveMorph = 0.0f;

    const auto a = render (plain);
    const auto b = render (configured);
    REQUIRE (a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        REQUIRE (a[i] == Catch::Approx (b[i]).margin (kRenderFloor));
}

TEST_CASE ("morph at one is the second waveform", "[capabilities]")
{
    auto morphed = simplePatch (Waveform::saw);
    morphed.oscs[0].waveformB = Waveform::square;
    morphed.oscs[0].waveMorph = 1.0f;

    const auto plainSquare = render (simplePatch (Waveform::square));
    CHECK (loudnessDistanceDb (render (morphed), plainSquare) < 0.5);
}

TEST_CASE ("morph is continuous between the two", "[capabilities]")
{
    const auto brightnessAt = [] (float morph)
    {
        auto p = simplePatch (Waveform::sine);
        p.oscs[0].waveformB = Waveform::saw;
        p.oscs[0].waveMorph = morph;
        return meanCentroidHz (render (p));
    };

    const auto low = brightnessAt (0.0f);
    const auto mid = brightnessAt (0.5f);
    const auto high = brightnessAt (1.0f);

    CHECK (low < mid);
    CHECK (mid < high);
}

// --- search scope ----------------------------------------------------------

TEST_CASE ("refinement searches only what the deliverable can carry", "[capabilities]")
{
    auto patch = simplePatch (Waveform::saw);
    patch.oscs[1].enabled = true;
    patch.oscs[1].level = 0.5f;

    const auto scope = Refine::scopeFor (patch);
    const auto has = [&scope] (const std::string& needle)
    {
        for (const auto& p : scope)
            if (p == needle)
                return true;
        return false;
    };

    // Enabled oscillators are searched; disabled ones contribute nothing, so
    // searching them is wasted dimensionality.
    CHECK (has ("oscs.0.level"));
    CHECK (has ("oscs.1.level"));
    CHECK_FALSE (has ("oscs.2.level"));

    // Nothing the preset cannot carry, stated as names rather than as a count
    // so a parameter reintroduced by accident is named in the failure.
    CHECK_FALSE (has ("oscs.0.reverb_send"));
    CHECK_FALSE (has ("oscs.0.filter.cutoff_hz"));
    CHECK_FALSE (has ("oscs.0.frame_position"));

    // And an oscillator envelope's sustain, which is structure rather than a
    // value to polish: one for a body that has to stay, zero for a transient
    // that has to go. Searched, it turned the body's placeholder decay into a
    // cliff -- sustain pulled to 0.68 with the decay left at ten milliseconds,
    // three decibels gone in a tenth of a blink. A listener reported that
    // before any axis here did, which is the second time an oscillator envelope
    // has been given freedom it did not need and spent it badly.
    auto shaped = simplePatch (Waveform::saw);
    shaped.oscs[0].envEnabled = true;
    const auto envScope = Refine::scopeFor (shaped);
    const auto hasEnv = [&envScope] (const std::string& needle)
    {
        for (const auto& p : envScope)
            if (p == needle)
                return true;
        return false;
    };
    CHECK (hasEnv ("oscs.0.env.attack"));
    CHECK (hasEnv ("oscs.0.env.decay"));
    CHECK_FALSE (hasEnv ("oscs.0.env.sustain"));
}
