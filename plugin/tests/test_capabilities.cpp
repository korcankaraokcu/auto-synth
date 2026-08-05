// The oscillator capabilities added after the first working plugin: a filter
// per oscillator, a second LFO slot, wavetable morph, and a per-oscillator
// reverb send.
//
// Each is checked the same way -- against the patch with that one feature
// switched off -- because the failure mode these guard against is not a wrong
// number but a control that does nothing at all. Three of them were, in fact,
// silently doing nothing, and every test in the suite still passed.

#include "Helpers.h"

#include "fit/Refine.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autotest;
using namespace autosynth;

// --- per-oscillator filter -------------------------------------------------

TEST_CASE ("a per-oscillator filter changes only its own oscillator", "[capabilities]")
{
    auto base = simplePatch (Waveform::saw);
    base.oscs[1].enabled = true;
    base.oscs[1].level = 1.0f;
    base.oscs[1].semitones = 12;

    auto shaped = base;
    shaped.oscs[0].filterEnabled = true;
    shaped.oscs[0].filter.type = FilterType::lowpass;
    shaped.oscs[0].filter.cutoffHz = 300.0f;

    const auto plain = render (base);
    const auto out = render (shaped);

    CHECK (meanCentroidHz (out) < meanCentroidHz (plain));

    // The octave above must survive: a global filter would have taken it too.
    CHECK (bandEnergy (out, 400.0, 500.0) > 0.4 * bandEnergy (plain, 400.0, 500.0));
}

TEST_CASE ("a disabled per-oscillator filter is an exact no-op", "[capabilities]")
{
    auto patch = simplePatch (Waveform::saw);
    auto configured = patch;
    configured.oscs[0].filterEnabled = false;
    configured.oscs[0].filter.type = FilterType::lowpass;
    configured.oscs[0].filter.cutoffHz = 200.0f;

    const auto a = render (patch);
    const auto b = render (configured);
    REQUIRE (a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        REQUIRE (a[i] == Catch::Approx (b[i]));
}

TEST_CASE ("a per-oscillator filter type of off is an exact no-op", "[capabilities]")
{
    auto patch = simplePatch (Waveform::saw);
    auto configured = patch;
    configured.oscs[0].filterEnabled = true;
    configured.oscs[0].filter.type = FilterType::off;

    const auto a = render (patch);
    const auto b = render (configured);
    REQUIRE (a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        REQUIRE (a[i] == Catch::Approx (b[i]));
}

TEST_CASE ("per-oscillator filters can differ between oscillators", "[capabilities]")
{
    // The whole reason the filter is per-oscillator: two layers with different
    // tone. If the implementation shared one filter instance between them, the
    // second would inherit the first's state and this would collapse.
    auto both = simplePatch (Waveform::saw);
    both.oscs[1].enabled = true;
    both.oscs[1].level = 1.0f;
    both.oscs[1].semitones = 12;

    both.oscs[0].filterEnabled = true;
    both.oscs[0].filter.type = FilterType::lowpass;
    both.oscs[0].filter.cutoffHz = 400.0f;

    both.oscs[1].filterEnabled = true;
    both.oscs[1].filter.type = FilterType::highpass;
    both.oscs[1].filter.cutoffHz = 2000.0f;

    auto swapped = both;
    std::swap (swapped.oscs[0].filter, swapped.oscs[1].filter);

    CHECK (centroidDistanceOctaves (render (both), render (swapped)) > 0.05);
}

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
        REQUIRE (a[i] == Catch::Approx (b[i]));
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
        REQUIRE (a[i] == Catch::Approx (b[i]));
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

// --- reverb send -----------------------------------------------------------

TEST_CASE ("the reverb send is per-oscillator", "[capabilities]")
{
    // One layer drenched, another dry -- the reason to want reverb "on an
    // oscillator" at all. A single shared send would make these identical.
    auto base = simplePatch (Waveform::saw);
    base.ampEnv = { 0.001f, 0.06f, 0.0f, 0.01f, 0.0f };
    base.oscs[1].enabled = true;
    base.oscs[1].level = 1.0f;
    base.oscs[1].semitones = 12;
    base.reverb = { true, 0.8f, 0.3f, 0.8f };

    auto firstWet = base;
    firstWet.oscs[0].reverbSend = 1.0f;
    firstWet.oscs[1].reverbSend = 0.0f;

    auto secondWet = base;
    secondWet.oscs[0].reverbSend = 0.0f;
    secondWet.oscs[1].reverbSend = 1.0f;

    // The tails carry different material, so they must sound different.
    CHECK (centroidDistanceOctaves (render (firstWet), render (secondWet)) > 0.05);
}

TEST_CASE ("a disabled reverb is an exact no-op regardless of send", "[capabilities]")
{
    auto patch = simplePatch (Waveform::saw);
    patch.oscs[0].reverbSend = 1.0f;

    auto configured = patch;
    configured.reverb = { false, 0.9f, 0.5f, 1.0f };

    const auto a = render (patch);
    const auto b = render (configured);
    REQUIRE (a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        REQUIRE (a[i] == Catch::Approx (b[i]));
}

TEST_CASE ("reverb level acts as a return gain, not a dry/wet balance", "[capabilities]")
{
    // The dry path must not lose anything when the send is raised: the mix is
    // already whatever the sends did not take. If `level` crossfaded instead,
    // raising it would thin the direct sound.
    auto dry = simplePatch (Waveform::saw);
    dry.ampEnv = { 0.001f, 0.06f, 0.0f, 0.01f, 0.0f };

    auto wet = dry;
    wet.reverb = { true, 0.8f, 0.3f, 0.8f };
    wet.oscs[0].reverbSend = 1.0f;

    const auto dryOut = render (dry);
    const auto wetOut = render (wet);

    // Early in the note, before any tail has built up, the two must agree
    // closely -- the reverb adds, it does not subtract.
    const auto head = (size_t) (0.02 * kSampleRate);
    double dryHead = 0.0, wetHead = 0.0;
    for (size_t i = 0; i < head && i < dryOut.size(); ++i)
    {
        dryHead += std::abs (dryOut[i]);
        wetHead += std::abs (wetOut[i]);
    }
    CHECK (wetHead > 0.8 * dryHead);
}

TEST_CASE ("the reverb send is in refinement scope for every enabled oscillator", "[capabilities]")
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

    CHECK (has ("oscs.0.reverb_send"));
    CHECK (has ("oscs.1.reverb_send"));
    // Disabled oscillators contribute nothing, so searching them is wasted
    // dimensionality.
    CHECK_FALSE (has ("oscs.2.reverb_send"));
}
