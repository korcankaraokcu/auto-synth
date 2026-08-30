// The ground-truth recovery harness itself.
//
// These are cheap guards on the measurement, not the measurement. The real
// numbers come from `autosynth_vital --eval`, which is slow enough that running
// it in the test suite would make the suite useless. What is asserted here is that
// the harness still measures something: that the fitter beats the control, and
// that the scoring cannot silently degrade into comparing a signal with itself.

#include "Helpers.h"

#include "eval/Recovery.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autotest;
using namespace autosynth;

TEST_CASE ("a random patch is audible and varied", "[recovery]")
{
    juce::Random rng (7);
    int audible = 0;
    int distinctWaveforms = 0;
    Waveform first = Waveform::sine;

    for (int i = 0; i < 12; ++i)
    {
        const auto patch = Recovery::randomPatch (rng);
        REQUIRE (patch.activeOscCount() >= 1);
        REQUIRE (patch.rootHz > 100.0f);
        REQUIRE (patch.rootHz < 450.0f);
        // Noise is excluded from targets deliberately: the fitter can recover a
        // noise *level* but never a realisation, so including it would measure
        // the noise rather than the fit.
        REQUIRE (patch.noiseLevel == Catch::Approx (0.0f));

        if (peakOf (render (patch, patch.rootHz)) > 0.05f)
            ++audible;

        if (i == 0)
            first = patch.oscs[0].waveform;
        else if (patch.oscs[0].waveform != first)
            ++distinctWaveforms;
    }

    // Most should be audible; the harness resamples the rest rather than
    // scoring silence.
    CHECK (audible >= 8);
    // Discrete parameters must actually vary, or the harness would only ever
    // pose one kind of question.
    CHECK (distinctWaveforms > 0);
}

TEST_CASE ("random patch generation is reproducible", "[recovery]")
{
    // A harness whose targets moved between runs could not be used to compare
    // two versions of the fitter, which is the only thing it is for.
    juce::Random a (3), b (3);
    for (int i = 0; i < 4; ++i)
    {
        const auto pa = Recovery::randomPatch (a);
        const auto pb = Recovery::randomPatch (b);
        CHECK (pa.toJson() == pb.toJson());
    }
}

TEST_CASE ("scoring a signal against itself is zero", "[recovery]")
{
    const auto audio = render (simplePatch (Waveform::saw));
    const auto s = Recovery::score (audio, audio, kSampleRate);

    CHECK (s.spectral == Catch::Approx (0.0).margin (1.0e-9));
    CHECK (s.loudnessDb == Catch::Approx (0.0).margin (1.0e-9));
    CHECK (s.centroidOctaves == Catch::Approx (0.0).margin (1.0e-9));
}

TEST_CASE ("scoring separates a close match from a distant one", "[recovery]")
{
    const auto target = render (simplePatch (Waveform::saw));

    auto nearPatch = simplePatch (Waveform::saw);
    nearPatch.oscs[0].cents = 4.0f;
    const auto near = Recovery::score (render (nearPatch), target, kSampleRate);

    auto farPatch = simplePatch (Waveform::sine);
    farPatch.rootHz = 440.0f;
    const auto far = Recovery::score (render (farPatch, 440.0), target, kSampleRate);

    CHECK (near.spectral < far.spectral);
    CHECK (near.centroidOctaves < far.centroidOctaves);
}

TEST_CASE ("the fitter beats the control", "[recovery][.slow]")
{
    // The headline claim, at the smallest trial count that still means
    // anything. Tagged [.slow] and skipped by default: even unrefined this is
    // several seconds, and refinement costs about eight seconds per second of
    // audio.
    //
    // A fitter that cannot beat an untouched default patch has learned nothing,
    // and no absolute distance says whether it did.
    Recovery::Options options;
    options.trials = 6;
    options.refine = false;
    options.renderer = renderer();
    const auto summary = Recovery::run (options);

    REQUIRE (summary.trials == 6);
    CHECK (summary.fit.spectral < summary.control.spectral);
    CHECK (summary.fit.loudnessDb < summary.control.loudnessDb);
    CHECK (summary.fit.centroidOctaves < summary.control.centroidOctaves);
}
