// The IR is the contract: analysis produces one, the engine renders one,
// exporters translate one. If it does not survive a round trip through JSON
// then nothing built on it can be trusted, so this is checked first.

#include "Helpers.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autotest;
using namespace autosynth;

TEST_CASE ("a default patch is renderable", "[ir]")
{
    const Patch patch;
    CHECK (patch.oscs.size() == (size_t) kNumOsc);
    CHECK (patch.lfos.size() == (size_t) kNumLfo);
    CHECK (patch.activeOscCount() >= 1);
    CHECK (patch.rootHz > 0.0f);
}

TEST_CASE ("active oscillator count follows enabled and level", "[ir]")
{
    auto patch = simplePatch();
    CHECK (patch.activeOscCount() == 1);

    patch.oscs[1].enabled = true;
    patch.oscs[1].level = 0.5f;
    CHECK (patch.activeOscCount() == 2);

    // Enabled but silent does not count. Refinement can drive a level to zero,
    // and a patch that still claimed the oscillator would misreport itself.
    patch.oscs[1].level = 0.0f;
    CHECK (patch.activeOscCount() == 1);
}

TEST_CASE ("a patch round-trips through JSON", "[ir]")
{
    auto patch = simplePatch (Waveform::square);
    patch.oscs[0].semitones = -12;
    patch.oscs[0].cents = 7.5f;
    patch.oscs[0].unisonVoices = 5;
    patch.oscs[0].unisonDetune = 22.0f;
    patch.oscs[0].waveformB = Waveform::triangle;
    patch.oscs[0].waveMorph = 0.35f;
    patch.oscs[0].reverbSend = 0.6f;
    patch.oscs[0].filterEnabled = true;
    patch.oscs[0].filter.type = FilterType::bandpass;
    patch.oscs[0].filter.cutoffHz = 900.0f;
    patch.ampEnv = { 0.02f, 0.3f, 0.4f, 0.15f, 0.8f };
    patch.filter.type = FilterType::highpass;
    patch.filter.cutoffHz = 1234.0f;
    patch.filter.envAmount = 1.5f;
    patch.lfos[0] = { LfoShape::square, LfoDest::pitch, 4.25f, 0.3f, 0.1f, 0.25f };
    patch.lfos[1] = { LfoShape::triangle, LfoDest::amp, 2.5f, 0.4f, 0.0f, 0.0f };
    patch.delay = { true, 0.31f, 0.42f, 0.53f };
    patch.reverb = { true, 0.65f, 0.35f, 0.75f };
    patch.noiseLevel = 0.12f;
    patch.masterLevel = 0.66f;
    patch.rootHz = 261.63f;
    patch.name = "round trip";

    juce::String error;
    const auto back = Patch::fromJsonString (patch.toJson(), &error);
    REQUIRE (error.isEmpty());

    CHECK (back.name == patch.name);
    CHECK (back.rootHz == Catch::Approx (patch.rootHz));
    CHECK (back.noiseLevel == Catch::Approx (patch.noiseLevel));
    CHECK (back.masterLevel == Catch::Approx (patch.masterLevel));

    CHECK (back.oscs[0].waveform == patch.oscs[0].waveform);
    CHECK (back.oscs[0].semitones == patch.oscs[0].semitones);
    CHECK (back.oscs[0].cents == Catch::Approx (patch.oscs[0].cents));
    CHECK (back.oscs[0].unisonVoices == patch.oscs[0].unisonVoices);
    CHECK (back.oscs[0].waveformB == patch.oscs[0].waveformB);
    CHECK (back.oscs[0].waveMorph == Catch::Approx (patch.oscs[0].waveMorph));
    CHECK (back.oscs[0].reverbSend == Catch::Approx (patch.oscs[0].reverbSend));
    CHECK (back.oscs[0].filterEnabled == patch.oscs[0].filterEnabled);
    CHECK (back.oscs[0].filter.type == patch.oscs[0].filter.type);
    CHECK (back.oscs[0].filter.cutoffHz == Catch::Approx (patch.oscs[0].filter.cutoffHz));

    CHECK (back.ampEnv.curve == Catch::Approx (patch.ampEnv.curve));
    CHECK (back.filter.type == patch.filter.type);
    CHECK (back.filter.envAmount == Catch::Approx (patch.filter.envAmount));

    for (int i = 0; i < kNumLfo; ++i)
    {
        INFO ("lfo " << i);
        CHECK (back.lfos[(size_t) i].shape == patch.lfos[(size_t) i].shape);
        CHECK (back.lfos[(size_t) i].dest == patch.lfos[(size_t) i].dest);
        CHECK (back.lfos[(size_t) i].rateHz == Catch::Approx (patch.lfos[(size_t) i].rateHz));
        CHECK (back.lfos[(size_t) i].depth == Catch::Approx (patch.lfos[(size_t) i].depth));
        CHECK (back.lfos[(size_t) i].phase == Catch::Approx (patch.lfos[(size_t) i].phase));
    }

    CHECK (back.delay.enabled == patch.delay.enabled);
    CHECK (back.delay.time == Catch::Approx (patch.delay.time));
    CHECK (back.delay.feedback == Catch::Approx (patch.delay.feedback));
    CHECK (back.delay.mix == Catch::Approx (patch.delay.mix));

    CHECK (back.reverb.enabled == patch.reverb.enabled);
    CHECK (back.reverb.size == Catch::Approx (patch.reverb.size));
    CHECK (back.reverb.damp == Catch::Approx (patch.reverb.damp));
    CHECK (back.reverb.level == Catch::Approx (patch.reverb.level));
}

TEST_CASE ("a round-tripped patch renders identically", "[ir]")
{
    // The stronger statement. Field equality can miss something the renderer
    // reads but the comparison above forgot to check.
    auto patch = simplePatch (Waveform::saw);
    patch.reverb = { true, 0.7f, 0.4f, 0.8f };
    patch.oscs[0].reverbSend = 0.9f;
    patch.delay = { true, 0.2f, 0.4f, 0.5f };
    patch.lfos[0] = { LfoShape::sine, LfoDest::pitch, 5.0f, 0.3f, 0.0f, 0.0f };

    juce::String error;
    const auto back = Patch::fromJsonString (patch.toJson(), &error);
    REQUIRE (error.isEmpty());

    const auto a = render (patch);
    const auto b = render (back);
    REQUIRE (a.size() == b.size());

    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = juce::jmax (worst, (double) std::abs (a[i] - b[i]));
    CHECK (worst < 1.0e-6);
}

TEST_CASE ("enum names survive as strings, not indices", "[ir]")
{
    // Serialising enums positionally would make every saved patch depend on
    // declaration order, so inserting a waveform would silently rewrite files
    // already on disk.
    auto patch = simplePatch (Waveform::pulse);
    patch.filter.type = FilterType::bandpass;
    patch.lfos[0].shape = LfoShape::saw;
    patch.lfos[0].dest = LfoDest::cutoff;

    const auto json = patch.toJson();
    CHECK (json.contains ("pulse"));
    CHECK (json.contains ("bandpass"));
    CHECK (json.contains ("saw"));
    CHECK (json.contains ("cutoff"));
}

TEST_CASE ("malformed JSON is reported, not silently accepted", "[ir]")
{
    juce::String error;
    Patch::fromJsonString ("{ this is not json", &error);
    CHECK (error.isNotEmpty());
}

TEST_CASE ("the legacy single-lfo form still loads", "[ir]")
{
    // Patches written before the second slot existed are still valid files.
    // A reader that rejected them would strand every patch anyone had saved.
    const juce::String legacy = R"({
        "name": "legacy",
        "root_hz": 220.0,
        "noise_level": 0.0,
        "master_level": 0.8,
        "oscs": [{"enabled": true, "waveform": "saw", "level": 1.0}],
        "amp_env": {"attack": 0.01, "decay": 0.2, "sustain": 0.7, "release": 0.2},
        "filter": {"type": "off"},
        "lfo": {"shape": "sine", "dest": "pitch", "rate_hz": 6.0, "depth": 0.25}
    })";

    juce::String error;
    const auto patch = Patch::fromJsonString (legacy, &error);
    REQUIRE (error.isEmpty());
    CHECK (patch.lfos[0].dest == LfoDest::pitch);
    CHECK (patch.lfos[0].rateHz == Catch::Approx (6.0f));
    CHECK (patch.lfos[0].depth == Catch::Approx (0.25f));
    // The slot that did not exist in the old format must come back inert.
    CHECK (patch.lfos[1].dest == LfoDest::none);
}
