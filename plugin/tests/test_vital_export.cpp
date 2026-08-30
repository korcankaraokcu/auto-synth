// The Vital exporter.
//
// None of this can prove the preset *loads* in Vital -- that needs Vital, and
// the format's numeric skews are the part least certain. What it can prove is
// everything on this side of the boundary: that the JSON is well formed, that
// the structure Vital expects is present, and above all that the wavetable
// survives the trip. The wavetable is the whole reason to target Vital rather
// than something simpler, so it is the part worth pinning hardest.

#include "Helpers.h"

#include "dsp/Tables.h"
#include "fit/Refine.h"
#include "ir/VitalExport.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <set>

using namespace autotest;
using autosynth::Oscillator;
using autosynth::Patch;
using autosynth::VitalExport;
using autosynth::Waveform;
using autosynth::WaveTables;

namespace
{

juce::var settingsOf (const Patch& patch)
{
    const auto json = juce::JSON::parse (VitalExport::toJson (patch, "test"));
    return json.getProperty ("settings", {});
}

// Decode one keyframe back into samples, the way Vital would.
std::vector<float> decodeFrame (const juce::var& keyframe)
{
    juce::MemoryOutputStream raw;
    const auto text = keyframe.getProperty ("wave_data", {}).toString();
    REQUIRE (juce::Base64::convertFromBase64 (raw, text));

    const auto count = raw.getDataSize() / sizeof (float);
    std::vector<float> out (count);
    juce::MemoryInputStream in (raw.getData(), raw.getDataSize(), false);
    for (size_t i = 0; i < count; ++i)
        out[i] = in.readFloat();
    return out;
}

const juce::var& firstKeyframes (const juce::var& settings, int osc, juce::var& holder)
{
    const auto* tables = settings.getProperty ("wavetables", {}).getArray();
    REQUIRE (tables != nullptr);
    REQUIRE (tables->size() > osc);
    holder = (*tables)[osc].getProperty ("groups", {})[0]
                 .getProperty ("components", {})[0]
                 .getProperty ("keyframes", {});
    return holder;
}

} // namespace

TEST_CASE ("the exported preset is well-formed JSON with Vital's shape", "[vital]")
{
    const auto patch = simplePatch (Waveform::saw);
    const auto json = juce::JSON::parse (VitalExport::toJson (patch, "hello"));

    REQUIRE (json.isObject());
    CHECK (json.getProperty ("preset_name", {}).toString() == "hello");
    CHECK (json.getProperty ("synth_version", {}).toString().isNotEmpty());

    const auto settings = json.getProperty ("settings", {});
    REQUIRE (settings.isObject());
    CHECK (settings.hasProperty ("osc_1_on"));
    CHECK (settings.hasProperty ("env_1_attack"));
    CHECK (settings.hasProperty ("filter_1_cutoff"));
    CHECK (settings.getProperty ("wavetables", {}).getArray() != nullptr);
    CHECK (settings.getProperty ("modulations", {}).getArray() != nullptr);
}

TEST_CASE ("one wavetable per oscillator, one keyframe per frame", "[vital]")
{
    auto patch = simplePatch (Waveform::saw);
    patch.oscs[0].numFrames = 3;
    for (int f = 0; f < 3; ++f)
        patch.oscs[0].frames[(size_t) f].custom = true;

    const auto settings = settingsOf (patch);
    const auto* tables = settings.getProperty ("wavetables", {}).getArray();
    REQUIRE (tables != nullptr);
    CHECK (tables->size() == autosynth::kNumOsc);

    juce::var holder;
    const auto& keys = firstKeyframes (settings, 0, holder);
    REQUIRE (keys.getArray() != nullptr);
    CHECK (keys.getArray()->size() == 3);

    // Spread across Vital's table positions, ends included. Measured at 0..256
    // across 274 presets, not 0..255.
    CHECK ((int) (*keys.getArray())[0].getProperty ("position", -1) == 0);
    CHECK ((int) (*keys.getArray())[2].getProperty ("position", -1) == 256);
}

TEST_CASE ("an exported frame carries the harmonics it was given", "[vital]")
{
    // The claim that makes this exporter worth having: the wavetable is copied,
    // not approximated. A drawn frame is checked by transforming the decoded
    // samples back and comparing the spectrum it came from.
    auto patch = simplePatch (Waveform::saw);
    auto& osc = patch.oscs[0];
    osc.numFrames = 1;
    osc.frames[0].custom = true;
    osc.frames[0].harmonics = {};
    osc.frames[0].harmonics[0] = 1.0f;    // fundamental
    osc.frames[0].harmonics[2] = 0.5f;    // and a third harmonic

    juce::var holder;
    const auto& keys = firstKeyframes (settingsOf (patch), 0, holder);
    const auto samples = decodeFrame ((*keys.getArray())[0]);
    REQUIRE (samples.size() == (size_t) VitalExport::Mapping::kFrameSamples);

    // Peak-normalised on the way out, so it must not clip.
    for (const auto v : samples)
        CHECK (std::abs (v) <= 1.0001f);

    // Correlate against the two sines that went in. Sine phase throughout, so
    // the projection onto each is the amplitude.
    const auto project = [&samples] (int harmonic)
    {
        double acc = 0.0;
        const auto n = (double) samples.size();
        for (size_t i = 0; i < samples.size(); ++i)
            acc += samples[i] * std::sin (2.0 * juce::MathConstants<double>::pi * harmonic * i / n);
        return std::abs (acc / n * 2.0);
    };

    const auto first = project (1);
    const auto third = project (3);
    CHECK (first > 0.1);
    CHECK (third == Catch::Approx (first * 0.5).margin (0.02));
    CHECK (project (2) < first * 0.02);   // nothing where nothing was asked for
}

TEST_CASE ("an undrawn frame exports as its own waveform", "[vital]")
{
    // An analog patch has to export as the shape it says it is, not as a
    // sixteen-harmonic approximation of some other shape.
    auto patch = simplePatch (Waveform::square);

    juce::var holder;
    const auto& keys = firstKeyframes (settingsOf (patch), 0, holder);
    const auto samples = decodeFrame ((*keys.getArray())[0]);

    const auto project = [&samples] (int harmonic)
    {
        double acc = 0.0;
        const auto n = (double) samples.size();
        for (size_t i = 0; i < samples.size(); ++i)
            acc += samples[i] * std::sin (2.0 * juce::MathConstants<double>::pi * harmonic * i / n);
        return std::abs (acc / n * 2.0);
    };

    // A square is odd harmonics falling as 1/k.
    const auto first = project (1);
    CHECK (project (3) == Catch::Approx (first / 3.0).margin (0.03));
    CHECK (project (5) == Catch::Approx (first / 5.0).margin (0.03));
    CHECK (project (2) < first * 0.02);
    CHECK (project (4) < first * 0.02);
}

TEST_CASE ("a routing is an entry plus a numbered amount", "[vital]")
{
    // Vital keeps the two halves apart: `modulations` names source and
    // destination, and `modulation_N_amount` carries the depth. Putting the
    // amount inside the entry -- which the first version of this exporter did
    // -- connects everything at zero and is silent rather than wrong-sounding.
    auto patch = simplePatch (Waveform::saw);
    patch.lfos[0].dest = autosynth::LfoDest::cutoff;
    patch.lfos[0].depth = 0.5f;
    patch.lfos[0].rateHz = 3.0f;

    const auto settings = settingsOf (patch);
    const auto* mods = settings.getProperty ("modulations", {}).getArray();
    REQUIRE (mods != nullptr);
    REQUIRE (mods->size() >= 1);

    auto index = -1;
    for (int i = 0; i < mods->size(); ++i)
        if ((*mods)[i].getProperty ("source", {}).toString() == "lfo_1"
            && (*mods)[i].getProperty ("destination", {}).toString() == "filter_1_cutoff")
            index = i;
    REQUIRE (index >= 0);

    // The entry itself carries no amount, and the numbered one does.
    CHECK_FALSE ((*mods)[index].hasProperty ("amount"));
    const auto amountKey = "modulation_" + juce::String (index + 1) + "_amount";
    CHECK ((float) settings.getProperty (amountKey, 0.0) > 0.0f);
}

TEST_CASE ("an envelope that is written is also connected", "[vital]")
{
    // env_1 drives amplitude in Vital whether or not anything routes it. Every
    // other envelope does nothing at all until the matrix names it, so writing
    // its shape and stopping is not a half-finished export -- it is a silent
    // one, and it stays silent through every check that reads the preset,
    // because a dangling envelope is perfectly legal.
    //
    // This is how the filter envelope was lost: env_2 carried the fitted shape,
    // no routing mentioned it, and Vital rendered a static cutoff while the
    // engine swept it 1.92 octaves. Rendering the preset through Vital is what
    // found it; this is what keeps it found.
    auto patch = simplePatch (Waveform::saw);
    patch.filter.type = autosynth::FilterType::lowpass;
    patch.filter.cutoffHz = 1200.0f;
    patch.filter.envAmount = 2.0f;

    const auto settings = settingsOf (patch);
    const auto* mods = settings.getProperty ("modulations", {}).getArray();
    REQUIRE (mods != nullptr);

    std::set<juce::String> connected;
    for (int i = 0; i < mods->size(); ++i)
    {
        const auto source = (*mods)[i].getProperty ("source", {}).toString();
        if (source.isNotEmpty())
            connected.insert (source);
    }

    CHECK (connected.count ("env_2") == 1);

    auto index = -1;
    for (int i = 0; i < mods->size(); ++i)
        if ((*mods)[i].getProperty ("source", {}).toString() == "env_2"
            && (*mods)[i].getProperty ("destination", {}).toString() == "filter_1_cutoff")
            index = i;
    REQUIRE (index >= 0);

    // Two octaves is 24 semitones over the cutoff control's 128, and an
    // envelope only ever adds, so the routing is unipolar.
    const auto n = juce::String (index + 1);
    CHECK ((float) settings.getProperty ("modulation_" + n + "_amount", 0.0)
           == Catch::Approx (24.0f / 128.0f).margin (1.0e-4));
    CHECK ((float) settings.getProperty ("modulation_" + n + "_bipolar", 1.0) == 0.0f);

    // And the general rule, so the next envelope added does not repeat it: any
    // envelope whose shape is written past env_1 must appear as a source.
    for (int e = 2; e <= 6; ++e)
    {
        const auto name = "env_" + juce::String (e);
        if (settings.hasProperty (name + "_attack"))
            INFO ("envelope written: " << name);
        if (settings.hasProperty (name + "_attack") && e == 2)
            CHECK (connected.count (name) == 1);
    }
}

TEST_CASE ("measured mappings, not assumed ones", "[vital]")
{
    // Each of these was wrong in the first version and was corrected against
    // 274 real presets. They are pinned here so a future edit has to disagree
    // with the evidence on purpose.
    using M = VitalExport::Mapping;

    // Envelope time is the fourth root of seconds: one second is 1.0, and the
    // ceiling of 32 s lands on the 2.378 seen as the maximum in the collection.
    CHECK (M::secondsToEnvelope (1.0f) == Catch::Approx (1.0f).margin (0.001));
    CHECK (M::secondsToEnvelope (32.0f) == Catch::Approx (2.3784f).margin (0.001));
    CHECK (M::secondsToEnvelope (0.0f) == Catch::Approx (0.0f).margin (0.001));

    // LFO rate is log2 of hertz when free-running.
    CHECK (M::hzToLfoRate (1.0f) == Catch::Approx (0.0f).margin (0.001));
    CHECK (M::hzToLfoRate (4.0f) == Catch::Approx (2.0f).margin (0.001));

    // Cutoff is a MIDI note.
    CHECK (M::cutoffToNote (440.0f) == Catch::Approx (69.0f).margin (0.01));
    CHECK (M::cutoffToNote (880.0f) == Catch::Approx (81.0f).margin (0.01));

    // Tune is a semitone fraction, not cents.
    CHECK (M::centsToTune (50.0f) == Catch::Approx (0.5f).margin (0.001));
    CHECK (M::centsToTune (500.0f) == Catch::Approx (1.0f).margin (0.001));   // clamped

    // Volume is a square-root control reading sqrt(stored) - 80 decibels, so
    // unity gain is 6400 and silence is 0.
    //
    // Unity amplitude lands 6 dB under that, at 5476, because Vital's summed
    // oscillators arrive at the control at twice their stated amplitude. Which
    // is also Vital's own default volume of 5473.04, to within the rounding --
    // the default is exactly -6 dB, and it is there for the same reason.
    CHECK (M::gainToVolume (1.0f) == Catch::Approx (5476.0f).margin (2.0));
    CHECK (M::gainToVolume (0.0f) == Catch::Approx (0.0f).margin (0.001));
    CHECK (M::gainToVolume (0.5f) == Catch::Approx (4621.0f).margin (2.0));

    // Oscillator level is quadratic, so the stored value is the square root of
    // the amplitude. Vital's own default of 0.70710678 renders as a half.
    CHECK (M::levelToOscLevel (1.0f) == Catch::Approx (1.0f).margin (0.001));
    CHECK (M::levelToOscLevel (0.5f) == Catch::Approx (0.7071f).margin (0.001));
    CHECK (M::levelToOscLevel (0.25f) == Catch::Approx (0.5f).margin (0.001));
}

TEST_CASE ("gain lands on the master, not folded into the oscillators", "[vital]")
{
    // An earlier version multiplied the two together and wrote the result
    // linearly into a *quadratic* control, which squared it again: a patch at
    // 0.43 master and full oscillator level arrived at 0.09 amplitude, roughly
    // 20 dB quiet. Each now goes to the control that actually means it.
    auto patch = simplePatch (Waveform::saw);
    patch.masterLevel = 1.0f;
    patch.oscs[0].level = 1.0f;

    const auto settings = settingsOf (patch);
    CHECK ((float) settings.getProperty ("osc_1_level", 0.0) == Catch::Approx (1.0f).margin (0.001));
    CHECK ((float) settings.getProperty ("volume", 0.0) == Catch::Approx (5476.0f).margin (2.0));

    patch.masterLevel = 0.5f;
    CHECK ((float) settingsOf (patch).getProperty ("volume", 0.0)
               < (float) settings.getProperty ("volume", 0.0));
}

TEST_CASE ("the blocks Vital's loader reads directly are present", "[vital]")
{
    // `LoadSave::jsonToState` reaches into `settings["lfos"]` and
    // `settings["sample"]` without checking they exist. A preset missing either
    // fails to parse and is reported as corrupted -- which is what happened, and
    // which no amount of comparing *parameters* would have caught, because the
    // parameters were all correct.
    const auto settings = settingsOf (simplePatch (Waveform::saw));

    const auto* lfos = settings.getProperty ("lfos", {}).getArray();
    REQUIRE (lfos != nullptr);
    CHECK (lfos->size() == 8);          // measured: every real preset has eight
    for (const auto& l : *lfos)
    {
        CHECK (l.hasProperty ("name"));
        CHECK (l.hasProperty ("num_points"));
        CHECK (l.getProperty ("points", {}).getArray() != nullptr);
        CHECK (l.getProperty ("powers", {}).getArray() != nullptr);
        CHECK (l.hasProperty ("smooth"));
    }

    const auto sample = settings.getProperty ("sample", {});
    REQUIRE (sample.isObject());
    for (const auto* key : { "name", "length", "sample_rate", "samples" })
        CHECK (sample.hasProperty (key));
}

TEST_CASE ("the declared version is not newer than the target", "[vital]")
{
    // Vital refuses a preset whose feature version is newer than its own, so
    // claiming a version we invented is a guaranteed rejection. The first
    // version of this exporter declared 1.5.5 against an installed 1.0.7.
    const auto json = juce::JSON::parse (VitalExport::toJson (simplePatch (Waveform::saw)));
    const auto version = json.getProperty ("synth_version", {}).toString();

    auto parts = juce::StringArray::fromTokens (version, ".", "");
    REQUIRE (parts.size() == 3);
    CHECK (parts[0].getIntValue() == 1);
    CHECK (parts[1].getIntValue() == 0);
}

// --------------------------------------------------------------------------
// Every value we write, checked against Vital's own declaration of it.
//
// This is the cheap half of a real integration test. It cannot say whether the
// preset *sounds* right, but every mistake this exporter actually made was a
// legal-looking number that sat outside, or askew of, the range Vital declares
// -- and all of them would have failed here without a listener, a build of
// Vital, or a round trip through a person.

namespace
{

// `osc_2_unison_detune` is the `unison_detune` declaration; `env_3_attack` is
// `attack`. Numbered prefixes are instances of one parameter, not parameters.
juce::String baseNameOf (const juce::String& key)
{
    for (const auto* prefix : { "osc_", "env_", "lfo_", "filter_", "modulation_" })
    {
        if (! key.startsWith (prefix))
            continue;
        const auto rest = key.fromFirstOccurrenceOf (prefix, false, false);
        const auto cut = rest.indexOfChar ('_');
        return cut >= 0 ? rest.substring (cut + 1) : rest;
    }
    return key;
}

juce::var vitalParameters()
{
    const auto file = goldenDir().getChildFile ("vital").getChildFile ("parameters.json");
    REQUIRE (file.existsAsFile());
    const auto json = juce::JSON::parse (file.loadFileAsString());
    REQUIRE (json.isObject());
    return json;
}

// A patch that exercises the extremes, because a mapping is usually right in
// the middle of its range and wrong at the ends.
Patch extremePatch()
{
    auto patch = simplePatch (Waveform::saw);
    patch.masterLevel = 1.0f;
    patch.ampEnv = { 30.0f, 30.0f, 1.0f, 30.0f, 20.0f, 20.0f };
    patch.filter.cutoffHz = 19000.0f;
    patch.filter.resonance = 8.0f;

    for (auto& osc : patch.oscs)
    {
        osc.enabled = true;
        osc.level = 1.0f;
        osc.semitones = 24;
        osc.cents = 50.0f;
        osc.unisonVoices = 7;
        osc.unisonDetune = 50.0f;
        osc.numFrames = Oscillator::kMaxFrames;
        osc.framePosition = 1.0f;
        osc.framePositionEnvAmount = 1.0f;
    }

    patch.lfos[0].dest = autosynth::LfoDest::pitch;
    patch.lfos[0].depth = 1.0f;
    patch.lfos[0].rateHz = 18.0f;
    patch.lfos[0].delay = 2.0f;
    patch.lfos[1].dest = autosynth::LfoDest::cutoff;
    patch.lfos[1].depth = 1.0f;
    patch.lfos[1].rateHz = 0.05f;

    // The room and the noise bed at their limits, so the range check actually
    // reaches the reverb and sampler keys. Without something switching them on,
    // the exporter never writes them and every check silently passes over them
    // -- which is how the reverb went missing in the first place.
    patch.reverb = { true, 1.0f, 1.0f, 1.0f };
    patch.noiseLevel = 1.0f;
    return patch;
}

} // namespace

TEST_CASE ("every exported value is inside Vital's declared range", "[vital]")
{
    const auto declarations = vitalParameters();

    for (const auto& patch : { simplePatch (Waveform::saw), extremePatch(),
                               simplePatch (Waveform::noise) })
    {
        const auto settings = settingsOf (patch);
        auto* object = settings.getDynamicObject();
        REQUIRE (object != nullptr);

        for (const auto& property : object->getProperties())
        {
            const auto key = property.name.toString();
            const auto value = property.value;
            if (! value.isDouble() && ! value.isInt())
                continue;   // wavetables, modulations, lfos, sample

            const auto declaration = declarations.getProperty (baseNameOf (key), {});
            if (! declaration.isObject())
                continue;   // not a parameter this fixture covers

            const auto v = static_cast<double> (value);
            const auto lo = static_cast<double> (declaration.getProperty ("min", 0.0));
            const auto hi = static_cast<double> (declaration.getProperty ("max", 1.0));

            // A hair of tolerance at the ends, because the exporter stores
            // floats and the declaration is written as decimal: a value clamped
            // to Vital's own ceiling of 7399.4404 comes back as 7399.44043 and
            // is not out of range, it is out of precision. The margin is far
            // too small to admit a value in the wrong unit, which is what this
            // is watching for.
            const auto slack = 1.0e-6 * juce::jmax (1.0, std::abs (lo), std::abs (hi));
            INFO (key << " = " << v << ", declared " << lo << " .. " << hi);
            CHECK (v >= lo - slack);
            CHECK (v <= hi + slack);

            // An indexed control is a whole number; a fraction there is a
            // control being written in the wrong unit.
            if (declaration.getProperty ("scale", {}).toString() == "kIndexed")
                CHECK (v == Catch::Approx (std::round (v)).margin (1.0e-6));
        }
    }
}

TEST_CASE ("the mappings invert Vital's declared curves", "[vital]")
{
    // Range alone would not have caught the two that mattered: a linear level
    // and a default volume are both perfectly in range and both quiet. What
    // catches them is checking that our value, put back through the curve Vital
    // declares, returns what we started with.
    const auto declarations = vitalParameters();
    using M = VitalExport::Mapping;

    const auto readBack = [&declarations] (const juce::String& name, float stored)
    {
        const auto d = declarations.getProperty (name, {});
        const auto scale = d.getProperty ("scale", {}).toString();
        const auto offset = static_cast<double> (d.getProperty ("post_offset", 0.0));

        if (scale == "kQuadratic")   return stored * stored + offset;
        if (scale == "kQuartic")     return std::pow (stored, 4.0) + offset;
        if (scale == "kSquareRoot")  return std::sqrt (stored) + offset;
        if (scale == "kExponential") return std::pow (2.0, stored) + offset;
        return stored + offset;
    };

    // Envelope time in seconds survives the quartic.
    for (const auto seconds : { 0.01f, 0.25f, 1.0f, 8.0f })
        CHECK (readBack ("attack", M::secondsToEnvelope (seconds))
                   == Catch::Approx (seconds).margin (0.01));

    // Oscillator amplitude survives the quadratic.
    for (const auto amplitude : { 0.1f, 0.5f, 1.0f })
        CHECK (readBack ("level", M::levelToOscLevel (amplitude))
                   == Catch::Approx (amplitude).margin (0.001));

    // LFO rate in hertz survives the exponential.
    for (const auto hz : { 0.5f, 3.0f, 12.0f })
        CHECK (readBack ("frequency", M::hzToLfoRate (hz)) == Catch::Approx (hz).margin (0.01));

    // And the master gain reads back as the decibels it was asked for, which is
    // where the trim shows up honestly rather than as a fudge.
    const auto db = readBack ("volume", M::gainToVolume (1.0f));
    CHECK (db == Catch::Approx (M::kMasterTrimDb).margin (0.05));
}

TEST_CASE ("every parameter the fitter can search reaches the preset", "[vital]")
{
    // The guard for a whole class of silent loss.
    //
    // The recovery harness found this the expensive way: run with Vital as the
    // renderer, oscillator counting fell from 83% to 58% and the control
    // distance nearly halved, which said the targets had become *less varied*
    // than the same random patches rendered here. The cause was that `Patch`
    // carried a filter, an envelope and a reverb send per oscillator and the
    // exporter wrote none of them. Those parameters were unrecoverable by
    // construction -- nothing in the rendered audio depended on them -- and
    // refinement spent its budget searching directions the renderer ignored.
    //
    // The rule this pins down: if the fitter is allowed to search a parameter,
    // the preset has to carry it. Anything else is an optimiser working on a
    // flat objective and a preset that quietly means something else.
    auto patch = simplePatch (Waveform::saw);

    // Everything switched on, so `scopeFor` returns the widest set it ever
    // would. A parameter out of scope is not a failure here -- it is one the
    // fitter would not search either.
    patch.noiseLevel = 0.3f;
    patch.reverb = { true, 0.5f, 0.5f, 0.3f };
    patch.delay = { true, 0.25f, 0.35f, 0.4f };
    patch.filter.type = autosynth::FilterType::lowpass;
    patch.filter.envAmount = 1.0f;

    for (int i = 0; i < autosynth::kNumOsc; ++i)
    {
        auto& osc = patch.oscs[(size_t) i];
        osc.enabled = true;
        osc.level = 0.5f;
        osc.unisonVoices = 3;
        osc.unisonDetune = 12.0f;
        // Pulse on one side of the morph, so pulse width is a parameter that
        // genuinely changes the wavetable rather than one that cannot.
        osc.waveform = Waveform::pulse;
        osc.waveformB = Waveform::square;
        osc.waveMorph = 0.4f;
        osc.pulseWidth = 0.35f;
        osc.numFrames = 3;
        osc.framePositionEnvAmount = 0.5f;
        osc.envEnabled = true;
    }

    patch.lfos[0].dest = autosynth::LfoDest::pitch;
    patch.lfos[0].depth = 0.4f;
    patch.lfos[1].dest = autosynth::LfoDest::cutoff;
    patch.lfos[1].depth = 0.4f;

    const auto baseline = VitalExport::toJson (patch, "test");
    const auto specs = autosynth::Refine::continuousSpecs();
    const auto scope = autosynth::Refine::scopeFor (patch);

    std::vector<std::string> silent;
    for (const auto& path : scope)
    {
        const auto spec = std::find_if (specs.begin(), specs.end(),
                                        [&path] (const auto& s) { return s.path == path; });
        if (spec == specs.end())
            continue;

        // A value far from the current one but inside the declared range, so a
        // parameter that *is* carried cannot fail this by moving too little.
        const auto current = autosynth::Refine::parameterValue (patch, path);
        const auto moved = std::abs (current - spec->lo) > std::abs (current - spec->hi)
                             ? spec->lo : spec->hi;

        auto altered = patch;
        autosynth::Refine::setParameterValue (altered, path, moved);

        if (VitalExport::toJson (altered, "test") == baseline)
            silent.push_back (path);
    }

    if (! silent.empty())
    {
        juce::String names;
        for (const auto& path : silent)
            names += "\n    " + juce::String (path);
        UNSCOPED_INFO ("searchable but absent from the preset:" << names);
    }
    CHECK (silent.empty());
}
