#include "ir/VitalExport.h"

#include "dsp/Tables.h"

#include <cmath>
#include <juce_dsp/juce_dsp.h>

namespace autosynth
{
namespace
{

// One frame of a Vital wavetable: 2048 samples, base64 of little-endian float.
//
// Built from the harmonics rather than resampled from our own 4096-point table.
// Resampling would need a filter and would land a sample short at the seam,
// which is where a wavetable is least forgiving -- the loop point is heard on
// every cycle.
juce::String frameToBase64 (const std::array<float, Oscillator::kFrameHarmonics>& harmonics)
{
    constexpr auto n = VitalExport::Mapping::kFrameSamples;
    constexpr auto order = 11;   // 2^11 == 2048
    static_assert (1 << order == n, "frame size and FFT order must agree");

    std::vector<juce::dsp::Complex<float>> spectrum (n, { 0.0f, 0.0f });
    for (size_t k = 1; k <= harmonics.size() && k < static_cast<size_t> (n / 2); ++k)
    {
        const auto a = harmonics[k - 1] * 0.5f;
        spectrum[k] = { 0.0f, -a };
        spectrum[static_cast<size_t> (n) - k] = { 0.0f, a };
    }

    std::vector<juce::dsp::Complex<float>> time (n);
    juce::dsp::FFT fft (order);
    fft.perform (spectrum.data(), time.data(), true);

    std::vector<float> samples (static_cast<size_t> (n));
    float peak = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        samples[static_cast<size_t> (i)] = time[static_cast<size_t> (i)].real();
        peak = juce::jmax (peak, std::abs (samples[static_cast<size_t> (i)]));
    }
    if (peak > 1.0e-9f)
        for (auto& v : samples)
            v /= peak;

    juce::MemoryOutputStream raw;
    for (const auto v : samples)
        raw.writeFloat (v);   // little-endian, which is what Vital reads

    return juce::Base64::toBase64 (raw.getData(), raw.getDataSize());
}

// The harmonics a frame stands for: its own if drawn, its generator's if not.
std::array<float, Oscillator::kFrameHarmonics> harmonicsOf (const Oscillator& osc, int frame)
{
    const auto& f = osc.frames[static_cast<size_t> (juce::jlimit (0, Oscillator::kMaxFrames - 1, frame))];
    if (f.custom)
        return f.harmonics;

    std::array<float, Oscillator::kFrameHarmonics> out {};
    if (osc.waveform == Waveform::noise)
    {
        // Vital's noise is a separate oscillator type, not a wavetable. A flat
        // spectrum is the closest a wavetable gets, and it is at least the right
        // colour.
        out.fill (1.0f);
        return out;
    }

    const auto amps = WaveTables::blendedHarmonics (osc.waveform, osc.waveformB, osc.waveMorph,
                                                    osc.pulseWidth, Oscillator::kFrameHarmonics);
    for (size_t k = 0; k < out.size() && k < amps.size(); ++k)
        out[k] = amps[k];
    return out;
}

juce::var wavetableFor (const Oscillator& osc, const juce::String& name)
{
    juce::Array<juce::var> keyframes;
    const auto count = juce::jlimit (1, Oscillator::kMaxFrames, osc.numFrames);
    for (int f = 0; f < count; ++f)
    {
        auto* key = new juce::DynamicObject();
        key->setProperty ("position",
                          count > 1 ? (f * static_cast<int> (VitalExport::Mapping::kWaveFramePositions))
                                          / (count - 1)
                                    : 0);
        key->setProperty ("wave_data", frameToBase64 (harmonicsOf (osc, f)));
        keyframes.add (juce::var (key));
    }

    // Exactly the four keys a Wave Source carries, and no more.
    //
    // The first version added nine others -- `audio_file`, `num_points`,
    // `window_size` and friends -- collected by counting keys across *all*
    // component types in the preset library. They belong to Audio File Source
    // and Line Source, and a Wave Source carrying them is what Vital rejected
    // as a corrupted file. Checked properly: all 510 Wave Source components in
    // 299 presets have this key set and only this one, and all 2063 of their
    // keyframes have `position` and `wave_data` and nothing else.
    auto* component = new juce::DynamicObject();
    component->setProperty ("type", "Wave Source");
    component->setProperty ("keyframes", juce::var (keyframes));
    component->setProperty ("interpolation", 1);          // linear, as our engine does
    component->setProperty ("interpolation_style", 1);

    juce::Array<juce::var> components;
    components.add (juce::var (component));

    auto* group = new juce::DynamicObject();
    group->setProperty ("components", juce::var (components));

    juce::Array<juce::var> groups;
    groups.add (juce::var (group));

    auto* table = new juce::DynamicObject();
    table->setProperty ("name", name);
    table->setProperty ("author", "");
    table->setProperty ("version", VitalExport::Mapping::kSynthVersion);
    table->setProperty ("groups", juce::var (groups));
    table->setProperty ("full_normalize", true);
    table->setProperty ("remove_all_dc", true);
    return juce::var (table);
}

void addAdsr (juce::DynamicObject& settings, const juce::String& prefix, const Adsr& env)
{
    const auto t = VitalExport::Mapping::secondsToEnvelope;
    settings.setProperty (prefix + "_attack", t (env.attack));
    settings.setProperty (prefix + "_decay", t (env.decay));
    settings.setProperty (prefix + "_sustain", juce::jlimit (0.0f, 1.0f, env.sustain));
    settings.setProperty (prefix + "_release", t (env.release));

    // Vital's power controls bend each segment; negative is the exponential
    // direction. Ours is a single positive "toward exponential", so the sign is
    // applied per segment.
    settings.setProperty (prefix + "_attack_power", -env.attackCurve);
    settings.setProperty (prefix + "_decay_power", -env.curve);
    settings.setProperty (prefix + "_release_power", -env.curve);
}

// The eight drawn LFO shapes Vital expects to find, and the sampler's sample.
//
// Neither is optional. `LoadSave::jsonToState` reads `settings["lfos"]` and
// `settings["sample"]` directly, so a preset without them fails to parse and is
// reported as corrupted -- which is exactly what happened, and what no amount of
// comparing the *parameters* would have found. Reading the loader was the only
// way to learn that a preset is not just a bag of settings.
//
// The shapes are Vital's default triangle: three points, no curvature. Ours are
// chosen by `LfoShape` in the parameters instead, so these are the canvas rather
// than the choice.
juce::var defaultLfoShapes()
{
    juce::Array<juce::var> lfos;
    for (int i = 0; i < 8; ++i)
    {
        juce::Array<juce::var> points;
        for (const auto v : { 0.0, 1.0, 0.5, 0.0, 1.0, 1.0 })
            points.add (v);

        juce::Array<juce::var> powers;
        for (int p = 0; p < 3; ++p)
            powers.add (0.0);

        auto* shape = new juce::DynamicObject();
        shape->setProperty ("name", "Triangle");
        shape->setProperty ("num_points", 3);
        shape->setProperty ("points", juce::var (points));
        shape->setProperty ("powers", juce::var (powers));
        shape->setProperty ("smooth", false);
        lfos.add (juce::var (shape));
    }
    return juce::var (lfos);
}

// A silent stub. The sampler is off in everything this exports, but the block
// has to exist and has to decode: measured against real presets, `samples` is
// base64 of little-endian 16-bit -- 44100 frames arrive as 88200 bytes.
juce::var silentSample()
{
    constexpr int frames = 8;
    juce::MemoryOutputStream raw;
    for (int i = 0; i < frames; ++i)
        raw.writeShort (0);

    auto* sample = new juce::DynamicObject();
    sample->setProperty ("name", "Init");
    sample->setProperty ("length", frames);
    sample->setProperty ("sample_rate", 44100);
    sample->setProperty ("samples", juce::Base64::toBase64 (raw.getData(), raw.getDataSize()));
    return juce::var (sample);
}

// Vital's routing lives in two places: an entry in `modulations` naming source
// and destination, and a numbered `modulation_N_*` block carrying the amount.
// The first version of this exporter put the amount inside the entry, where
// nothing reads it -- every routing would have connected at zero.
struct Routing
{
    juce::String source, destination;
    float amount = 0.0f;
    bool bipolar = true;
};

} // namespace

float VitalExport::Mapping::secondsToEnvelope (float seconds) noexcept
{
    return juce::jlimit (0.0f, 2.3784f, std::pow (juce::jmax (0.0f, seconds), 0.25f));
}

float VitalExport::Mapping::hzToLfoRate (float hz) noexcept
{
    // Exponential on -7 to 9, so the stored value is the base-two logarithm of
    // the rate. Vital displays it as a period in seconds, which is the same
    // number read the other way up.
    return juce::jlimit (-7.0f, 9.0f, std::log2 (juce::jmax (hz, 1.0e-3f)));
}

float VitalExport::Mapping::gainToVolume (float linearGain) noexcept
{
    if (linearGain <= 1.0e-5f)
        return 0.0f;
    const auto db = 20.0f * std::log10 (linearGain) + kExportMakeupDb;
    return juce::jlimit (0.0f, 7399.4404f, (db + 80.0f) * (db + 80.0f));
}

float VitalExport::Mapping::levelToOscLevel (float linearLevel) noexcept
{
    return juce::jlimit (0.0f, 1.0f, std::sqrt (juce::jmax (0.0f, linearLevel)));
}

float VitalExport::Mapping::cutoffToNote (float hz) noexcept
{
    return juce::jlimit (8.0f, 136.0f, 69.0f + 12.0f * std::log2 (juce::jmax (hz, 1.0f) / 440.0f));
}

float VitalExport::Mapping::centsToTune (float cents) noexcept
{
    return juce::jlimit (-1.0f, 1.0f, cents / 100.0f);
}

float VitalExport::Mapping::detuneToUnison (float cents) noexcept
{
    // Quadratic, so the stored value is the square root of the percentage.
    return juce::jlimit (0.0f, 10.0f, std::sqrt (juce::jmax (0.0f, cents)));
}

float VitalExport::Mapping::resonanceToNormalised (float q) noexcept
{
    return juce::jlimit (0.0f, 1.0f, (q - 0.5f) / 7.5f);
}

juce::String VitalExport::toJson (const Patch& patch, const juce::String& presetName)
{
    auto* settings = new juce::DynamicObject();

    settings->setProperty ("volume", Mapping::gainToVolume (patch.masterLevel));
    settings->setProperty ("polyphony", 8);
    settings->setProperty ("voice_transpose", 0);

    juce::Array<juce::var> wavetables;
    for (int i = 0; i < kNumOsc; ++i)
    {
        const auto& osc = patch.oscs[static_cast<size_t> (i)];
        const auto n = juce::String (i + 1);
        const auto on = osc.enabled && osc.level > 1.0e-4f;

        settings->setProperty ("osc_" + n + "_on", on ? 1.0f : 0.0f);
        settings->setProperty ("osc_" + n + "_level", Mapping::levelToOscLevel (osc.level));
        settings->setProperty ("osc_" + n + "_transpose", static_cast<float> (osc.semitones));
        settings->setProperty ("osc_" + n + "_tune", Mapping::centsToTune (osc.cents));
        settings->setProperty ("osc_" + n + "_unison_voices",
                               static_cast<float> (juce::jlimit (1, 16, osc.unisonVoices)));
        settings->setProperty ("osc_" + n + "_unison_detune",
                               Mapping::detuneToUnison (osc.unisonDetune));
        settings->setProperty ("osc_" + n + "_wave_frame",
                               juce::jlimit (0.0f, Mapping::kWaveFramePositions,
                                             osc.framePosition * Mapping::kWaveFramePositions));

        wavetables.add (wavetableFor (osc, "autosynth osc " + n));
    }
    settings->setProperty ("wavetables", juce::var (wavetables));

    addAdsr (*settings, "env_1", patch.ampEnv);
    addAdsr (*settings, "env_2", patch.filter.env);
    addAdsr (*settings, "env_3", patch.oscs[0].framePositionEnv);

    settings->setProperty ("filter_1_on", patch.filter.type == FilterType::off ? 0.0f : 1.0f);
    settings->setProperty ("filter_1_cutoff", Mapping::cutoffToNote (patch.filter.cutoffHz));
    settings->setProperty ("filter_1_resonance",
                           Mapping::resonanceToNormalised (patch.filter.resonance));
    settings->setProperty ("filter_1_model", patch.filter.type == FilterType::highpass ? 1 : 0);

    for (int i = 0; i < kNumLfo; ++i)
    {
        const auto& lfo = patch.lfos[static_cast<size_t> (i)];
        const auto n = juce::String (i + 1);
        settings->setProperty ("lfo_" + n + "_sync", 0.0f);   // free-running, so the rate is in Hz
        settings->setProperty ("lfo_" + n + "_frequency", Mapping::hzToLfoRate (lfo.rateHz));
        settings->setProperty ("lfo_" + n + "_delay_time", juce::jmax (0.0f, lfo.delay));
        settings->setProperty ("lfo_" + n + "_phase", juce::jlimit (0.0f, 1.0f, lfo.phase));
    }

    std::vector<Routing> routings;
    for (int i = 0; i < kNumLfo; ++i)
    {
        const auto& lfo = patch.lfos[static_cast<size_t> (i)];
        if (lfo.dest == LfoDest::none || lfo.depth <= 1.0e-4f)
            continue;

        const auto source = "lfo_" + juce::String (i + 1);
        switch (lfo.dest)
        {
            case LfoDest::pitch:
                for (int o = 0; o < kNumOsc; ++o)
                    if (patch.oscs[static_cast<size_t> (o)].enabled)
                        routings.push_back ({ source, "osc_" + juce::String (o + 1) + "_transpose",
                                              lfo.depth * kLfoPitchSemitones / 48.0f, true });
                break;
            case LfoDest::amp:
                // Per oscillator rather than at the master: modulating `volume`
                // would move a square-root decibel control, where the same depth
                // means different things at different settings.
                for (int o = 0; o < kNumOsc; ++o)
                    if (patch.oscs[static_cast<size_t> (o)].enabled)
                        routings.push_back ({ source, "osc_" + juce::String (o + 1) + "_level",
                                              lfo.depth, true });
                break;
            case LfoDest::cutoff:
                routings.push_back ({ source, "filter_1_cutoff",
                                      lfo.depth * kLfoCutoffOctaves * 12.0f / 128.0f, true });
                break;
            case LfoDest::lfoRate:
                routings.push_back ({ source,
                                      "lfo_" + juce::String (((i + 1) % kNumLfo) + 1) + "_frequency",
                                      lfo.depth, true });
                break;
            case LfoDest::lfoDepth:
            case LfoDest::none:
                break;
        }
    }

    // The wavetable sweep: an envelope here, a matrix entry there.
    for (int i = 0; i < kNumOsc; ++i)
    {
        const auto& osc = patch.oscs[static_cast<size_t> (i)];
        if (! osc.enabled || std::abs (osc.framePositionEnvAmount) <= 1.0e-4f || osc.numFrames < 2)
            continue;
        routings.push_back ({ "env_3", "osc_" + juce::String (i + 1) + "_wave_frame",
                              juce::jlimit (-1.0f, 1.0f, osc.framePositionEnvAmount), false });
    }

    juce::Array<juce::var> modulations;
    for (size_t i = 0; i < routings.size(); ++i)
    {
        const auto& r = routings[i];
        auto* entry = new juce::DynamicObject();
        entry->setProperty ("source", r.source);
        entry->setProperty ("destination", r.destination);
        modulations.add (juce::var (entry));

        const auto n = juce::String (static_cast<int> (i) + 1);
        settings->setProperty ("modulation_" + n + "_amount", r.amount);
        settings->setProperty ("modulation_" + n + "_bipolar", r.bipolar ? 1.0f : 0.0f);
        settings->setProperty ("modulation_" + n + "_bypass", 0.0f);
        settings->setProperty ("modulation_" + n + "_power", 0.0f);
        settings->setProperty ("modulation_" + n + "_stereo", 0.0f);
    }
    settings->setProperty ("modulations", juce::var (modulations));
    settings->setProperty ("lfos", defaultLfoShapes());
    settings->setProperty ("sample", silentSample());

    auto* root = new juce::DynamicObject();
    root->setProperty ("synth_version", Mapping::kSynthVersion);
    root->setProperty ("preset_name", presetName.isNotEmpty() ? presetName : patch.name);
    root->setProperty ("preset_style", "");
    root->setProperty ("author", "");
    root->setProperty ("comments", "");
    root->setProperty ("macro1", "MACRO 1");
    root->setProperty ("macro2", "MACRO 2");
    root->setProperty ("macro3", "MACRO 3");
    root->setProperty ("macro4", "MACRO 4");
    root->setProperty ("settings", juce::var (settings));

    // One line, which is what Vital writes and therefore the shape its parser
    // is exercised against every day.
    return juce::JSON::toString (juce::var (root), true);
}

bool VitalExport::writeTo (const Patch& patch, const juce::File& file, juce::String* errorOut)
{
    const auto json = toJson (patch, file.getFileNameWithoutExtension());
    if (! file.replaceWithText (json))
    {
        if (errorOut != nullptr)
            *errorOut = "could not write " + file.getFullPathName();
        return false;
    }
    return true;
}

} // namespace autosynth
