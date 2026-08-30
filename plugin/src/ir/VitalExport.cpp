#include "ir/VitalExport.h"

#include "dsp/Reverb.h"
#include "dsp/Tables.h"

#include <array>
#include <cmath>
#include <juce_dsp/juce_dsp.h>

namespace autosynth
{
namespace
{

// env_1 is the voice amplitude, env_2 the filter and env_3 the wavetable
// position, so the per-oscillator envelopes start at four. Vital has six, which
// is exactly enough for those three plus one per oscillator.
constexpr int kFirstOscEnvelope = 4;

// Added to an attack power to undo Vital squaring the envelope. See addAdsr.
constexpr float kSquaringCompensation = 2.5f;

// Vital has four random LFOs, and its lfo_N_frequency control spans -7 to 9 --
// sixteen octaves -- which is what a drift in octaves is expressed against.
constexpr int kNumRandomLfos = 4;
constexpr float kLfoRateSpan = 16.0f;

// One frame of a Vital wavetable: 2048 samples, base64 of little-endian float.
//
// Built from the harmonics rather than resampled from our own 4096-point table.
// Resampling would need a filter and would land a sample short at the seam,
// which is where a wavetable is least forgiving -- the loop point is heard on
// every cycle.
juce::String frameToBase64 (const std::vector<float>& harmonics)
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
//
// A drawn frame is sixteen numbers because that is what was measured into it. A
// generated one is not: a saw is a saw, not its first sixteen harmonics, and
// truncating it there threw away everything above the sixteenth -- at 220 Hz,
// the whole band above 3.5 kHz. Exported that way a saw and a saw drawn out as
// sixteen numbers rendered to the same audio, which is how the loss was found.
// Generated frames therefore carry every harmonic the frame can hold.
std::vector<float> harmonicsOf (const Oscillator& osc, int frame)
{
    const auto& f = osc.frames[static_cast<size_t> (juce::jlimit (0, Oscillator::kMaxFrames - 1, frame))];
    if (f.custom)
        return { f.harmonics.begin(), f.harmonics.end() };

    // One short of the frame's own Nyquist, which is what a 2048-point frame
    // can represent. Vital band-limits per octave when it plays.
    constexpr int kGeneratedHarmonics = VitalExport::Mapping::kFrameSamples / 2 - 1;

    if (osc.waveform == Waveform::noise)
    {
        // Vital's noise is a separate oscillator type, not a wavetable. A flat
        // spectrum is the closest a wavetable gets, and it is at least the right
        // colour.
        return std::vector<float> ((size_t) kGeneratedHarmonics, 1.0f);
    }

    return WaveTables::blendedHarmonics (osc.waveform, osc.waveformB, osc.waveMorph,
                                         osc.pulseWidth, kGeneratedHarmonics);
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
    component->setProperty ("interpolation", 1);          // linear
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
    settings.setProperty (prefix + "_sustain",
                          std::sqrt (juce::jlimit (0.0f, 1.0f, env.sustain)));
    settings.setProperty (prefix + "_release", t (env.release));

    // Vital's power controls bend each segment, and the sign was backwards.
    //
    // Assumed rather than measured, and wrong: sweeping the attack power
    // through -5, -2.5, 0, +2 and +4 and rendering each shows the segment
    // getting *slower* as it goes negative and faster as it goes positive,
    // which is the opposite of what this wrote. Negative is the direction that
    // dwells at the start; ours is a positive "toward exponential" meaning a
    // fast start, so the sign carries straight across.
    //
    // Plus extra concavity, because the envelope Vital applies is squared.
    //
    // The sustain is handled by writing its square root; the attack needs the
    // same idea applied to a shape rather than a number. Squaring a rise makes
    // it slower early, so the curve written has to be the *square root* of the
    // one fitted -- and the square root of the patch model's
    // exponential-approach shape is close to the same shape with a constant
    // added to its rate.
    //
    // Measured rather than derived: with the correction absent, both presets
    // sit at 0.02 of their peak fifty milliseconds in where the recordings are
    // at 0.45 and 0.25.
    // Clamped to the control's own range: an envelope curve of 20 plus the
    // compensation is 22.5, which the range check caught before Vital did.
    const auto power = [] (float value)
    {
        return juce::jlimit (-20.0f, 20.0f, value);
    };

    settings.setProperty (prefix + "_attack_power",
                          power (env.attackCurve + kSquaringCompensation));
    settings.setProperty (prefix + "_decay_power", power (-env.curve));
    settings.setProperty (prefix + "_release_power", power (-env.curve));
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

// The sample block. Measured against real presets, `samples` is base64 of
// little-endian 16-bit -- 44100 frames arrive as 88200 bytes.
//
// It has to exist and has to decode even when unused, because Vital's loader
// reads it directly. When the patch has a noise bed it stops being a stub and
// becomes the noise: Vital's oscillators are wavetables and have no noise mode,
// so the sampler is the only continuous broadband source it owns. Rendering
// through Vital measured the cost of leaving it silent -- the violin's noise
// read 0.171 as fitted and 0.048 in Vital, which is most of what "it sounds a
// bit different" was.
//
// One second, looped. The loop is inaudible because the content is noise, and a
// fixed seed keeps the export reproducible -- an exporter that emits different
// bytes each run cannot have a golden fixture.
juce::var sampleBlock (bool withNoise)
{
    constexpr int silentFrames = 8;
    constexpr int noiseRate = 44100;
    const auto frames = withNoise ? noiseRate : silentFrames;

    juce::MemoryOutputStream raw;
    if (withNoise)
    {
        // Uniform rather than Gaussian, and full scale, to match the engine's
        // own generator: it adds `noise * level` with noise uniform on [-1, 1],
        // so the amplitude asked for is the amplitude that arrives.
        juce::Random random (0x5eed);
        for (int i = 0; i < frames; ++i)
            raw.writeShort ((short) juce::jlimit (-32767, 32767,
                                                  (int) (random.nextFloat() * 65534.0f) - 32767));
    }
    else
    {
        for (int i = 0; i < frames; ++i)
            raw.writeShort (0);
    }

    auto* sample = new juce::DynamicObject();
    sample->setProperty ("name", withNoise ? "autosynth noise" : "Init");
    sample->setProperty ("length", frames);
    sample->setProperty ("sample_rate", noiseRate);
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
    const auto db = 20.0f * std::log10 (linearGain) + kMasterTrimDb;
    return juce::jlimit (0.0f, 7399.4404f, (db + 80.0f) * (db + 80.0f));
}

float VitalExport::Mapping::levelToOscLevel (float linearLevel) noexcept
{
    return juce::jlimit (0.0f, 1.0f, std::sqrt (juce::jmax (0.0f, linearLevel)));
}

VitalExport::Mapping::LevelSwing
VitalExport::Mapping::levelModulation (float level, float depth) noexcept
{
    LevelSwing out;
    const auto amplitude = juce::jlimit (0.0f, 1.0f, level);
    const auto d = juce::jlimit (0.0f, 0.95f, depth);

    if (d <= 1.0e-4f)
    {
        out.level = levelToOscLevel (amplitude);
        return out;
    }

    const auto r = std::sqrt ((1.0f + d) / (1.0f - d));
    const auto k = (r - 1.0f) / (r + 1.0f);
    out.level = std::sqrt (amplitude / juce::jmax (1.0e-6f, 1.0f - k * k));
    out.amount = k * out.level;
    return out;
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
    // Quadratic, so the stored value is the square root of the percentage --
    // and the percentage is not cents. Measured through the plug-in: two voices
    // written at 8% render 30.5 cents apart and at 12% render 48.1, so the
    // control spans four cents per percent and 400 cents at the top of its
    // range. Read as cents it detuned everything four times too far, which at
    // the fitter's usual values is the difference between a beat and a chord.
    constexpr float kCentsPerPercent = 4.0f;
    return juce::jlimit (0.0f, 10.0f,
                         std::sqrt (juce::jmax (0.0f, cents / kCentsPerPercent)));
}

float VitalExport::Mapping::resonanceToNormalised (float q) noexcept
{
    return juce::jlimit (0.0f, 1.0f, (q - 0.5f) / 7.5f);
}

juce::String VitalExport::toJson (const Patch& patch, const juce::String& presetName)
{
    auto* settings = new juce::DynamicObject();

    // Written at the end, once the reverb crossfade is known: matching this
    // engine's reverb *send* with Vital's dry/wet means giving the master back
    // whatever the crossfade took off the dry.
    auto reverbWet = 0.0f;

    settings->setProperty ("polyphony", 8);
    settings->setProperty ("voice_transpose", 0);

    // How much amplitude modulation each oscillator carries, before anything is
    // written, because the level and the modulation amount have to be solved
    // together -- see Mapping::levelModulation.
    auto ampDepth = 0.0f;
    for (int l = 0; l < kNumLfo; ++l)
    {
        const auto& lfo = patch.lfos[static_cast<size_t> (l)];
        if (lfo.dest == LfoDest::amp && lfo.depth > 1.0e-4f)
            ampDepth += lfo.depth;
    }

    std::array<Mapping::LevelSwing, kNumOsc> swing {};
    auto headroom = 1.0f;
    for (int i = 0; i < kNumOsc; ++i)
    {
        const auto& osc = patch.oscs[static_cast<size_t> (i)];
        swing[static_cast<size_t> (i)] = Mapping::levelModulation (osc.level, ampDepth);
        const auto peak = swing[static_cast<size_t> (i)].level
                        + swing[static_cast<size_t> (i)].amount;
        headroom = juce::jmax (headroom, peak);
    }

    // Vital's level control stops at one, so a patch whose level plus its
    // modulation would go past it is scaled down as a whole and the master is
    // given the difference back. Scaling one oscillator alone would change the
    // balance between them, and the master cannot fix that per oscillator.
    // Amplitude is the square of the stored value, so the master owes the
    // square of the scale.
    const auto levelScale = 1.0f / headroom;
    for (auto& s : swing)
    {
        s.level *= levelScale;
        s.amount *= levelScale;
    }

    juce::Array<juce::var> wavetables;
    for (int i = 0; i < kNumOsc; ++i)
    {
        const auto& osc = patch.oscs[static_cast<size_t> (i)];
        const auto n = juce::String (i + 1);
        const auto on = osc.enabled && osc.level > 1.0e-4f;

        settings->setProperty ("osc_" + n + "_on", on ? 1.0f : 0.0f);

        // Vital randomises each note's starting phase by default. The patch
        // model has no such thing, so switching it off is the faithful
        // translation rather than a preference.
        //
        // It also decides whether the preset can be *fitted* at all. Rendering
        // one patch six times through Vital with random phase on gave renders
        // differing by 0.39 at the sample -- on a signal peaking at 0.22, which
        // is more noise than signal -- so an optimiser measuring Vital would be
        // reading the phase lottery rather than the parameters. With it off the
        // renders are identical.
        settings->setProperty ("osc_" + n + "_random_phase", 0.0f);
        settings->setProperty ("osc_" + n + "_phase", 0.0f);
        // The oscillator's own amplitude envelope, when it has one.
        //
        // env_1 is the voice's amplitude and applies to everything; this is the
        // per-oscillator envelope on top of it, and Vital has six of them
        // against a need for three, so each gets its own. The level
        // itself drops to zero and the envelope's *amount* carries it, because
        // an envelope that scales an oscillator has to be able to silence it --
        // modulation adds to the parameter rather than multiplying it.
        //
        // The shape is not identical. Vital's level control is quadratic, so an
        // envelope driving it produces an amplitude following the square of the
        // curve where the fitted envelope is the curve. That matters less than
        // it sounds: what the export owes the fitter is that the parameter *has
        // an effect*, so the optimiser can use it, and refinement rendering
        // through Vital absorbs the difference by adjusting the times and
        // powers.
        const auto oscEnvIndex = kFirstOscEnvelope + i;
        const auto usesOwnEnvelope = on && osc.envEnabled;
        settings->setProperty ("osc_" + n + "_level",
                               usesOwnEnvelope ? 0.0f
                                               : swing[static_cast<size_t> (i)].level);
        if (usesOwnEnvelope)
            addAdsr (*settings, "env_" + juce::String (oscEnvIndex), osc.env);
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

        // An LFO with no depth is not part of the patch, so its rate is not
        // written. It reads as harmless -- nothing is routed from it -- but
        // Vital runs every LFO whether or not anything listens, and two patches
        // differing only in the rate of a silent one render about a
        // ten-thousandth apart. Inaudible, and still a difference nobody asked
        // for; leaving the slot at Vital's default keeps it at nothing.
        if (lfo.dest == LfoDest::none || lfo.depth <= 1.0e-4f)
            continue;

        const auto n = juce::String (i + 1);
        settings->setProperty ("lfo_" + n + "_sync", 0.0f);   // free-running, so the rate is in Hz
        settings->setProperty ("lfo_" + n + "_frequency", Mapping::hzToLfoRate (lfo.rateHz));
        settings->setProperty ("lfo_" + n + "_delay_time", juce::jmax (0.0f, lfo.delay));
        settings->setProperty ("lfo_" + n + "_phase", juce::jlimit (0.0f, 1.0f, lfo.phase));
    }

    // Rate drift, as one of Vital's random LFOs.
    //
    // These sit outside the eight ordinary LFOs and cost none of them, which is
    // the whole reason the wander is a field on an LFO here rather than a slot
    // spent pointing a second LFO at the first. Style is left at Vital's
    // default, which is the smooth one; a sample-and-hold would step the rate
    // rather than drift it.
    //
    // The amount is the drift in octaves over the frequency control's own span
    // of sixteen, and bipolar because a rate that only ever rose would not be
    // wander, it would be a ramp.
    auto randomsUsed = 0;

    std::vector<Routing> routings;

    for (int i = 0; i < kNumLfo && randomsUsed < kNumRandomLfos; ++i)
    {
        const auto& lfo = patch.lfos[static_cast<size_t> (i)];
        if (lfo.dest == LfoDest::none || lfo.depth <= 1.0e-4f)
            continue;
        if (lfo.rateWander <= 1.0e-4f || lfo.wanderRateHz <= 1.0e-4f)
            continue;

        const auto r = juce::String (++randomsUsed);
        settings->setProperty ("random_" + r + "_sync", 0.0f);   // free-running, rate in Hz
        settings->setProperty ("random_" + r + "_frequency",
                               Mapping::hzToLfoRate (lfo.wanderRateHz));

        routings.push_back ({ "random_" + r,
                              "lfo_" + juce::String (i + 1) + "_frequency",
                              juce::jlimit (0.0f, 1.0f, lfo.rateWander / kLfoRateSpan),
                              true });
    }

    // The filter envelope, which was written and then left dangling: env_2 held
    // the right shape and nothing connected it to anything, so every export
    // rendered a static cutoff while the fit swept it. On the clarinet that is
    // 1.92 octaves of missing brightness, and it read as harmonics two to eight
    // sitting 3 to 13 dB low -- a difference no amount of reading the preset
    // would have shown, because the preset was legal.
    //
    // Scaled like the LFO route below: octaves to semitones, over the cutoff
    // parameter's own span of 128. Unipolar, because an ADSR in Vital runs zero
    // to one and ours is added to the cutoff rather than swung around it.
    if (patch.filter.type != FilterType::off && std::abs (patch.filter.envAmount) > 1.0e-4f)
        routings.push_back ({ "env_2", "filter_1_cutoff",
                              juce::jlimit (-1.0f, 1.0f,
                                            patch.filter.envAmount * 12.0f / 128.0f),
                              false });

    // Each oscillator's own envelope, driving its level. The amount is the
    // level the oscillator would otherwise have sat at, because the level
    // itself was written as zero: modulation adds, so the envelope has to
    // supply the whole of it for a closed envelope to mean silence.
    for (int i = 0; i < kNumOsc; ++i)
    {
        const auto& osc = patch.oscs[static_cast<size_t> (i)];
        if (! osc.enabled || osc.level <= 1.0e-4f || ! osc.envEnabled)
            continue;

        routings.push_back ({ "env_" + juce::String (kFirstOscEnvelope + i),
                              "osc_" + juce::String (i + 1) + "_level",
                              juce::jlimit (0.0f, 1.0f, swing[static_cast<size_t> (i)].level),
                              false });
    }

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
                                              swing[static_cast<size_t> (o)].amount
                                                  * (ampDepth > 1.0e-4f ? lfo.depth / ampDepth : 0.0f),
                                              true });
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

    // The noise bed, as the sampler.
    //
    // Routed to filter 1 rather than left on Vital's default of the effects
    // bus: the patch model puts noise into the mix *before* the filter, so an
    // unfiltered noise bed is a different sound -- brighter, and unaffected by
    // the filter envelope that shapes everything else.
    // The room.
    //
    // Dropping this was the last thing that made an exported preset stop
    // sounding like the fit, and the most obvious once heard: the note simply
    // ended. Both library patches carry a reverb at a return gain around 0.1
    // and an RT60 near a second, and the exported preset went to digital
    // silence 50 ms after note-off, because the fitted amplitude release is
    // 5 ms and the tail it was fitted alongside was never written.
    //
    // Vital states the tail as a decay time in seconds where ours states a room
    // size, so the size goes back through the reverb's own RT60 relation rather
    // than being copied across as if the two controls meant the same thing.
    // `reverb_size` is left at its default for the same reason: it scales the
    // delay lines here, and putting the size in both places would count the
    // tail twice.
    const auto reverbOn = patch.reverb.enabled && patch.reverb.level > 1.0e-4f;
    settings->setProperty ("reverb_on", reverbOn ? 1.0f : 0.0f);
    if (reverbOn)
    {
        // Ours is a return gain *added* to the dry; Vital's is a crossfade, and
        // the return is applied after a comb bank with a large gain of its own.
        // Writing the return gain straight in as a mix left the tail 14 dB
        // under the source recording -- present in the file, inaudible in the
        // room -- so the two are related by a measured factor rather than
        // treated as the same quantity.
        const auto ratio = juce::jlimit (0.0f, 9.0f,
                                         Mapping::kReverbReturnToRatio * patch.reverb.level);
        reverbWet = ratio / (1.0f + ratio);
        settings->setProperty ("reverb_dry_wet", juce::jlimit (0.0f, 1.0f, reverbWet));

        const auto rt60 = Reverb::rt60ForSize (patch.reverb.size)
                        * Mapping::kReverbDecayCorrection;
        settings->setProperty ("reverb_decay_time",
                               juce::jlimit (-6.0f, 6.0f,
                                             (float) std::log2 (juce::jmax (1.0e-3, rt60))));

        // Damping is a one-pole lowpass on our tail and a high shelf on
        // Vital's, so this is a match in direction and rough degree rather than
        // in filter shape. Both library patches sit near zero damping, so the
        // approximation is not currently carrying any weight.
        settings->setProperty ("reverb_high_shelf_gain",
                               juce::jlimit (-6.0f, 0.0f, -6.0f * patch.reverb.damp));

        // No chorus in the reverb, which is both faithful and the difference
        // between a measurable render and an unmeasurable one.
        //
        // Vital modulates its reverb with a chorus that free-runs at a quarter
        // of a hertz by default -- a four second cycle -- and it is not reset by
        // a note, by `reset()`, by reloading the preset, or by reallocating the
        // plug-in's resources. So two renders of *one* patch differ by whatever
        // that cycle has moved between them: measured, 0.072 at the sample on a
        // peak of 0.43, and in the terms the objective is built from, a spectral
        // distance of 0.057 and a loudness distance of two decibels. The fit's
        // own loudness error is about four and a half, so nearly half of that
        // term was the phase of a chorus nobody asked for.
        //
        // It alternated with every other render, which is what a four second
        // cycle sampled every two seconds does, and it was there through every
        // comparison made before it was found. Switched off, two renders of one
        // patch agree to 0.0001.
        //
        // This engine's reverb is a Schroeder network with no modulation at all,
        // so writing a chorused reverb was never faithful to the thing being
        // fitted.
        settings->setProperty ("reverb_chorus_amount", 0.0f);
    }

    // Written here rather than at the top only because the reverb decides how
    // much of the dry Vital keeps. At these mix levels the crossfade takes
    // well under a decibel, and compensating for it was measured to overshoot:
    // Vital's dry falls far more slowly than `1 - wet`, so undoing that much
    // put the whole patch several decibels over the level it was fitted at.
    settings->setProperty ("volume",
                           Mapping::gainToVolume (patch.masterLevel / (levelScale * levelScale)));

    // The delay, which had been dropped as silently as the reverb was.
    //
    // Vital states the time as a frequency: the control is exponential over
    // -2..9, so the delay is one over two to that power, and `delay_sync` has
    // to say "not tempo" or the number is read as a division instead.
    const auto delayOn = patch.delay.enabled && patch.delay.mix > 1.0e-4f;
    settings->setProperty ("delay_on", delayOn ? 1.0f : 0.0f);
    if (delayOn)
    {
        settings->setProperty ("delay_dry_wet", juce::jlimit (0.0f, 1.0f, patch.delay.mix));
        settings->setProperty ("delay_feedback",
                               juce::jlimit (-1.0f, 1.0f, patch.delay.feedback));
        settings->setProperty ("delay_sync", 0.0f);
        settings->setProperty ("delay_aux_sync", 0.0f);

        const auto seconds = juce::jmax (1.0f / 512.0f, patch.delay.time);
        const auto frequency = juce::jlimit (-2.0f, 9.0f, (float) -std::log2 (seconds));
        settings->setProperty ("delay_frequency", frequency);
        settings->setProperty ("delay_aux_frequency", frequency);
    }

    const auto hasNoise = patch.noiseLevel > 1.0e-4f;
    settings->setProperty ("sample", sampleBlock (hasNoise));
    settings->setProperty ("sample_on", hasNoise ? 1.0f : 0.0f);
    if (hasNoise)
    {
        settings->setProperty ("sample_level", Mapping::levelToOscLevel (patch.noiseLevel));
        settings->setProperty ("sample_destination", 0.0f);   // filter 1
        settings->setProperty ("sample_keytrack", 0.0f);      // noise has no pitch to track
        settings->setProperty ("sample_loop", 1.0f);
    }

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
