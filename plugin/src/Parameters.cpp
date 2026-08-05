#include "Parameters.h"

#include <cmath>

namespace autosynth
{
namespace params
{

juce::String idFor (const juce::String& path)
{
    return path.replaceCharacter ('.', '_');
}

namespace
{

// Skewed so the midpoint of the slider travel lands on the geometric mean.
// Times and frequencies are perceived logarithmically; a linear cutoff slider
// spends most of its range above 9 kHz where nothing interesting happens.
juce::NormalisableRange<float> logRange (float min, float max)
{
    juce::NormalisableRange<float> range (min, max);
    range.setSkewForCentre (std::sqrt (min * max));
    return range;
}

const juce::StringArray kWaveforms { "Sine", "Triangle", "Saw", "Square", "Pulse" };
const juce::StringArray kFilterTypes { "Off", "Low pass", "High pass", "Band pass" };
const juce::StringArray kLfoShapes { "Sine", "Triangle", "Saw", "Square" };
const juce::StringArray kLfoDests { "None", "Pitch", "Amp", "Cutoff" };

using Layout = juce::AudioProcessorValueTreeState::ParameterLayout;

void addFloat (Layout& layout, const juce::String& path, const juce::String& name,
               juce::NormalisableRange<float> range, float defaultValue,
               const juce::String& suffix = {})
{
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { idFor (path), 1 }, name, range, defaultValue,
        juce::AudioParameterFloatAttributes().withLabel (suffix)));
}

void addAdsr (Layout& layout, const juce::String& prefix, const juce::String& label,
              const Adsr& defaults)
{
    addFloat (layout, prefix + ".attack",  label + " attack",  logRange (0.001f, 2.0f), defaults.attack, "s");
    addFloat (layout, prefix + ".decay",   label + " decay",   logRange (0.005f, 4.0f), defaults.decay, "s");
    addFloat (layout, prefix + ".sustain", label + " sustain", { 0.0f, 1.0f }, defaults.sustain);
    addFloat (layout, prefix + ".release", label + " release", logRange (0.005f, 4.0f), defaults.release, "s");
    addFloat (layout, prefix + ".curve",   label + " curve",   { 0.0f, 8.0f }, defaults.curve);
}

// Reading and writing go through one table so they cannot drift apart. Adding a
// parameter means adding it here and in ir.py, and nowhere else.
struct Accessor
{
    juce::String path;
    std::function<float (const Patch&)> get;
    std::function<void (Patch&, float)> set;
};

Adsr* adsrIn (Patch& patch, const juce::String& prefix)
{
    if (prefix == "amp_env")
        return &patch.ampEnv;
    if (prefix == "filter.env")
        return &patch.filter.env;
    if (prefix.startsWith ("oscs."))
    {
        const auto index = prefix[5] - '0';
        if (index < 0 || index >= kNumOsc)
            return nullptr;
        auto& osc = patch.oscs[static_cast<size_t> (index)];
        return prefix.endsWith (".filter.env") ? &osc.filter.env : &osc.env;
    }
    return nullptr;
}

void addAdsrAccessors (std::vector<Accessor>& out, const juce::String& prefix)
{
    const auto make = [prefix] (const char* leaf,
                                float Adsr::*member)
    {
        return Accessor {
            prefix + "." + leaf,
            [prefix, member] (const Patch& p)
            {
                auto& mutablePatch = const_cast<Patch&> (p);
                const auto* a = adsrIn (mutablePatch, prefix);
                return a != nullptr ? a->*member : 0.0f;
            },
            [prefix, member] (Patch& p, float v)
            {
                if (auto* a = adsrIn (p, prefix))
                    a->*member = v;
            }
        };
    };

    out.push_back (make ("attack", &Adsr::attack));
    out.push_back (make ("decay", &Adsr::decay));
    out.push_back (make ("sustain", &Adsr::sustain));
    out.push_back (make ("release", &Adsr::release));
    out.push_back (make ("curve", &Adsr::curve));
}

const std::vector<Accessor>& accessors()
{
    static const std::vector<Accessor> table = [] {
        std::vector<Accessor> out;

        for (int i = 0; i < kNumOsc; ++i)
        {
            const auto p = "oscs." + juce::String (i);
            const auto index = static_cast<size_t> (i);

            out.push_back ({ p + ".enabled",
                             [index] (const Patch& x) { return x.oscs[index].enabled ? 1.0f : 0.0f; },
                             [index] (Patch& x, float v) { x.oscs[index].enabled = v >= 0.5f; } });
            out.push_back ({ p + ".waveform",
                             [index] (const Patch& x) { return static_cast<float> (x.oscs[index].waveform); },
                             [index] (Patch& x, float v) { x.oscs[index].waveform =
                                 static_cast<Waveform> (juce::jlimit (0, 4, static_cast<int> (std::lround (v)))); } });
            out.push_back ({ p + ".semitones",
                             [index] (const Patch& x) { return static_cast<float> (x.oscs[index].semitones); },
                             [index] (Patch& x, float v) { x.oscs[index].semitones = static_cast<int> (std::lround (v)); } });
            out.push_back ({ p + ".cents",
                             [index] (const Patch& x) { return x.oscs[index].cents; },
                             [index] (Patch& x, float v) { x.oscs[index].cents = v; } });
            out.push_back ({ p + ".level",
                             [index] (const Patch& x) { return x.oscs[index].level; },
                             [index] (Patch& x, float v) { x.oscs[index].level = v; } });
            out.push_back ({ p + ".pulse_width",
                             [index] (const Patch& x) { return x.oscs[index].pulseWidth; },
                             [index] (Patch& x, float v) { x.oscs[index].pulseWidth = v; } });
            out.push_back ({ p + ".unison_voices",
                             [index] (const Patch& x) { return static_cast<float> (x.oscs[index].unisonVoices); },
                             [index] (Patch& x, float v) { x.oscs[index].unisonVoices =
                                 juce::jlimit (1, kMaxUnison, static_cast<int> (std::lround (v))); } });
            out.push_back ({ p + ".unison_detune",
                             [index] (const Patch& x) { return x.oscs[index].unisonDetune; },
                             [index] (Patch& x, float v) { x.oscs[index].unisonDetune = v; } });
            out.push_back ({ p + ".waveform_b",
                             [index] (const Patch& x) { return static_cast<float> (x.oscs[index].waveformB); },
                             [index] (Patch& x, float v) { x.oscs[index].waveformB =
                                 static_cast<Waveform> (juce::jlimit (0, 4, static_cast<int> (std::lround (v)))); } });
            out.push_back ({ p + ".wave_morph",
                             [index] (const Patch& x) { return x.oscs[index].waveMorph; },
                             [index] (Patch& x, float v) { x.oscs[index].waveMorph = v; } });
            out.push_back ({ p + ".reverb_send",
                             [index] (const Patch& x) { return x.oscs[index].reverbSend; },
                             [index] (Patch& x, float v) { x.oscs[index].reverbSend = v; } });
            out.push_back ({ p + ".env_enabled",
                             [index] (const Patch& x) { return x.oscs[index].envEnabled ? 1.0f : 0.0f; },
                             [index] (Patch& x, float v) { x.oscs[index].envEnabled = v >= 0.5f; } });
            addAdsrAccessors (out, p + ".env");

            out.push_back ({ p + ".filter_enabled",
                             [index] (const Patch& x) { return x.oscs[index].filterEnabled ? 1.0f : 0.0f; },
                             [index] (Patch& x, float v) { x.oscs[index].filterEnabled = v >= 0.5f; } });
            out.push_back ({ p + ".filter.type",
                             [index] (const Patch& x) { return static_cast<float> (x.oscs[index].filter.type); },
                             [index] (Patch& x, float v) { x.oscs[index].filter.type =
                                 static_cast<FilterType> (juce::jlimit (0, 3, static_cast<int> (std::lround (v)))); } });
            out.push_back ({ p + ".filter.cutoff_hz",
                             [index] (const Patch& x) { return x.oscs[index].filter.cutoffHz; },
                             [index] (Patch& x, float v) { x.oscs[index].filter.cutoffHz = v; } });
            out.push_back ({ p + ".filter.resonance",
                             [index] (const Patch& x) { return x.oscs[index].filter.resonance; },
                             [index] (Patch& x, float v) { x.oscs[index].filter.resonance = v; } });
            out.push_back ({ p + ".filter.env_amount",
                             [index] (const Patch& x) { return x.oscs[index].filter.envAmount; },
                             [index] (Patch& x, float v) { x.oscs[index].filter.envAmount = v; } });
            addAdsrAccessors (out, p + ".filter.env");
        }

        addAdsrAccessors (out, "amp_env");

        out.push_back ({ "filter.type",
                         [] (const Patch& x) { return static_cast<float> (x.filter.type); },
                         [] (Patch& x, float v) { x.filter.type =
                             static_cast<FilterType> (juce::jlimit (0, 3, static_cast<int> (std::lround (v)))); } });
        out.push_back ({ "filter.cutoff_hz",
                         [] (const Patch& x) { return x.filter.cutoffHz; },
                         [] (Patch& x, float v) { x.filter.cutoffHz = v; } });
        out.push_back ({ "filter.resonance",
                         [] (const Patch& x) { return x.filter.resonance; },
                         [] (Patch& x, float v) { x.filter.resonance = v; } });
        out.push_back ({ "filter.env_amount",
                         [] (const Patch& x) { return x.filter.envAmount; },
                         [] (Patch& x, float v) { x.filter.envAmount = v; } });
        addAdsrAccessors (out, "filter.env");

        for (int i = 0; i < kNumLfo; ++i)
        {
            const auto p = "lfos." + juce::String (i);
            const auto index = static_cast<size_t> (i);

            out.push_back ({ p + ".shape",
                             [index] (const Patch& x) { return static_cast<float> (x.lfos[index].shape); },
                             [index] (Patch& x, float v) { x.lfos[index].shape =
                                 static_cast<LfoShape> (juce::jlimit (0, 3, static_cast<int> (std::lround (v)))); } });
            out.push_back ({ p + ".dest",
                             [index] (const Patch& x) { return static_cast<float> (x.lfos[index].dest); },
                             [index] (Patch& x, float v) { x.lfos[index].dest =
                                 static_cast<LfoDest> (juce::jlimit (0, 3, static_cast<int> (std::lround (v)))); } });
            out.push_back ({ p + ".rate_hz",
                             [index] (const Patch& x) { return x.lfos[index].rateHz; },
                             [index] (Patch& x, float v) { x.lfos[index].rateHz = v; } });
            out.push_back ({ p + ".depth",
                             [index] (const Patch& x) { return x.lfos[index].depth; },
                             [index] (Patch& x, float v) { x.lfos[index].depth = v; } });
            out.push_back ({ p + ".delay",
                             [index] (const Patch& x) { return x.lfos[index].delay; },
                             [index] (Patch& x, float v) { x.lfos[index].delay = v; } });
            out.push_back ({ p + ".phase",
                             [index] (const Patch& x) { return x.lfos[index].phase; },
                             [index] (Patch& x, float v) { x.lfos[index].phase = v; } });
        }

        out.push_back ({ "delay.enabled",
                         [] (const Patch& x) { return x.delay.enabled ? 1.0f : 0.0f; },
                         [] (Patch& x, float v) { x.delay.enabled = v >= 0.5f; } });
        out.push_back ({ "delay.time",
                         [] (const Patch& x) { return x.delay.time; },
                         [] (Patch& x, float v) { x.delay.time = v; } });
        out.push_back ({ "delay.feedback",
                         [] (const Patch& x) { return x.delay.feedback; },
                         [] (Patch& x, float v) { x.delay.feedback = v; } });
        out.push_back ({ "delay.mix",
                         [] (const Patch& x) { return x.delay.mix; },
                         [] (Patch& x, float v) { x.delay.mix = v; } });

        out.push_back ({ "reverb.enabled",
                         [] (const Patch& x) { return x.reverb.enabled ? 1.0f : 0.0f; },
                         [] (Patch& x, float v) { x.reverb.enabled = v >= 0.5f; } });
        out.push_back ({ "reverb.size",
                         [] (const Patch& x) { return x.reverb.size; },
                         [] (Patch& x, float v) { x.reverb.size = v; } });
        out.push_back ({ "reverb.damp",
                         [] (const Patch& x) { return x.reverb.damp; },
                         [] (Patch& x, float v) { x.reverb.damp = v; } });
        out.push_back ({ "reverb.level",
                         [] (const Patch& x) { return x.reverb.level; },
                         [] (Patch& x, float v) { x.reverb.level = v; } });

        out.push_back ({ "noise_level",
                         [] (const Patch& x) { return x.noiseLevel; },
                         [] (Patch& x, float v) { x.noiseLevel = v; } });
        out.push_back ({ "master_level",
                         [] (const Patch& x) { return x.masterLevel; },
                         [] (Patch& x, float v) { x.masterLevel = v; } });

        return out;
    }();
    return table;
}

} // namespace

juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
{
    Layout layout;
    const Patch defaults;

    for (int i = 0; i < kNumOsc; ++i)
    {
        const auto p = "oscs." + juce::String (i);
        const auto label = "Osc " + juce::String (i + 1) + " ";
        const auto& osc = defaults.oscs[static_cast<size_t> (i)];

        layout.add (std::make_unique<juce::AudioParameterBool> (
            juce::ParameterID { idFor (p + ".enabled"), 1 }, label + "on", osc.enabled));
        layout.add (std::make_unique<juce::AudioParameterChoice> (
            juce::ParameterID { idFor (p + ".waveform"), 1 }, label + "wave",
            kWaveforms, static_cast<int> (osc.waveform)));
        layout.add (std::make_unique<juce::AudioParameterInt> (
            juce::ParameterID { idFor (p + ".semitones"), 1 }, label + "semitones",
            -24, 24, osc.semitones));
        addFloat (layout, p + ".cents", label + "cents", { -50.0f, 50.0f }, osc.cents, "c");
        addFloat (layout, p + ".level", label + "level", { 0.0f, 1.0f }, osc.level);
        addFloat (layout, p + ".pulse_width", label + "width", { 0.05f, 0.95f }, osc.pulseWidth);
        layout.add (std::make_unique<juce::AudioParameterInt> (
            juce::ParameterID { idFor (p + ".unison_voices"), 1 }, label + "unison",
            1, kMaxUnison, osc.unisonVoices));
        addFloat (layout, p + ".unison_detune", label + "detune", { 0.0f, 50.0f }, osc.unisonDetune, "c");
        layout.add (std::make_unique<juce::AudioParameterChoice> (
            juce::ParameterID { idFor (p + ".waveform_b"), 1 }, label + "wave B",
            kWaveforms, static_cast<int> (osc.waveformB)));
        addFloat (layout, p + ".wave_morph", label + "morph", { 0.0f, 1.0f }, osc.waveMorph);
        addFloat (layout, p + ".reverb_send", label + "verb send", { 0.0f, 1.0f }, osc.reverbSend);
        layout.add (std::make_unique<juce::AudioParameterBool> (
            juce::ParameterID { idFor (p + ".env_enabled"), 1 }, label + "own env", osc.envEnabled));
        addAdsr (layout, p + ".env", label + "env", osc.env);

        layout.add (std::make_unique<juce::AudioParameterBool> (
            juce::ParameterID { idFor (p + ".filter_enabled"), 1 }, label + "filter on",
            osc.filterEnabled));
        layout.add (std::make_unique<juce::AudioParameterChoice> (
            juce::ParameterID { idFor (p + ".filter.type"), 1 }, label + "filter type",
            kFilterTypes, static_cast<int> (osc.filter.type)));
        addFloat (layout, p + ".filter.cutoff_hz", label + "filter cutoff",
                  logRange (30.0f, 18000.0f), osc.filter.cutoffHz, "Hz");
        addFloat (layout, p + ".filter.resonance", label + "filter res",
                  logRange (0.5f, 8.0f), osc.filter.resonance);
        addFloat (layout, p + ".filter.env_amount", label + "filter env amt",
                  { -4.0f, 4.0f }, osc.filter.envAmount, "oct");
        addAdsr (layout, p + ".filter.env", label + "filter env", osc.filter.env);
    }

    addAdsr (layout, "amp_env", "Amp", defaults.ampEnv);

    layout.add (std::make_unique<juce::AudioParameterChoice> (
        juce::ParameterID { idFor ("filter.type"), 1 }, "Filter type",
        kFilterTypes, static_cast<int> (defaults.filter.type)));
    addFloat (layout, "filter.cutoff_hz", "Filter cutoff", logRange (30.0f, 18000.0f),
              defaults.filter.cutoffHz, "Hz");
    addFloat (layout, "filter.resonance", "Filter resonance", logRange (0.5f, 8.0f),
              defaults.filter.resonance);
    addFloat (layout, "filter.env_amount", "Filter env amount", { -4.0f, 4.0f },
              defaults.filter.envAmount, "oct");
    addAdsr (layout, "filter.env", "Filter env", defaults.filter.env);

    for (int i = 0; i < kNumLfo; ++i)
    {
        const auto p = "lfos." + juce::String (i);
        const auto label = "LFO " + juce::String (i + 1) + " ";
        const auto& l = defaults.lfos[static_cast<size_t> (i)];

        layout.add (std::make_unique<juce::AudioParameterChoice> (
            juce::ParameterID { idFor (p + ".shape"), 1 }, label + "shape",
            kLfoShapes, static_cast<int> (l.shape)));
        layout.add (std::make_unique<juce::AudioParameterChoice> (
            juce::ParameterID { idFor (p + ".dest"), 1 }, label + "target",
            kLfoDests, static_cast<int> (l.dest)));
        addFloat (layout, p + ".rate_hz", label + "rate", logRange (0.05f, 20.0f), l.rateHz, "Hz");
        addFloat (layout, p + ".depth", label + "depth", { 0.0f, 1.0f }, l.depth);
        addFloat (layout, p + ".delay", label + "fade in", { 0.0f, 2.0f }, l.delay, "s");
        addFloat (layout, p + ".phase", label + "phase", { 0.0f, 1.0f }, l.phase);
    }

    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { idFor ("reverb.enabled"), 1 }, "Reverb on", defaults.reverb.enabled));
    addFloat (layout, "reverb.size", "Reverb size", { 0.0f, 1.0f }, defaults.reverb.size);
    addFloat (layout, "reverb.damp", "Reverb damp", { 0.0f, 1.0f }, defaults.reverb.damp);
    addFloat (layout, "reverb.level", "Reverb level", { 0.0f, 1.0f }, defaults.reverb.level);

    layout.add (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { idFor ("delay.enabled"), 1 }, "Delay on", defaults.delay.enabled));
    addFloat (layout, "delay.time", "Delay time", logRange (0.01f, 1.0f), defaults.delay.time, "s");
    addFloat (layout, "delay.feedback", "Delay feedback", { 0.0f, 0.85f }, defaults.delay.feedback);
    addFloat (layout, "delay.mix", "Delay mix", { 0.0f, 1.0f }, defaults.delay.mix);

    addFloat (layout, "noise_level", "Noise", { 0.0f, 1.0f }, defaults.noiseLevel);
    addFloat (layout, "master_level", "Output", { 0.0f, 1.0f }, defaults.masterLevel);

    return layout;
}

void applyPatch (juce::AudioProcessorValueTreeState& state, const Patch& patch)
{
    for (const auto& accessor : accessors())
    {
        if (auto* parameter = state.getParameter (idFor (accessor.path)))
        {
            const auto raw = accessor.get (patch);
            parameter->setValueNotifyingHost (parameter->convertTo0to1 (raw));
        }
    }
}

Patch toPatch (const juce::AudioProcessorValueTreeState& state, const Patch& previous)
{
    // Starts from `previous` so metadata that is not a parameter -- the root
    // pitch the patch was fitted at, its name -- survives a knob turn.
    auto patch = previous;
    for (const auto& accessor : accessors())
        if (auto* parameter = state.getParameter (idFor (accessor.path)))
            accessor.set (patch, parameter->convertFrom0to1 (parameter->getValue()));
    return patch;
}

} // namespace params
} // namespace autosynth
