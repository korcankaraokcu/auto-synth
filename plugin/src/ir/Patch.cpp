#include "ir/Patch.h"

namespace autosynth
{
namespace
{

float getFloat (const juce::var& obj, const char* key, float fallback)
{
    if (auto* o = obj.getDynamicObject())
        if (o->hasProperty (key))
            return static_cast<float> (double (o->getProperty (key)));
    return fallback;
}

int getInt (const juce::var& obj, const char* key, int fallback)
{
    if (auto* o = obj.getDynamicObject())
        if (o->hasProperty (key))
            return static_cast<int> (o->getProperty (key));
    return fallback;
}

bool getBool (const juce::var& obj, const char* key, bool fallback)
{
    if (auto* o = obj.getDynamicObject())
        if (o->hasProperty (key))
            return static_cast<bool> (o->getProperty (key));
    return fallback;
}

juce::String getString (const juce::var& obj, const char* key, const juce::String& fallback)
{
    if (auto* o = obj.getDynamicObject())
        if (o->hasProperty (key))
            return o->getProperty (key).toString();
    return fallback;
}

juce::var getChild (const juce::var& obj, const char* key)
{
    if (auto* o = obj.getDynamicObject())
        if (o->hasProperty (key))
            return o->getProperty (key);
    return juce::var();
}

// Enums travel as strings in the JSON so a patch stays readable and stays
// valid if the Python enum order ever changes. Matching on the string is the
// whole point -- an index would silently remap.
template <typename T, size_t N>
T parseEnum (const juce::String& text, const std::array<const char*, N>& names, T fallback)
{
    for (size_t i = 0; i < N; ++i)
        if (text == names[i])
            return static_cast<T> (i);
    return fallback;
}

const std::array<const char*, 6> kWaveformNames { "sine", "triangle", "saw", "square",
                                                  "pulse", "noise" };
const std::array<const char*, 4> kFilterNames { "off", "lowpass", "highpass", "bandpass" };
const std::array<const char*, 4> kLfoShapeNames { "sine", "triangle", "saw", "square" };
const std::array<const char*, 6> kLfoDestNames { "none", "pitch", "amp", "cutoff",
                                                 "lfo_rate", "lfo_depth" };

Adsr parseAdsr (const juce::var& v, const Adsr& fallback)
{
    if (! v.isObject())
        return fallback;
    Adsr a;
    a.attack = getFloat (v, "attack", fallback.attack);
    a.decay = getFloat (v, "decay", fallback.decay);
    a.sustain = getFloat (v, "sustain", fallback.sustain);
    a.release = getFloat (v, "release", fallback.release);
    a.curve = getFloat (v, "curve", fallback.curve);
    a.attackCurve = getFloat (v, "attack_curve", fallback.attackCurve);
    return a;
}

} // namespace

int Patch::activeOscCount() const
{
    int n = 0;
    for (const auto& o : oscs)
        if (o.enabled && o.level > 1.0e-4f)
            ++n;
    return n;
}

Patch Patch::fromJson (const juce::var& json)
{
    Patch p;
    if (! json.isObject())
        return p;

    if (auto* oscArray = getChild (json, "oscs").getArray())
    {
        for (int i = 0; i < juce::jmin (kNumOsc, oscArray->size()); ++i)
        {
            const juce::var& v = (*oscArray)[i];
            Oscillator o;
            o.enabled = getBool (v, "enabled", o.enabled);
            o.waveform = parseEnum (getString (v, "waveform", "saw"), kWaveformNames, Waveform::saw);
            o.semitones = getInt (v, "semitones", o.semitones);
            o.cents = getFloat (v, "cents", o.cents);
            o.level = getFloat (v, "level", o.level);
            o.pulseWidth = getFloat (v, "pulse_width", o.pulseWidth);
            o.unisonVoices = juce::jlimit (1, kMaxUnison, getInt (v, "unison_voices", o.unisonVoices));
            o.unisonDetune = getFloat (v, "unison_detune", o.unisonDetune);
            o.waveformB = parseEnum (getString (v, "waveform_b", getString (v, "waveform", "saw")),
                                     kWaveformNames, o.waveform);
            o.waveMorph = getFloat (v, "wave_morph", o.waveMorph);
            o.reverbSend = getFloat (v, "reverb_send", o.reverbSend);
            o.numFrames = juce::jlimit (1, Oscillator::kMaxFrames,
                                        getInt (v, "num_frames", o.numFrames));
            o.framePosition = getFloat (v, "frame_position", o.framePosition);
            o.framePositionEnvAmount = getFloat (v, "frame_position_env_amount",
                                                 o.framePositionEnvAmount);
            o.framePositionEnv = parseAdsr (getChild (v, "frame_position_env"), o.framePositionEnv);
            // Frames are an array of arrays: one harmonic series per drawn
            // frame, and `null` for a frame still generated from the waveform
            // above. A missing array leaves every frame generated, which is
            // what an analog patch is.
            if (auto* rows = getChild (v, "frames").getArray())
            {
                for (int f = 0; f < juce::jmin (Oscillator::kMaxFrames, rows->size()); ++f)
                {
                    auto& frame = o.frames[static_cast<size_t> (f)];
                    if (auto* amps = (*rows)[f].getArray())
                    {
                        frame.custom = true;
                        const auto n = juce::jmin (Oscillator::kFrameHarmonics, amps->size());
                        for (int k = 0; k < n; ++k)
                            frame.harmonics[static_cast<size_t> (k)] =
                                static_cast<float> (static_cast<double> ((*amps)[k]));
                    }
                }
            }
            o.envEnabled = getBool (v, "env_enabled", o.envEnabled);
            o.env = parseAdsr (getChild (v, "env"), o.env);
            o.filterEnabled = getBool (v, "filter_enabled", o.filterEnabled);
            if (auto oscFilter = getChild (v, "filter"); oscFilter.isObject())
            {
                o.filter.type = parseEnum (getString (oscFilter, "type", "lowpass"),
                                           kFilterNames, FilterType::lowpass);
                o.filter.cutoffHz = getFloat (oscFilter, "cutoff_hz", o.filter.cutoffHz);
                o.filter.resonance = getFloat (oscFilter, "resonance", o.filter.resonance);
                o.filter.envAmount = getFloat (oscFilter, "env_amount", o.filter.envAmount);
                o.filter.env = parseAdsr (getChild (oscFilter, "env"), o.filter.env);
            }
            p.oscs[static_cast<size_t> (i)] = o;
        }
    }

    p.ampEnv = parseAdsr (getChild (json, "amp_env"), p.ampEnv);

    if (auto filterVar = getChild (json, "filter"); filterVar.isObject())
    {
        p.filter.type = parseEnum (getString (filterVar, "type", "lowpass"), kFilterNames, FilterType::lowpass);
        p.filter.cutoffHz = getFloat (filterVar, "cutoff_hz", p.filter.cutoffHz);
        p.filter.resonance = getFloat (filterVar, "resonance", p.filter.resonance);
        p.filter.envAmount = getFloat (filterVar, "env_amount", p.filter.envAmount);
        p.filter.env = parseAdsr (getChild (filterVar, "env"), p.filter.env);
    }

    const auto parseLfo = [] (const juce::var& v, Lfo& out)
    {
        out.shape = parseEnum (getString (v, "shape", "sine"), kLfoShapeNames, LfoShape::sine);
        out.dest = parseEnum (getString (v, "dest", "none"), kLfoDestNames, LfoDest::none);
        out.rateHz = getFloat (v, "rate_hz", out.rateHz);
        out.depth = getFloat (v, "depth", out.depth);
        out.delay = getFloat (v, "delay", out.delay);
        out.phase = getFloat (v, "phase", out.phase);
    };

    // Accepts the older single-`lfo` form as well. Patches written before the
    // second slot existed are still valid files, and a reader that rejected
    // them would strand every patch anyone had saved.
    if (auto* lfoArray = getChild (json, "lfos").getArray())
    {
        for (int i = 0; i < juce::jmin (kNumLfo, lfoArray->size()); ++i)
            parseLfo ((*lfoArray)[i], p.lfos[static_cast<size_t> (i)]);
    }
    else if (auto lfoVar = getChild (json, "lfo"); lfoVar.isObject())
    {
        parseLfo (lfoVar, p.lfos[0]);
    }

    if (auto delayVar = getChild (json, "delay"); delayVar.isObject())
    {
        p.delay.enabled = getBool (delayVar, "enabled", p.delay.enabled);
        p.delay.time = getFloat (delayVar, "time", p.delay.time);
        p.delay.feedback = getFloat (delayVar, "feedback", p.delay.feedback);
        p.delay.mix = getFloat (delayVar, "mix", p.delay.mix);
    }

    if (auto reverbVar = getChild (json, "reverb"); reverbVar.isObject())
    {
        p.reverb.enabled = getBool (reverbVar, "enabled", p.reverb.enabled);
        p.reverb.size = getFloat (reverbVar, "size", p.reverb.size);
        p.reverb.damp = getFloat (reverbVar, "damp", p.reverb.damp);
        p.reverb.level = getFloat (reverbVar, "level", p.reverb.level);
    }

    p.noiseLevel = getFloat (json, "noise_level", p.noiseLevel);
    p.masterLevel = getFloat (json, "master_level", p.masterLevel);
    p.rootHz = getFloat (json, "root_hz", p.rootHz);
    p.name = getString (json, "name", p.name);
    return p;
}

namespace
{
juce::var adsrToVar (const Adsr& a)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("attack", a.attack);
    obj->setProperty ("decay", a.decay);
    obj->setProperty ("sustain", a.sustain);
    obj->setProperty ("release", a.release);
    obj->setProperty ("curve", a.curve);
    obj->setProperty ("attack_curve", a.attackCurve);
    return juce::var (obj);
}
} // namespace

juce::String Patch::toJson() const
{
    juce::Array<juce::var> oscArray;
    for (const auto& o : oscs)
    {
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("enabled", o.enabled);
        obj->setProperty ("waveform", kWaveformNames[static_cast<size_t> (o.waveform)]);
        obj->setProperty ("semitones", o.semitones);
        obj->setProperty ("cents", o.cents);
        obj->setProperty ("level", o.level);
        obj->setProperty ("pulse_width", o.pulseWidth);
        obj->setProperty ("unison_voices", o.unisonVoices);
        obj->setProperty ("unison_detune", o.unisonDetune);
        obj->setProperty ("waveform_b", kWaveformNames[static_cast<size_t> (o.waveformB)]);
        obj->setProperty ("wave_morph", o.waveMorph);
        obj->setProperty ("reverb_send", o.reverbSend);
        obj->setProperty ("num_frames", o.numFrames);
        obj->setProperty ("frame_position", o.framePosition);
        obj->setProperty ("frame_position_env_amount", o.framePositionEnvAmount);
        obj->setProperty ("frame_position_env", adsrToVar (o.framePositionEnv));
        // Only the drawn frames are written out. A patch of five analog
        // oscillators should not carry eighty rows of numbers describing a saw
        // that is already named on the line above.
        juce::Array<juce::var> frameArray;
        for (int f = 0; f < o.numFrames; ++f)
        {
            const auto& frame = o.frames[static_cast<size_t> (f)];
            if (! frame.custom)
            {
                frameArray.add (juce::var());
                continue;
            }
            juce::Array<juce::var> amps;
            for (const auto a : frame.harmonics)
                amps.add (a);
            frameArray.add (juce::var (amps));
        }
        obj->setProperty ("frames", juce::var (frameArray));
        obj->setProperty ("env_enabled", o.envEnabled);
        obj->setProperty ("env", adsrToVar (o.env));

        auto* oscFilterObj = new juce::DynamicObject();
        oscFilterObj->setProperty ("type", kFilterNames[static_cast<size_t> (o.filter.type)]);
        oscFilterObj->setProperty ("cutoff_hz", o.filter.cutoffHz);
        oscFilterObj->setProperty ("resonance", o.filter.resonance);
        oscFilterObj->setProperty ("env_amount", o.filter.envAmount);
        oscFilterObj->setProperty ("env", adsrToVar (o.filter.env));
        obj->setProperty ("filter_enabled", o.filterEnabled);
        obj->setProperty ("filter", juce::var (oscFilterObj));
        oscArray.add (juce::var (obj));
    }

    auto* filterObj = new juce::DynamicObject();
    filterObj->setProperty ("type", kFilterNames[static_cast<size_t> (filter.type)]);
    filterObj->setProperty ("cutoff_hz", filter.cutoffHz);
    filterObj->setProperty ("resonance", filter.resonance);
    filterObj->setProperty ("env_amount", filter.envAmount);
    filterObj->setProperty ("env", adsrToVar (filter.env));

    juce::Array<juce::var> lfoArray;
    for (const auto& l : lfos)
    {
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("shape", kLfoShapeNames[static_cast<size_t> (l.shape)]);
        obj->setProperty ("dest", kLfoDestNames[static_cast<size_t> (l.dest)]);
        obj->setProperty ("rate_hz", l.rateHz);
        obj->setProperty ("depth", l.depth);
        obj->setProperty ("delay", l.delay);
        obj->setProperty ("phase", l.phase);
        lfoArray.add (juce::var (obj));
    }

    auto* delayObj = new juce::DynamicObject();
    delayObj->setProperty ("enabled", delay.enabled);
    delayObj->setProperty ("time", delay.time);
    delayObj->setProperty ("feedback", delay.feedback);
    delayObj->setProperty ("mix", delay.mix);

    auto* reverbObj = new juce::DynamicObject();
    reverbObj->setProperty ("enabled", reverb.enabled);
    reverbObj->setProperty ("size", reverb.size);
    reverbObj->setProperty ("damp", reverb.damp);
    reverbObj->setProperty ("level", reverb.level);

    auto* root = new juce::DynamicObject();
    root->setProperty ("oscs", oscArray);
    root->setProperty ("amp_env", adsrToVar (ampEnv));
    root->setProperty ("filter", juce::var (filterObj));
    root->setProperty ("lfos", lfoArray);
    root->setProperty ("delay", juce::var (delayObj));
    root->setProperty ("reverb", juce::var (reverbObj));
    root->setProperty ("noise_level", noiseLevel);
    root->setProperty ("master_level", masterLevel);
    root->setProperty ("root_hz", rootHz);
    root->setProperty ("name", name);
    return juce::JSON::toString (juce::var (root), false);
}

Patch Patch::fromJsonString (const juce::String& text, juce::String* error)
{
    juce::var parsed;
    const auto result = juce::JSON::parse (text, parsed);
    if (result.failed())
    {
        if (error != nullptr)
            *error = result.getErrorMessage();
        return {};
    }
    return fromJson (parsed);
}

Patch Patch::fromFile (const juce::File& file, juce::String* error)
{
    if (! file.existsAsFile())
    {
        if (error != nullptr)
            *error = "no such file: " + file.getFullPathName();
        return {};
    }
    return fromJsonString (file.loadFileAsString(), error);
}

} // namespace autosynth
