#include "fit/Refine.h"

#include "analysis/Stft.h"
#include "dsp/Voice.h"
#include "fit/CmaEs.h"
#include "fit/EnvelopeFit.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace autosynth
{
namespace
{

// Two scales rather than three during the search: the third costs ~40% more per
// evaluation and moves the ranking of candidates very little.
constexpr int kSearchScales[] = { 512, 2048 };
constexpr double kDynamicRangeDb = 80.0;

// Floors stop a term that begins near-perfect from acquiring an enormous weight
// and dominating the search.
const std::map<std::string, double> kFloors {
    { "spectral", 0.10 }, { "loudness", 0.50 }, { "centroid", 0.05 }
};

std::vector<float> envelopeDb (const float* samples, int numSamples, double sampleRate)
{
    auto env = Stft::loudnessEnvelope (samples, numSamples, 256);
    const auto peak = env.empty() ? 0.0f : *std::max_element (env.begin(), env.end());
    if (peak <= 1.0e-12f)
    {
        std::fill (env.begin(), env.end(), static_cast<float> (-kDynamicRangeDb));
        return env;
    }
    for (auto& v : env)
        v = static_cast<float> (std::max (20.0 * std::log10 (v / peak + 1.0e-12), -kDynamicRangeDb));
    return env;
}

struct TargetFeatures
{
    std::vector<Stft::Result> spectrograms;
    std::vector<float> loudnessDb;
    std::vector<float> centroidLog2;

    TargetFeatures (const float* samples, int numSamples, double sampleRate)
    {
        for (auto scale : kSearchScales)
            spectrograms.push_back (Stft::magnitudeSpectrogram (samples, numSamples, scale,
                                                                std::max (1, scale / 4), sampleRate));
        loudnessDb = envelopeDb (samples, numSamples, sampleRate);

        const auto centroid = Stft::spectralCentroid (spectrograms.back());
        centroidLog2.resize (centroid.size());
        for (size_t i = 0; i < centroid.size(); ++i)
            centroidLog2[i] = static_cast<float> (std::log2 (std::max (centroid[i], 20.0f)));
    }
};

Refine::Loss lossComponents (const TargetFeatures& target, const float* samples,
                             int numSamples, double sampleRate)
{
    Refine::Loss loss;

    for (size_t s = 0; s < std::size (kSearchScales); ++s)
    {
        const auto scale = kSearchScales[s];
        const auto spectrogram = Stft::magnitudeSpectrogram (samples, numSamples, scale,
                                                             std::max (1, scale / 4), sampleRate);
        const auto& reference = target.spectrograms[s];
        const auto frames = std::min (spectrogram.numFrames, reference.numFrames);
        const auto bins = std::min (spectrogram.numBins, reference.numBins);
        if (frames <= 0 || bins <= 0)
            continue;

        double linear = 0.0, logarithmic = 0.0;
        for (int f = 0; f < frames; ++f)
        {
            const auto* a = reference.frame (f);
            const auto* b = spectrogram.frame (f);
            for (int k = 0; k < bins; ++k)
            {
                linear += std::abs (a[k] - b[k]);
                logarithmic += std::abs (std::log (a[k] + 1.0e-7) - std::log (b[k] + 1.0e-7));
            }
        }
        const auto count = static_cast<double> (frames) * bins;
        loss.spectral += linear / count + logarithmic / count;
    }
    loss.spectral /= static_cast<double> (std::size (kSearchScales));

    const auto loud = envelopeDb (samples, numSamples, sampleRate);
    const auto loudCount = std::min (loud.size(), target.loudnessDb.size());
    for (size_t i = 0; i < loudCount; ++i)
        loss.loudness += std::abs (target.loudnessDb[i] - loud[i]);
    loss.loudness /= std::max<size_t> (loudCount, 1);

    const auto spectrogram = Stft::magnitudeSpectrogram (samples, numSamples, kSearchScales[1],
                                                         kSearchScales[1] / 4, sampleRate);
    const auto centroid = Stft::spectralCentroid (spectrogram);
    const auto centroidCount = std::min (centroid.size(), target.centroidLog2.size());
    for (size_t i = 0; i < centroidCount; ++i)
        loss.centroid += std::abs (target.centroidLog2[i]
                                   - static_cast<float> (std::log2 (std::max (centroid[i], 20.0f))));
    loss.centroid /= std::max<size_t> (centroidCount, 1);

    return loss;
}

// --- flat parameter view ---------------------------------------------------
//
// Mirrors ir.PARAM_SPECS for the continuous parameters only. Encoding is
// normalised to [0, 1] with the same log/linear mapping the Python side uses,
// so a scope entry means the same thing on both sides.

struct Spec
{
    std::string path;
    double lo;
    double hi;
    bool logScale;
};

double encode (const Spec& spec, double value)
{
    if (spec.logScale)
    {
        const auto lo = std::log (spec.lo), hi = std::log (spec.hi);
        return juce::jlimit (0.0, 1.0, (std::log (std::max (value, 1.0e-9)) - lo) / (hi - lo));
    }
    return juce::jlimit (0.0, 1.0, (value - spec.lo) / (spec.hi - spec.lo));
}

double decode (const Spec& spec, double x)
{
    x = juce::jlimit (0.0, 1.0, x);
    if (spec.logScale)
    {
        const auto lo = std::log (spec.lo), hi = std::log (spec.hi);
        return std::exp (lo + x * (hi - lo));
    }
    return spec.lo + x * (spec.hi - spec.lo);
}

const std::map<std::string, Spec>& specTable()
{
    static const std::map<std::string, Spec> table = [] {
        std::map<std::string, Spec> t;
        const auto addAdsr = [&t] (const std::string& prefix, double maxAttack)
        {
            t[prefix + ".attack"] = { prefix + ".attack", 0.001, maxAttack, true };
            t[prefix + ".decay"] = { prefix + ".decay", 0.005, 4.0, true };
            t[prefix + ".sustain"] = { prefix + ".sustain", 0.0, 1.0, false };
            t[prefix + ".release"] = { prefix + ".release", 0.005, 4.0, true };
            t[prefix + ".curve"] = { prefix + ".curve", 0.0, 8.0, false };
        };

        for (int i = 0; i < kNumOsc; ++i)
        {
            const auto p = "oscs." + std::to_string (i);
            t[p + ".cents"] = { p + ".cents", -50.0, 50.0, false };
            t[p + ".level"] = { p + ".level", 0.0, 1.0, false };
            t[p + ".pulse_width"] = { p + ".pulse_width", 0.05, 0.95, false };
            t[p + ".unison_detune"] = { p + ".unison_detune", 0.0, 50.0, false };
            t[p + ".wave_morph"] = { p + ".wave_morph", 0.0, 1.0, false };
            t[p + ".reverb_send"] = { p + ".reverb_send", 0.0, 1.0, false };
            addAdsr (p + ".env", 2.0);
            t[p + ".filter.cutoff_hz"] = { p + ".filter.cutoff_hz", 30.0, 18000.0, true };
            t[p + ".filter.resonance"] = { p + ".filter.resonance", 0.5, 8.0, true };
            t[p + ".filter.env_amount"] = { p + ".filter.env_amount", -4.0, 4.0, false };
            addAdsr (p + ".filter.env", 2.0);
        }
        addAdsr("amp_env", 2.0);
        addAdsr("filter.env", 2.0);
        t["filter.cutoff_hz"] = { "filter.cutoff_hz", 30.0, 18000.0, true };
        t["filter.resonance"] = { "filter.resonance", 0.5, 8.0, true };
        t["filter.env_amount"] = { "filter.env_amount", -4.0, 4.0, false };
        for (int i = 0; i < kNumLfo; ++i)
        {
            const auto p = "lfos." + std::to_string (i);
            t[p + ".rate_hz"] = { p + ".rate_hz", 0.05, 20.0, true };
            t[p + ".depth"] = { p + ".depth", 0.0, 1.0, false };
            t[p + ".delay"] = { p + ".delay", 0.0, 2.0, false };
            t[p + ".phase"] = { p + ".phase", 0.0, 1.0, false };
        }
            t["reverb.size"] = { "reverb.size", 0.0, 1.0, false };
        t["reverb.damp"] = { "reverb.damp", 0.0, 1.0, false };
        t["reverb.level"] = { "reverb.level", 0.0, 1.0, false };
        t["delay.time"] = { "delay.time", 0.01, 1.0, true };
        t["delay.feedback"] = { "delay.feedback", 0.0, 0.85, false };
        t["delay.mix"] = { "delay.mix", 0.0, 1.0, false };
        t["noise_level"] = { "noise_level", 0.0, 1.0, false };
        t["master_level"] = { "master_level", 0.0, 1.0, false };
        return t;
    }();
    return table;
}

Adsr* adsrFor (Patch& patch, const std::string& prefix)
{
    if (prefix == "amp_env")
        return &patch.ampEnv;
    if (prefix == "filter.env")
        return &patch.filter.env;
    if (prefix.rfind ("oscs.", 0) == 0)
    {
        const auto index = prefix[5] - '0';
        if (index < 0 || index >= kNumOsc)
            return nullptr;
        auto& osc = patch.oscs[static_cast<size_t> (index)];
        const auto isFilterEnv = prefix.size() > 11
                              && prefix.compare (prefix.size() - 11, 11, ".filter.env") == 0;
        return isFilterEnv ? &osc.filter.env : &osc.env;
    }
    return nullptr;
}

double getParameter (const Patch& patch, const std::string& path)
{
    auto& mutablePatch = const_cast<Patch&> (patch);
    const auto dot = path.rfind ('.');
    const auto prefix = path.substr (0, dot);
    const auto leaf = path.substr (dot + 1);

    if (auto* adsr = adsrFor (mutablePatch, prefix))
    {
        if (leaf == "attack")  return adsr->attack;
        if (leaf == "decay")   return adsr->decay;
        if (leaf == "sustain") return adsr->sustain;
        if (leaf == "release") return adsr->release;
        if (leaf == "curve")   return adsr->curve;
    }
    if (path == "filter.cutoff_hz")  return patch.filter.cutoffHz;
    if (path == "filter.resonance")  return patch.filter.resonance;
    if (path == "filter.env_amount") return patch.filter.envAmount;
    if (prefix.size() == 6 && prefix.rfind ("lfos.", 0) == 0)
    {
        const auto& l = patch.lfos[static_cast<size_t> (prefix[5] - '0')];
        if (leaf == "rate_hz") return l.rateHz;
        if (leaf == "depth")   return l.depth;
        if (leaf == "delay")   return l.delay;
        if (leaf == "phase")   return l.phase;
    }
    if (path == "reverb.size")       return patch.reverb.size;
    if (path == "reverb.damp")       return patch.reverb.damp;
    if (path == "reverb.level")      return patch.reverb.level;
    if (path == "delay.time")        return patch.delay.time;
    if (path == "delay.feedback")    return patch.delay.feedback;
    if (path == "delay.mix")         return patch.delay.mix;
    if (path == "noise_level")       return patch.noiseLevel;
    if (path == "master_level")      return patch.masterLevel;

    if (prefix.rfind ("oscs.", 0) == 0 && prefix.size() == 6)
    {
        const auto& osc = patch.oscs[static_cast<size_t> (prefix[5] - '0')];
        if (leaf == "cents")         return osc.cents;
        if (leaf == "level")         return osc.level;
        if (leaf == "pulse_width")   return osc.pulseWidth;
        if (leaf == "unison_detune") return osc.unisonDetune;
        if (leaf == "wave_morph")    return osc.waveMorph;
        if (leaf == "reverb_send")   return osc.reverbSend;
    }
    if (prefix.size() == 13 && prefix.rfind ("oscs.", 0) == 0
        && prefix.compare (6, 7, ".filter") == 0)
    {
        const auto& osc = patch.oscs[static_cast<size_t> (prefix[5] - '0')];
        if (leaf == "cutoff_hz")  return osc.filter.cutoffHz;
        if (leaf == "resonance")  return osc.filter.resonance;
        if (leaf == "env_amount") return osc.filter.envAmount;
    }
    return 0.0;
}

void setParameter (Patch& patch, const std::string& path, double value)
{
    const auto dot = path.rfind ('.');
    const auto prefix = path.substr (0, dot);
    const auto leaf = path.substr (dot + 1);

    if (auto* adsr = adsrFor (patch, prefix))
    {
        if (leaf == "attack")  { adsr->attack = static_cast<float> (value); return; }
        if (leaf == "decay")   { adsr->decay = static_cast<float> (value); return; }
        if (leaf == "sustain") { adsr->sustain = static_cast<float> (value); return; }
        if (leaf == "release") { adsr->release = static_cast<float> (value); return; }
        if (leaf == "curve")   { adsr->curve = static_cast<float> (value); return; }
    }
    if (path == "filter.cutoff_hz")  { patch.filter.cutoffHz = static_cast<float> (value); return; }
    if (path == "filter.resonance")  { patch.filter.resonance = static_cast<float> (value); return; }
    if (path == "filter.env_amount") { patch.filter.envAmount = static_cast<float> (value); return; }
    if (prefix.size() == 6 && prefix.rfind ("lfos.", 0) == 0)
    {
        auto& l = patch.lfos[static_cast<size_t> (prefix[5] - '0')];
        if (leaf == "rate_hz")    { l.rateHz = static_cast<float> (value); return; }
        if (leaf == "depth")      { l.depth = static_cast<float> (value); return; }
        if (leaf == "delay")      { l.delay = static_cast<float> (value); return; }
        if (leaf == "phase")      { l.phase = static_cast<float> (value); return; }
    }
    if (path == "reverb.size")       { patch.reverb.size = static_cast<float> (value); return; }
    if (path == "reverb.damp")       { patch.reverb.damp = static_cast<float> (value); return; }
    if (path == "reverb.level")      { patch.reverb.level = static_cast<float> (value); return; }
    if (path == "delay.time")        { patch.delay.time = static_cast<float> (value); return; }
    if (path == "delay.feedback")    { patch.delay.feedback = static_cast<float> (value); return; }
    if (path == "delay.mix")         { patch.delay.mix = static_cast<float> (value); return; }
    if (path == "noise_level")       { patch.noiseLevel = static_cast<float> (value); return; }
    if (path == "master_level")      { patch.masterLevel = static_cast<float> (value); return; }

    if (prefix.rfind ("oscs.", 0) == 0 && prefix.size() == 6)
    {
        auto& osc = patch.oscs[static_cast<size_t> (prefix[5] - '0')];
        if (leaf == "cents")         osc.cents = static_cast<float> (value);
        else if (leaf == "level")    osc.level = static_cast<float> (value);
        else if (leaf == "pulse_width")   osc.pulseWidth = static_cast<float> (value);
        else if (leaf == "unison_detune") osc.unisonDetune = static_cast<float> (value);
        else if (leaf == "wave_morph")    osc.waveMorph = static_cast<float> (value);
        else if (leaf == "reverb_send")   osc.reverbSend = static_cast<float> (value);
        return;
    }
    if (prefix.size() == 13 && prefix.rfind ("oscs.", 0) == 0
        && prefix.compare (6, 7, ".filter") == 0)
    {
        auto& osc = patch.oscs[static_cast<size_t> (prefix[5] - '0')];
        if (leaf == "cutoff_hz")       osc.filter.cutoffHz = static_cast<float> (value);
        else if (leaf == "resonance")  osc.filter.resonance = static_cast<float> (value);
        else if (leaf == "env_amount") osc.filter.envAmount = static_cast<float> (value);
    }
}

} // namespace

std::vector<std::string> Refine::scopeFor (const Patch& patch)
{
    std::vector<std::string> paths;
    const auto& table = specTable();

    for (int i = 0; i < kNumOsc; ++i)
    {
        const auto& osc = patch.oscs[static_cast<size_t> (i)];
        if (! osc.enabled || osc.level <= 1.0e-4f)
            continue;
        const auto p = "oscs." + std::to_string (i);
        paths.push_back (p + ".cents");
        paths.push_back (p + ".level");
        paths.push_back (p + ".pulse_width");
        paths.push_back (p + ".unison_detune");
        paths.push_back (p + ".wave_morph");
        paths.push_back (p + ".reverb_send");
        if (osc.envEnabled)
            for (const char* leaf : { "attack", "decay", "sustain", "release", "curve" })
                paths.push_back (p + ".env." + leaf);

        if (osc.filterEnabled)
        {
            paths.push_back (p + ".filter.cutoff_hz");
            paths.push_back (p + ".filter.resonance");
            paths.push_back (p + ".filter.env_amount");
            for (const char* leaf : { "attack", "decay", "sustain", "release", "curve" })
                paths.push_back (p + ".filter.env." + leaf);
        }
    }

    for (const char* leaf : { "attack", "decay", "sustain", "release", "curve" })
        paths.push_back (std::string ("amp_env.") + leaf);

    paths.push_back ("filter.cutoff_hz");
    paths.push_back ("filter.resonance");
    paths.push_back ("filter.env_amount");
    if (patch.filter.envAmount > 0.0f)
        for (const char* leaf : { "attack", "decay", "sustain", "release", "curve" })
            paths.push_back (std::string ("filter.env.") + leaf);

    for (int i = 0; i < kNumLfo; ++i)
    {
        if (patch.lfos[static_cast<size_t> (i)].depth <= 0.0f)
            continue;
        // Phase included deliberately: analysis recovers rate and shape well but
        // depth only approximately -- the f0 tracker's own window low-passes
        // vibrato before it is measured -- and phase not reliably at all.
        const auto p = "lfos." + std::to_string (i);
        for (const char* leaf : { ".rate_hz", ".depth", ".delay", ".phase" })
            paths.push_back (p + leaf);
    }

    if (patch.reverb.enabled)
    {
        paths.push_back ("reverb.size");
        paths.push_back ("reverb.damp");
        paths.push_back ("reverb.level");
    }

    if (patch.delay.enabled)
    {
        paths.push_back ("delay.time");
        paths.push_back ("delay.feedback");
        paths.push_back ("delay.mix");
    }

    paths.push_back ("noise_level");
    paths.push_back ("master_level");

    std::vector<std::string> valid;
    for (auto& path : paths)
        if (table.find (path) != table.end())
            valid.push_back (path);
    return valid;
}

Refine::Result Refine::run (const Patch& patch, const float* target, int numSamples,
                            double sampleRate, const Options& options)
{
    Result result;
    result.patch = patch;

    const auto scope = scopeFor (patch);
    if (scope.size() < 2 || numSamples <= 0)
        return result;

    const auto& table = specTable();
    const auto duration = numSamples / sampleRate;
    const auto gate = options.gateSeconds >= 0.0 ? options.gateSeconds : duration;

    std::vector<double> x0 (scope.size());
    for (size_t i = 0; i < scope.size(); ++i)
        x0[i] = encode (table.at (scope[i]), getParameter (patch, scope[i]));

    const auto build = [&] (const std::vector<double>& x)
    {
        auto out = patch;
        for (size_t i = 0; i < scope.size(); ++i)
            setParameter (out, scope[i], decode (table.at (scope[i]), x[i]));
        return out;
    };

    const TargetFeatures features (target, numSamples, sampleRate);

    Engine engine;
    engine.prepare (sampleRate, 512);

    const auto measure = [&] (const Patch& candidate)
    {
        engine.setPatch (candidate);
        juce::AudioBuffer<float> buffer;
        engine.renderOffline (buffer, candidate.rootHz, duration, gate);
        return lossComponents (features, buffer.getReadPointer (0), buffer.getNumSamples(), sampleRate);
    };

    // Adaptive weights: each term normalised by its own value at the starting
    // patch, so all three contribute equally and the optimiser is asked to
    // improve them *proportionally*.
    //
    // Fixed weights were tried and are subtly wrong -- they encode an assumption
    // about typical error magnitudes, and material that violates it gets the
    // wrong trade. With a pure spectral objective the optimiser reliably spent
    // the envelope to buy spectral accuracy.
    const auto start = measure (patch);
    const auto weightFor = [] (double value, const char* key)
    {
        return 1.0 / std::max (value, kFloors.at (key));
    };
    const auto wSpectral = weightFor (start.spectral, "spectral");
    const auto wLoudness = weightFor (start.loudness, "loudness");
    const auto wCentroid = weightFor (start.centroid, "centroid");

    const auto combine = [&] (const Loss& loss)
    {
        return (wSpectral * loss.spectral + wLoudness * loss.loudness
                + wCentroid * loss.centroid) / 3.0;
    };

    CmaEs::Options cmaOptions;
    cmaOptions.maxEvaluations = options.maxEvaluations;
    cmaOptions.populationSize = options.populationSize;
    cmaOptions.sigma = options.sigma;
    cmaOptions.seed = options.seed;

    const auto objective = [&] (const std::vector<std::vector<double>>& population)
    {
        std::vector<double> values;
        values.reserve (population.size());
        for (const auto& candidate : population)
            values.push_back (combine (measure (build (candidate))));
        return values;
    };

    const auto search = CmaEs::minimise (x0, cmaOptions, objective);

    result.initialLoss = search.initialValue;
    result.finalLoss = search.bestValue;
    result.evaluations = search.evaluations;
    result.improved = search.improved;

    // Never worse than what it was given. A refinement stage that occasionally
    // regresses is worse than none: it invalidates every number downstream.
    if (search.improved)
        result.patch = build (search.best);
    return result;
}

} // namespace autosynth
