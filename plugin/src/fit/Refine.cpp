#include "fit/Refine.h"

#include "analysis/Stft.h"
#include "dsp/Envelope.h"
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
    { "spectral", 0.10 }, { "loudness", 0.50 }, { "centroid", 0.05 },
    // Level is a median over the note, so half a decibel is already below what
    // anyone would call a difference.
    { "level", 0.50 },
    // The trajectory terms are errors against a measurement rather than
    // distances, so their floors are the tolerance below which the diagnostic
    // would call them equal: 30 ms of attack, and a decibel of movement.
    { "attack", 0.030 }, { "drift", 1.00 }, { "wobble", 0.50 },
    // A twelfth of full scale: below that, two onsets sound like each other.
    { "onset", 0.08 }
};

// Loudness in dB against a *fixed* reference, not against the signal's own peak.
//
// Normalising each signal by its own peak made this term completely blind to
// level: a candidate ten decibels too quiet scored exactly as well as one that
// matched. Nothing else in the objective anchors absolute level either -- the
// spectral log-term actively prefers a quiet render, because a fit carries more
// energy than the target in the many near-silent bins and turning everything
// down moves all of them closer to the target's floor.
//
// The result was refinement reliably making patches quieter: measured on two
// library samples it dropped them 8 to 11 dB below the source while its own
// loss improved, so a refined fit sounded worse than the unrefined one it
// started from.
//
// Passing the target's peak as the reference for both makes the term mean what
// its name says. The floor stays relative to that same reference, so a decayed
// tail is still compared against a floor rather than against digital silence.
std::vector<float> envelopeDb (const float* samples, int numSamples, double sampleRate,
                               float referencePeak)
{
    juce::ignoreUnused (sampleRate);
    auto env = Stft::loudnessEnvelope (samples, numSamples, 256);
    if (referencePeak <= 1.0e-12f)
    {
        std::fill (env.begin(), env.end(), static_cast<float> (-kDynamicRangeDb));
        return env;
    }
    for (auto& v : env)
        v = static_cast<float> (std::max (20.0 * std::log10 (v / referencePeak + 1.0e-12),
                                          -kDynamicRangeDb));
    return env;
}

// The note's own loudness: the median of the frames before it is released.
//
// A median rather than a mean, so that the attack transient and the tail both
// stay out of it, and only over the note because the tail is where a fit and a
// recording disagree most and by the largest margin.
//
// This exists because nothing else in the objective anchors absolute level
// reliably. The loudness term is a mean over *all* frames, so a tail 20 dB too
// loud outweighs a body 3 dB too quiet and the cheapest way to reduce it is to
// turn the whole patch down. And the spectral term pulls the other way -- it
// falls monotonically as a fit gets louder, measured at 2.36 to 2.27 over a
// 7 dB rise. Between the two, the level a listener hears was whatever those
// biases happened to cancel to, which is why the same material fitted from
// three different seeds came back 6.9 dB quiet, correct, and correct again.
double bodyDb (const std::vector<float>& envelopeInDb, double gate, double sampleRate)
{
    if (envelopeInDb.empty())
        return -kDynamicRangeDb;

    const auto dt = 256.0 / sampleRate;
    const auto gateIndex = juce::jlimit (1, static_cast<int> (envelopeInDb.size()),
                                         static_cast<int> (std::lround (gate / dt)));

    std::vector<float> body (envelopeInDb.begin(), envelopeInDb.begin() + gateIndex);
    std::sort (body.begin(), body.end());
    return body[body.size() / 2];
}

float envelopePeak (const float* samples, int numSamples)
{
    const auto env = Stft::loudnessEnvelope (samples, numSamples, 256);
    return env.empty() ? 0.0f : *std::max_element (env.begin(), env.end());
}

// --- the trajectory measurements -------------------------------------------
//
// The diagnostic's own quantities, computed here the same way, so that a change
// which improves the loss is a change that improves what the diagnostic reports
// rather than something orthogonal to it.

constexpr int kDriftHarmonics = 6;

// Mean absolute change in the first few harmonics between the note's early and
// late thirds, in decibels, each profile peak-normalised first so this is a
// statement about balance rather than level.
//
// Read straight off the spectrogram at multiples of the known fundamental
// instead of from tracked partials. The diagnostic tracks partials, which costs
// far too much to do a hundred and ninety-two times, and the harmonic bins are
// the same measurement to within the vibrato that both signals share.
double harmonicDriftDb (const Stft::Result& spectrogram, double f0)
{
    if (f0 < 20.0 || spectrogram.numFrames < 6 || spectrogram.numBins < 4)
        return 0.0;

    const auto binNear = [&spectrogram] (double hz)
    {
        auto best = 1;
        auto bestDistance = std::abs (spectrogram.frequencies[1] - hz);
        for (int k = 2; k < spectrogram.numBins; ++k)
        {
            const auto distance = std::abs (spectrogram.frequencies[(size_t) k] - hz);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = k;
            }
        }
        return best;
    };

    const auto profileOver = [&] (int from, int to)
    {
        std::array<double, kDriftHarmonics> profile {};
        for (int h = 0; h < kDriftHarmonics; ++h)
        {
            const auto bin = binNear (f0 * (h + 1));
            if (bin <= 0 || bin >= spectrogram.numBins)
                continue;

            double acc = 0.0;
            for (int t = from; t < to; ++t)
                acc += spectrogram.frame (t)[bin];
            profile[(size_t) h] = acc / std::max (1, to - from);
        }

        const auto peak = *std::max_element (profile.begin(), profile.end());
        if (peak > 1.0e-12)
            for (auto& v : profile)
                v /= peak;
        return profile;
    };

    const auto third = std::max (1, spectrogram.numFrames / 3);
    const auto early = profileOver (0, third);
    const auto late = profileOver (spectrogram.numFrames - third, spectrogram.numFrames);

    const auto asDb = [] (double v) { return 20.0 * std::log10 (std::max (v, 1.0e-4)); };
    double drift = 0.0;
    for (size_t h = 0; h < early.size(); ++h)
        drift += std::abs (asDb (late[h]) - asDb (early[h]));
    return drift / (double) early.size();
}

struct Contour
{
    double attackSeconds = 0.0;
    double wobbleDb = 0.0;
    double onset = 0.0;
};

Contour contourOf (const float* samples, int numSamples, double sampleRate, double gate)
{
    Contour out;
    const auto env = Stft::loudnessEnvelope (samples, numSamples, 256);
    if (env.size() < 8)
        return out;

    const auto fps = sampleRate / 256.0;
    std::vector<float> times (env.size());
    for (size_t i = 0; i < times.size(); ++i)
        times[i] = static_cast<float> ((double) i / fps);

    out.attackSeconds = EnvelopeFit::attackSeconds (env, times);

    // How much of the note is there a twentieth of a second in, against the
    // loudest it ever gets. Attack *time* is the moment a threshold is crossed
    // and says nothing about the shape of the rise, so an onset that fades in
    // and one that arrives can measure the same -- which is how an envelope came
    // to be described as too slow while its attack read fine.
    {
        const auto loudest = *std::max_element (env.begin(), env.end());
        const auto at = static_cast<size_t> (0.05 * fps);
        if (loudest > 1.0e-9f && at < env.size())
            out.onset = env[at] / loudest;
    }

    // Wobble as the rms deviation of the sustained middle in decibels, measured
    // over the same window the diagnostic uses -- past the attack and short of
    // the release, so neither reads as movement.
    const auto lo = (size_t) std::max (0.0, juce::jlimit (0.0, gate * 0.7, 0.3) * fps);
    const auto hi = (size_t) std::max (0.0, std::min ((double) env.size(), gate * 0.95 * fps));
    if (hi <= lo + 8)
        return out;

    const auto peak = *std::max_element (env.begin(), env.end());
    if (peak <= 1.0e-9f)
        return out;

    double mean = 0.0;
    std::vector<double> db;
    db.reserve (hi - lo);
    for (auto i = lo; i < hi; ++i)
    {
        db.push_back (20.0 * std::log10 (std::max<double> (env[i], peak * 1.0e-4) / peak));
        mean += db.back();
    }
    mean /= (double) db.size();

    double sumSquares = 0.0;
    for (const auto v : db)
        sumSquares += (v - mean) * (v - mean);
    out.wobbleDb = std::sqrt (sumSquares / (double) db.size());
    return out;
}

struct TargetFeatures
{
    std::vector<Stft::Result> spectrograms;
    std::vector<float> loudnessDb;
    double bodyLevelDb = 0.0;
    std::vector<float> centroidLog2;
    float loudnessPeak = 0.0f;
    double attackSeconds = 0.0;
    double driftDb = 0.0;
    double wobbleDb = 0.0;
    double onset = 0.0;

    TargetFeatures (const float* samples, int numSamples, double sampleRate,
                    double gate, double f0)
    {
        loudnessPeak = envelopePeak (samples, numSamples);
        for (auto scale : kSearchScales)
            spectrograms.push_back (Stft::magnitudeSpectrogram (samples, numSamples, scale,
                                                                std::max (1, scale / 4), sampleRate));
        loudnessDb = envelopeDb (samples, numSamples, sampleRate, loudnessPeak);
        bodyLevelDb = bodyDb (loudnessDb, gate, sampleRate);

        const auto centroid = Stft::spectralCentroid (spectrograms.back());
        centroidLog2.resize (centroid.size());
        for (size_t i = 0; i < centroid.size(); ++i)
            centroidLog2[i] = static_cast<float> (std::log2 (std::max (centroid[i], 20.0f)));

        const auto contour = contourOf (samples, numSamples, sampleRate, gate);
        attackSeconds = contour.attackSeconds;
        wobbleDb = contour.wobbleDb;
        onset = contour.onset;
        driftDb = harmonicDriftDb (spectrograms.back(), f0);
    }
};

Refine::Loss lossComponents (const TargetFeatures& target, const float* samples,
                             int numSamples, double sampleRate, double gate, double f0)
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

    const auto loud = envelopeDb (samples, numSamples, sampleRate, target.loudnessPeak);
    const auto loudCount = std::min (loud.size(), target.loudnessDb.size());
    for (size_t i = 0; i < loudCount; ++i)
        loss.loudness += std::abs (target.loudnessDb[i] - loud[i]);
    loss.loudness /= std::max<size_t> (loudCount, 1);

    // Absolute, because `loud` is already referenced to the *target's* peak.
    loss.level = std::abs (target.bodyLevelDb - bodyDb (loud, gate, sampleRate));

    const auto spectrogram = Stft::magnitudeSpectrogram (samples, numSamples, kSearchScales[1],
                                                         kSearchScales[1] / 4, sampleRate);
    const auto centroid = Stft::spectralCentroid (spectrogram);
    const auto centroidCount = std::min (centroid.size(), target.centroidLog2.size());
    for (size_t i = 0; i < centroidCount; ++i)
        loss.centroid += std::abs (target.centroidLog2[i]
                                   - static_cast<float> (std::log2 (std::max (centroid[i], 20.0f))));
    loss.centroid /= std::max<size_t> (centroidCount, 1);

    // The trajectory terms. Each is the distance between a measurement of the
    // candidate and the same measurement of the target, in the unit the
    // diagnostic reports it in.
    const auto contour = contourOf (samples, numSamples, sampleRate, gate);
    loss.attack = std::abs (contour.attackSeconds - target.attackSeconds);
    loss.wobble = std::abs (contour.wobbleDb - target.wobbleDb);
    loss.onset = std::abs (contour.onset - target.onset);
    loss.drift = std::abs (harmonicDriftDb (spectrogram, f0) - target.driftDb);

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
            // Much narrower than the IR allows.
            //
            // Analysis pins pitch to within a few cents, and refinement given
            // the full range walks away from it: a clarinet fitted at exactly
            // the right pitch came out 31 cents sharp after refinement, which
            // is plainly audible while barely moving a magnitude spectrum --
            // the bins are wider than the error. Detuning is a precision knob
            // here, not a way to buy spectral distance.
            t[p + ".cents"] = { p + ".cents", -15.0, 15.0, false };
            t[p + ".level"] = { p + ".level", 0.0, 1.0, false };
            t[p + ".pulse_width"] = { p + ".pulse_width", 0.05, 0.95, false };
            t[p + ".unison_detune"] = { p + ".unison_detune", 0.0, 50.0, false };
            t[p + ".wave_morph"] = { p + ".wave_morph", 0.0, 1.0, false };
            addAdsr (p + ".env", 2.0);
        }
        addAdsr("amp_env", 2.0);
        // Only the amplitude envelope's, and only now that there is a
        // measurement of what it does: how much of the note is present fifty
        // milliseconds in. `fitAttackCurve` picks it from the contour and
        // picks too gently -- a fit reached 0.20 of its peak there where the
        // clarinet recording is at 0.45 -- and before the onset term below
        // there was nothing for a search to improve.
        t["amp_env.attack_curve"] = { "amp_env.attack_curve", 0.0, 8.0, false };
        addAdsr("filter.env", 2.0);
        t["filter.cutoff_hz"] = { "filter.cutoff_hz", 30.0, 18000.0, true };
        t["filter.resonance"] = { "filter.resonance", 0.5, 8.0, true };
        t["filter.env_amount"] = { "filter.env_amount", -4.0, 4.0, false };
        for (int i = 0; i < kNumLfo; ++i)
        {
            const auto p = "lfos." + std::to_string (i);
            // Capped well below the IR limit. Twenty hertz is not a
            // modulation rate, it is roughness, and refinement drove a
            // correctly-measured 3.3 Hz tremolo straight to the ceiling
            // because smearing the spectrum lowered the distance. Analysis
            // measures rate accurately now -- on a violin, 6.0 Hz against a
            // true 5.8 -- so this only needs room to polish.
            t[p + ".rate_hz"] = { p + ".rate_hz", 0.05, 12.0, true };
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
        // Bounded for the same reason PartialFit caps it: filling empty bins
        // lowers a log-spectral error, so an unbounded noise level is a free
        // win for the objective and a loss for the ear.
        t["noise_level"] = { "noise_level", 0.0, 0.25, false };
        t["master_level"] = { "master_level", 0.0, kMaxMasterLevel, false };
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
        return &patch.oscs[static_cast<size_t> (index)].env;
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
        if (leaf == "attack_curve") return adsr->attackCurve;
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
        if (leaf == "attack_curve") { adsr->attackCurve = static_cast<float> (value); return; }
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
        return;
    }
}

} // namespace

std::vector<Refine::ParamSpec> Refine::continuousSpecs()
{
    std::vector<ParamSpec> out;
    out.reserve (specTable().size());
    for (const auto& entry : specTable())
        out.push_back ({ entry.second.path, entry.second.lo, entry.second.hi,
                         entry.second.logScale });
    return out;
}

double Refine::parameterValue (const Patch& patch, const std::string& path)
{
    return getParameter (patch, path);
}

void Refine::setParameterValue (Patch& patch, const std::string& path, double value)
{
    setParameter (patch, path, value);
}

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

        // The wavetable is absent on purpose, frames and position alike. The
        // frames are a measurement, and the position is the trajectory that
        // measurement was taken along -- refining either would be re-deciding
        // structure with a distance metric, which is the split this project
        // keeps: analysis owns what was measured, refinement owns precision.
        // It is the same reason the LFO rate is not in this list.
        if (osc.envEnabled)
            for (const char* leaf : { "attack", "decay", "sustain", "release", "curve" })
                paths.push_back (p + ".env." + leaf);
    }

    for (const char* leaf : { "attack", "decay", "sustain", "release", "curve", "attack_curve" })
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
        // Rate is measured, not searched.
        //
        // Analysis reads it straight off the trajectory and gets it right --
        // 6.0 Hz against a true 5.8 on a violin, 3.3 on a clarinet. Refinement
        // then moved a correct 4.4 Hz vibrato down to 1.6 Hz and tripled its
        // depth, because a slow deep wobble smears the spectrum in a way the
        // objective rewards and the ear does not.
        //
        // That is the structure/precision split applied to modulation: a
        // quantity analysis measures well belongs to analysis. Depth and phase
        // stay searchable -- the f0 tracker's own window low-passes vibrato
        // before it is measured, so depth is only ever approximate, and phase
        // is not recovered reliably at all.
        //
        // Depth is out too, for the same reason and with the same evidence.
        // Analysis measured a violin's vibrato at 5 cents; refinement returned
        // 39, which is a wobble no player produces and the first thing a
        // listener objects to. Under-stating vibrato is a knob turn away;
        // over-stating it makes the patch unusable, and the knob is now
        // labelled so it can be found.
        const auto p = "lfos." + std::to_string (i);
        for (const char* leaf : { ".delay", ".phase" })
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

namespace
{

// Where the note is released: what the caller said, or what the target shows.
//
// Shared by `run` and `measure` so that a score and the search that produced it
// are taken against the same note. Scoring a patch against a different gate
// from the one it was fitted at answers a different question.
//
// Detected from the target when the caller does not say, which is the common
// case. The old default was `duration` -- hold the note for the whole file --
// and that is wrong for anything that stops before the end, which is every real
// recording: candidates were rendered still sounding while the target had gone
// quiet seconds earlier, so the only way to fit was to mangle the envelope into
// faking a release it was never allowed to perform. On two library samples
// refinement made the fit three to five times worse this way.
//
// The recovery harness never showed it because it passes the gate explicitly --
// the one caller that did.
double gateFor (const float* target, int numSamples, double sampleRate, double requested)
{
    if (requested >= 0.0)
        return requested;

    const auto duration = numSamples / sampleRate;
    const auto rms = Stft::loudnessEnvelope (target, numSamples, 256);
    std::vector<float> times (rms.size());
    for (size_t i = 0; i < rms.size(); ++i)
        times[i] = static_cast<float> (i * 256.0 / sampleRate);

    const auto detected = EnvelopeFit::detectGate (rms, times);
    return detected.oneShot ? duration : detected.time;
}

} // namespace

Refine::Loss Refine::measure (const Patch& patch, const float* target, int numSamples,
                              double sampleRate, const Options& options)
{
    if (numSamples <= 0 || ! options.renderer)
        return {};

    const auto duration = numSamples / sampleRate;
    const auto gate = gateFor (target, numSamples, sampleRate, options.gateSeconds);

    const TargetFeatures features (target, numSamples, sampleRate, gate, patch.rootHz);
    const auto rendered = options.renderer (patch, duration, gate);
    return lossComponents (features, rendered.data(), (int) rendered.size(), sampleRate,
                           gate, patch.rootHz);
}

Refine::Result Refine::run (const Patch& patch, const float* target, int numSamples,
                            double sampleRate, const Options& options)
{
    Result result;
    result.patch = patch;

    // Nothing to optimise against without a synth. Refinement is a closed loop
    // -- render, measure, adjust -- so a missing renderer is not a degraded
    // mode, it is no mode at all.
    const auto scope = scopeFor (patch);
    if (scope.size() < 2 || numSamples <= 0 || ! options.renderer)
        return result;

    const auto& table = specTable();
    const auto duration = numSamples / sampleRate;

    const auto gate = gateFor (target, numSamples, sampleRate, options.gateSeconds);
    result.gateSeconds = gate;

    // Noise is measured, and refinement may only sharpen it.
    //
    // It is set closed-loop before this runs -- render, measure the energy
    // between the harmonics, scale, repeat -- against the same quantity
    // `autosynth_diff` reports as noisiness. Left at its full range, refinement
    // undid that: a violin calibrated to 0.05 came back at 0.015 with its bow
    // stripped off, because broadband energy lowers a log-spectral error
    // wherever the harmonic fit is imperfect and the optimiser always finds
    // that trade.
    //
    // Removing it from the search entirely was tried and costs more than it
    // saves -- CMA-ES loses a dimension and lands elsewhere, which put the
    // harness's brightness error up by a third on patches whose noise was
    // already zero. Half to double, like the measured attack before it.
    constexpr double kNoiseSearchFactor = 2.0;
    std::vector<Spec> specs;
    specs.reserve (scope.size());
    for (const auto& path : scope)
    {
        auto spec = table.at (path);
        if (path == "noise_level")
        {
            const auto measured = getParameter (patch, path);
            spec.lo = juce::jmax (spec.lo, measured / kNoiseSearchFactor);
            spec.hi = juce::jmin (spec.hi, measured * kNoiseSearchFactor);
            if (spec.hi <= spec.lo)
                spec.hi = spec.lo + 1.0e-6;
        }
        specs.push_back (spec);
    }

    // Attack is derived, not searched.
    //
    // It is a measurement: the analysis knows how long the note took to reach
    // nine tenths of the level it holds, and a spectral distance is nearly
    // blind to the first tenth of a second and will spend it buying accuracy
    // elsewhere. Given its full range the optimiser moved a violin's attack to
    // 0.12 s and a clarinet's to 0.50 -- opposite directions, both away from
    // the truth.
    //
    // Bounding the *parameter* was tried and is not enough, because the
    // rendered attack depends on the attack and the curve *together* and the
    // curve is legitimately searchable. Held to half-to-double, a violin whose
    // envelope should measure 0.40 s still rendered at 0.29, because refinement
    // had flattened the curve underneath it.
    //
    // So what is held fixed is the thing that was measured -- the crossing time
    // -- and the attack is re-derived from it against whatever curve the
    // candidate chose. CMA-ES keeps the curve; the envelope keeps its shape in
    // time.
    struct DerivedAttack
    {
        std::string prefix;   // the envelope, without the trailing field
        float crossingSeconds = 0.0f;
    };

    const auto envelopeAt = [] (const Patch& p, const std::string& prefix)
    {
        Adsr env;
        env.attack = static_cast<float> (getParameter (p, prefix + ".attack"));
        env.decay = static_cast<float> (getParameter (p, prefix + ".decay"));
        env.sustain = static_cast<float> (getParameter (p, prefix + ".sustain"));
        env.release = static_cast<float> (getParameter (p, prefix + ".release"));
        env.curve = static_cast<float> (getParameter (p, prefix + ".curve"));
        return env;
    };

    // What each envelope currently measures as, which is what has to survive.
    const auto crossingOf = [&] (const Adsr& env)
    {
        constexpr double fps = 200.0;
        const auto span = juce::jmax (gate, 0.5);
        const auto count = juce::jlimit (16, 4000, static_cast<int> (std::lround (span * fps)));
        std::vector<float> times (static_cast<size_t> (count)), rendered (static_cast<size_t> (count));
        for (int i = 0; i < count; ++i)
        {
            times[static_cast<size_t> (i)] = static_cast<float> (i / fps);
            rendered[static_cast<size_t> (i)] = Envelope::evaluate (env, times[static_cast<size_t> (i)], gate);
        }
        return EnvelopeFit::attackSeconds (rendered, times);
    };

    std::vector<DerivedAttack> derived;
    std::vector<std::string> searched;
    std::vector<Spec> searchedSpecs;
    for (size_t i = 0; i < scope.size(); ++i)
    {
        const auto& path = scope[i];
        if (path.size() >= 7 && path.compare (path.size() - 7, 7, ".attack") == 0)
        {
            const auto prefix = path.substr (0, path.size() - 7);
            derived.push_back ({ prefix, crossingOf (envelopeAt (patch, prefix)) });
            continue;
        }
        searched.push_back (path);
        searchedSpecs.push_back (specs[i]);
    }

    std::vector<double> x0 (searched.size());
    for (size_t i = 0; i < searched.size(); ++i)
        x0[i] = encode (searchedSpecs[i], getParameter (patch, searched[i]));

    const auto build = [&] (const std::vector<double>& x)
    {
        auto out = patch;
        for (size_t i = 0; i < searched.size(); ++i)
            setParameter (out, searched[i], decode (searchedSpecs[i], x[i]));

        for (const auto& d : derived)
            setParameter (out, d.prefix + ".attack",
                          EnvelopeFit::attackMeasuring (envelopeAt (out, d.prefix), gate,
                                                        d.crossingSeconds));
        return out;
    };

    const TargetFeatures features (target, numSamples, sampleRate, gate, patch.rootHz);

    const auto measure = [&] (const Patch& candidate)
    {
        const auto rendered = options.renderer (candidate, duration, gate);
        return lossComponents (features, rendered.data(), (int) rendered.size(), sampleRate,
                               gate, candidate.rootHz);
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
    const auto wAttack = weightFor (start.attack, "attack");
    const auto wDrift = weightFor (start.drift, "drift");
    const auto wWobble = weightFor (start.wobble, "wobble");
    const auto wOnset = weightFor (start.onset, "onset");
    const auto wLevel = weightFor (start.level, "level");

    const auto combine = [&] (const Loss& loss)
    {
        return (wSpectral * loss.spectral + wLoudness * loss.loudness
                + wCentroid * loss.centroid + wAttack * loss.attack
                + wDrift * loss.drift + wWobble * loss.wobble
                + wOnset * loss.onset + wLevel * loss.level) / 8.0;
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
