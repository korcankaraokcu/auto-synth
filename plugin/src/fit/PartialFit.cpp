#include "fit/PartialFit.h"

#include "analysis/Grouping.h"
#include "analysis/Partials.h"
#include "analysis/Stft.h"
#include "analysis/Yin.h"
#include "fit/EffectsFit.h"
#include "fit/EnvelopeFit.h"
#include "fit/FilterFit.h"
#include "fit/Modulation.h"
#include "fit/Nnls.h"
#include "fit/WaveformFit.h"

#include <algorithm>
#include <cmath>

namespace autosynth
{
namespace
{

// Smallest power-of-two window that still resolves f0, matching
// analysis.harmonics.window_for_f0. Six bins per harmonic spacing: three was
// tried and measurably degraded waveform recovery, because a Hann main lobe is
// about four bins wide and the peak-picking window then overlaps its
// neighbour's leakage.
int windowForF0 (double f0Hz, double sampleRate, int lo = 1024, int hi = 8192)
{
    if (f0Hz <= 0.0)
        return hi;
    const auto needed = 6.0 * sampleRate / f0Hz;
    auto n = lo;
    while (n < needed && n < hi)
        n *= 2;
    return juce::jlimit (lo, hi, n);
}

// Energy-weighted mean harmonic profile, peak-normalised. Weighted so the quiet
// tail -- where partial amplitudes are least reliable -- does not get an equal
// vote in the waveform decision.
std::vector<float> meanProfile (const std::vector<float>& H, int numHarmonics, int numFrames)
{
    std::vector<float> profile (static_cast<size_t> (juce::jmax (1, numHarmonics)), 1.0f);
    if (numHarmonics <= 0 || numFrames <= 0)
        return profile;

    std::vector<double> frameWeight (static_cast<size_t> (numFrames), 0.0);
    double total = 0.0;
    for (int t = 0; t < numFrames; ++t)
    {
        for (int k = 0; k < numHarmonics; ++k)
            frameWeight[static_cast<size_t> (t)] +=
                H[static_cast<size_t> (k) * numFrames + t];
        total += frameWeight[static_cast<size_t> (t)];
    }
    if (total <= 1.0e-12)
        return profile;

    for (int k = 0; k < numHarmonics; ++k)
    {
        double acc = 0.0;
        for (int t = 0; t < numFrames; ++t)
            acc += H[static_cast<size_t> (k) * numFrames + t]
                 * frameWeight[static_cast<size_t> (t)] / total;
        profile[static_cast<size_t> (k)] = static_cast<float> (acc);
    }

    const auto peak = *std::max_element (profile.begin(), profile.end());
    if (peak > 1.0e-12f)
        for (auto& v : profile)
            v /= peak;
    return profile;
}

std::vector<double> spectralFeatures (const float* samples, int numSamples, double sampleRate)
{
    const auto spectrogram = Stft::magnitudeSpectrogram (samples, numSamples, 1024, 512, sampleRate);
    std::vector<double> out (spectrogram.magnitude.size());
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = spectrogram.magnitude[i];
    return out;
}

} // namespace

Patch PartialFit::calibrateLevels (Patch patch, const float* target, int numSamples,
                                   double sampleRate, double gateSeconds)
{
    std::vector<int> active;
    for (int i = 0; i < kNumOsc; ++i)
        if (patch.oscs[static_cast<size_t> (i)].enabled
            && patch.oscs[static_cast<size_t> (i)].level > 1.0e-4f)
            active.push_back (i);
    if (active.empty())
        return patch;

    const auto duration = numSamples / sampleRate;
    const auto hasNoise = patch.noiseLevel > 1.0e-6f;
    const auto numColumns = static_cast<int> (active.size()) + (hasNoise ? 1 : 0);

    Engine engine;
    engine.prepare (sampleRate, 512);

    const auto renderSolo = [&] (int oscIndex, bool noiseOnly)
    {
        auto probe = patch;
        probe.masterLevel = 1.0f;
        probe.noiseLevel = 0.0f;
        for (auto& osc : probe.oscs)
        {
            osc.enabled = false;
            osc.level = 0.0f;
        }
        if (noiseOnly)
            probe.noiseLevel = 1.0f;
        else
        {
            probe.oscs[static_cast<size_t> (oscIndex)].enabled = true;
            probe.oscs[static_cast<size_t> (oscIndex)].level = 1.0f;
        }

        engine.setPatch (probe);
        juce::AudioBuffer<float> buffer;
        engine.renderOffline (buffer, probe.rootHz, duration, gateSeconds);
        return spectralFeatures (buffer.getReadPointer (0), buffer.getNumSamples(), sampleRate);
    };

    std::vector<std::vector<double>> columns;
    columns.reserve (static_cast<size_t> (numColumns));
    for (auto index : active)
        columns.push_back (renderSolo (index, false));
    if (hasNoise)
        columns.push_back (renderSolo (0, true));

    const auto observed = spectralFeatures (target, numSamples, sampleRate);

    size_t rows = observed.size();
    for (const auto& c : columns)
        rows = std::min (rows, c.size());
    if (rows == 0)
        return patch;

    std::vector<double> A (rows * static_cast<size_t> (numColumns));
    for (int j = 0; j < numColumns; ++j)
        std::copy (columns[static_cast<size_t> (j)].begin(),
                   columns[static_cast<size_t> (j)].begin() + static_cast<long> (rows),
                   A.begin() + static_cast<long> (j * rows));

    std::vector<double> b (observed.begin(), observed.begin() + static_cast<long> (rows));
    const auto gains = nnls::solve (A, b, static_cast<int> (rows), numColumns);

    // Scale by the loudest *oscillator*, never by the noise gain. Broadband
    // noise has energy in every bin, so in a magnitude-domain least squares it
    // can absorb an arbitrary amount of a poorly-fitted harmonic sound -- which
    // drove every oscillator level to zero and switched whole patches off.
    double scale = 0.0;
    for (size_t i = 0; i < active.size(); ++i)
        scale = std::max (scale, gains[i]);
    if (scale <= 1.0e-9)
        return patch;

    for (size_t i = 0; i < active.size(); ++i)
        patch.oscs[static_cast<size_t> (active[i])].level =
            static_cast<float> (juce::jlimit (0.0, 1.0, gains[i] / scale));
    if (hasNoise)
        patch.noiseLevel = static_cast<float> (juce::jlimit (0.0, 1.0, gains.back() / scale));
    patch.masterLevel = static_cast<float> (juce::jlimit (0.0, 1.0, scale));

    // A source solved to zero is one the analysis proposed and the mix
    // rejected. The loudest is always kept: "no oscillators" is never a better
    // description of a pitched sound than one oscillator.
    size_t loudest = 0;
    for (size_t i = 1; i < active.size(); ++i)
        if (gains[i] > gains[loudest])
            loudest = i;
    for (size_t i = 0; i < active.size(); ++i)
        if (i != loudest && patch.oscs[static_cast<size_t> (active[i])].level <= 1.0e-3f)
            patch.oscs[static_cast<size_t> (active[i])].enabled = false;

    return patch;
}

Patch PartialFit::fit (const float* samples, int numSamples, double sampleRate,
                       const Options& options)
{
    Patch patch;
    patch.name = "partialfit";
    for (auto& osc : patch.oscs)
    {
        osc.enabled = false;
        osc.level = 0.0f;
    }

    const auto rms = Stft::loudnessEnvelope (samples, numSamples, options.hop);
    std::vector<float> times (rms.size());
    for (size_t i = 0; i < rms.size(); ++i)
        times[i] = static_cast<float> (i * options.hop / sampleRate);

    double gateTime;
    bool oneShot;
    if (options.gateSeconds >= 0.0)
    {
        gateTime = options.gateSeconds;
        oneShot = false;
    }
    else
    {
        const auto detected = EnvelopeFit::detectGate (rms, times);
        gateTime = detected.time;
        oneShot = detected.oneShot;
    }

    const auto globalEnv = EnvelopeFit::fitAdsr (rms, times, gateTime, 0.05f, oneShot);

    float peak = 0.0f;
    for (int i = 0; i < numSamples; ++i)
        peak = juce::jmax (peak, std::abs (samples[i]));
    patch.masterLevel = juce::jlimit (0.05f, 1.0f, peak);

    double f0Hint = 0.0, confidence = 0.0;
    Yin::estimate (samples, numSamples, sampleRate, f0Hint, confidence, options.hop);

    PartialTracker::Options trackOptions;
    trackOptions.hop = options.hop;
    trackOptions.fftSize = windowForF0 (f0Hint > 0.0 ? f0Hint : 100.0, sampleRate);
    const auto partialSet = PartialTracker::track (samples, numSamples, sampleRate, trackOptions);
    const auto groups = Grouping::group (partialSet, options.maxOscillators, options.tolCents);

    if (groups.empty() || f0Hint <= 0.0 || confidence <= 0.0)
    {
        // Nothing periodic to model: hand it to the noise source rather than
        // inventing an oscillator to explain broadband energy.
        patch.noiseLevel = 0.5f;
        patch.filter.type = FilterType::off;
        patch.ampEnv = globalEnv;
        return patch;
    }

    // Root is the lowest source, so every other oscillator transposes upward
    // and stays inside the +/-24 semitone range the IR allows.
    auto root = groups.front().f0;
    for (const auto& g : groups)
        root = std::min (root, g.f0);
    patch.rootHz = static_cast<float> (options.noteHz > 0.0 ? options.noteHz : root);

    // --- filter, from the dominant source ---------------------------------
    auto dominant = groups.front();
    for (const auto& g : groups)
        if (g.energy() > dominant.energy())
            dominant = g;

    const auto profile = meanProfile (dominant.H, dominant.numHarmonics, dominant.numFrames);
    const auto anchor = WaveformFit::matchWithCutoff (profile, dominant.f0, sampleRate);
    const auto trajectory = FilterFit::estimateCutoffTrajectory (
        dominant.H, dominant.numHarmonics, dominant.numFrames, dominant.f0, sampleRate);
    const auto split = FilterFit::trajectoryToEnv (trajectory.cutoffHz, anchor.cutoffHz);

    patch.filter.type = FilterType::lowpass;
    patch.filter.cutoffHz = juce::jlimit (30.0f, 18000.0f, split.baseCutoffHz);
    patch.filter.resonance = FilterFit::kDefaultQ;
    patch.filter.envAmount = juce::jlimit (0.0f, 4.0f, split.envAmountOctaves);
    if (split.envAmountOctaves > 0.0f)
    {
        std::vector<float> shapeTimes (split.shape.size());
        for (size_t i = 0; i < shapeTimes.size(); ++i)
            shapeTimes[i] = i < partialSet.times.size() ? partialSet.times[i]
                                                        : static_cast<float> (i * options.hop / sampleRate);
        patch.filter.env = EnvelopeFit::fitAdsr (split.shape, shapeTimes, gateTime, 0.05f, oneShot);
    }

    // --- sources -> oscillators --------------------------------------------
    auto ordered = groups;
    std::sort (ordered.begin(), ordered.end(),
               [] (const HarmonicGroup& a, const HarmonicGroup& b) { return a.energy() > b.energy(); });

    const auto multi = ordered.size() > 1;
    const auto count = juce::jmin (static_cast<int> (ordered.size()), options.maxOscillators);

    for (int i = 0; i < count; ++i)
    {
        const auto& g = ordered[static_cast<size_t> (i)];
        auto& osc = patch.oscs[static_cast<size_t> (i)];
        osc.enabled = true;
        osc.level = 1.0f;

        const auto interval = 12.0 * std::log2 (g.f0 / patch.rootHz);
        const auto semis = juce::jlimit (-24, 24, static_cast<int> (std::lround (interval)));
        osc.semitones = semis;
        // The remainder is real detuning, not rounding error.
        osc.cents = static_cast<float> (juce::jlimit (-50.0, 50.0, (interval - semis) * 100.0));

        // Waveform is read after removing the filter, so a bright source behind
        // a closed filter is not mistaken for a dull one.
        const auto flat = FilterFit::deconvolve (g.H, g.numHarmonics, g.numFrames,
                                                 trajectory.cutoffHz, g.f0);
        const auto match = WaveformFit::match (meanProfile (flat, g.numHarmonics, g.numFrames));
        osc.waveform = match.waveform;
        osc.pulseWidth = match.pulseWidth;

        // Unison is visible only because partials are tracked individually.
        Grouping::estimateUnison (g, osc.unisonVoices, osc.unisonDetune);

        if (multi)
        {
            osc.envEnabled = true;
            std::vector<float> total (static_cast<size_t> (g.numFrames), 0.0f);
            for (int t = 0; t < g.numFrames; ++t)
                for (int k = 0; k < g.numHarmonics; ++k)
                    total[static_cast<size_t> (t)] += g.harmonic (k)[t];

            std::vector<float> groupTimes (total.size());
            for (size_t j = 0; j < groupTimes.size(); ++j)
                groupTimes[j] = j < partialSet.times.size() ? partialSet.times[j]
                                                            : static_cast<float> (j * options.hop / sampleRate);
            osc.env = EnvelopeFit::fitAdsr (total, groupTimes, gateTime, 0.05f, oneShot);
        }
    }

    if (multi)
    {
        // The patch-wide envelope becomes a pass-through supplying only the
        // note-off release, so per-oscillator envelopes are not multiplied by a
        // second shape they did not account for.
        patch.ampEnv = { 0.001f, 0.005f, 1.0f, globalEnv.release, globalEnv.curve };
    }
    else
    {
        patch.ampEnv = globalEnv;
    }

    // The IR carries one LFO, so vibrato, tremolo and filter wobble compete and
    // only the most convincingly periodic survives.
    // Two slots now, so vibrato and tremolo can coexist instead of competing.
    // Filled most-convincing-first; unused slots stay routed to none.
    const auto detected = Modulation::bestSeveral (
        Modulation::extract (samples, numSamples, sampleRate, options.hop));
    for (size_t i = 0; i < detected.size() && i < patch.lfos.size(); ++i)
        patch.lfos[i] = detected[i];

    // Delay is fitted, reverb is not -- see fit/EffectsFit.h for why only one
    // half of "effects" is tractable this way.
    patch.delay = EffectsFit::fitDelay (samples, numSamples, sampleRate, options.hop);

    double claimed = 0.0;
    for (const auto& g : groups)
        claimed += g.energy();
    const auto totalPartialEnergy = juce::jmax (partialSet.totalEnergy(), 1.0e-12f);
    const auto residual = 1.0 - claimed / totalPartialEnergy;
    patch.noiseLevel = static_cast<float> (juce::jlimit (0.0, 1.0, residual)) * 0.5f;

    return calibrateLevels (patch, samples, numSamples, sampleRate, gateTime);
}

} // namespace autosynth
