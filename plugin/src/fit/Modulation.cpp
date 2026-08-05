#include "fit/Modulation.h"

#include "analysis/Stft.h"
#include "analysis/Yin.h"
#include "fit/NdFilters.h"

#include <algorithm>
#include <cmath>
#include <juce_dsp/juce_dsp.h>
#include <limits>

namespace autosynth
{
namespace
{

// Remove the slow component, leaving whatever oscillates. The window has to be
// long compared with an LFO period and short compared with an envelope; half a
// second sits between the two for every rate this looks for.
std::vector<float> detrend (const std::vector<float>& y, double dt, double seconds)
{
    const auto span = std::max (3, static_cast<int> (std::lround (seconds / std::max (dt, 1.0e-9))));
    if (static_cast<int> (y.size()) <= span)
    {
        double mean = 0.0;
        for (auto v : y)
            mean += v;
        mean /= std::max<size_t> (y.size(), 1);
        std::vector<float> out (y.size());
        for (size_t i = 0; i < y.size(); ++i)
            out[i] = y[i] - static_cast<float> (mean);
        return out;
    }

    const auto smooth = nd::uniformFilter1d (y, span);
    std::vector<float> out (y.size());
    for (size_t i = 0; i < y.size(); ++i)
        out[i] = y[i] - smooth[i];
    return out;
}

int nextPowerOfTwo (int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

// Strongest periodic component. Deliberately does not report amplitude:
// reading it off the FFT peak needs the window's coherent gain undone and then
// a leakage correction when the rate falls between bins, two corrections that
// are easy to get subtly wrong and did produce depths ~1.6x too large.
// `matchShape` recovers amplitude by least-squares projection instead.
void dominantRate (const std::vector<float>& y, double dt,
                   double& rateOut, double& concentrationOut)
{
    rateOut = 0.0;
    concentrationOut = 0.0;
    const auto n = static_cast<int> (y.size());
    if (n < 16)
        return;

    const auto fftSize = nextPowerOfTwo (n);
    const auto order = static_cast<int> (std::log2 (static_cast<double> (fftSize)));
    juce::dsp::FFT fft (order);

    // Hann window over the *actual* trajectory length, zero-padded to the FFT
    // size -- matching numpy, which windows n points and transforms n points.
    std::vector<float> buffer (static_cast<size_t> (fftSize) * 2, 0.0f);
    const auto window = Stft::periodicHann (n);
    for (int i = 0; i < n; ++i)
        buffer[static_cast<size_t> (i)] = y[static_cast<size_t> (i)] * window[static_cast<size_t> (i)];

    fft.performRealOnlyForwardTransform (buffer.data(), true);

    const auto numBins = fftSize / 2 + 1;
    std::vector<double> spectrum (static_cast<size_t> (numBins));
    for (int b = 0; b < numBins; ++b)
    {
        const auto re = buffer[static_cast<size_t> (b) * 2];
        const auto im = buffer[static_cast<size_t> (b) * 2 + 1];
        spectrum[static_cast<size_t> (b)] = std::sqrt (re * re + im * im);
    }

    const auto binHz = 1.0 / (fftSize * dt);
    const auto duration = n * dt;
    // Require enough cycles to distinguish periodicity from a slow drift.
    const auto lowest = std::max (Modulation::kMinRateHz,
                                  Modulation::kMinCycles / std::max (duration, 1.0e-9));

    int peak = -1;
    for (int b = 1; b < numBins; ++b)
    {
        const auto freq = b * binHz;
        if (freq < lowest || freq > Modulation::kMaxRateHz)
            continue;
        if (peak < 0 || spectrum[static_cast<size_t> (b)] > spectrum[static_cast<size_t> (peak)])
            peak = b;
    }
    if (peak < 0)
        return;

    double total = 0.0;
    for (int b = 1; b < numBins; ++b)
        total += spectrum[static_cast<size_t> (b)];
    if (total <= 1.0e-12)
        return;

    double near = 0.0;
    for (int b = std::max (1, peak - 1); b <= std::min (numBins - 1, peak + 1); ++b)
        near += spectrum[static_cast<size_t> (b)];
    concentrationOut = near / total;

    // Refine to sub-bin accuracy. Bin spacing over a two-second trajectory is
    // ~0.5 Hz, and an error that size drifts the least-squares fit half a cycle
    // out of phase by the end -- which then reports the amplitude a third too
    // small. The peak position matters as much as the peak height.
    auto rate = peak * binHz;
    if (peak > 0 && peak < numBins - 1)
    {
        const auto a = std::log (spectrum[static_cast<size_t> (peak - 1)] + 1.0e-12);
        const auto b = std::log (spectrum[static_cast<size_t> (peak)] + 1.0e-12);
        const auto c = std::log (spectrum[static_cast<size_t> (peak + 1)] + 1.0e-12);
        const auto denom = a - 2.0 * b + c;
        if (std::abs (denom) > 1.0e-12)
            rate += juce::jlimit (-0.5, 0.5, 0.5 * (a - c) / denom) * binHz;
    }
    rateOut = std::max (rate, 0.0);
}

float shapeValue (LfoShape shape, double phase)
{
    const auto twoPi = juce::MathConstants<double>::twoPi;
    const auto frac = phase - std::floor (phase);
    switch (shape)
    {
        case LfoShape::sine:     return static_cast<float> (std::sin (twoPi * phase));
        case LfoShape::triangle: return static_cast<float> (2.0 / juce::MathConstants<double>::pi
                                        * std::asin (juce::jlimit (-1.0, 1.0, std::sin (twoPi * phase))));
        case LfoShape::saw:      return static_cast<float> (2.0 * frac - 1.0);
        case LfoShape::square:   return std::sin (twoPi * phase) >= 0.0 ? 1.0f : -1.0f;
    }
    return 0.0f;
}

// Best-fitting shape, phase, amplitude and correlation. Amplitude comes out of
// the same least-squares projection that picks the shape: for a unit-amplitude
// candidate, <y,w>/<w,w> *is* the amplitude, with nothing to correct.
void matchShape (const std::vector<float>& y, double dt, double rate,
                 LfoShape& shapeOut, double& phaseOut, double& amplitudeOut,
                 double& correlationOut)
{
    shapeOut = LfoShape::sine;
    phaseOut = 0.0;
    amplitudeOut = 0.0;
    correlationOut = 0.0;

    double norm = 0.0;
    for (auto v : y)
        norm += static_cast<double> (v) * v;
    norm = std::sqrt (norm);
    if (norm <= 1.0e-12 || rate <= 0.0)
        return;

    const LfoShape shapes[] = { LfoShape::sine, LfoShape::triangle, LfoShape::saw, LfoShape::square };
    auto best = -std::numeric_limits<double>::infinity();

    for (int p = 0; p < 32; ++p)
    {
        const auto phase = static_cast<double> (p) / 32.0;
        for (auto shape : shapes)
        {
            double dot = 0.0, energy = 0.0;
            for (size_t i = 0; i < y.size(); ++i)
            {
                const auto w = shapeValue (shape, rate * (i * dt) + phase);
                dot += static_cast<double> (y[i]) * w;
                energy += static_cast<double> (w) * w;
            }
            if (energy <= 1.0e-12)
                continue;

            const auto correlation = dot / (norm * std::sqrt (energy));
            if (correlation > best)
            {
                best = correlation;
                shapeOut = shape;
                phaseOut = phase;
                amplitudeOut = std::abs (dot / energy);
                correlationOut = correlation;
            }
        }
    }
}

double fadeIn (const std::vector<float>& y, double dt, double rate)
{
    if (rate <= 0.0 || y.size() < 8)
        return 0.0;
    const auto span = std::max (3, static_cast<int> (std::lround ((1.0 / rate) / std::max (dt, 1.0e-9))));

    std::vector<float> magnitude (y.size());
    for (size_t i = 0; i < y.size(); ++i)
        magnitude[i] = std::abs (y[i]);
    const auto envelope = nd::uniformFilter1d (magnitude, span);

    const auto peak = *std::max_element (envelope.begin(), envelope.end());
    if (peak <= 1.0e-12f)
        return 0.0;
    for (size_t i = 0; i < envelope.size(); ++i)
        if (envelope[i] >= 0.7f * peak)
            return static_cast<double> (i) * dt;
    return 0.0;
}

} // namespace

Modulation::Detected Modulation::analyseTrajectory (const std::vector<float>& y, double dt,
                                                    LfoDest dest)
{
    Detected out;
    if (y.size() < 16)
        return out;
    for (auto v : y)
        if (! std::isfinite (v))
            return out;

    const auto detrended = detrend (y, dt, kDetrendSeconds);

    double rate = 0.0, concentration = 0.0;
    dominantRate (detrended, dt, rate, concentration);
    if (rate <= 0.0 || concentration < kMinConcentration)
        return out;

    LfoShape shape = LfoShape::sine;
    double phase = 0.0, amplitude = 0.0, correlation = 0.0;
    matchShape (detrended, dt, rate, shape, phase, amplitude, correlation);
    if (amplitude <= 0.0 || correlation < kMinCorrelation)
        return out;

    // The modulation has to be worth a parameter. A ramp leaves a small
    // oscillating residue after detrending, and fitting that residue reports a
    // confident LFO whose depth is a fraction of a percent of the trajectory it
    // supposedly explains.
    const auto lo = *std::min_element (y.begin(), y.end());
    const auto hi = *std::max_element (y.begin(), y.end());
    const auto span = static_cast<double> (hi - lo);
    if (span > 1.0e-12 && amplitude < kMinRelativeAmplitude * span)
        return out;

    out.found = true;
    out.dest = dest;
    out.rateHz = rate;
    out.amplitude = amplitude;
    out.shape = shape;
    out.phase = phase;
    out.delay = fadeIn (detrended, dt, rate);
    out.concentration = concentration;
    return out;
}

Lfo Modulation::toLfo (const Detected& detected)
{
    Lfo lfo;
    if (! detected.found)
    {
        lfo.dest = LfoDest::none;
        lfo.depth = 0.0f;
        return lfo;
    }

    // Depth is normalised by the engine's full-scale range for each
    // destination, which is how the engine reads it back.
    double depth;
    if (detected.dest == LfoDest::pitch)
        depth = detected.amplitude / (100.0 * kLfoPitchSemitones);
    else if (detected.dest == LfoDest::cutoff)
        depth = detected.amplitude / kLfoCutoffOctaves;
    else
        depth = detected.amplitude;

    lfo.shape = detected.shape;
    lfo.dest = detected.dest;
    lfo.rateHz = static_cast<float> (juce::jlimit (0.05, 20.0, detected.rateHz));
    lfo.depth = static_cast<float> (juce::jlimit (0.0, 1.0, depth));
    lfo.delay = static_cast<float> (juce::jlimit (0.0, 2.0, detected.delay));
    lfo.phase = static_cast<float> (juce::jlimit (0.0, 1.0, detected.phase));
    return lfo;
}

Modulation::Trajectories Modulation::extract (const float* samples, int numSamples,
                                              double sampleRate, int hop)
{
    Trajectories out;
    out.dt = hop / sampleRate;

    const auto pitch = Yin::track (samples, numSamples, sampleRate, hop);
    int voiced = 0;
    for (size_t i = 0; i < pitch.f0.size(); ++i)
        if (pitch.confidence[i] > 0.5f && pitch.f0[i] > 0.0f)
            ++voiced;

    if (voiced > 16)
    {
        std::vector<float> voicedF0;
        for (size_t i = 0; i < pitch.f0.size(); ++i)
            if (pitch.confidence[i] > 0.5f && pitch.f0[i] > 0.0f)
                voicedF0.push_back (pitch.f0[i]);
        std::sort (voicedF0.begin(), voicedF0.end());
        const auto mid = voicedF0.size() / 2;
        const auto reference = (voicedF0.size() % 2 == 1)
                             ? static_cast<double> (voicedF0[mid])
                             : 0.5 * (voicedF0[mid - 1] + voicedF0[mid]);

        out.pitchCents.assign (pitch.f0.size(), 0.0f);
        std::vector<int> valid;
        for (size_t i = 0; i < pitch.f0.size(); ++i)
        {
            if (pitch.confidence[i] > 0.5f && pitch.f0[i] > 0.0f)
            {
                out.pitchCents[i] = static_cast<float> (1200.0 * std::log2 (pitch.f0[i] / reference));
                valid.push_back (static_cast<int> (i));
            }
        }

        // Interpolate across unvoiced frames rather than leaving zeros: a hole
        // reads to the FFT as a broadband transient and buries the narrow peak
        // the detector is looking for.
        if (! valid.empty() && valid.size() < out.pitchCents.size())
        {
            for (int i = 0; i < static_cast<int> (out.pitchCents.size()); ++i)
            {
                if (pitch.confidence[static_cast<size_t> (i)] > 0.5f
                    && pitch.f0[static_cast<size_t> (i)] > 0.0f)
                    continue;

                if (i <= valid.front())
                    out.pitchCents[static_cast<size_t> (i)] = out.pitchCents[static_cast<size_t> (valid.front())];
                else if (i >= valid.back())
                    out.pitchCents[static_cast<size_t> (i)] = out.pitchCents[static_cast<size_t> (valid.back())];
                else
                {
                    const auto upper = std::lower_bound (valid.begin(), valid.end(), i);
                    const auto hiIndex = *upper;
                    const auto loIndex = *(upper - 1);
                    const auto denom = static_cast<float> (hiIndex - loIndex);
                    const auto frac = denom > 0.0f ? (i - loIndex) / denom : 0.0f;
                    const auto a = out.pitchCents[static_cast<size_t> (loIndex)];
                    const auto b = out.pitchCents[static_cast<size_t> (hiIndex)];
                    out.pitchCents[static_cast<size_t> (i)] = a + (b - a) * frac;
                }
            }
        }
        out.hasPitch = true;
    }

    const auto rms = Stft::loudnessEnvelope (samples, numSamples, hop);
    const auto peak = rms.empty() ? 0.0f : *std::max_element (rms.begin(), rms.end());
    if (peak > 1.0e-9f)
    {
        const auto span = std::max (3, static_cast<int> (0.5 / out.dt));
        const auto smooth = nd::uniformFilter1d (rms, span);
        out.ampRelative.assign (rms.size(), 0.0f);
        for (size_t i = 0; i < rms.size(); ++i)
            out.ampRelative[i] = rms[i] / std::max (smooth[i], 1.0e-9f) - 1.0f;
    }

    const auto spectrogram = Stft::magnitudeSpectrogram (samples, numSamples, 2048, hop, sampleRate);
    const auto centroid = Stft::spectralCentroid (spectrogram);
    const auto centroidPeak = centroid.empty() ? 0.0f : *std::max_element (centroid.begin(), centroid.end());
    if (centroidPeak > 20.0f)
    {
        out.centroidOctaves.assign (centroid.size(), 0.0f);
        for (size_t i = 0; i < centroid.size(); ++i)
            out.centroidOctaves[i] = static_cast<float> (std::log2 (std::max (centroid[i], 20.0f)));
    }

    return out;
}

std::vector<Lfo> Modulation::bestSeveral (const Trajectories& trajectories, int maxCount)
{
    std::vector<Detected> found;
    if (trajectories.hasPitch)
        if (const auto d = analyseTrajectory (trajectories.pitchCents, trajectories.dt, LfoDest::pitch); d.found)
            found.push_back (d);
    if (! trajectories.ampRelative.empty())
        if (const auto d = analyseTrajectory (trajectories.ampRelative, trajectories.dt, LfoDest::amp); d.found)
            found.push_back (d);
    if (! trajectories.centroidOctaves.empty())
        if (const auto d = analyseTrajectory (trajectories.centroidOctaves, trajectories.dt, LfoDest::cutoff); d.found)
            found.push_back (d);

    std::sort (found.begin(), found.end(),
               [] (const Detected& a, const Detected& b)
               { return a.concentration > b.concentration; });

    std::vector<Lfo> out;
    for (int i = 0; i < std::min (maxCount, static_cast<int> (found.size())); ++i)
        out.push_back (toLfo (found[static_cast<size_t> (i)]));
    return out;
}

Lfo Modulation::best (const Trajectories& trajectories)
{
    std::vector<Detected> found;

    if (trajectories.hasPitch)
        if (const auto d = analyseTrajectory (trajectories.pitchCents, trajectories.dt, LfoDest::pitch); d.found)
            found.push_back (d);
    if (! trajectories.ampRelative.empty())
        if (const auto d = analyseTrajectory (trajectories.ampRelative, trajectories.dt, LfoDest::amp); d.found)
            found.push_back (d);
    if (! trajectories.centroidOctaves.empty())
        if (const auto d = analyseTrajectory (trajectories.centroidOctaves, trajectories.dt, LfoDest::cutoff); d.found)
            found.push_back (d);

    if (found.empty())
    {
        Lfo lfo;
        lfo.dest = LfoDest::none;
        lfo.depth = 0.0f;
        return lfo;
    }

    auto bestIt = std::max_element (found.begin(), found.end(),
                                    [] (const Detected& a, const Detected& b)
                                    { return a.concentration < b.concentration; });
    return toLfo (*bestIt);
}

} // namespace autosynth
