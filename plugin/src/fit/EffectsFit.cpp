#include "fit/EffectsFit.h"

#include "analysis/Stft.h"
#include "fit/NdFilters.h"

#include <algorithm>
#include <cmath>
#include <juce_dsp/juce_dsp.h>
#include <numeric>

namespace autosynth
{
namespace
{

// Autocorrelation of a zero-mean signal, normalised so lag 0 is 1.
std::vector<float> normalisedAutocorrelation (const std::vector<float>& input)
{
    const auto n = static_cast<int> (input.size());
    std::vector<float> out (input.size(), 0.0f);
    if (n < 2)
        return out;

    const auto mean = std::accumulate (input.begin(), input.end(), 0.0)
                    / static_cast<double> (n);

    auto size = 2;
    while (size < 2 * n)
        size <<= 1;
    const auto order = static_cast<int> (std::log2 (static_cast<double> (size)));

    std::vector<float> buffer (static_cast<size_t> (size) * 2, 0.0f);
    for (int i = 0; i < n; ++i)
        buffer[static_cast<size_t> (i)] = static_cast<float> (input[static_cast<size_t> (i)] - mean);

    juce::dsp::FFT fft (order);
    fft.performRealOnlyForwardTransform (buffer.data(), true);

    // Power spectrum, then back: X * conj(X) is real, so the imaginary part is
    // zeroed rather than computed.
    const auto numBins = size / 2 + 1;
    for (int b = 0; b < numBins; ++b)
    {
        const auto re = buffer[static_cast<size_t> (b) * 2];
        const auto im = buffer[static_cast<size_t> (b) * 2 + 1];
        buffer[static_cast<size_t> (b) * 2] = re * re + im * im;
        buffer[static_cast<size_t> (b) * 2 + 1] = 0.0f;
    }
    fft.performRealOnlyInverseTransform (buffer.data());

    const auto zeroLag = buffer[0];
    if (zeroLag <= 1.0e-12f)
        return out;

    for (int i = 0; i < n; ++i)
        out[static_cast<size_t> (i)] = buffer[static_cast<size_t> (i)] / zeroLag;
    return out;
}

} // namespace

EffectsFit::DelayEstimate EffectsFit::detectDelay (const float* samples, int numSamples,
                                                   double sampleRate, int hop,
                                                   double smoothSeconds)
{
    DelayEstimate estimate;

    auto envelope = Stft::loudnessEnvelope (samples, numSamples, hop);
    if (envelope.size() < 32)
        return estimate;

    const auto dt = hop / sampleRate;
    const auto span = std::max (1, static_cast<int> (std::lround (smoothSeconds / dt)));
    if (span > 1)
        envelope = nd::uniformFilter1d (envelope, span);

    const auto correlation = normalisedAutocorrelation (envelope);
    const auto size = static_cast<int> (correlation.size());
    const auto lo = std::max (1, static_cast<int> (std::lround (kMinDelaySeconds / dt)));
    const auto hi = std::min (size - 1, static_cast<int> (std::lround (kMaxDelaySeconds / dt)));
    if (hi <= lo)
        return estimate;

    // Skip the main lobe. An envelope is smooth, so its autocorrelation
    // descends slowly from 1.0 and is still ~0.9 several frames later -- taking
    // the maximum over a lag band just reports the left edge of that slope,
    // which "detects" a ~20 ms delay in everything including dry material.
    auto i = 1;
    while (i < size - 1 && correlation[static_cast<size_t> (i)]
                            <= correlation[static_cast<size_t> (i - 1)])
        ++i;
    const auto start = std::max (i, lo);
    if (start >= hi)
        return estimate;

    // And require a genuine local maximum. A single decaying note has a
    // monotonic autocorrelation with no interior peak, which is the rejection
    // wanted.
    auto peak = -1;
    auto strength = 0.0;
    for (int j = start + 1; j < hi; ++j)
    {
        const auto c = static_cast<double> (correlation[static_cast<size_t> (j)]);
        if (c > correlation[static_cast<size_t> (j - 1)]
            && c >= correlation[static_cast<size_t> (j + 1)] && c > strength)
        {
            peak = j;
            strength = c;
        }
    }

    if (peak < 0 || strength < kMinPeakRatio)
        return estimate;

    // Feedback from how much the second repeat retains relative to the first.
    const auto second = 2 * peak;
    auto ratio = 0.0;
    if (second < size)
        ratio = juce::jlimit (0.0, 1.0,
                              correlation[static_cast<size_t> (second)] / std::max (strength, 1.0e-9));

    estimate.found = true;
    estimate.time = juce::jlimit (kMinDelaySeconds, kMaxDelaySeconds, peak * dt);
    // A proxy, not the coefficient itself; CMA-ES refines it, so a rough
    // mapping is enough to start from.
    estimate.feedback = juce::jlimit (0.0, 0.85, std::sqrt (ratio));
    estimate.mix = juce::jlimit (0.0, 1.0, strength);
    estimate.strength = strength;
    return estimate;
}

DelayParams EffectsFit::fitDelay (const float* samples, int numSamples, double sampleRate, int hop)
{
    DelayParams params;
    const auto estimate = detectDelay (samples, numSamples, sampleRate, hop);
    if (! estimate.found)
    {
        params.enabled = false;
        params.mix = 0.0f;
        return params;
    }

    params.enabled = true;
    params.time = static_cast<float> (estimate.time);
    params.feedback = static_cast<float> (estimate.feedback);
    params.mix = static_cast<float> (estimate.mix);
    return params;
}

} // namespace autosynth
