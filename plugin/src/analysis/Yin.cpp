#include "analysis/Yin.h"

#include "analysis/Stft.h"

#include <algorithm>
#include <cmath>
#include <juce_dsp/juce_dsp.h>

namespace autosynth
{
namespace
{
const char* kNoteNames[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };

int nextPowerOfTwo (int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

void difference (const float* frame, int frameSize, int maxTau,
                      std::vector<float>& out, std::vector<float>& scratch,
                      juce::dsp::FFT*& fft, int& fftOrder)
{
    const auto w = frameSize - maxTau;
    out.assign (static_cast<size_t> (maxTau), 0.0f);

    // Running sum of squares, so the two energy terms are O(1) per lag.
    std::vector<double> cumsq (static_cast<size_t> (frameSize) + 1, 0.0);
    for (int i = 0; i < frameSize; ++i)
        cumsq[static_cast<size_t> (i) + 1] =
            cumsq[static_cast<size_t> (i)] + static_cast<double> (frame[i]) * frame[i];

    const auto energyHead = cumsq[static_cast<size_t> (w)] - cumsq[0];

    // r(tau) = sum_j x[j] x[j+tau], computed as a convolution with the
    // reversed head of the frame -- O(n log n) instead of O(n * maxTau).
    const auto n = nextPowerOfTwo (frameSize + w);
    const auto order = static_cast<int> (std::log2 (static_cast<double> (n)));
    if (fft == nullptr || order != fftOrder)
    {
        delete fft;
        fft = new juce::dsp::FFT (order);
        fftOrder = order;
    }

    scratch.assign (static_cast<size_t> (n) * 2, 0.0f);
    std::vector<float> reversed (static_cast<size_t> (n) * 2, 0.0f);
    for (int i = 0; i < frameSize; ++i)
        scratch[static_cast<size_t> (i)] = frame[i];
    for (int i = 0; i < w; ++i)
        reversed[static_cast<size_t> (i)] = frame[w - 1 - i];

    fft->performRealOnlyForwardTransform (scratch.data(), true);
    fft->performRealOnlyForwardTransform (reversed.data(), true);

    const auto numBins = n / 2 + 1;
    std::vector<float> product (static_cast<size_t> (n) * 2, 0.0f);
    for (int b = 0; b < numBins; ++b)
    {
        const auto ar = scratch[static_cast<size_t> (b) * 2];
        const auto ai = scratch[static_cast<size_t> (b) * 2 + 1];
        const auto br = reversed[static_cast<size_t> (b) * 2];
        const auto bi = reversed[static_cast<size_t> (b) * 2 + 1];
        product[static_cast<size_t> (b) * 2] = ar * br - ai * bi;
        product[static_cast<size_t> (b) * 2 + 1] = ar * bi + ai * br;
    }
    fft->performRealOnlyInverseTransform (product.data());

    for (int tau = 0; tau < maxTau; ++tau)
    {
        const auto energyLag = cumsq[static_cast<size_t> (tau + w)] - cumsq[static_cast<size_t> (tau)];
        const auto corr = static_cast<double> (product[static_cast<size_t> (w - 1 + tau)]);
        out[static_cast<size_t> (tau)] =
            static_cast<float> (energyHead + energyLag - 2.0 * corr);
    }
}

void cumulativeMeanNormalised (std::vector<float>& d)
{
    if (d.size() < 2)
        return;
    double running = 0.0;
    const auto first = d[0];
    juce::ignoreUnused (first);
    d[0] = 1.0f;
    for (size_t tau = 1; tau < d.size(); ++tau)
    {
        running += d[tau];
        d[tau] = static_cast<float> (d[tau] * static_cast<double> (tau) / juce::jmax (running, 1.0e-12));
    }
}

void pickTau (const std::vector<float>& dp, int tauMin, double threshold,
                   double& tauOut, double& confidenceOut)
{
    const auto size = static_cast<int> (dp.size());
    int tau = -1;
    for (int t = tauMin; t < size; ++t)
    {
        if (dp[static_cast<size_t> (t)] < threshold)
        {
            tau = t;
            break;
        }
    }

    if (tau >= 0)
    {
        // Walk to the bottom of the dip rather than taking its leading edge.
        while (tau + 1 < size && dp[static_cast<size_t> (tau + 1)] < dp[static_cast<size_t> (tau)])
            ++tau;
    }
    else
    {
        tau = tauMin;
        for (int t = tauMin; t < size; ++t)
            if (dp[static_cast<size_t> (t)] < dp[static_cast<size_t> (tau)])
                tau = t;
    }

    confidenceOut = juce::jlimit (0.0, 1.0, 1.0 - static_cast<double> (dp[static_cast<size_t> (tau)]));

    auto refined = static_cast<double> (tau);
    if (tau > 0 && tau < size - 1)
    {
        const auto a = static_cast<double> (dp[static_cast<size_t> (tau - 1)]);
        const auto b = static_cast<double> (dp[static_cast<size_t> (tau)]);
        const auto c = static_cast<double> (dp[static_cast<size_t> (tau + 1)]);
        const auto denom = a - 2.0 * b + c;
        if (std::abs (denom) > 1.0e-12)
            refined = tau + 0.5 * (a - c) / denom;
    }
    tauOut = refined;
}

} // namespace

Yin::Track Yin::track (const float* samples, int numSamples, double sampleRate,
                       int hop, double fmin, double fmax, double threshold)
{
    Track result;

    const auto tauMin = juce::jmax (2, static_cast<int> (sampleRate / fmax));
    const auto maxTau = juce::jmin (static_cast<int> (sampleRate / fmin) + 2, 4096);
    const auto windowSize = maxTau * 2;

    const auto numFrames = Stft::numFramesFor (numSamples, windowSize, hop);
    result.f0.assign (static_cast<size_t> (numFrames), 0.0f);
    result.confidence.assign (static_cast<size_t> (numFrames), 0.0f);

    // Frame RMS, computed on the same framing YIN uses, for the silence gate.
    std::vector<float> rms (static_cast<size_t> (numFrames), 0.0f);
    std::vector<float> frame (static_cast<size_t> (windowSize), 0.0f);
    float peakRms = 0.0f;

    for (int f = 0; f < numFrames; ++f)
    {
        double sum = 0.0;
        const auto start = f * hop;
        for (int i = 0; i < windowSize; ++i)
        {
            const auto index = start + i;
            const auto value = (index < numSamples) ? samples[index] : 0.0f;
            sum += static_cast<double> (value) * value;
        }
        rms[static_cast<size_t> (f)] = static_cast<float> (std::sqrt (sum / windowSize + 1.0e-20));
        peakRms = juce::jmax (peakRms, rms[static_cast<size_t> (f)]);
    }
    const auto floorRms = juce::jmax (1.0e-5f, 0.02f * peakRms);

    juce::dsp::FFT* fft = nullptr;
    int fftOrder = -1;
    std::vector<float> diff, scratch;

    for (int f = 0; f < numFrames; ++f)
    {
        if (rms[static_cast<size_t> (f)] < floorRms)
            continue;

        const auto start = f * hop;
        for (int i = 0; i < windowSize; ++i)
        {
            const auto index = start + i;
            frame[static_cast<size_t> (i)] = (index < numSamples) ? samples[index] : 0.0f;
        }

        difference (frame.data(), windowSize, maxTau, diff, scratch, fft, fftOrder);
        cumulativeMeanNormalised (diff);

        double tau = 0.0, confidence = 0.0;
        pickTau (diff, tauMin, threshold, tau, confidence);

        if (tau > 0.0)
        {
            const auto hz = sampleRate / tau;
            if (hz >= fmin && hz <= fmax)
            {
                result.f0[static_cast<size_t> (f)] = static_cast<float> (hz);
                result.confidence[static_cast<size_t> (f)] = static_cast<float> (confidence);
            }
        }
    }

    delete fft;
    return result;
}

void Yin::estimate (const float* samples, int numSamples, double sampleRate,
                    double& f0Out, double& confidenceOut, int hop, double minConfidence)
{
    const auto t = track (samples, numSamples, sampleRate, hop);

    std::vector<float> good;
    double confidenceSum = 0.0;
    for (size_t i = 0; i < t.f0.size(); ++i)
    {
        if (t.confidence[i] >= minConfidence)
        {
            good.push_back (t.f0[i]);
            confidenceSum += t.confidence[i];
        }
    }

    if (good.empty())
    {
        f0Out = 0.0;
        confidenceOut = 0.0;
        return;
    }

    std::sort (good.begin(), good.end());
    const auto mid = good.size() / 2;
    // numpy's median averages the middle pair for even counts.
    f0Out = (good.size() % 2 == 1)
          ? good[mid]
          : 0.5 * (static_cast<double> (good[mid - 1]) + good[mid]);
    confidenceOut = confidenceSum / static_cast<double> (good.size());
}

juce::String Yin::noteName (double hz, double& centsOut)
{
    if (hz <= 0.0)
    {
        centsOut = 0.0;
        return "-";
    }
    const auto midi = 69.0 + 12.0 * std::log2 (hz / 440.0);
    const auto nearest = static_cast<int> (std::lround (midi));
    centsOut = (midi - nearest) * 100.0;

    auto index = nearest % 12;
    if (index < 0)
        index += 12;
    return juce::String (kNoteNames[index]) + juce::String (nearest / 12 - 1);
}

} // namespace autosynth
