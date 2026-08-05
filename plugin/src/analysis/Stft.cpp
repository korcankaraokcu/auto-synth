#include "analysis/Stft.h"

#include <cmath>

namespace autosynth
{

std::vector<float> Stft::periodicHann (int size)
{
    std::vector<float> window (static_cast<size_t> (size));
    // numpy's hanning(N) is symmetric over N points; taking the first N of
    // hanning(N+1) gives the periodic window, which is what an STFT wants and
    // what the Python side uses.
    const auto denom = static_cast<double> (size); // (N+1) - 1
    for (int i = 0; i < size; ++i)
        window[static_cast<size_t> (i)] =
            static_cast<float> (0.5 - 0.5 * std::cos (2.0 * juce::MathConstants<double>::pi * i / denom));
    return window;
}

int Stft::numFramesFor (int numSamples, int fftSize, int hop) noexcept
{
    if (numSamples < fftSize)
        return 1; // the Python side zero-pads up to one full frame
    return 1 + (numSamples - fftSize) / hop;
}

Stft::Result Stft::magnitudeSpectrogram (const float* samples, int numSamples,
                                         int fftSize, int hop, double sampleRate)
{
    Result result;
    result.numBins = fftSize / 2 + 1;
    result.numFrames = numFramesFor (numSamples, fftSize, hop);

    const auto window = periodicHann (fftSize);
    double windowSum = 0.0;
    for (auto w : window)
        windowSum += w;
    const auto scale = static_cast<float> (2.0 / juce::jmax (windowSum, 1.0e-12));

    result.frequencies.resize (static_cast<size_t> (result.numBins));
    for (int b = 0; b < result.numBins; ++b)
        result.frequencies[static_cast<size_t> (b)] =
            static_cast<float> (b * sampleRate / fftSize);

    result.times.resize (static_cast<size_t> (result.numFrames));
    for (int f = 0; f < result.numFrames; ++f)
        result.times[static_cast<size_t> (f)] =
            static_cast<float> ((f * hop + fftSize / 2.0) / sampleRate);

    result.magnitude.assign (static_cast<size_t> (result.numFrames) * static_cast<size_t> (result.numBins), 0.0f);

    const auto order = static_cast<int> (std::log2 (static_cast<double> (fftSize)));
    juce::dsp::FFT fft (order);
    std::vector<float> scratch (static_cast<size_t> (fftSize) * 2);

    for (int f = 0; f < result.numFrames; ++f)
    {
        std::fill (scratch.begin(), scratch.end(), 0.0f);
        const auto start = f * hop;
        for (int i = 0; i < fftSize; ++i)
        {
            const auto index = start + i;
            const auto value = (index < numSamples) ? samples[index] : 0.0f;
            scratch[static_cast<size_t> (i)] = value * window[static_cast<size_t> (i)];
        }

        fft.performRealOnlyForwardTransform (scratch.data(), true);

        auto* out = result.magnitude.data() + static_cast<size_t> (f) * static_cast<size_t> (result.numBins);
        for (int b = 0; b < result.numBins; ++b)
        {
            const auto re = scratch[static_cast<size_t> (b) * 2];
            const auto im = scratch[static_cast<size_t> (b) * 2 + 1];
            out[b] = std::sqrt (re * re + im * im) * scale;
        }
    }

    return result;
}

std::vector<float> Stft::loudnessEnvelope (const float* samples, int numSamples,
                                           int hop, int windowSize)
{
    const auto numFrames = numFramesFor (numSamples, windowSize, hop);
    std::vector<float> out (static_cast<size_t> (numFrames), 0.0f);

    for (int f = 0; f < numFrames; ++f)
    {
        double sum = 0.0;
        const auto start = f * hop;
        for (int i = 0; i < windowSize; ++i)
        {
            const auto index = start + i;
            const auto value = (index < numSamples) ? static_cast<double> (samples[index]) : 0.0;
            sum += value * value;
        }
        out[static_cast<size_t> (f)] =
            static_cast<float> (std::sqrt (sum / windowSize + 1.0e-20));
    }
    return out;
}

std::vector<float> Stft::spectralCentroid (const Result& spectrogram)
{
    std::vector<float> out (static_cast<size_t> (spectrogram.numFrames), 0.0f);
    for (int f = 0; f < spectrogram.numFrames; ++f)
    {
        const auto* mag = spectrogram.frame (f);
        double energy = 1.0e-12;
        double weighted = 0.0;
        for (int b = 0; b < spectrogram.numBins; ++b)
        {
            energy += mag[b];
            weighted += mag[b] * spectrogram.frequencies[static_cast<size_t> (b)];
        }
        out[static_cast<size_t> (f)] = static_cast<float> (weighted / energy);
    }
    return out;
}

} // namespace autosynth
