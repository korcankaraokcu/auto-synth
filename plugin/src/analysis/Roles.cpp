#include "analysis/Roles.h"

#include "analysis/Stft.h"

#include <algorithm>
#include <cmath>

namespace autosynth
{

double Roles::noiseShare (const float* samples, int numSamples, double sampleRate,
                          double f0Hz, int fftSize, int hop)
{
    if (samples == nullptr || numSamples <= 0 || f0Hz <= 20.0)
        return 0.0;

    const auto spectrogram = Stft::magnitudeSpectrogram (samples, numSamples, fftSize, hop, sampleRate);
    if (spectrogram.numFrames <= 0 || spectrogram.numBins <= 1)
        return 0.0;

    const auto binHz = sampleRate / fftSize;

    double harmonic = 0.0, floorEnergy = 0.0;
    for (int t = 0; t < spectrogram.numFrames; ++t)
    {
        const auto* frame = spectrogram.frame (t);
        for (int k = 1; k < spectrogram.numBins; ++k)
        {
            const auto freq = k * binHz;
            const auto nearest = std::max (1.0, std::round (freq / f0Hz));
            const auto distanceBins = std::abs (freq - nearest * f0Hz) / binHz;
            (distanceBins <= kHarmonicGuardBins ? harmonic : floorEnergy) += frame[k];
        }
    }

    const auto total = harmonic + floorEnergy;
    return total > 1.0e-9 ? floorEnergy / total : 0.0;
}

} // namespace autosynth
