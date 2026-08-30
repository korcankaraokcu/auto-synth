#include "dsp/Tables.h"

#include <cmath>

namespace autosynth
{

void WaveTables::harmonicAmplitudes (Waveform waveform, int numHarmonics, float pulseWidth,
                                     std::vector<float>& amps, bool& cosinePhase)
{
    amps.assign (static_cast<size_t> (numHarmonics), 0.0f);
    cosinePhase = false;

    for (int k = 1; k <= numHarmonics; ++k)
    {
        const auto i = static_cast<size_t> (k - 1);
        const auto kf = static_cast<float> (k);

        switch (waveform)
        {
            case Waveform::sine:
                amps[i] = (k == 1) ? 1.0f : 0.0f;
                break;
            case Waveform::saw:
                amps[i] = 1.0f / kf;
                break;
            case Waveform::square:
                amps[i] = (k % 2 == 1) ? 1.0f / kf : 0.0f;
                break;
            case Waveform::triangle:
                if (k % 2 == 1)
                {
                    const auto sign = (((k - 1) / 2) % 2 == 0) ? 1.0f : -1.0f;
                    amps[i] = sign / (kf * kf);
                }
                break;
            case Waveform::pulse:
            {
                const auto d = juce::jlimit (0.01f, 0.99f, pulseWidth);
                amps[i] = 2.0f * std::sin (kf * juce::MathConstants<float>::pi * d)
                        / (kf * juce::MathConstants<float>::pi);
                cosinePhase = true;
                break;
            }
            case Waveform::noise:
                // Flat, which is what noise looks like on a harmonic grid. It
                // never reaches a table -- the voice short-circuits it -- but
                // the fitter reads profiles through this function and a case
                // that fell through would silently return silence.
                amps[i] = 1.0f;
                break;
        }
    }
}

std::vector<float> WaveTables::blendedHarmonics (Waveform a, Waveform b, float morph,
                                                 float pulseWidth, int numHarmonics)
{
    const auto n = juce::jmax (1, numHarmonics);
    std::vector<float> first, second;
    bool cosinePhase = false;
    harmonicAmplitudes (a, n, pulseWidth, first, cosinePhase);
    harmonicAmplitudes (b, n, pulseWidth, second, cosinePhase);

    const auto blend = juce::jlimit (0.0f, 1.0f, morph);
    std::vector<float> out (static_cast<size_t> (n), 0.0f);
    float peak = 0.0f;
    for (size_t k = 0; k < out.size(); ++k)
    {
        out[k] = std::abs (first[k]) * (1.0f - blend) + std::abs (second[k]) * blend;
        peak = juce::jmax (peak, out[k]);
    }
    if (peak > 1.0e-12f)
        for (auto& v : out)
            v /= peak;
    return out;
}

} // namespace autosynth
