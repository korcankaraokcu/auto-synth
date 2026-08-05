#pragma once

#include "ir/Patch.h"
#include <cmath>

namespace autosynth
{

// Topology-preserving state variable filter (Zavalishin), 2-pole.
//
// **This is the known seam between the two engines.** The Python engine filters
// in the frequency domain -- per-frame magnitude response applied inside an
// STFT -- because that is parallel over time, differentiable, and above all
// because it is the same model the analysis front-end fits, which removes a
// whole class of systematic error from the fitter.
//
// None of that survives contact with realtime: an STFT filter needs the whole
// signal, and a plugin has 128 samples and no future. So this is a genuinely
// different filter, matched in magnitude response but not in phase, and with
// resonance behaviour the frequency-domain version does not model at all.
//
// The two will therefore *not* produce identical output, and no amount of care
// will make them. What matters is that they stay spectrally close, which is
// what the conformance test asserts -- render the same patch through both and
// compare. That test is the reason the headless renderer exists.
class Svf
{
public:
    void prepare (double sr) noexcept
    {
        sampleRate = sr;
        reset();
    }

    void reset() noexcept
    {
        ic1eq = 0.0f;
        ic2eq = 0.0f;
    }

    void setType (FilterType t) noexcept { type = t; }

    void update (float cutoffHz, float resonance) noexcept
    {
        const auto nyquist = static_cast<float> (sampleRate * 0.5);
        const auto fc = juce::jlimit (20.0f, nyquist * 0.98f, cutoffHz);
        g = std::tan (juce::MathConstants<float>::pi * fc / static_cast<float> (sampleRate));
        // Q maps to the damping term; 0.707 is critically damped (no peak).
        k = 1.0f / juce::jmax (resonance, 0.05f);
        a1 = 1.0f / (1.0f + g * (g + k));
        a2 = g * a1;
    }

    float processSample (float x) noexcept
    {
        if (type == FilterType::off)
            return x;

        const auto v3 = x - ic2eq;
        const auto v1 = a1 * ic1eq + a2 * v3;
        const auto v2 = ic2eq + g * v1;
        ic1eq = 2.0f * v1 - ic1eq;
        ic2eq = 2.0f * v2 - ic2eq;

        switch (type)
        {
            case FilterType::lowpass:  return v2;
            case FilterType::bandpass: return v1;
            case FilterType::highpass: return x - k * v1 - v2;
            case FilterType::off:      break;
        }
        return x;
    }

private:
    double sampleRate = 44100.0;
    FilterType type = FilterType::lowpass;
    float g = 0.0f, k = 1.4142f, a1 = 0.0f, a2 = 0.0f;
    float ic1eq = 0.0f, ic2eq = 0.0f;
};

} // namespace autosynth
