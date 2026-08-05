#pragma once

#include "ir/Patch.h"
#include <cmath>

namespace autosynth
{

// Magnitude response of the 2-pole analog prototype, matching
// engine/filters.analog_magnitude.
//
// Distinct from `Svf`, and deliberately so. Svf is what *plays* audio; this is
// what *fitting* reasons about. The fitter needs to ask "what would a filter at
// this cutoff do to this harmonic?" thousands of times without rendering
// anything, and it needs to give the same answer as the Python fitter did.
//
// Keeping them separate is also honest about the seam: the fitting model and
// the playback model are not the same filter, which is exactly what the engine
// conformance test measures.
inline float analogMagnitude (FilterType type, float frequencyHz, float cutoffHz, float q) noexcept
{
    if (type == FilterType::off)
        return 1.0f;

    const auto w = frequencyHz / juce::jmax (cutoffHz, 1.0e-6f);
    const auto w2 = w * w;
    const auto qq = juce::jmax (q, 1.0e-6f);
    const auto denom = std::sqrt ((1.0f - w2) * (1.0f - w2) + (w / qq) * (w / qq)) + 1.0e-12f;

    switch (type)
    {
        case FilterType::lowpass:  return 1.0f / denom;
        case FilterType::highpass: return w2 / denom;
        case FilterType::bandpass: return (w / qq) / denom;
        case FilterType::off:      break;
    }
    return 1.0f;
}

} // namespace autosynth
