#pragma once

#include "ir/Patch.h"
#include <cmath>

namespace autosynth
{

// Magnitude response of the 2-pole analog prototype, matching
// engine/filters.analog_magnitude.
//
// This is what *fitting* reasons about, and it is not what plays: the fitter
// asks "what would a filter at this cutoff do to this harmonic?" thousands of
// times without rendering anything, while the sound is made by Vital's own
// filters. So it is a model of a filter rather than the filter, close enough to
// choose a cutoff from and never used to produce a sample.
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
