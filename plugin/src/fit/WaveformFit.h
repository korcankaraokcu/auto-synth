#pragma once

#include "ir/Patch.h"
#include <vector>

namespace autosynth
{

// Port of autosynth/fit/waveform.py.
//
// Comparison happens in the log-amplitude domain, weighted by observed
// loudness. Linear comparison is dominated by the fundamental, which every
// waveform has in roughly the same proportion -- what distinguishes a saw from
// a square lives in harmonics 20 dB down, and only a log comparison sees it.
class WaveformFit
{
public:
    // Stops at 0.45: a pulse at 50% duty is bit-for-bit a square wave, so
    // including it puts two exactly-equivalent points in the search space.
    static constexpr int kNumPulseWidths = 8;

    struct Match
    {
        Waveform waveform = Waveform::saw;
        float pulseWidth = 0.5f;
        float cutoffHz = 0.0f; // only set by matchWithCutoff
        double error = 0.0;
    };

    // Use only when the filter has already been divided out. With the filter
    // still present, waveform and cutoff are confounded and must be searched
    // together.
    static Match match (const std::vector<float>& profile);

    // Jointly chooses waveform and lowpass cutoff.
    //
    // This is also the only thing that makes the *absolute* cutoff identifiable
    // at all: source spectrum times filter response is a blind deconvolution,
    // and "bright source, closed filter" is the same signal as "dull source,
    // open filter". Constraining the source to one of a few known waveforms is
    // what breaks the tie.
    static Match matchWithCutoff (const std::vector<float>& profile, double f0Hz,
                                  double sampleRate, float q = 0.707f, int numGrid = 48);

    static std::vector<float> normalise (std::vector<float> v);
    static double profileError (const std::vector<float>& observed,
                                const std::vector<float>& model);
    static float pulseWidthAt (int index) noexcept;
};

} // namespace autosynth
