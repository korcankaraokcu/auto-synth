#pragma once

#include <juce_core/juce_core.h>
#include <vector>

namespace autosynth
{

// Port of autosynth/analysis/f0.py.
//
// YIN rather than plain autocorrelation because the cumulative mean normalised
// difference is what suppresses octave errors, and an octave error here is not
// a small mistake -- it makes the entire harmonic grid wrong and produces a
// confident, plausible fit to a sound nobody played.
class Yin
{
public:
    static constexpr double kDefaultFMin = 30.0;
    static constexpr double kDefaultFMax = 2000.0;
    static constexpr double kThreshold = 0.15;

    struct Track
    {
        std::vector<float> f0;         // Hz, 0 where unvoiced
        std::vector<float> confidence; // [0, 1]
    };

    static Track track (const float* samples, int numSamples, double sampleRate,
                        int hop = 256,
                        double fmin = kDefaultFMin, double fmax = kDefaultFMax,
                        double threshold = kThreshold);

    // Median over confident frames. Assumes a roughly static pitch, which is
    // true for one-shot notes and a documented limitation for glides.
    static void estimate (const float* samples, int numSamples, double sampleRate,
                          double& f0Out, double& confidenceOut,
                          int hop = 256, double minConfidence = 0.5);

    // Nearest equal-tempered note name plus offset in cents. Reported, never
    // snapped to: a sample 30 cents flat should render 30 cents flat.
    static juce::String noteName (double hz, double& centsOut);

    // The difference function, CMND and dip-picking live in the .cpp rather
    // than here: they need juce::dsp::FFT, and pulling that into the header
    // would make every translation unit that wants an f0 estimate depend on
    // the whole DSP module.
};

} // namespace autosynth
