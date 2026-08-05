#pragma once

#include <juce_core/juce_core.h>
#include <limits>
#include <vector>

namespace autosynth
{

// Port of autosynth/analysis/partials.py -- sinusoidal partial tracking
// (McAulay-Quatieri).
//
// This is what replaces a fixed harmonic grid. Sampling H at integer multiples
// of one detected fundamental can only see sources harmonically related to it;
// two oscillators a fifth apart put partials on half-integer multiples, which
// fall between the grid points and are simply absent. Tracking measures
// partials where they actually are, and deciding which belong together is left
// to Grouping.
struct Partial
{
    std::vector<int> frames;
    std::vector<float> freqs;
    std::vector<float> amps;

    int length() const noexcept { return static_cast<int> (frames.size()); }

    // Amplitude-weighted: a partial's frequency estimate is least reliable
    // where it is quietest, which is at birth, death, and through the tail.
    float meanFreq() const noexcept;
    float energy() const noexcept;
    float peakAmp() const noexcept;
};

struct PartialSet
{
    std::vector<Partial> partials;
    std::vector<float> times;
    double sampleRate = 48000.0;
    int fftSize = 2048;
    int hop = 256;

    int numFrames() const noexcept { return static_cast<int> (times.size()); }
    float totalEnergy() const noexcept;
};

class PartialTracker
{
public:
    // Defaults were measured, not guessed. A permissive floor (-70 dB) with a
    // short minimum length (4 frames) tracked 700+ partials on filtered, noisy
    // material where a few dozen exist, and grouping then locked onto
    // fundamentals that were pure noise artefacts.
    struct Options
    {
        int fftSize = 2048;
        int hop = 256;
        double tolCents = 60.0;
        int maxSleep = 2;
        int minFrames = 12;
        double floorDb = -55.0;
        int maxPeaks = 80;
    };

    static PartialSet track (const float* samples, int numSamples, double sampleRate,
                             const Options& options = {});

    // Spectral peaks in one frame, refined to sub-bin accuracy by parabolic
    // interpolation on the *log* magnitude -- a Hann main lobe is close to a
    // parabola in dB, and the same fit in linear magnitude is biased.
    static void findPeaks (const float* magnitude, int numBins, double sampleRate,
                           int fftSize, double floorDb, int maxPeaks,
                           std::vector<float>& freqsOut, std::vector<float>& ampsOut);
};

} // namespace autosynth
