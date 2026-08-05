#pragma once

#include <juce_dsp/juce_dsp.h>
#include <vector>

namespace autosynth
{

// Port of autosynth/analysis/stft.py.
//
// Conventions are copied exactly, not approximately, because everything
// downstream is calibrated against them:
//
//   * periodic Hann, `np.hanning(n+1)[:n]` -- not the symmetric window;
//   * magnitude scaled by 2/sum(window), so a full-amplitude sinusoid reads
//     as ~1 and absolute thresholds mean the same thing on both sides;
//   * frame centres at (i*hop + n_fft/2)/sr.
//
// A half-window offset or a symmetric-vs-periodic mismatch would not look like
// a bug -- it would look like a slightly worse fitter.
class Stft
{
public:
    // Frames are laid out contiguously: frame i occupies [i*numBins, (i+1)*numBins).
    struct Result
    {
        std::vector<float> magnitude;  // [numFrames * numBins]
        std::vector<float> frequencies; // [numBins]
        std::vector<float> times;       // [numFrames], seconds
        int numFrames = 0;
        int numBins = 0;

        const float* frame (int index) const noexcept
        {
            return magnitude.data() + static_cast<size_t> (index) * static_cast<size_t> (numBins);
        }
    };

    static Result magnitudeSpectrogram (const float* samples, int numSamples,
                                        int fftSize, int hop, double sampleRate);

    // Frame-wise RMS. The amplitude trajectory an amp envelope must explain.
    static std::vector<float> loudnessEnvelope (const float* samples, int numSamples,
                                                int hop = 256, int window = 1024);

    // Per-frame spectral centroid in Hz -- the brightness trajectory.
    static std::vector<float> spectralCentroid (const Result& spectrogram);

    // `np.hanning(size + 1)[:size]`.
    static std::vector<float> periodicHann (int size);

    static int numFramesFor (int numSamples, int fftSize, int hop) noexcept;
};

} // namespace autosynth
