#pragma once

// Shared scaffolding for the test suite: patch builders, offline rendering,
// the fixture loader and the handful of spectral metrics the assertions are
// phrased in.
//
// The metrics are deliberately the same three the refinement objective uses --
// loudness, brightness and a multi-scale spectral distance -- so a bound
// written here means the same thing as a bound written there.

#include "analysis/Stft.h"
#include "dsp/Voice.h"
#include "ir/Patch.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <cmath>
#include <string>
#include <vector>

namespace autotest
{

constexpr double kSampleRate = 48000.0;
constexpr double kDuration = 1.0;
constexpr double kGate = 0.7;

// --- fixtures --------------------------------------------------------------

inline juce::File goldenDir()
{
    return juce::File (juce::String (AUTOSYNTH_GOLDEN_DIR));
}

inline std::vector<float> readWav (const juce::File& file, double* sampleRateOut = nullptr)
{
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (file));
    if (reader == nullptr)
        return {};

    const auto numSamples = static_cast<int> (reader->lengthInSamples);
    juce::AudioBuffer<float> buffer (static_cast<int> (reader->numChannels), numSamples);
    reader->read (&buffer, 0, numSamples, 0, true, true);

    // Downmix exactly as the probe and audio.read_wav do.
    if (buffer.getNumChannels() > 1)
    {
        for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
            buffer.addFrom (0, 0, buffer, ch, 0, numSamples);
        buffer.applyGain (0, 0, numSamples, 1.0f / buffer.getNumChannels());
    }

    if (sampleRateOut != nullptr)
        *sampleRateOut = reader->sampleRate;

    const auto* data = buffer.getReadPointer (0);
    return std::vector<float> (data, data + numSamples);
}

inline juce::var readJson (const juce::File& file)
{
    return juce::JSON::parse (file.loadFileAsString());
}

// --- rendering -------------------------------------------------------------

inline std::vector<float> render (const autosynth::Patch& patch,
                                  double noteHz = 0.0,
                                  double duration = kDuration,
                                  double gate = kGate,
                                  const std::array<float, autosynth::kNumOsc>* monitor = nullptr)
{
    autosynth::Engine engine;
    engine.prepare (kSampleRate, 512);
    engine.setPatch (patch);
    if (monitor != nullptr)
        engine.setMonitorMask (*monitor);

    juce::AudioBuffer<float> buffer;
    engine.renderOffline (buffer, noteHz > 0.0 ? noteHz : patch.rootHz, duration, gate);

    const auto* data = buffer.getReadPointer (0);
    std::vector<float> out (data, data + buffer.getNumSamples());

    // Peak-limit, matching autosynth_render and audio.write_wav. Without this
    // a patch that clips would be compared against a reference that did not.
    float peak = 0.0f;
    for (auto v : out)
        peak = juce::jmax (peak, std::abs (v));
    if (peak > 1.0f)
        for (auto& v : out)
            v /= peak;

    return out;
}

// --- patch builders --------------------------------------------------------

// A single saw at 220 Hz with everything else out of the way. Every engine
// test starts here and switches on exactly the one thing it is about.
inline autosynth::Patch simplePatch (autosynth::Waveform waveform = autosynth::Waveform::saw)
{
    autosynth::Patch patch;
    for (auto& osc : patch.oscs)
    {
        osc.enabled = false;
        osc.level = 0.0f;
    }
    patch.oscs[0].enabled = true;
    patch.oscs[0].level = 1.0f;
    patch.oscs[0].waveform = waveform;
    patch.ampEnv = { 0.01f, 0.2f, 0.7f, 0.2f, 0.0f };
    patch.filter.type = autosynth::FilterType::off;
    for (auto& lfo : patch.lfos)
    {
        lfo.dest = autosynth::LfoDest::none;
        lfo.depth = 0.0f;
    }
    patch.noiseLevel = 0.0f;
    patch.rootHz = 220.0f;
    return patch;
}

// --- metrics ---------------------------------------------------------------

inline float peakOf (const std::vector<float>& x)
{
    float peak = 0.0f;
    for (auto v : x)
        peak = juce::jmax (peak, std::abs (v));
    return peak;
}

inline double sumAbs (const std::vector<float>& x)
{
    double acc = 0.0;
    for (auto v : x)
        acc += std::abs (v);
    return acc;
}

inline std::vector<float> loudnessOf (const std::vector<float>& x, int hop = 256)
{
    return autosynth::Stft::loudnessEnvelope (x.data(), static_cast<int> (x.size()), hop);
}

// Mean absolute difference of the two loudness envelopes in dB, with a floor
// 80 dB below each signal's own peak.
//
// The floor is not cosmetic. Without it, comparing anything against a decayed
// tail means comparing against digital silence, and the difference runs to
// ~140 dB -- a number that swamps every real difference in the same average.
inline double loudnessDistanceDb (const std::vector<float>& a, const std::vector<float>& b)
{
    const auto la = loudnessOf (a);
    const auto lb = loudnessOf (b);
    const auto n = juce::jmin (la.size(), lb.size());
    if (n == 0)
        return 0.0;

    const auto floorOf = [] (const std::vector<float>& v)
    {
        float peak = 0.0f;
        for (auto s : v)
            peak = juce::jmax (peak, s);
        return juce::jmax (1.0e-10f, peak * 1.0e-4f); // -80 dB
    };
    const auto fa = floorOf (la), fb = floorOf (lb);

    double acc = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const auto da = 20.0 * std::log10 (juce::jmax (la[i], fa));
        const auto db = 20.0 * std::log10 (juce::jmax (lb[i], fb));
        acc += std::abs (da - db);
    }
    return acc / static_cast<double> (n);
}

// Mean absolute difference of spectral centroid, in octaves.
inline double centroidDistanceOctaves (const std::vector<float>& a, const std::vector<float>& b)
{
    const auto sa = autosynth::Stft::magnitudeSpectrogram (a.data(), (int) a.size(), 2048, 256, kSampleRate);
    const auto sb = autosynth::Stft::magnitudeSpectrogram (b.data(), (int) b.size(), 2048, 256, kSampleRate);
    const auto ca = autosynth::Stft::spectralCentroid (sa);
    const auto cb = autosynth::Stft::spectralCentroid (sb);
    const auto n = juce::jmin (ca.size(), cb.size());
    if (n == 0)
        return 0.0;

    double acc = 0.0;
    int counted = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (ca[i] < 20.0f || cb[i] < 20.0f)
            continue; // silence has no meaningful brightness
        acc += std::abs (std::log2 (ca[i] / cb[i]));
        ++counted;
    }
    return counted > 0 ? acc / counted : 0.0;
}

// Mean spectral centroid in Hz, for "is this brighter than that?" assertions.
inline double meanCentroidHz (const std::vector<float>& x)
{
    const auto s = autosynth::Stft::magnitudeSpectrogram (x.data(), (int) x.size(), 2048, 256, kSampleRate);
    const auto c = autosynth::Stft::spectralCentroid (s);
    double acc = 0.0;
    int counted = 0;
    for (auto v : c)
        if (v > 20.0f) { acc += v; ++counted; }
    return counted > 0 ? acc / counted : 0.0;
}

// Energy in a band, for asserting that a filter did what it claims.
inline double bandEnergy (const std::vector<float>& x, double lowHz, double highHz)
{
    const auto s = autosynth::Stft::magnitudeSpectrogram (x.data(), (int) x.size(), 2048, 256, kSampleRate);
    const auto binHz = kSampleRate / 2048.0;
    const auto lo = juce::jlimit (0, s.numBins - 1, (int) std::floor (lowHz / binHz));
    const auto hi = juce::jlimit (0, s.numBins - 1, (int) std::ceil (highHz / binHz));

    double acc = 0.0;
    for (int t = 0; t < s.numFrames; ++t)
    {
        const auto* frame = s.frame (t);
        for (int k = lo; k <= hi; ++k)
            acc += frame[k];
    }
    return acc;
}

} // namespace autotest
