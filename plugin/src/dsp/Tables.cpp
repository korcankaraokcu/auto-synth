#include "dsp/Tables.h"

#include <juce_dsp/juce_dsp.h>
#include <cmath>

namespace autosynth
{
namespace
{
constexpr int kFftOrder = 12; // 2^12 == WaveTables::kTableSize
constexpr int kNonPulseWaveforms = 4; // sine, triangle, saw, square

// Pulse widths stop at 0.45 because a 50% pulse is bit-for-bit a square wave.
// The fitter's grid matches; keeping the duplicate would put two identical
// points in the search space for no expressive gain.
float pulseWidthAt (int index) noexcept
{
    return 0.10f + 0.35f * static_cast<float> (index)
                 / static_cast<float> (WaveTables::kNumPulseWidths - 1);
}
} // namespace

int WaveTables::octaveIndexFor (double frequencyHz) noexcept
{
    if (frequencyHz <= kLowestHz)
        return 0;
    const auto octave = static_cast<int> (std::floor (std::log2 (frequencyHz / kLowestHz)));
    return juce::jlimit (0, kNumOctaves - 1, octave);
}

int WaveTables::pulseIndexFor (float pulseWidth) noexcept
{
    const auto t = (juce::jlimit (0.10f, 0.45f, pulseWidth) - 0.10f) / 0.35f;
    return juce::jlimit (0, kNumPulseWidths - 1,
                         static_cast<int> (std::lround (t * (kNumPulseWidths - 1))));
}

size_t WaveTables::offsetFor (Waveform waveform, int pulseIndex, int octave) const noexcept
{
    const auto table = waveform == Waveform::pulse
                     ? kNonPulseWaveforms * kNumOctaves + pulseIndex * kNumOctaves + octave
                     : static_cast<int> (waveform) * kNumOctaves + octave;
    return static_cast<size_t> (table) * kTableSize;
}

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

// Shared spectrum-to-table step. Sine phase throughout, matching every fixed
// shape except the pulse, so a custom frame and a classic waveform of the same
// harmonic content are the same signal.
//
// Deliberately not normalised: the caller scales a whole frame set by one
// factor. Normalising each table on its own would flatten the differences
// between frames, and it would do it by crest factor rather than by loudness --
// a frame with more harmonics has a taller peak at the same energy, so it would
// come out quieter, and moving the frame position would sound like a volume
// change rather than a timbre change.
void WaveTables::buildFromHarmonics (const float* amplitudes, int numAmplitudes,
                                     int maxHarmonics, float* out)
{
    juce::dsp::FFT fft (kFftOrder);
    buildFromHarmonics (amplitudes, numAmplitudes, maxHarmonics, out, fft);
}

void WaveTables::buildFromHarmonics (const float* amplitudes, int numAmplitudes,
                                     int maxHarmonics, float* out, const juce::dsp::FFT& fft)
{
    const auto limit = juce::jlimit (1, kTableSize / 2 - 1,
                                     juce::jmin (numAmplitudes, maxHarmonics));

    std::vector<juce::dsp::Complex<float>> spectrum (kTableSize, { 0.0f, 0.0f });
    for (int k = 1; k <= limit; ++k)
    {
        const auto a = amplitudes[k - 1] * 0.5f;
        spectrum[static_cast<size_t> (k)] = { 0.0f, -a };
        spectrum[static_cast<size_t> (kTableSize - k)] = { 0.0f, a };
    }

    std::vector<juce::dsp::Complex<float>> time (kTableSize);
    fft.perform (spectrum.data(), time.data(), true);

    for (int i = 0; i < kTableSize; ++i)
        out[i] = time[static_cast<size_t> (i)].real();
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

bool WaveTables::FrameTables::matches (const std::array<Oscillator::Frame, Oscillator::kMaxFrames>& frames,
                                      int numFrames, double sampleRate) const noexcept
{
    if (! juce::approximatelyEqual (builtRate, sampleRate) || builtCount != numFrames)
        return false;
    for (int f = 0; f < numFrames; ++f)
    {
        const auto& a = frames[static_cast<size_t> (f)];
        const auto& b = builtFrom[static_cast<size_t> (f)];
        if (a.custom != b.custom)
            return false;
        if (! a.custom)
            continue;
        for (size_t k = 0; k < a.harmonics.size(); ++k)
            if (! juce::approximatelyEqual (a.harmonics[k], b.harmonics[k]))
                return false;
    }
    return true;
}

void WaveTables::FrameTables::build (const std::array<Oscillator::Frame, Oscillator::kMaxFrames>& frames,
                                     int numFrames, double sampleRate)
{
    const auto count = juce::jlimit (1, Oscillator::kMaxFrames, numFrames);
    if (sampleRate <= 0.0 || matches (frames, count, sampleRate))
        return;

    builtFrom = frames;
    builtCount = count;
    builtRate = sampleRate;
    slot.fill (-1);

    // Only drawn frames get storage. A generated frame is one of the shared
    // shapes and already has a mipmap, so building a private copy of it would
    // cost a megabyte and an FFT to arrive back where we started.
    int numCustom = 0;
    for (int f = 0; f < count; ++f)
        if (frames[static_cast<size_t> (f)].custom)
            slot[static_cast<size_t> (f)] = numCustom++;

    storage.assign (static_cast<size_t> (numCustom) * kNumOctaves * kTableSize, 0.0f);
    if (numCustom == 0)
        return;

    // How many harmonics survive in each octave. A frame holds sixteen, so
    // every octave low enough to fit all sixteen produces the *same* table --
    // six of the eleven at 48 kHz. Building each one anyway cost 5 ms per
    // oscillator, which is far too slow to do while someone drags a bar: the
    // audio thread try-locks the patch and outputs silence when it loses, so a
    // slow rebuild is audible as dropouts while editing.
    std::array<int, kNumOctaves> limitFor {};
    const auto nyquist = sampleRate * 0.5;
    for (int octave = 0; octave < kNumOctaves; ++octave)
    {
        const auto topHz = kLowestHz * std::pow (2.0, octave + 1);
        const auto allowed = static_cast<int> (nyquist / juce::jmax (topHz, 1.0));
        limitFor[static_cast<size_t> (octave)] =
            juce::jlimit (1, Oscillator::kFrameHarmonics, allowed);
    }

    juce::dsp::FFT fft (kFftOrder);
    for (int f = 0; f < count; ++f)
    {
        const auto index = slot[static_cast<size_t> (f)];
        if (index < 0)
            continue;

        for (int octave = 0; octave < kNumOctaves; ++octave)
        {
            const auto offset = (static_cast<size_t> (index) * kNumOctaves + octave) * kTableSize;
            auto* out = storage.data() + offset;

            const auto limit = limitFor[static_cast<size_t> (octave)];
            if (octave > 0 && limit == limitFor[static_cast<size_t> (octave - 1)])
            {
                std::copy (out - kTableSize, out, out);
                continue;
            }

            buildFromHarmonics (frames[static_cast<size_t> (f)].harmonics.data(),
                                Oscillator::kFrameHarmonics, limit, out, fft);
        }
    }

    // One scale factor for the whole set, so relative loudness between frames
    // survives and the oscillator still peaks at 1 like every fixed shape.
    float peak = 0.0f;
    for (const auto v : storage)
        peak = juce::jmax (peak, std::abs (v));
    if (peak > 1.0e-12f)
        for (auto& v : storage)
            v /= peak;
}

const float* WaveTables::FrameTables::tableFor (int frame, double frequencyHz) const noexcept
{
    if (frame < 0 || frame >= Oscillator::kMaxFrames)
        return nullptr;
    const auto index = slot[static_cast<size_t> (frame)];
    if (index < 0 || storage.empty())
        return nullptr;
    const auto octave = octaveIndexFor (frequencyHz);
    return storage.data() + (static_cast<size_t> (index) * kNumOctaves + octave) * kTableSize;
}

void WaveTables::buildTable (Waveform waveform, int numHarmonics, float pulseWidth, float* out)
{
    numHarmonics = juce::jlimit (1, kTableSize / 2 - 1, numHarmonics);

    std::vector<float> amps;
    bool cosinePhase = false;
    harmonicAmplitudes (waveform, numHarmonics, pulseWidth, amps, cosinePhase);

    // Build the spectrum and inverse-transform. A real output needs conjugate
    // symmetry: bin k and bin N-k carry conjugate values. Overall scale is
    // irrelevant because the table is peak-normalised afterwards, so the FFT's
    // normalisation convention does not have to be reasoned about.
    std::vector<juce::dsp::Complex<float>> spectrum (kTableSize, { 0.0f, 0.0f });
    for (int k = 1; k <= numHarmonics; ++k)
    {
        const auto a = amps[static_cast<size_t> (k - 1)] * 0.5f;
        if (cosinePhase)
        {
            spectrum[static_cast<size_t> (k)] = { a, 0.0f };
            spectrum[static_cast<size_t> (kTableSize - k)] = { a, 0.0f };
        }
        else
        {
            spectrum[static_cast<size_t> (k)] = { 0.0f, -a };
            spectrum[static_cast<size_t> (kTableSize - k)] = { 0.0f, a };
        }
    }

    std::vector<juce::dsp::Complex<float>> time (kTableSize);
    juce::dsp::FFT fft (kFftOrder);
    fft.perform (spectrum.data(), time.data(), true);

    float peak = 0.0f;
    for (int i = 0; i < kTableSize; ++i)
    {
        out[i] = time[static_cast<size_t> (i)].real();
        peak = juce::jmax (peak, std::abs (out[i]));
    }
    if (peak > 1.0e-12f)
        for (int i = 0; i < kTableSize; ++i)
            out[i] /= peak;
}

void WaveTables::prepare (double sampleRate)
{
    if (sampleRate <= 0.0 || juce::approximatelyEqual (sampleRate, preparedRate))
        return;

    preparedRate = sampleRate;
    const auto nyquist = sampleRate * 0.5;

    const auto numTables = static_cast<size_t> (kNonPulseWaveforms * kNumOctaves
                                                + kNumPulseWidths * kNumOctaves);
    storage.assign (numTables * kTableSize, 0.0f);

    const auto build = [&] (Waveform waveform, int pulseIndex, float pulseWidth)
    {
        for (int octave = 0; octave < kNumOctaves; ++octave)
        {
            // Band-limit for the *top* of the octave, not its bottom, or notes
            // in the upper half of each band alias.
            const auto topHz = kLowestHz * std::pow (2.0, octave + 1);
            const auto harmonics = static_cast<int> (nyquist / juce::jmax (topHz, 1.0));
            buildTable (waveform, juce::jmax (1, harmonics), pulseWidth,
                        storage.data() + offsetFor (waveform, pulseIndex, octave));
        }
    };

    build (Waveform::sine, 0, 0.5f);
    build (Waveform::triangle, 0, 0.5f);
    build (Waveform::saw, 0, 0.5f);
    build (Waveform::square, 0, 0.5f);
    for (int p = 0; p < kNumPulseWidths; ++p)
        build (Waveform::pulse, p, pulseWidthAt (p));
}

const float* WaveTables::tableFor (Waveform waveform, float pulseWidth,
                                   double frequencyHz) const noexcept
{
    jassert (isPrepared());
    const auto pulseIndex = waveform == Waveform::pulse ? pulseIndexFor (pulseWidth) : 0;
    return storage.data() + offsetFor (waveform, pulseIndex, octaveIndexFor (frequencyHz));
}

float WaveTables::lookup (const float* table, double phase) noexcept
{
    auto frac = phase - std::floor (phase);
    const auto scaled = frac * kTableSize;
    const auto i0 = static_cast<int> (scaled);
    const auto t = static_cast<float> (scaled - i0);
    const auto a = table[i0 & (kTableSize - 1)];
    const auto b = table[(i0 + 1) & (kTableSize - 1)];
    return a + (b - a) * t;
}

} // namespace autosynth
