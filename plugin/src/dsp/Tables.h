#pragma once

#include "ir/Patch.h"

#include <juce_dsp/juce_dsp.h>
#include <vector>

namespace autosynth
{

// Band-limited wavetables, built in the frequency domain from each waveform's
// Fourier series and inverse-transformed -- the same construction as
// engine/tables.py, so a table here holds the same harmonic spectrum as there.
//
// Two things differ, and both are forced by running in realtime.
//
// The Python engine rebuilds a table per note because it renders offline. Doing
// that inside a note-on would allocate and run an FFT on the audio thread, so
// tables are built once per (waveform, pulse width, octave) into a mipmap and
// playback picks the level whose harmonic count fits under Nyquist.
//
// And the harmonic count depends on the sample rate, which is not known until
// the host says so -- hence `prepare` rather than a static singleton. Building
// for a fixed worst case would either alias at 44.1 kHz or throw away the top
// octave at 96 kHz.
class WaveTables
{
public:
    static constexpr int kTableSize = 4096;
    static constexpr int kNumOctaves = 11;      // ~20 Hz to ~20 kHz
    static constexpr int kNumPulseWidths = 8;   // 0.10 .. 0.45, matching the fitter
    static constexpr double kLowestHz = 20.0;

    // Safe to call from prepareToPlay; allocates. Never call from the audio
    // thread.
    void prepare (double sampleRate);

    bool isPrepared() const noexcept { return ! storage.empty(); }

    // Returns a pointer to kTableSize floats. `pulseWidth` is ignored unless
    // the waveform is pulse.
    const float* tableFor (Waveform waveform, float pulseWidth, double frequencyHz) const noexcept;

    // Linear interpolation on a phase in cycles; only the fractional part is
    // used. Matches tables.lookup in the Python engine.
    static float lookup (const float* table, double phase) noexcept;

    // Public because the waveform fitter compares observed harmonic profiles
    // against these same series -- fitter and engine must agree on what a "saw"
    // is, and the only way to guarantee that is to share the definition.
    static void harmonicAmplitudes (Waveform waveform, int numHarmonics, float pulseWidth,
                                    std::vector<float>& amps, bool& cosinePhase);

    // Band-limited mipmap for the *drawn* frames of one oscillator.
    //
    // Separate from the shared tables above because these depend on the patch,
    // not just on the sample rate: every edited or fitted frame is its own
    // spectrum. A frame that has not been drawn on is one of the shared shapes
    // and gets no storage here -- `tableFor` returns null for it and the caller
    // falls back to the mipmap it already has.
    //
    // Building runs an FFT per drawn frame per octave, so it must not happen on
    // the audio thread. It does not: frame data changes only when a patch is
    // loaded, analysed or edited, all off-thread, and `matches` skips the
    // rebuild otherwise. Host automation changes scalars, never frames --
    // sixteen harmonics times sixteen frames is not something to expose as
    // knobs.
    class FrameTables
    {
    public:
        void build (const std::array<Oscillator::Frame, Oscillator::kMaxFrames>& frames,
                    int numFrames, double sampleRate);

        bool matches (const std::array<Oscillator::Frame, Oscillator::kMaxFrames>& frames,
                      int numFrames, double sampleRate) const noexcept;

        // Null when that frame is generated rather than drawn.
        const float* tableFor (int frame, double frequencyHz) const noexcept;

    private:
        std::vector<float> storage;
        std::array<Oscillator::Frame, Oscillator::kMaxFrames> builtFrom {};
        std::array<int, Oscillator::kMaxFrames> slot {};
        int builtCount = 0;
        double builtRate = 0.0;
    };

    // The harmonic series a generated frame stands for: the blend of two fixed
    // shapes, peak-normalised. One definition, used by the oscillator that
    // plays it, the fitter that scores against it and the editor that seeds a
    // frame from it -- three places that must agree on what "saw" means, and
    // the only way to guarantee that is to share the arithmetic.
    static std::vector<float> blendedHarmonics (Waveform a, Waveform b, float morph,
                                                float pulseWidth, int numHarmonics);

    // Builds a band-limited table from an arbitrary harmonic series rather than
    // one of the fixed shapes. Public so the fitter can render a candidate
    // frame without going through an oscillator.
    static void buildFromHarmonics (const float* amplitudes, int numAmplitudes,
                                    int maxHarmonics, float* out);

private:
    static int octaveIndexFor (double frequencyHz) noexcept;
    static int pulseIndexFor (float pulseWidth) noexcept;
    static void buildTable (Waveform waveform, int numHarmonics, float pulseWidth, float* out);
    static void buildFromHarmonics (const float* amplitudes, int numAmplitudes,
                                    int maxHarmonics, float* out, const juce::dsp::FFT& fft);

    size_t offsetFor (Waveform waveform, int pulseIndex, int octave) const noexcept;

    std::vector<float> storage;
    double preparedRate = 0.0;
};

} // namespace autosynth
