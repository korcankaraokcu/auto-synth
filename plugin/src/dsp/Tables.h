#pragma once

#include "ir/Patch.h"
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

private:
    static int octaveIndexFor (double frequencyHz) noexcept;
    static int pulseIndexFor (float pulseWidth) noexcept;
    static void buildTable (Waveform waveform, int numHarmonics, float pulseWidth, float* out);

    size_t offsetFor (Waveform waveform, int pulseIndex, int octave) const noexcept;

    std::vector<float> storage;
    double preparedRate = 0.0;
};

} // namespace autosynth
