#pragma once

#include "ir/Patch.h"

namespace autosynth

{

// Writes a patch as a Vital preset.
//
// This is the first exporter, and it is the reason the IR is the deliverable
// rather than the engine: nothing here touches synthesis. It translates one
// description into another, and a second target would be another file this
// size.
//
// Vital was the right first choice because its oscillators are wavetables and
// ours now are too, so the part that would otherwise be a lossy approximation is
// a direct copy: sixteen harmonic amplitudes inverse-transform into the 2048
// samples per frame it expects, and a frame nobody has drawn on is its own
// waveform's series.
//
// The numeric mappings below were *measured*, not guessed. A first version was
// written from assumption and got most of them wrong -- envelope times as
// seconds, LFO rate in hertz, tune in cents, the master gain as a fraction, and
// modulation amounts stored inside the routing entry where Vital never looks.
// They were corrected against 274 real presets, and the evidence for each is
// recorded with it so the next person can check rather than trust.
class VitalExport
{
public:
    struct Mapping
    {
        // Envelope times are the fourth root of seconds.
        //
        // Measured: across 274 presets the stored decay spans 0.27 to 2.37 and
        // clusters hard on 1.0. As seconds that would mean no preset in the
        // collection decays faster than a quarter second, which is plainly
        // false. As a fourth root it means 0.006 s to 31.4 s, and the maximum
        // lands on 32 -- Vital's documented ceiling -- with the default at
        // exactly one second.
        static float secondsToEnvelope (float seconds) noexcept;

        // LFO rate is log2 of hertz, and only when the LFO is not synced to
        // tempo. Measured: stored values run -5.32 to 9.0, which is 0.025 Hz to
        // 512 Hz; 251 of 274 presets sit at 1.0 with `sync` at 1, which is a
        // tempo division rather than a frequency.
        static float hzToLfoRate (float hz) noexcept;

        // Filter cutoff is a MIDI note number: 440 Hz is 69. Measured range 8
        // to 136 across the collection, which is 20 Hz to 21 kHz.
        static float cutoffToNote (float hz) noexcept;

        // Tune is a *semitone* fraction, not cents. Measured: -0.287 to 1.0
        // across the collection, clustering on 0.
        static float centsToTune (float cents) noexcept;

        // Unison detune is quadratic on 0 to 10 and displayed as a percentage,
        // so the percentage is the square of the stored value -- which is why
        // 124 presets sit on 4.4721, the square root of twenty, for a default of
        // 20%. Our cents are read as that percentage.
        static float detuneToUnison (float cents) noexcept;

        // Master volume is a square-root control displayed in decibels with an
        // offset of -80: the decibel reading is sqrt(stored) - 80, so silence is
        // 0, unity is 6400 and the ceiling of 7399.44 is +6 dB. Left at its
        // default this exported everything about 6 dB quiet.
        static float gainToVolume (float linearGain) noexcept;

        // Makeup applied to the master on the way out, in decibels.
        //
        // Our `masterLevel` is normalised against the *source sample's peak*,
        // which is a fact about the recording rather than a decision about how
        // loud the patch should be -- a quietly-recorded note should not export
        // as a quiet preset. Vital's own default sits at -6 dB, which is its
        // headroom convention, so a peak-normalised patch belongs there rather
        // than 6 dB under it.
        static constexpr float kExportMakeupDb = 6.0f;

        // Oscillator level is *quadratic*: the amplitude is the square of the
        // stored value, and its default of 0.70710678 is exactly the value that
        // renders as a half. Written linearly, a level of 0.3 arrived as 0.09.
        static float levelToOscLevel (float linearLevel) noexcept;

        // Resonance is normalised 0 to 1; ours is a Q between 0.5 and 8.
        static float resonanceToNormalised (float q) noexcept;

        // Samples per wavetable frame. Vital is 2048 -- confirmed by decoding
        // 125 real Wave Source keyframes, every one of them 2048 floats.
        static constexpr int kFrameSamples = 2048;

        // Where a wavetable position lives. Measured 0 to 256.
        static constexpr float kWaveFramePositions = 256.0f;

        // The version written into the preset. Vital only ever *adds* settings
        // keys -- comparing 0.6, 0.8, 0.9 and 1.0 presets on one machine, no key
        // present in an older preset is absent from a newer one -- so a preset
        // written to the 1.0 vocabulary loads in later versions with the newer
        // parameters left at their defaults.
        static constexpr const char* kSynthVersion = "1.0.7";
    };

    // The preset as JSON text, ready to write to a `.vital` file.
    static juce::String toJson (const Patch& patch, const juce::String& presetName = {});

    static bool writeTo (const Patch& patch, const juce::File& file, juce::String* errorOut = nullptr);
};

} // namespace autosynth
