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

        // Trim applied to the master on the way out, in decibels.
        //
        // Vital's oscillators arrive at the volume control at twice their
        // stated amplitude. One oscillator at full level renders a peak of
        // 1.998 with the control at 0 dB, and a pair at half level each renders
        // 1.998 as well, so the factor is on the sum rather than on each of
        // them. Asking for the patch's gain literally renders it 6 dB hot.
        //
        // That is not merely loud, because Vital limits its own output at about
        // 2.1: the same patch at master 0.5 and 0.8 renders peaks of 1.998 and
        // 2.100 -- four decibels asked for, a twentieth of a decibel delivered
        // -- and the difference comes back as odd harmonics on what was a sine.
        // The export was 12 dB over that ceiling, so every loud preset was being
        // saturated by the synth, and nothing here could see it while the
        // measurements were taken against an engine of our own.
        static constexpr float kMasterTrimDb = -6.0f;

        // Our reverb `level` is a return gain; Vital's `reverb_dry_wet` is a
        // crossfade. This is the measured factor between them, and it is well
        // above one because the return is applied *after* a comb bank with a
        // considerable gain of its own, so a return of 0.1 is nothing like a
        // mix of 0.1.
        //
        // Measured against the recordings rather than against a fit. One
        // tenth of a second after the release, the tail of both source samples
        // sits at about 0.21 of the sustained level -- 0.218 for the clarinet,
        // 0.211 for the violin, which is closer agreement than expected from
        // two unrelated instruments. Vital's tail rises very nearly linearly
        // with the crossfade, so the crossfade that lands on 0.21 is what this
        // multiplies our return gain to reach.
        static constexpr float kReverbReturnToRatio = 8.0f;

        // What the master owes the dry signal at a given return ratio.
        //
        // Vital's reverb is a crossfade, so raising it takes the dry away where
        // ours leaves the dry alone and adds to it. Uncompensated that makes
        // the reverb a second volume control, and an optimiser allowed to
        // search both will reach for it: a clarinet whose return analysis
        // measured at 0.028 came back from refinement at 0.304, because
        // drowning the note was the cheapest way to bring its level down.
        //
        // Measured rather than derived. A pure crossfade would cost 1/(1-wet);
        // the dry falls at half that in decibels, because the wet feeds energy
        // back into the note while it is still sounding. Rendering one patch at
        // seven return gains gives 0.28, 0.97, 2.20, 4.15, 5.39 and 6.67 dB
        // against a square root's 0.64, 1.46, 2.55, 4.15, 5.31 and 6.99 --
        // within half a decibel everywhere, and exact in the middle where the
        // fits actually sit.
        static float dryLossForRatio (float ratio) noexcept;

        // Vital's reverb rings about twice as long as the decay time it is
        // given, so the time it is given is half the one that was fitted.
        //
        // Measured by exporting the same patch at room sizes of 0.0, 0.3 and
        // 0.6 -- asking for 0.61 s, 0.89 s and 1.52 s -- and measuring the
        // rendered decay at 1.19 s, 1.75 s and 3.42 s. That is 1.96x, 1.97x and
        // 2.24x, stable enough across a factor of three to be a convention
        // rather than a coincidence. A fourth point at 0.9 was discarded: the
        // tail had not fallen far enough inside the render to measure.
        static constexpr float kReverbDecayCorrection = 0.5f;

        // Oscillator level is *quadratic*: the amplitude is the square of the
        // stored value, and its default of 0.70710678 is exactly the value that
        // renders as a half. Written linearly, a level of 0.3 arrived as 0.09.
        static float levelToOscLevel (float linearLevel) noexcept;

        // The stored level and modulation amount that swing the amplitude by
        // +/- `depth` around `level`.
        //
        // Needed because Vital's level control is *quadratic* and its
        // modulation *adds* to the stored value. Writing the depth in
        // directly, as the first version did, gets both wrong at once: the
        // swing lands on the square of the parameter rather than on the
        // amplitude, and the top of it clips against the control's ceiling of
        // one. A fitted tremolo of 0.30 on a level of 0.95 lost its whole upper
        // half that way and arrived about a decibel shallow.
        //
        // Solving it is arithmetic rather than taste. For an amplitude ratio of
        // (1+d)/(1-d) between the peak and the trough, the stored values need a
        // ratio of r = sqrt((1+d)/(1-d)), which fixes the amount as a fraction
        // k = (r-1)/(r+1) of the level; the level then follows from wanting the
        // *mean* amplitude to stay where it was. At a depth of zero this
        // returns exactly `levelToOscLevel` and nothing.
        struct LevelSwing { float level = 0.0f; float amount = 0.0f; };
        static LevelSwing levelModulation (float level, float depth) noexcept;

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
