#pragma once

#include <array>
#include <juce_core/juce_core.h>

namespace autosynth
{

// Mirrors autosynth/ir.py. The Python side is the source of truth for the
// representation; this is a reader for it, so field names and defaults must
// match exactly or round-tripping a patch silently changes the sound.
constexpr int kNumOsc = 3;

// Two LFO slots, each with its own destination -- a two-slot modulation matrix.
// One was a real limitation: vibrato and tremolo competed for the single slot
// and only the more periodic survived, so a sound with both could never be
// represented.
constexpr int kNumLfo = 2;

enum class Waveform { sine, triangle, saw, square, pulse };
enum class FilterType { off, lowpass, highpass, bandpass };
enum class LfoShape { sine, triangle, saw, square };
enum class LfoDest { none, pitch, amp, cutoff };

struct Adsr
{
    float attack = 0.01f;
    float decay = 0.20f;
    float sustain = 0.70f;
    float release = 0.30f;
    // 0 is linear, larger bends toward exponential. Not cosmetic: a linear-only
    // envelope mis-states a plucked decay by 5-7 dB and then hits digital
    // silence while the real tail is still audible.
    float curve = 0.0f;
};

struct Filter
{
    FilterType type = FilterType::lowpass;
    float cutoffHz = 8000.0f;
    float resonance = 0.707f;
    float envAmount = 0.0f;
    Adsr env { 0.005f, 0.30f, 0.30f, 0.20f, 0.0f };
};

struct Oscillator
{
    bool enabled = true;
    Waveform waveform = Waveform::saw;
    int semitones = 0;
    float cents = 0.0f;
    float level = 0.7f;
    float pulseWidth = 0.5f;
    int unisonVoices = 1;
    float unisonDetune = 0.0f;

    // Continuous blend from `waveform` toward `waveformB`. Five fixed shapes
    // are the real ceiling on this synth, not the oscillator count; a blend
    // turns the waveform into a continuum, and unlike a discrete choice it is
    // something CMA-ES can actually search.
    Waveform waveformB = Waveform::saw;
    float waveMorph = 0.0f;

    // How much of this oscillator feeds the shared reverb. Per-oscillator so
    // one layer can be drenched while another stays dry, which is the whole
    // reason to want reverb "on an oscillator".
    float reverbSend = 0.0f;
    bool envEnabled = false;
    Adsr env { 0.005f, 0.3f, 1.0f, 0.1f, 0.0f };

    // Per-oscillator filter, applied before this oscillator is summed into the
    // mix. Filtering the sum instead would make every oscillator share a tone
    // again, which defeats the point of having more than one.
    bool filterEnabled = false;
    Filter filter {};
};

struct Lfo
{
    LfoShape shape = LfoShape::sine;
    LfoDest dest = LfoDest::none;
    float rateHz = 5.0f;
    float depth = 0.0f;
    float delay = 0.0f;
    float phase = 0.0f;
};

// Named DelayParams rather than Delay so the struct and the `Delay` DSP class
// can coexist without qualification at every use site.
struct DelayParams
{
    bool enabled = false;
    float time = 0.25f;      // seconds
    float feedback = 0.35f;
    float mix = 0.0f;
};

// One shared reverb, fed by per-oscillator sends.
//
// So an oscillator controls *how much* reverb it gets, but not what kind: the
// size, damping and return are common to all three. Three instances would put
// each oscillator in a different room, which is unusual for one instrument --
// a real source sits in one space, and layers of a patch normally read as one
// sound rather than three.
//
// The cost argument is weaker than it looks and is not the reason: the reverb
// lives on the Engine, not the Voice, so three of them would be three
// instances in total rather than three per voice. Polyphony does not multiply
// it.
struct ReverbParams
{
    bool enabled = false;
    float size = 0.5f;
    float damp = 0.5f;
    float level = 0.0f;  // return gain, not a dry/wet balance
};

struct Patch
{
    std::array<Oscillator, kNumOsc> oscs {};
    Adsr ampEnv {};
    Filter filter {};
    std::array<Lfo, kNumLfo> lfos {};
    DelayParams delay {};
    ReverbParams reverb {};
    float noiseLevel = 0.0f;
    float masterLevel = 0.8f;
    // Playback metadata, not a synthesis parameter: the pitch the patch was
    // fitted at. Never optimised, never exposed as a knob.
    float rootHz = 220.0f;
    juce::String name { "untitled" };

    int activeOscCount() const;

    static Patch fromJson (const juce::var& json);

    // Round-trips back to the same format ir.py writes. Needed because the
    // plugin persists its state as the patch JSON, and an analysed patch has no
    // source file to fall back on -- without this a session would reopen empty.
    juce::String toJson() const;
    static Patch fromJsonString (const juce::String& text, juce::String* error = nullptr);
    static Patch fromFile (const juce::File& file, juce::String* error = nullptr);
};

// Depth 1.0 maps to these ranges, matching engine/synth.py.
constexpr float kLfoPitchSemitones = 2.0f;
constexpr float kLfoCutoffOctaves = 2.0f;
constexpr int kMaxUnison = 7;

} // namespace autosynth
