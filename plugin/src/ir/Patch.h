#pragma once

#include <array>
#include <functional>
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
//
// A third was tried, to give a *modulator* -- an LFO pointed at another LFO's
// rate -- a slot of its own. It was reverted, and the reason is worth keeping:
// the recovery harness randomises every continuous parameter, so a third slot
// changes how many random draws a target consumes and therefore *which targets
// get generated*. Its control distance moved from 2.28 to 2.46 on patches that
// were supposed to be unchanged, which silently invalidates every number this
// project has recorded. A modulator now competes for one of the two slots on
// merit, which is the same rule everything else here follows.
constexpr int kNumLfo = 2;

// `noise` is a generator like the rest, not a special case bolted on beside
// them. Selecting it gives the oscillator's noise everything an oscillator
// already has -- its own level, envelope, filter and reverb send -- which is
// exactly what the global noise bed lacks and what two failed attempts to bolt
// an envelope onto that bed were reaching for.
enum class Waveform { sine, triangle, saw, square, pulse, noise };
enum class FilterType { off, lowpass, highpass, bandpass };
enum class LfoShape { sine, triangle, saw, square };
// `lfoRate` and `lfoDepth` point at the *other* LFO slot, which is unambiguous
// because there are two. That is how Vital's matrix does it, and it is the
// cheapest way to build wander out of periodic parts: a slow LFO on a faster
// one's rate spreads its spectrum instead of leaving a single spike.
//
// Which matters because a single spike is what a listener hears as mechanical.
// Measured on a clarinet, the recording's amplitude spectrum is spread from 0.4
// to 3.6 Hz with no dominant peak, while the fit put everything at 3.2 -- and it
// was described as having more vibrato than the original despite measuring
// *shallower*. Regularity, not depth, is what reads as synthetic.
enum class LfoDest { none, pitch, amp, cutoff, lfoRate, lfoDepth };

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

    // The attack gets its own, and defaults to linear so nothing already
    // written changes.
    //
    // Sharing `curve` was tried and reverted. A linear attack really is too
    // quiet early -- a fit whose envelope reached nine tenths at exactly the
    // right moment was still 16 dB below its target at note-on, and the note
    // seemed to emerge from silence rather than start -- but one number cannot
    // shape an attack, a decay and a release at once. Asked to, the fit lands
    // on a compromise that suits none of them: a clarinet's came back at 5.5
    // where it had been 2.5, and a listener judged both samples worse.
    float attackCurve = 0.0f;
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
    // are a coarse net for a real instrument, and a blend turns them into a
    // continuum; unlike a discrete choice it is also something CMA-ES can
    // search.
    //
    // Together with `pulseWidth` above, these four fields are the *generator*
    // for every frame of the wavetable below that nobody has drawn on -- which,
    // for an ordinary analog patch, is the only frame there is.
    Waveform waveformB = Waveform::saw;
    float waveMorph = 0.0f;

    // --- wavetable ---------------------------------------------------------
    //
    // Every oscillator is a wavetable. There is no switch, because there is
    // nothing to switch between: a saw *is* a one-frame table whose frame has
    // not been drawn on, exactly as it is in Vital, and the fields above are
    // that frame's generator.
    //
    // A frame stays generated until someone edits it. That is what keeps the
    // analog shapes analog: a generated frame is built from its waveform's full
    // Fourier series and band-limited per octave, so a saw at 220 Hz keeps all
    // seventy-five harmonics that fit under Nyquist. Storing it as sixteen
    // numbers instead would cut it at the sixteenth -- measured, that is 1.7
    // octaves of brightness gone, and every classic preset would sound muffled
    // for no gain.
    //
    // Editing one converts it, and from then on it is sixteen harmonic
    // amplitudes. Sixteen rather than a full spectrum because the deliverable
    // is an *editable* patch: a 256-point table is a wavetable dump, which is
    // the thing this project exists not to produce, and sixteen numbers is a
    // curve a person can read and drag. It also maps straight onto a Vital
    // wavetable for export.
    struct Frame
    {
        bool custom = false;
        std::array<float, 16> harmonics {};
    };

    static constexpr int kFrameHarmonics = 16;
    static constexpr int kMaxFrames = 16;

    // How many of `frames` are in use. One is an ordinary analog oscillator;
    // more is a table the note travels through. The fitter spends the extra
    // frames only when the tone measurably moves -- see WavetableFit.
    int numFrames = 1;
    std::array<Frame, kMaxFrames> frames {};

    // Where between the frames the oscillator sits: 0 is the first frame, 1 the
    // last. `framePositionEnvAmount` is how far the envelope drags it over the
    // course of the note.
    float framePosition = 0.0f;
    float framePositionEnvAmount = 0.0f;
    Adsr framePositionEnv { 0.05f, 0.4f, 0.6f, 0.2f, 0.0f };

    bool envEnabled = false;
    Adsr env { 0.005f, 0.3f, 1.0f, 0.1f, 0.0f };
};

struct Lfo
{
    LfoShape shape = LfoShape::sine;
    LfoDest dest = LfoDest::none;
    float rateHz = 5.0f;
    float depth = 0.0f;
    float delay = 0.0f;
    float phase = 0.0f;

    // How far this LFO's own rate drifts, in octaves, and how fast it drifts.
    //
    // A property of the LFO rather than a second LFO pointed at it, which is
    // how this was expressed before and why it almost never survived: with two
    // slots, spending one on a modulator meant losing a violin's tremolo to
    // smear its vibrato, and the fitter rightly declined that trade nearly
    // every time. Vital's random LFOs are separate from its eight ordinary
    // ones, so there the wander is free -- expressing it as a field rather than
    // a slot is what lets that be true here too.
    //
    // Measured, not searched. `Modulation::detectWander` reports both numbers
    // and refinement leaves them alone, on the same rule as every other
    // measurement.
    float rateWander = 0.0f;
    float wanderRateHz = 0.0f;
};

struct DelayParams
{
    bool enabled = false;
    float time = 0.25f;      // seconds
    float feedback = 0.35f;
    float mix = 0.0f;
};

// One reverb for the patch.
//
// Three would put each oscillator in a different room, which is unusual for one
// instrument -- a real source sits in one space, and layers of a patch normally
// read as one sound rather than three. Vital has one too, which settles it.
//
// `level` is a return gain rather than a dry/wet balance. Vital's control is a
// crossfade, so `VitalExport` converts between them by a measured ratio.
struct ReverbParams
{
    bool enabled = false;
    float size = 0.5f;
    float damp = 0.5f;
    float level = 0.0f;  // return gain, not a dry/wet balance
};

// How a patch becomes audio.
//
// There is no synth in this repository any more: the deliverable is a Vital
// preset, so the thing that says what a patch sounds like is Vital, and the
// caller that owns a plug-in host owns the renderer. Analysis, refinement and
// the recovery harness all take one of these rather than each reaching for an
// engine of their own -- which is also what stops any of them optimising
// against a sound the preset will not make.
//
// Returns mono samples at the caller's own rate.
using Renderer = std::function<std::vector<float> (const struct Patch& patch,
                                                   double durationSeconds,
                                                   double gateSeconds)>;

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
