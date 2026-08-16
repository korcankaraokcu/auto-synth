#pragma once

#include "dsp/Delay.h"
#include "dsp/Reverb.h"
#include "dsp/Envelope.h"
#include "dsp/Svf.h"
#include "dsp/Tables.h"
#include "ir/Patch.h"

#include <array>
#include <juce_audio_basics/juce_audio_basics.h>

namespace autosynth
{

// One polyphonic voice: the signal path of engine/synth.py for a single note.
//
// Signal order is identical to the reference engine and that ordering matters:
// oscillators (with per-oscillator envelopes) -> noise -> amp envelope and amp
// LFO -> filter -> master. Moving the filter before the amp envelope, which
// looks equivalent, is not: the filter is nonlinear in its state and would ring
// differently against a different input level.
class Voice
{
public:
    void prepare (double sampleRate, const WaveTables* sharedTables,
                  const std::array<WaveTables::FrameTables, kNumOsc>* oscFrameTables);
    void setPatch (const Patch& patch);

    // Monitoring gain per oscillator, from the editor's solo/mute buttons.
    // Deliberately *not* part of the patch: soloing is a way of listening, not
    // a property of the sound, and a saved patch should not remember that you
    // muted oscillator 2 while working on it.
    void setMonitorMask (const std::array<float, kNumOsc>& mask) noexcept { monitor = mask; }

    void noteOn (int midiNote, float velocity);
    void noteOff();
    void reset();

    bool isActive() const noexcept { return active; }
    int getMidiNote() const noexcept { return midiNote; }

    // Adds this voice's dry output into `out` and its reverb-send contribution
    // into `send`. Additive so the engine can sum voices without a scratch
    // buffer each, and separate so the shared reverb sees only what the sends
    // actually routed to it.
    void render (float* out, float* send, int numSamples);

private:
    struct OscState
    {
        std::array<double, kMaxUnison> phase {};
        std::array<double, kMaxUnison> detuneRatio {};
        Envelope env;
        Envelope filterEnv;
        Envelope frameEnv;
        Svf filter;
        // Its own generator, seeded per oscillator, so two noise oscillators in
        // one patch are two different sounds rather than one at double level.
        juce::Random noise;
        int numVoices = 1;
        float gain = 0.0f;
        bool useEnv = false;
        bool useFilter = false;
    };

    float baseFrequency() const noexcept;

    Patch patch {};
    const WaveTables* tables = nullptr;
    // Owned by the Engine, one per oscillator slot: unlike the fixed tables
    // these depend on the patch, so they cannot be shared across instruments.
    const std::array<WaveTables::FrameTables, kNumOsc>* frameTables = nullptr;
    double sampleRate = 44100.0;

    std::array<OscState, kNumOsc> oscs {};
    std::array<float, kNumOsc> monitor { 1.0f, 1.0f, 1.0f };
    Envelope ampEnv;
    Envelope filterEnv;
    std::array<LfoOsc, kNumLfo> lfos;
    Svf filter;
    // The send path needs its own filter instance: a filter carries state, so
    // running two signals through one would cross-contaminate them.
    Svf sendFilter;
    juce::Random noise;

    int midiNote = 60;
    float velocity = 1.0f;
    bool active = false;
};

// Voice allocator. Patch changes are applied to idle voices immediately and to
// sounding voices on their next note-on, so editing a parameter never restarts
// a held note.
class Engine
{
public:
    static constexpr int kMaxVoices = 16;

    void prepare (double sampleRate, int maxBlockSize);
    void setPatch (const Patch& patch);
    const Patch& getPatch() const noexcept { return patch; }
    void setMonitorMask (const std::array<float, kNumOsc>& mask);

    void noteOn (int midiNote, float velocity);
    void noteOff (int midiNote);
    void allNotesOff();

    void render (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi);

    using FrameTableSet = std::array<WaveTables::FrameTables, kNumOsc>;

    // Builds the drawn-frame mipmaps for `p` without touching this engine.
    //
    // Exists so the caller can do the FFTs *before* taking whatever lock keeps
    // the audio thread out, and then hand them over as a pointer swap. Built in
    // place it measured 1.5 ms with three drawn frames, and the audio callback
    // try-locks and outputs silence when it loses -- which turned dragging a
    // harmonic bar into several dropped blocks a second.
    static void buildFrameTables (const Patch& p, double sampleRate, FrameTableSet& out);

    // Takes ownership of tables built by the call above. Cheap: a vector move
    // per oscillator. Call it before `setPatch`, which will then find them
    // already correct and do nothing.
    void adoptFrameTables (FrameTableSet&& built);

    // Offline render of a single note, used by the headless renderer and the
    // conformance test. Mirrors `autosynth render`.
    void renderOffline (juce::AudioBuffer<float>& out, double noteHz,
                        double durationSeconds, double gateSeconds);

private:
    void renderBlock (float* out, int numSamples);

    void rebuildFrameTables();

    std::array<Voice, kMaxVoices> voices {};
    WaveTables tables;
    // Rebuilt only when the frame data actually differs, which happens on patch
    // load and analysis -- both off the audio thread. Automation changes
    // scalars, never frames, so `setPatch` from a parameter callback costs a
    // comparison and no allocation.
    FrameTableSet frameTables {};
    // One delay for the whole instrument, not one per voice: a delay repeats
    // what came out of the mix, and per-voice copies would multiply the tail by
    // the polyphony.
    Delay delay;
    Reverb reverb;
    std::vector<float> sendBuffer;
    Patch patch {};
    std::array<float, kNumOsc> monitor { 1.0f, 1.0f, 1.0f };
    double sampleRate = 44100.0;
};

} // namespace autosynth
