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
    void prepare (double sampleRate, const WaveTables* sharedTables);
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
        Svf filter;
        int numVoices = 1;
        float gain = 0.0f;
        bool useEnv = false;
        bool useFilter = false;
    };

    float baseFrequency() const noexcept;

    Patch patch {};
    const WaveTables* tables = nullptr;
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

    // Offline render of a single note, used by the headless renderer and the
    // conformance test. Mirrors `autosynth render`.
    void renderOffline (juce::AudioBuffer<float>& out, double noteHz,
                        double durationSeconds, double gateSeconds);

private:
    void renderBlock (float* out, int numSamples);

    std::array<Voice, kMaxVoices> voices {};
    WaveTables tables;
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
