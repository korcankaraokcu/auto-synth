#pragma once

#include "Parameters.h"
#include "dsp/Voice.h"
#include <juce_audio_processors/juce_audio_processors.h>

namespace autosynth
{

class AutoSynthProcessor : public juce::AudioProcessor,
                           private juce::AudioProcessorValueTreeState::Listener
{
public:
    AutoSynthProcessor();
    ~AutoSynthProcessor() override = default;

    void prepareToPlay (double sampleRate, int maxBlockSize) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override;

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return "default"; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    // Patch loading. `loadPatchFromJson` is what the editor's drag-and-drop and
    // file chooser both go through.
    bool loadPatchFromJson (const juce::String& json, juce::String& errorOut);
    bool loadPatchFromFile (const juce::File& file, juce::String& errorOut);

    // Analyse an audio file and load the resulting patch. Runs on a background
    // thread: fitting takes seconds and refinement takes seconds more, and
    // neither may block the message thread, let alone the audio thread.
    void analyseFileAsync (const juce::File& file, bool refine);
    bool isAnalysing() const noexcept { return analysing.load(); }
    juce::String getStatus() const;
    juce::String getLoadedPatchName() const;
    int getActiveOscCount() const;

    juce::AudioProcessorValueTreeState& getState() noexcept { return state; }

    // The on-screen keyboard's state lives here, not in the editor.
    //
    // A MidiKeyboardComponent only writes note events into its
    // MidiKeyboardState; something has to pump that state into the processor's
    // MIDI buffer or the notes go nowhere. With the state owned by the editor
    // there was no way to do that, so the keyboard drew and clicked and made no
    // sound -- and closing the editor would have discarded held notes.
    juce::MidiKeyboardState& getKeyboardState() noexcept { return keyboardState; }

    // -- monitoring ------------------------------------------------------
    // Solo/mute are listening controls, not patch parameters, so they live
    // here rather than in the IR and are never serialised.
    void setSolo (int osc, bool shouldSolo);
    void setMute (int osc, bool shouldMute);
    bool isSolo (int osc) const noexcept;
    bool isMute (int osc) const noexcept;

    // -- A/B -------------------------------------------------------------
    // Plays the stored source or a render of the current patch straight out,
    // so "is this a good fit" stops being a guess.
    enum class Playing { none, source, rebuilt };
    void startPlayback (Playing what);
    void stopPlayback();
    Playing getPlayback() const noexcept { return playing.load(); }
    bool hasSource() const noexcept { return sourceLength.load() > 0; }

    // Re-renders the patch and refreshes the comparison spectra. Called after
    // analysis and after a patch load; safe to call from the message thread.
    void refreshComparison();

    // -- spectrum overlay -------------------------------------------------
    struct Spectra
    {
        std::vector<float> frequencies; // Hz, log-spaced
        std::vector<float> sourceDb;
        std::vector<float> fitDb;
        bool valid = false;
    };
    Spectra getSpectra() const;

    // A copy of what the engine is playing. The editor needs the wavetable
    // frames to draw them, and they are the one part of a patch that is not a
    // parameter -- there is nothing in the value tree to read them from.
    Patch getPatchSnapshot() const;

    // Edits one harmonic of one frame, converting that frame from generated to
    // drawn on the first touch. Not a parameter, so it does not go through the
    // value tree: sixteen amplitudes times sixteen frames times three
    // oscillators is not something to hand a host, and a table rebuild is an
    // FFT that must not land on the audio thread. This runs on the message
    // thread and hands the finished tables to the engine under the patch lock.
    void setFrameHarmonic (int oscIndex, int frameIndex, int harmonic, float amount);

    std::function<void()> onPatchChanged;

private:
    void applyPatch (const Patch& patch);
    void pushPatchToParameters (const Patch& patch);
    void pullPatchFromParameters();
    void parameterChanged (const juce::String& id, float value) override;

    void applyMonitorMask();
    void renderComparison();

    juce::MidiKeyboardState keyboardState;

    // Source audio is kept after analysis so it can be replayed for A/B and
    // re-spectrum'd when the patch changes. Stored at its own sample rate and
    // read with a ratio, rather than resampled -- monitoring does not need
    // better than linear interpolation.
    juce::AudioBuffer<float> sourceAudio;
    double sourceRate = 48000.0;
    std::atomic<int> sourceLength { 0 };

    juce::AudioBuffer<float> rebuiltAudio;

    std::atomic<Playing> playing { Playing::none };
    std::atomic<int> playPosition { 0 };
    juce::CriticalSection playbackLock;

    std::array<bool, kNumOsc> solo {};
    std::array<bool, kNumOsc> mute {};

    mutable juce::CriticalSection spectraLock;
    Spectra spectra;
    juce::AudioProcessorValueTreeState state;
    // Guards the two-way sync. Writing a patch into the parameters fires the
    // listener, which would otherwise immediately read them back out and
    // rebuild the patch -- harmless in effect but wasteful, and it would
    // clobber non-parameter metadata on every load.
    std::atomic<bool> syncing { false };

    Engine engine;
    mutable juce::CriticalSection patchLock;
    std::atomic<bool> analysing { false };
    juce::CriticalSection statusLock;
    juce::String status { "no patch loaded" };
    std::unique_ptr<juce::ThreadPool> pool;
    double currentSampleRate = 48000.0;
    juce::String loadedJson;
    juce::String patchName { "(none)" };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AutoSynthProcessor)
};

} // namespace autosynth
