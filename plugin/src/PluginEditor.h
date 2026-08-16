#pragma once

#include "ControlGroup.h"
#include "LfoStrip.h"
#include "OscillatorStrip.h"
#include "ParamKnob.h"
#include "PluginProcessor.h"
#include "SpectrumView.h"

// MidiKeyboardComponent lives here, not in juce_audio_processors. Without it
// the member declaration below is a parse error that MSVC reports as six
// unrelated failures elsewhere in the file.
#include <juce_audio_utils/juce_audio_utils.h>

namespace autosynth
{

// Drop a sample, see what the fitter made of it, edit it, hear it.
//
// Three oscillator strips are always present, dimmed when off. Everything in
// the IR is host-automatable; what gets a knob on screen is the subset worth
// reaching for while listening.
class AutoSynthEditor : public juce::AudioProcessorEditor,
                        public juce::FileDragAndDropTarget
{
public:
    explicit AutoSynthEditor (AutoSynthProcessor&);
    ~AutoSynthEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

    bool isInterestedInFileDrag (const juce::StringArray& files) override;
    void fileDragEnter (const juce::StringArray&, int, int) override;
    void fileDragExit (const juce::StringArray&) override;
    void filesDropped (const juce::StringArray& files, int x, int y) override;

private:
    void refresh();
    void chooseFile();
    void exportVital();
    void handleFile (const juce::File& file);
    static bool isSupported (const juce::String& path);

    AutoSynthProcessor& processor;
    juce::TextButton loadButton { "Load sample or patch..." };
    juce::TextButton exportButton { "Export Vital" };
    juce::ToggleButton refineToggle { "Refine (slower, closer)" };
    juce::Label titleLabel;
    juce::Label statusLabel;

    juce::TextButton playSourceButton { "play original" };
    juce::TextButton playFitButton { "play fit" };
    juce::TextButton stopButton { "stop" };

    // The parameter area scrolls; the transport above it and the keyboard below
    // it stay pinned. There are more knobs here than fit on a laptop screen,
    // and the two things you reach for while auditioning -- loading a file and
    // playing a note -- are exactly the two that must never scroll away.
    //
    // `content` is laid out at its natural full height and the viewport shows a
    // window onto it, so nothing is ever clipped, only out of view.
    juce::Viewport viewport;
    juce::Component content;

    SpectrumView spectrum;
    std::vector<std::unique_ptr<OscillatorStrip>> strips;
    std::vector<std::unique_ptr<LfoStrip>> lfoStrips;

    // Height `content` needs for everything it holds, computed by laying it
    // out. Kept so `resized` can size the viewed component before the viewport
    // decides whether a scrollbar is required.
    int layoutContent (int width, bool applyBounds);

    juce::MidiKeyboardComponent keyboard;
    std::unique_ptr<juce::FileChooser> chooser;
    bool dragHighlight = false;

    // The global controls, in labelled groups. Everything in the IR stays
    // automatable from the host; what gets a knob here is the subset worth
    // reaching for while listening, arranged so it is obvious what belongs to
    // what.
    std::vector<std::unique_ptr<ControlGroup>> groups;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AutoSynthEditor)
};

} // namespace autosynth
