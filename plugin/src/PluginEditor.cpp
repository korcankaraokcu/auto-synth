#include "PluginEditor.h"

namespace autosynth
{

AutoSynthEditor::AutoSynthEditor (AutoSynthProcessor& p)
    : AudioProcessorEditor (&p), processor (p),
      keyboard (p.getKeyboardState(), juce::MidiKeyboardComponent::horizontalKeyboard)
{
    titleLabel.setText ("auto-synth", juce::dontSendNotification);
    titleLabel.setFont (juce::FontOptions (22.0f, juce::Font::bold));
    addAndMakeVisible (titleLabel);

    statusLabel.setJustificationType (juce::Justification::centredLeft);
    statusLabel.setColour (juce::Label::textColourId, juce::Colours::lightgrey);
    statusLabel.setFont (juce::FontOptions (12.0f));
    addAndMakeVisible (statusLabel);

    loadButton.onClick = [this] { chooseFile(); };
    addAndMakeVisible (loadButton);

    refineToggle.setToggleState (true, juce::dontSendNotification);
    refineToggle.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible (refineToggle);

    playSourceButton.onClick = [this]
    {
        processor.startPlayback (AutoSynthProcessor::Playing::source);
    };
    playFitButton.onClick = [this]
    {
        processor.startPlayback (AutoSynthProcessor::Playing::rebuilt);
    };
    stopButton.onClick = [this] { processor.stopPlayback(); };
    addAndMakeVisible (playSourceButton);
    addAndMakeVisible (playFitButton);
    addAndMakeVisible (stopButton);

    viewport.setViewedComponent (&content, false);
    viewport.setScrollBarsShown (true, false);
    viewport.setScrollBarThickness (10);
    addAndMakeVisible (viewport);

    content.addAndMakeVisible (spectrum);

    for (int i = 0; i < kNumOsc; ++i)
    {
        auto strip = std::make_unique<OscillatorStrip> (processor, i);
        content.addAndMakeVisible (*strip);
        strips.push_back (std::move (strip));
    }

    for (int i = 0; i < kNumLfo; ++i)
    {
        auto strip = std::make_unique<LfoStrip> (processor, i);
        content.addAndMakeVisible (*strip);
        lfoStrips.push_back (std::move (strip));
    }

    // Each effect's on/off lives in the same box as the knobs it arms. The
    // reverb and the delay previously had no switch anywhere in the interface,
    // so their knobs were inert no matter what you did with them.
    const auto addGroup = [this] (juce::String title, juce::String toggleId,
                                  juce::String comboId,
                                  std::vector<ControlGroup::KnobDef> defs)
    {
        auto group = std::make_unique<ControlGroup> (processor, std::move (title),
                                                     std::move (toggleId),
                                                     std::move (comboId), std::move (defs));
        content.addAndMakeVisible (*group);
        groups.push_back (std::move (group));
    };

    addGroup ("Filter", {}, params::idFor ("filter.type"),
              { { params::idFor ("filter.cutoff_hz"), "cutoff" },
                { params::idFor ("filter.resonance"), "res" },
                { params::idFor ("filter.env_amount"), "env amt" } });

    addGroup ("Amp envelope", {}, {},
              { { params::idFor ("amp_env.attack"), "A" },
                { params::idFor ("amp_env.decay"), "D" },
                { params::idFor ("amp_env.sustain"), "S" },
                { params::idFor ("amp_env.release"), "R" } });

    addGroup ("Delay", params::idFor ("delay.enabled"), {},
              { { params::idFor ("delay.time"), "time" },
                { params::idFor ("delay.feedback"), "feedback" },
                { params::idFor ("delay.mix"), "mix" } });

    addGroup ("Reverb", params::idFor ("reverb.enabled"), {},
              { { params::idFor ("reverb.size"), "size" },
                { params::idFor ("reverb.damp"), "damp" },
                { params::idFor ("reverb.level"), "return" } });

    addGroup ("Output", {}, {},
              { { params::idFor ("noise_level"), "noise" },
                { params::idFor ("master_level"), "master" } });

    addAndMakeVisible (keyboard);
    keyboard.setAvailableRange (36, 84);

    // The SafePointer is built here and captured by value, rather than
    // constructed from `this` inside the nested lambda: in an init-capture of a
    // lambda nested inside another lambda, MSVC resolves unqualified `this` to
    // the enclosing closure rather than to the editor.
    //
    // It has to be a SafePointer at all because the patch can change from the
    // audio or message thread and the callback may still be queued after the
    // editor closes -- the destructor clears `onPatchChanged`, but a callAsync
    // already in flight cannot be recalled.
    const juce::Component::SafePointer<AutoSynthEditor> safe (this);
    processor.onPatchChanged = [safe]
    {
        juce::MessageManager::callAsync ([safe]
        {
            if (safe != nullptr)
                safe->refresh();
        });
    };

    // The default has to fit on a laptop, so it is shorter than the panel it
    // contains and opens already scrolled to the top. It was previously 1330
    // tall, which is taller than a 1080p desktop once the taskbar is taken out
    // -- the bottom rows were simply unreachable, including the global reverb
    // controls. Growing the window still reveals more at once.
    setResizable (true, true);
    setResizeLimits (900, 480, 2000, 1800);
    setSize (1100, 880);
    refresh();
}

AutoSynthEditor::~AutoSynthEditor()
{
    processor.onPatchChanged = nullptr;
}

void AutoSynthEditor::refresh()
{
    statusLabel.setText (processor.getLoadedPatchName()
                         + "   \xc2\xb7   " + juce::String (processor.getActiveOscCount()) + " osc"
                         + "   \xc2\xb7   " + processor.getStatus(),
                         juce::dontSendNotification);

    loadButton.setEnabled (! processor.isAnalysing());
    playSourceButton.setEnabled (processor.hasSource());

    spectrum.setSpectra (processor.getSpectra());
    for (auto& strip : strips)
        strip->refresh();
}

void AutoSynthEditor::chooseFile()
{
    chooser = std::make_unique<juce::FileChooser> ("Load a sample or patch", juce::File(),
                                                  "*.wav;*.aiff;*.aif;*.flac;*.json");
    const auto flags = juce::FileBrowserComponent::openMode
                     | juce::FileBrowserComponent::canSelectFiles;

    chooser->launchAsync (flags, [this] (const juce::FileChooser& fc)
    {
        const auto file = fc.getResult();
        if (file == juce::File())
            return;

        handleFile (file);
    });
}

bool AutoSynthEditor::isInterestedInFileDrag (const juce::StringArray& files)
{
    for (const auto& f : files)
        if (isSupported (f))
            return true;
    return false;
}

bool AutoSynthEditor::isSupported (const juce::String& path)
{
    return path.endsWithIgnoreCase (".json") || path.endsWithIgnoreCase (".wav")
        || path.endsWithIgnoreCase (".aiff") || path.endsWithIgnoreCase (".aif")
        || path.endsWithIgnoreCase (".flac");
}

void AutoSynthEditor::handleFile (const juce::File& file)
{
    if (file == juce::File())
        return;

    // A patch loads instantly; a sample has to be analysed, which takes seconds
    // and therefore goes to a background thread.
    if (file.hasFileExtension ("json"))
    {
        juce::String error;
        if (processor.loadPatchFromFile (file, error))
            processor.refreshComparison();
        else
            statusLabel.setText ("failed to load: " + error, juce::dontSendNotification);
        return;
    }

    processor.analyseFileAsync (file, refineToggle.getToggleState());
}

void AutoSynthEditor::fileDragEnter (const juce::StringArray&, int, int)
{
    dragHighlight = true;
    repaint();
}

void AutoSynthEditor::fileDragExit (const juce::StringArray&)
{
    dragHighlight = false;
    repaint();
}

void AutoSynthEditor::filesDropped (const juce::StringArray& files, int, int)
{
    dragHighlight = false;
    repaint();

    for (const auto& f : files)
    {
        if (! isSupported (f))
            continue;
        handleFile (juce::File (f));
        return;
    }
}

void AutoSynthEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff1b1d21));
    if (dragHighlight)
    {
        g.setColour (juce::Colour (0xff4a8fe7));
        g.drawRect (getLocalBounds().reduced (4), 3);
    }
}

int AutoSynthEditor::layoutContent (int width, bool applyBounds)
{
    // Runs twice per resize: once to measure, once to place. Measuring first is
    // what lets the viewport know whether a scrollbar is needed before the
    // width that depends on it is chosen.
    const auto place = [applyBounds] (juce::Component& c, juce::Rectangle<int> bounds)
    {
        if (applyBounds)
            c.setBounds (bounds);
    };

    auto y = 0;
    const auto take = [&y] (int height)
    {
        const auto top = y;
        y += height;
        return top;
    };

    place (spectrum, { 0, take (110), width, 110 });
    take (10);

    for (auto& strip : strips)
    {
        place (*strip, { 0, take (OscillatorStrip::kPreferredHeight), width,
                         OscillatorStrip::kPreferredHeight });
        take (6);
    }

    for (auto& strip : lfoStrips)
    {
        place (*strip, { 0, take (62), width, 62 });
        take (6);
    }

    take (4);

    // Flow the groups left to right, wrapping when the next one will not fit.
    // Groups differ in width because they differ in content, so a fixed grid
    // would either clip the wide ones or leave the narrow ones swimming.
    const auto groupHeight = ControlGroup::preferredHeight();
    auto rowTop = take (groupHeight);
    auto x = 0;
    for (auto& group : groups)
    {
        const auto groupWidth = juce::jmin (width, group->preferredWidth());
        if (x > 0 && x + groupWidth > width)
        {
            x = 0;
            take (6);
            rowTop = take (groupHeight);
        }

        place (*group, { x, rowTop, groupWidth, groupHeight });
        x += groupWidth + 6;
    }

    return y;
}

void AutoSynthEditor::resized()
{
    auto area = getLocalBounds().reduced (14);

    keyboard.setBounds (area.removeFromBottom (70));
    area.removeFromBottom (10);

    auto header = area.removeFromTop (32);
    loadButton.setBounds (header.removeFromRight (170).reduced (0, 2));
    titleLabel.setBounds (header.removeFromLeft (140));
    refineToggle.setBounds (header.reduced (8, 4));

    statusLabel.setBounds (area.removeFromTop (20));
    area.removeFromTop (6);

    auto transport = area.removeFromTop (24);
    playSourceButton.setBounds (transport.removeFromLeft (110).reduced (1));
    playFitButton.setBounds (transport.removeFromLeft (90).reduced (1));
    stopButton.setBounds (transport.removeFromLeft (64).reduced (1));

    area.removeFromTop (8);

    // Everything below the transport scrolls.
    viewport.setBounds (area);

    // Measure at the full width first. If the result does not fit, the vertical
    // scrollbar appears and steals width, so measure again narrower -- otherwise
    // the last column of knobs would sit underneath the scrollbar.
    auto contentWidth = area.getWidth();
    auto contentHeight = layoutContent (contentWidth, false);
    if (contentHeight > area.getHeight())
    {
        contentWidth = juce::jmax (1, contentWidth - viewport.getScrollBarThickness());
        contentHeight = layoutContent (contentWidth, false);
    }

    content.setSize (contentWidth, contentHeight);
    layoutContent (contentWidth, true);
}

} // namespace autosynth
