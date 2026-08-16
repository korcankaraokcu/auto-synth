#pragma once

#include "FrameView.h"
#include "ParamKnob.h"
#include "PluginProcessor.h"

namespace autosynth
{

// One oscillator's controls.
//
// Three of these are always shown, dimmed when the oscillator is off. Hiding
// unused slots was the alternative and would have thrown information away: that
// a patch uses one oscillator is a decision the fitter made, and an empty slot
// says so. It also leaves somewhere to click to add one by hand, which a tool
// claiming to produce *editable* patches needs.
class OscillatorStrip : public juce::Component
{
public:
    // Header, then three captioned rows. Published so the editor sizes the
    // scrolling panel from the strip rather than from a number kept in step
    // with it by hand.
    static constexpr int kCaptionHeight = 11;
    static constexpr int kRowHeight = 70;
    static constexpr int kPreferredHeight = 8 + 18 + 4 * kRowHeight;

    OscillatorStrip (AutoSynthProcessor& processorToUse, int oscIndex)
        : processor (processorToUse), index (oscIndex)
    {
        auto& state = processor.getState();
        const auto prefix = "oscs_" + juce::String (index) + "_";

        title.setText ("OSC " + juce::String (index + 1), juce::dontSendNotification);
        title.setFont (juce::FontOptions (12.0f, juce::Font::bold));
        title.setColour (juce::Label::textColourId, juce::Colours::white);
        addAndMakeVisible (title);

        addToggle (enabled, prefix + "enabled", "on");
        enabledAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
            state, prefix + "enabled", enabled);
        enabled.onStateChange = [this] { updateDimming(); };

        waveform.setJustificationType (juce::Justification::centredLeft);
        if (auto* parameter = state.getParameter (prefix + "waveform"))
            waveform.addItemList (parameter->getAllValueStrings(), 1);
        // A combo that acted on the wheel would silently change the waveform
        // while scrolling past it. Let the event through to the panel instead.
        waveform.setScrollWheelEnabled (false);
        addAndMakeVisible (waveform);
        waveformAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (
            state, prefix + "waveform", waveform);

        addKnob (semitones, semitonesLabel, prefix + "semitones", "semi", semitonesAttachment);
        addKnob (cents, centsLabel, prefix + "cents", "cent", centsAttachment);
        addKnob (level, levelLabel, prefix + "level", "level", levelAttachment);
        addKnob (unison, unisonLabel, prefix + "unison_voices", "unison", unisonAttachment);
        addKnob (detune, detuneLabel, prefix + "unison_detune", "detune", detuneAttachment);

        // Wave B and morph: the waveform becomes a continuum rather than a
        // choice of five, which is the actual ceiling on what this synth can
        // represent.
        waveformB.setJustificationType (juce::Justification::centredLeft);
        if (auto* parameter = state.getParameter (prefix + "waveform_b"))
            waveformB.addItemList (parameter->getAllValueStrings(), 1);
        waveformB.setScrollWheelEnabled (false);
        addAndMakeVisible (waveformB);
        waveformBAttachment = std::make_unique<ComboAttachment> (
            state, prefix + "waveform_b", waveformB);
        addKnob (morph, morphLabel, prefix + "wave_morph", "morph", morphAttachment);
        // Reads with its group caption as "REVERB / send": how much of *this*
        // oscillator is fed to the one shared reverb. Distinct from that
        // reverb's own size/damp/return in the global section, which every
        // oscillator shares.
        addKnob (reverbSend, reverbSendLabel, prefix + "reverb_send", "send",
                 reverbSendAttachment);

        // Solo and mute are not parameters -- they are ways of listening, and a
        // saved patch must not remember them.
        soloButton.setButtonText ("S");
        soloButton.setClickingTogglesState (true);
        soloButton.onClick = [this]
        {
            processor.setSolo (index, soloButton.getToggleState());
        };
        addAndMakeVisible (soloButton);

        muteButton.setButtonText ("M");
        muteButton.setClickingTogglesState (true);
        muteButton.onClick = [this]
        {
            processor.setMute (index, muteButton.getToggleState());
        };
        addAndMakeVisible (muteButton);

        addToggle (ownEnv, prefix + "env_enabled", "own env");
        ownEnvAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (
            state, prefix + "env_enabled", ownEnv);

        addKnob (envA, envALabel, prefix + "env_attack", "A", envAAttachment);
        addKnob (envD, envDLabel, prefix + "env_decay", "D", envDAttachment);
        addKnob (envS, envSLabel, prefix + "env_sustain", "S", envSAttachment);
        addKnob (envR, envRLabel, prefix + "env_release", "R", envRAttachment);

        // Per-oscillator filter. This is the row that lets one oscillator be
        // the dark body of a sound and another the bright attack.
        addToggle (ownFilter, prefix + "filter_enabled", "own filter");
        ownFilterAttachment = std::make_unique<ButtonAttachment> (
            state, prefix + "filter_enabled", ownFilter);

        filterType.setJustificationType (juce::Justification::centredLeft);
        if (auto* parameter = state.getParameter (prefix + "filter_type"))
            filterType.addItemList (parameter->getAllValueStrings(), 1);
        filterType.setScrollWheelEnabled (false);
        addAndMakeVisible (filterType);
        filterTypeAttachment = std::make_unique<ComboAttachment> (
            state, prefix + "filter_type", filterType);

        addKnob (filterCutoff, filterCutoffLabel, prefix + "filter_cutoff_hz", "cutoff",
                 filterCutoffAttachment);
        addKnob (filterRes, filterResLabel, prefix + "filter_resonance", "res",
                 filterResAttachment);
        addKnob (filterEnvAmt, filterEnvAmtLabel, prefix + "filter_env_amount", "env amt",
                 filterEnvAmtAttachment);
        addKnob (filterD, filterDLabel, prefix + "filter_env_decay", "f.D",
                 filterDAttachment);
        addKnob (filterS, filterSLabel, prefix + "filter_env_sustain", "f.S",
                 filterSAttachment);

        // Wavetable. There is no on/off, because there is nothing to switch
        // between: one frame, undrawn, *is* the wave blend above. Adding frames
        // and dragging bars is how it stops being one.
        addAndMakeVisible (frameView);
        frameView.onHarmonicDragged = [this] (int harmonic, float amount)
        {
            processor.setFrameHarmonic (index, editFrameIndex(), harmonic, amount);
        };

        addKnob (frameCount, frameCountLabel, prefix + "num_frames", "frames",
                 frameCountAttachment);
        frameCount.onValueChange = [this] { refresh(); };

        // Which frame the bars edit. Not a parameter: it is a view, like solo
        // and mute, and a saved patch must not remember which frame someone
        // happened to have open.
        editFrame.setSliderStyle (juce::Slider::RotaryVerticalDrag);
        editFrame.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 44, 14);
        editFrame.setRange (1.0, Oscillator::kMaxFrames, 1.0);
        editFrame.setValue (1.0, juce::dontSendNotification);
        editFrame.onValueChange = [this] { refresh(); };
        addAndMakeVisible (editFrame);
        editFrameLabel.setText ("edit", juce::dontSendNotification);
        editFrameLabel.setJustificationType (juce::Justification::centred);
        editFrameLabel.setColour (juce::Label::textColourId, juce::Colours::grey);
        editFrameLabel.setFont (juce::FontOptions (10.0f));
        addAndMakeVisible (editFrameLabel);

        addKnob (framePos, framePosLabel, prefix + "frame_position", "pos",
                 framePosAttachment);
        addKnob (frameEnvAmt, frameEnvAmtLabel, prefix + "frame_position_env_amount", "env amt",
                 frameEnvAmtAttachment);
        addKnob (frameA, frameALabel, prefix + "frame_position_env_attack", "sweep",
                 frameAAttachment);

        refresh();
        updateDimming();
    }

    void refresh()
    {
        const auto patch = processor.getPatchSnapshot();
        const auto& osc = patch.oscs[static_cast<size_t> (index)];
        editFrame.setRange (1.0, juce::jmax (1, osc.numFrames), 1.0);
        frameView.setFrames (osc.frames, osc.numFrames, editFrameIndex(), osc.framePosition,
                             WaveTables::blendedHarmonics (osc.waveform, osc.waveformB,
                                                           osc.waveMorph, osc.pulseWidth,
                                                           Oscillator::kFrameHarmonics));
        updateDimming();
    }

    void paint (juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat().reduced (1.0f);
        g.setColour (juce::Colour (enabled.getToggleState() ? 0xff20242a : 0xff191c20));
        g.fillRoundedRectangle (bounds, 4.0f);
        g.setColour (juce::Colour (0xff2c3138));
        g.drawRoundedRectangle (bounds, 4.0f, 1.0f);

        // Captions and dividers for the groups measured out in `resized`.
        // Without them the row reads as nine unrelated controls, and the two
        // wave selectors in particular look like a duplicate rather than the
        // two ends of a blend.
        for (const auto& group : groups)
        {
            g.setColour (juce::Colour (0xff6d8296));
            g.setFont (juce::FontOptions (9.0f, juce::Font::bold));
            g.drawText (group.caption.toUpperCase(),
                        group.area.withHeight (10).reduced (3, 0),
                        juce::Justification::centredLeft);

            if (group.divider)
            {
                g.setColour (juce::Colour (0xff2c3138));
                g.fillRect (group.area.getX() - 4, group.area.getY() + 2, 1,
                            group.area.getHeight() - 4);
            }
        }
    }

    void resized() override
    {
        groups.clear();
        auto area = getLocalBounds().reduced (6, 4);

        auto header = area.removeFromTop (18);
        title.setBounds (header.removeFromLeft (52));
        enabled.setBounds (header.removeFromRight (44));
        muteButton.setBounds (header.removeFromRight (26).reduced (1));
        soloButton.setBounds (header.removeFromRight (26).reduced (1));

        // Row one, in four groups: the oscillator itself, unison, the wave
        // blend, and the reverb send.
        auto row = area.removeFromTop (kRowHeight);

        auto tone = row.removeFromLeft (86 + 3 * 64);
        beginGroup (tone, "tone", false);
        waveform.setBounds (tone.removeFromLeft (86).reduced (2, 14));
        layoutKnob (tone, semitones, semitonesLabel);
        layoutKnob (tone, cents, centsLabel);
        layoutKnob (tone, level, levelLabel);

        row.removeFromLeft (8);
        auto unisonGroup = row.removeFromLeft (2 * 64);
        beginGroup (unisonGroup, "unison", true);
        layoutKnob (unisonGroup, unison, unisonLabel);
        layoutKnob (unisonGroup, detune, detuneLabel);

        // Wave B sits with morph, not beside the main selector. On its own it
        // reads as a second waveform choice, when what it actually is is the
        // far end of the blend that morph sweeps towards -- at morph 0 it is
        // inaudible, which is exactly the confusing part.
        row.removeFromLeft (8);
        auto morphGroup = row.removeFromLeft (86 + 64);
        beginGroup (morphGroup, "wave blend", true);
        waveformB.setBounds (morphGroup.removeFromLeft (86).reduced (2, 14));
        layoutKnob (morphGroup, morph, morphLabel);

        // One knob, but still its own group: a send is routing rather than
        // synthesis, so it does not belong beside the tone controls. The
        // caption carries the noun and the knob the verb -- "REVERB / send" --
        // because captioning it "to reverb" over a knob labelled "rev send"
        // said reverb twice and still left the direction unclear.
        row.removeFromLeft (8);
        auto sendGroup = row.removeFromLeft (64);
        beginGroup (sendGroup, "reverb", true);
        layoutKnob (sendGroup, reverbSend, reverbSendLabel);

        auto envRow = area.removeFromTop (kRowHeight);
        beginGroup (envRow, "own amp envelope", false);
        ownEnv.setBounds (envRow.removeFromLeft (86).reduced (2, 14));
        layoutKnob (envRow, envA, envALabel);
        layoutKnob (envRow, envD, envDLabel);
        layoutKnob (envRow, envS, envSLabel);
        layoutKnob (envRow, envR, envRLabel);

        auto filterRow = area.removeFromTop (kRowHeight);
        beginGroup (filterRow, "own filter", false);
        auto filterLeft = filterRow.removeFromLeft (86);
        ownFilter.setBounds (filterLeft.removeFromTop (24).reduced (2, 3));
        filterType.setBounds (filterLeft.reduced (2, 3));
        layoutKnob (filterRow, filterCutoff, filterCutoffLabel);
        layoutKnob (filterRow, filterRes, filterResLabel);
        layoutKnob (filterRow, filterEnvAmt, filterEnvAmtLabel);
        layoutKnob (filterRow, filterD, filterDLabel);
        layoutKnob (filterRow, filterS, filterSLabel);

        auto tableRow = area.removeFromTop (kRowHeight);
        beginGroup (tableRow, "wavetable", false);
        layoutKnob (tableRow, frameCount, frameCountLabel);
        layoutKnob (tableRow, editFrame, editFrameLabel);
        layoutKnob (tableRow, framePos, framePosLabel);
        layoutKnob (tableRow, frameEnvAmt, frameEnvAmtLabel);
        layoutKnob (tableRow, frameA, frameALabel);
        frameView.setBounds (tableRow.reduced (6, 4));
    }

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;
    using ComboAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    void addToggle (juce::ToggleButton& button, const juce::String&, const juce::String& text)
    {
        button.setButtonText (text);
        button.setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);
        addAndMakeVisible (button);
    }

    void addKnob (juce::Slider& slider, juce::Label& label, const juce::String& id,
                  const juce::String& text, std::unique_ptr<SliderAttachment>& attachment)
    {
        // Style and wheel behaviour both come from ParamKnob.
        addAndMakeVisible (slider);

        label.setText (text, juce::dontSendNotification);
        label.setJustificationType (juce::Justification::centred);
        label.setColour (juce::Label::textColourId, juce::Colours::grey);
        label.setFont (juce::FontOptions (10.0f));
        addAndMakeVisible (label);

        attachment = std::make_unique<SliderAttachment> (processor.getState(), id, slider);
    }

    // Records a group's rectangle for `paint` to caption, then reserves the
    // strip of it the caption will occupy so controls do not sit under it.
    void beginGroup (juce::Rectangle<int>& rect, const juce::String& caption, bool divider)
    {
        groups.push_back ({ rect, caption, divider });
        rect.removeFromTop (kCaptionHeight);
    }

    static void layoutKnob (juce::Rectangle<int>& row, juce::Slider& slider, juce::Label& label)
    {
        auto cell = row.removeFromLeft (64);
        label.setBounds (cell.removeFromTop (12));
        slider.setBounds (cell);
    }

    void updateDimming()
    {
        const auto on = enabled.getToggleState();
        const auto alpha = on ? 1.0f : 0.35f;
        for (auto* child : getChildren())
            if (child != &enabled && child != &title)
                child->setAlpha (alpha);
        repaint();
    }

    AutoSynthProcessor& processor;
    int index;

    juce::Label title;
    juce::ToggleButton enabled, ownEnv, ownFilter;
    FrameView frameView;

    int editFrameIndex() const
    {
        return juce::jlimit (0, Oscillator::kMaxFrames - 1,
                             static_cast<int> (editFrame.getValue()) - 1);
    }
    juce::TextButton soloButton, muteButton;
    juce::ComboBox waveform, waveformB, filterType;
    struct Group
    {
        juce::Rectangle<int> area;
        juce::String caption;
        bool divider = false;
    };
    std::vector<Group> groups;

    ParamKnob semitones, cents, level, unison, detune, morph, reverbSend,
                 envA, envD, envS, envR,
                 filterCutoff, filterRes, filterEnvAmt, filterD, filterS,
                 framePos, frameEnvAmt, frameA, frameCount;
    ParamKnob editFrame;
    juce::Label semitonesLabel, centsLabel, levelLabel, unisonLabel, detuneLabel, morphLabel,
                reverbSendLabel,
                envALabel, envDLabel, envSLabel, envRLabel,
                filterCutoffLabel, filterResLabel, filterEnvAmtLabel,
                filterDLabel, filterSLabel,
                framePosLabel, frameEnvAmtLabel, frameALabel,
                frameCountLabel, editFrameLabel;

    std::unique_ptr<ButtonAttachment> enabledAttachment, ownEnvAttachment, ownFilterAttachment;
    std::unique_ptr<ComboAttachment> waveformAttachment, waveformBAttachment,
                                     filterTypeAttachment;
    std::unique_ptr<SliderAttachment> semitonesAttachment, centsAttachment, levelAttachment,
                                      unisonAttachment, detuneAttachment, morphAttachment,
                                      reverbSendAttachment,
                                      envAAttachment, envDAttachment, envSAttachment,
                                      envRAttachment, filterCutoffAttachment,
                                      filterResAttachment, filterEnvAmtAttachment,
                                      filterDAttachment, filterSAttachment,
                                      framePosAttachment, frameEnvAmtAttachment,
                                      frameAAttachment, frameCountAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (OscillatorStrip)
};

} // namespace autosynth
