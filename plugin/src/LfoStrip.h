#pragma once

#include "ParamKnob.h"
#include "PluginProcessor.h"

namespace autosynth
{

// One LFO slot.
//
// Two of these exist because one was a real limitation: vibrato and tremolo
// competed for a single slot and only the more periodic one survived, so a
// sound with both could never be represented. Each slot carries its own
// destination, which makes the pair a two-slot modulation matrix.
class LfoStrip : public juce::Component
{
public:
    LfoStrip (AutoSynthProcessor& processorToUse, int lfoIndex)
        : processor (processorToUse), index (lfoIndex)
    {
        auto& state = processor.getState();
        const auto prefix = "lfos_" + juce::String (lfoIndex) + "_";

        title.setFont (juce::FontOptions (12.0f, juce::Font::bold));
        title.setColour (juce::Label::textColourId, juce::Colours::white);
        addAndMakeVisible (title);

        addCombo (shape, prefix + "shape", shapeAttachment);
        addCombo (dest, prefix + "dest", destAttachment);

        // Say what the LFO is *doing*, not just that it exists.
        //
        // "LFO 1" is the synth's word for it; "vibrato" is the player's, and a
        // user looking for vibrato scrolled straight past a strip that never
        // used the term. The destination already carries the answer, so the
        // title follows it.
        dest.onChange = [this] { updateRole(); };
        updateRole();
        addKnob (rate, rateLabel, prefix + "rate_hz", "rate", rateAttachment);
        addKnob (depth, depthLabel, prefix + "depth", "depth", depthAttachment);
        addKnob (fade, fadeLabel, prefix + "delay", "fade in", fadeAttachment);
    }

    // Called by the editor after a fit, since a new patch changes the
    // destination without the combo box firing.
    void refresh() { updateRole(); }

    void paint (juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat().reduced (1.0f);
        g.setColour (juce::Colour (0xff20242a));
        g.fillRoundedRectangle (bounds, 4.0f);
        g.setColour (juce::Colour (0xff2c3138));
        g.drawRoundedRectangle (bounds, 4.0f, 1.0f);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (6, 4);
        // Wide enough for the role name, not just "LFO 1".
        title.setBounds (area.removeFromLeft (118));

        auto combos = area.removeFromLeft (96);
        shape.setBounds (combos.removeFromTop (combos.getHeight() / 2).reduced (2, 4));
        dest.setBounds (combos.reduced (2, 4));

        layoutKnob (area, rate, rateLabel);
        layoutKnob (area, depth, depthLabel);
        layoutKnob (area, fade, fadeLabel);
    }

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ComboAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    void updateRole()
    {
        // Matches LfoDest: none, pitch, amp, cutoff.
        const char* roles[] = { "off", "vibrato", "tremolo", "filter sweep" };
        const auto selected = juce::jlimit (0, 3, dest.getSelectedItemIndex());
        title.setText ("LFO " + juce::String (index + 1) + ":  " + roles[selected],
                       juce::dontSendNotification);
        title.setColour (juce::Label::textColourId,
                         selected == 0 ? juce::Colours::grey : juce::Colours::white);
    }

    void addCombo (juce::ComboBox& box, const juce::String& id,
                   std::unique_ptr<ComboAttachment>& attachment)
    {
        box.setJustificationType (juce::Justification::centredLeft);
        if (auto* parameter = processor.getState().getParameter (id))
            box.addItemList (parameter->getAllValueStrings(), 1);
        // As in OscillatorStrip: scrolling past a combo must not change it.
        box.setScrollWheelEnabled (false);
        addAndMakeVisible (box);
        attachment = std::make_unique<ComboAttachment> (processor.getState(), id, box);
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

    static void layoutKnob (juce::Rectangle<int>& row, juce::Slider& slider, juce::Label& label)
    {
        auto cell = row.removeFromLeft (64);
        label.setBounds (cell.removeFromTop (12));
        slider.setBounds (cell);
    }

    AutoSynthProcessor& processor;
    const int index;
    juce::Label title;
    juce::ComboBox shape, dest;
    ParamKnob rate, depth, fade;
    juce::Label rateLabel, depthLabel, fadeLabel;
    std::unique_ptr<ComboAttachment> shapeAttachment, destAttachment;
    std::unique_ptr<SliderAttachment> rateAttachment, depthAttachment, fadeAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (LfoStrip)
};

} // namespace autosynth
