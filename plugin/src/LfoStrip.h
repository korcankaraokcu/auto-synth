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
        : processor (processorToUse)
    {
        auto& state = processor.getState();
        const auto prefix = "lfos_" + juce::String (lfoIndex) + "_";

        title.setText ("LFO " + juce::String (lfoIndex + 1), juce::dontSendNotification);
        title.setFont (juce::FontOptions (12.0f, juce::Font::bold));
        title.setColour (juce::Label::textColourId, juce::Colours::white);
        addAndMakeVisible (title);

        addCombo (shape, prefix + "shape", shapeAttachment);
        addCombo (dest, prefix + "dest", destAttachment);
        addKnob (rate, rateLabel, prefix + "rate_hz", "rate", rateAttachment);
        addKnob (depth, depthLabel, prefix + "depth", "depth", depthAttachment);
        addKnob (fade, fadeLabel, prefix + "delay", "fade in", fadeAttachment);
    }

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
        title.setBounds (area.removeFromLeft (52));

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
    juce::Label title;
    juce::ComboBox shape, dest;
    ParamKnob rate, depth, fade;
    juce::Label rateLabel, depthLabel, fadeLabel;
    std::unique_ptr<ComboAttachment> shapeAttachment, destAttachment;
    std::unique_ptr<SliderAttachment> rateAttachment, depthAttachment, fadeAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (LfoStrip)
};

} // namespace autosynth
