#pragma once

#include "ParamKnob.h"
#include "PluginProcessor.h"

namespace autosynth
{

// A titled box of related controls: an optional on/off, an optional type
// selector, and the knobs that belong to them.
//
// The global area was previously one flat wrapped grid of every visible
// parameter. Nineteen knobs in reading order, with names truncated to fit,
// gave no clue which three belonged to the reverb or that the delay had an
// on/off at all -- and the on/off was in fact missing entirely, so the reverb
// knobs did nothing however far you turned them.
//
// Grouping is what makes an effect discoverable: the switch that arms it sits
// inside the same box as the knobs it arms.
class ControlGroup : public juce::Component
{
public:
    struct KnobDef
    {
        juce::String id;    // parameter id, without the group prefix logic
        juce::String label; // short caption; the full name is on the tooltip
    };

    static constexpr int headerHeight = 15;
    static constexpr int bodyHeight = 84;
    static constexpr int toggleWidth = 58;
    static constexpr int comboWidth = 92;
    static constexpr int knobWidth = 72;

    ControlGroup (AutoSynthProcessor& processorToUse,
                  juce::String groupTitle,
                  juce::String toggleId,
                  juce::String comboId,
                  std::vector<KnobDef> knobDefs)
        : processor (processorToUse),
          title (std::move (groupTitle)),
          defs (std::move (knobDefs))
    {
        auto& state = processor.getState();

        if (toggleId.isNotEmpty())
        {
            toggle = std::make_unique<juce::ToggleButton> ("on");
            toggle->setColour (juce::ToggleButton::textColourId, juce::Colours::lightgrey);
            addAndMakeVisible (*toggle);
            toggleAttachment = std::make_unique<ButtonAttachment> (state, toggleId, *toggle);
        }

        if (comboId.isNotEmpty())
        {
            combo = std::make_unique<juce::ComboBox>();
            combo->setJustificationType (juce::Justification::centredLeft);
            if (auto* parameter = state.getParameter (comboId))
                combo->addItemList (parameter->getAllValueStrings(), 1);
            // Scrolling past a selector must not silently change it.
            combo->setScrollWheelEnabled (false);
            addAndMakeVisible (*combo);
            comboAttachment = std::make_unique<ComboAttachment> (state, comboId, *combo);
        }

        for (const auto& def : defs)
        {
            auto knob = std::make_unique<Knob>();
            knob->slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 62, 14);
            if (auto* parameter = state.getParameter (def.id))
                knob->slider.setTooltip (parameter->getName (64));
            addAndMakeVisible (knob->slider);

            knob->label.setText (def.label, juce::dontSendNotification);
            knob->label.setJustificationType (juce::Justification::centred);
            knob->label.setColour (juce::Label::textColourId, juce::Colours::grey);
            knob->label.setFont (juce::FontOptions (10.0f));
            addAndMakeVisible (knob->label);

            knob->attachment = std::make_unique<SliderAttachment> (state, def.id, knob->slider);
            knobs.push_back (std::move (knob));
        }
    }

    int preferredWidth() const
    {
        auto width = 12;
        if (toggle != nullptr) width += toggleWidth;
        if (combo != nullptr)  width += comboWidth;
        width += static_cast<int> (knobs.size()) * knobWidth;
        return width;
    }

    static int preferredHeight() { return headerHeight + bodyHeight; }

    void paint (juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat().reduced (0.5f);
        g.setColour (juce::Colour (0xff1d2126));
        g.fillRoundedRectangle (bounds, 4.0f);
        g.setColour (juce::Colour (0xff2c3138));
        g.drawRoundedRectangle (bounds, 4.0f, 1.0f);

        g.setColour (juce::Colour (0xff7f93a8));
        g.setFont (juce::FontOptions (10.0f, juce::Font::bold));
        g.drawText (title.toUpperCase(),
                    getLocalBounds().removeFromTop (headerHeight).reduced (8, 0),
                    juce::Justification::centredLeft);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (6, 3);
        area.removeFromTop (headerHeight - 3);

        if (toggle != nullptr)
            toggle->setBounds (area.removeFromLeft (toggleWidth).reduced (2, 30));
        if (combo != nullptr)
            combo->setBounds (area.removeFromLeft (comboWidth).reduced (3, 28));

        for (auto& knob : knobs)
        {
            auto cell = area.removeFromLeft (knobWidth);
            knob->label.setBounds (cell.removeFromTop (12));
            knob->slider.setBounds (cell);
        }
    }

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;
    using ComboAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    struct Knob
    {
        ParamKnob slider;
        juce::Label label;
        std::unique_ptr<SliderAttachment> attachment;
    };

    AutoSynthProcessor& processor;
    juce::String title;
    std::vector<KnobDef> defs;

    std::unique_ptr<juce::ToggleButton> toggle;
    std::unique_ptr<ButtonAttachment> toggleAttachment;
    std::unique_ptr<juce::ComboBox> combo;
    std::unique_ptr<ComboAttachment> comboAttachment;
    std::vector<std::unique_ptr<Knob>> knobs;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ControlGroup)
};

} // namespace autosynth
