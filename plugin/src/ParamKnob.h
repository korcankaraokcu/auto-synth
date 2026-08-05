#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

namespace autosynth
{

// A rotary knob that shares the mouse wheel with the panel it sits in.
//
// Hovering a knob and scrolling adjusts that knob, which is what the wheel is
// for and what every other synth does. But the parameter panel is taller than
// the window and an oscillator strip is close to wall-to-wall knobs, so if the
// knobs took every wheel event there would be almost nowhere left to scroll
// from -- the pointer is nearly always over one.
//
// Holding Alt hands the event to the parent instead. JUCE's default
// `Component::mouseWheelMove` forwards to the parent component, so the event
// travels up to the enclosing Viewport on its own; there is nothing to wire.
class ParamKnob : public juce::Slider
{
public:
    ParamKnob()
    {
        setSliderStyle (RotaryHorizontalVerticalDrag);
        setTextBoxStyle (TextBoxBelow, false, 54, 14);
        setColour (textBoxOutlineColourId, juce::Colours::transparentBlack);
        setColour (textBoxTextColourId, juce::Colours::lightgrey);
        setColour (rotarySliderFillColourId, juce::Colour (0xff4a8fe7));
        setColour (rotarySliderOutlineColourId, juce::Colour (0xff333a42));
        setColour (thumbColourId, juce::Colours::white);
    }

    void mouseWheelMove (const juce::MouseEvent& e,
                         const juce::MouseWheelDetails& wheel) override
    {
        if (e.mods.isAltDown())
        {
            juce::Component::mouseWheelMove (e, wheel);
            return;
        }

        juce::Slider::mouseWheelMove (e, wheel);
    }

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ParamKnob)
};

} // namespace autosynth
