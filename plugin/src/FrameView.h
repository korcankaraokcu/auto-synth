#pragma once

#include "ir/Patch.h"
#include <juce_gui_basics/juce_gui_basics.h>

namespace autosynth
{

// One frame of an oscillator's wavetable, drawn and edited as harmonic bars.
//
// Bars rather than a waveform trace because harmonics are the domain everything
// else in this project reasons in: the fitter compares harmonic profiles, the
// filter is divided out in harmonic terms, and `autosynth_diff` reports the
// error harmonic by harmonic. Drawing a time-domain squiggle would hide the one
// thing a person needs to see, and a drawn curve cannot always be represented
// in sixteen harmonics anyway, so what you drew and what you heard would differ.
//
// This is also the answer to the only real objection to having wavetables at
// all: a table is the one part of a fitted patch that cannot be read as a
// sentence. "Saw, morphed 40% toward a narrow pulse" is a description; sixteen
// numbers are not. Made visible and draggable, they are.
class FrameView : public juce::Component
{
public:
    FrameView() = default;

    // Called back with (harmonic index, new amplitude) as the user drags. The
    // owner is what turns that into a patch edit -- this component holds no
    // state that outlives a repaint.
    std::function<void (int, float)> onHarmonicDragged;

    void setFrames (const std::array<Oscillator::Frame, Oscillator::kMaxFrames>& newFrames,
                    int newNumFrames, int newSelected, float newPosition,
                    const std::vector<float>& generatedProfile)
    {
        frames = newFrames;
        numFrames = juce::jlimit (1, Oscillator::kMaxFrames, newNumFrames);
        selected = juce::jlimit (0, numFrames - 1, newSelected);
        position = juce::jlimit (0.0f, 1.0f, newPosition);
        generated = generatedProfile;
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat().reduced (1.0f);
        g.setColour (juce::Colour (0xff15181c));
        g.fillRoundedRectangle (bounds, 3.0f);

        const auto plot = bounds.reduced (3.0f);
        const auto slot = plot.getWidth() / static_cast<float> (Oscillator::kFrameHarmonics);

        // Where the note currently sits, so a table that moves can be read
        // against the frame being edited.
        if (numFrames > 1)
        {
            const auto scaled = position * static_cast<float> (numFrames - 1);
            const auto nearest = juce::jlimit (0, numFrames - 1, static_cast<int> (std::lround (scaled)));
            if (nearest == selected)
            {
                g.setColour (juce::Colour (0xff5a9ee0).withAlpha (0.10f));
                g.fillRoundedRectangle (bounds, 3.0f);
            }
        }

        for (int k = 0; k < Oscillator::kFrameHarmonics; ++k)
        {
            const auto x = plot.getX() + k * slot;
            const auto width = juce::jmax (1.0f, slot - 2.0f);

            // The frames either side, faint. The point of a multi-frame table
            // is the movement between frames, and editing one blind to the
            // others hides exactly that.
            for (int f = 0; f < numFrames; ++f)
            {
                if (f == selected)
                    continue;
                const auto h = plot.getHeight() * amplitudeAt (f, k);
                if (h < 0.5f)
                    continue;
                g.setColour (juce::Colour (0xff2f3b45));
                g.fillRect (juce::Rectangle<float> (x, plot.getBottom() - h, width, 1.5f));
            }

            const auto amount = amplitudeAt (selected, k);
            const auto height = plot.getHeight() * amount;
            const auto drawn = frames[static_cast<size_t> (selected)].custom;
            g.setColour (juce::Colour (drawn ? 0xff5a9ee0 : 0xff4a5f70));
            g.fillRect (juce::Rectangle<float> (x, plot.getBottom() - juce::jmax (1.0f, height),
                                                width, juce::jmax (1.0f, height)));
        }

        if (! frames[static_cast<size_t> (selected)].custom)
        {
            g.setColour (juce::Colour (0xff6d8296));
            g.setFont (juce::FontOptions (9.0f));
            g.drawText ("generated - drag to edit", plot.withHeight (12.0f),
                        juce::Justification::centredRight);
        }
    }

    void mouseDown (const juce::MouseEvent& e) override { drag (e); }
    void mouseDrag (const juce::MouseEvent& e) override { drag (e); }

private:
    // What the frame is worth right now: its own harmonics once drawn, and the
    // generator's otherwise, so switching a waveform moves the bars.
    float amplitudeAt (int frame, int k) const
    {
        const auto& f = frames[static_cast<size_t> (frame)];
        const auto value = f.custom
                         ? f.harmonics[static_cast<size_t> (k)]
                         : (k < static_cast<int> (generated.size()) ? generated[static_cast<size_t> (k)] : 0.0f);
        return juce::jlimit (0.0f, 1.0f, std::abs (value) / juce::jmax (1.0e-9f, peak()));
    }

    float peak() const
    {
        float p = 1.0e-9f;
        for (int f = 0; f < numFrames; ++f)
        {
            const auto& frame = frames[static_cast<size_t> (f)];
            if (frame.custom)
                for (const auto v : frame.harmonics)
                    p = juce::jmax (p, std::abs (v));
        }
        for (const auto v : generated)
            p = juce::jmax (p, std::abs (v));
        return p;
    }

    void drag (const juce::MouseEvent& e)
    {
        if (onHarmonicDragged == nullptr)
            return;

        const auto plot = getLocalBounds().toFloat().reduced (4.0f);
        const auto slot = plot.getWidth() / static_cast<float> (Oscillator::kFrameHarmonics);
        const auto k = juce::jlimit (0, Oscillator::kFrameHarmonics - 1,
                                     static_cast<int> ((e.position.x - plot.getX()) / juce::jmax (1.0f, slot)));
        const auto amount = juce::jlimit (0.0f, 1.0f,
                                          (plot.getBottom() - e.position.y) / juce::jmax (1.0f, plot.getHeight()));
        onHarmonicDragged (k, amount);
    }

    std::array<Oscillator::Frame, Oscillator::kMaxFrames> frames {};
    std::vector<float> generated;
    int numFrames = 1;
    int selected = 0;
    float position = 0.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FrameView)
};

} // namespace autosynth
