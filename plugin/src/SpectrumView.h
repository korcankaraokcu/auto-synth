#pragma once

#include "PluginProcessor.h"

namespace autosynth
{

// Source spectrum against fitted spectrum, on a log-frequency axis.
//
// The point is not decoration: it shows *where* the fit is losing the sound.
// A patch can score well overall and still be visibly missing the top two
// octaves or the noise floor, and no single distance number says which.
class SpectrumView : public juce::Component
{
public:
    void setSpectra (AutoSynthProcessor::Spectra newSpectra)
    {
        spectra = std::move (newSpectra);
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        auto bounds = getLocalBounds().toFloat().reduced (1.0f);
        g.setColour (juce::Colour (0xff141619));
        g.fillRoundedRectangle (bounds, 3.0f);

        if (! spectra.valid || spectra.frequencies.size() < 2)
        {
            g.setColour (juce::Colours::grey);
            g.setFont (juce::FontOptions (11.0f));
            g.drawText ("drop a sample to compare spectra", bounds, juce::Justification::centred);
            return;
        }

        // Gridlines at decade-ish landmarks, so the log axis is readable.
        g.setColour (juce::Colour (0xff24282e));
        for (auto hz : { 100.0f, 1000.0f, 10000.0f })
        {
            const auto x = xFor (hz, bounds);
            if (x > bounds.getX() && x < bounds.getRight())
                g.drawVerticalLine (juce::roundToInt (x), bounds.getY(), bounds.getBottom());
        }

        drawCurve (g, bounds, spectra.sourceDb, juce::Colour (0xff8a929c), 1.6f);
        drawCurve (g, bounds, spectra.fitDb, juce::Colour (0xff4a8fe7), 1.6f);

        g.setFont (juce::FontOptions (10.0f));
        g.setColour (juce::Colour (0xff8a929c));
        g.drawText ("source", bounds.reduced (6.0f), juce::Justification::topLeft);
        g.setColour (juce::Colour (0xff4a8fe7));
        g.drawText ("fit", bounds.reduced (6.0f).translated (0.0f, 12.0f),
                    juce::Justification::topLeft);
    }

private:
    static constexpr float kMinDb = -90.0f;
    static constexpr float kMaxDb = 0.0f;

    float xFor (float hz, juce::Rectangle<float> bounds) const
    {
        const auto lo = std::log (juce::jmax (spectra.frequencies.front(), 1.0f));
        const auto hi = std::log (juce::jmax (spectra.frequencies.back(), 2.0f));
        const auto t = (std::log (juce::jmax (hz, 1.0f)) - lo) / juce::jmax (hi - lo, 1.0e-6f);
        return bounds.getX() + juce::jlimit (0.0f, 1.0f, t) * bounds.getWidth();
    }

    void drawCurve (juce::Graphics& g, juce::Rectangle<float> bounds,
                    const std::vector<float>& db, juce::Colour colour, float thickness) const
    {
        if (db.size() != spectra.frequencies.size())
            return;

        // Normalised to the source's own peak so the two curves are compared on
        // shape, not on output level -- master gain is a separate question and
        // would otherwise swamp the comparison.
        auto peak = kMinDb;
        for (auto v : spectra.sourceDb)
            peak = juce::jmax (peak, v);
        for (auto v : spectra.fitDb)
            peak = juce::jmax (peak, v);

        juce::Path path;
        for (size_t i = 0; i < db.size(); ++i)
        {
            const auto x = xFor (spectra.frequencies[i], bounds);
            const auto norm = juce::jlimit (kMinDb, kMaxDb, db[i] - peak);
            const auto y = bounds.getBottom()
                         - (norm - kMinDb) / (kMaxDb - kMinDb) * bounds.getHeight();
            if (i == 0)
                path.startNewSubPath (x, y);
            else
                path.lineTo (x, y);
        }

        g.setColour (colour);
        g.strokePath (path, juce::PathStrokeType (thickness));
    }

    AutoSynthProcessor::Spectra spectra;
};

} // namespace autosynth
