#pragma once

#include "ir/Patch.h"
#include <array>
#include <vector>

namespace autosynth
{

// Schroeder reverb matching engine/reverb.py: eight parallel combs summed into
// four series allpasses, then a one-pole lowpass on the tail.
//
// The damping filter sits *after* the comb bank rather than inside each comb's
// feedback loop, which is where Freeverb puts it. That is not an approximation
// made here for convenience -- it is a constraint imported from the reference
// engine. A filter inside the loop makes the recurrence depend on the
// immediately preceding sample, which is fine in C++ and fatal in numpy, where
// a two-second render would become ~600k Python iterations and CMA-ES
// evaluates 192 renders per fit.
//
// Both engines therefore implement the same reverb, and the conformance test
// can hold them tightly together.
class Reverb
{
public:
    // Freeverb's tunings at 44100 Hz. Mutually prime, which is what stops the
    // comb resonances lining up into a metallic ring.
    static constexpr std::array<int, 8> kCombDelays {
        1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617 };
    static constexpr std::array<int, 4> kAllpassDelays { 556, 441, 341, 225 };
    static constexpr float kAllpassGain = 0.5f;
    static constexpr double kReferenceSampleRate = 44100.0;

    void prepare (double sr)
    {
        sampleRate = sr;
        const auto scale = sr / kReferenceSampleRate;

        for (size_t i = 0; i < combs.size(); ++i)
        {
            combs[i].buffer.assign (
                static_cast<size_t> (juce::jmax (1, static_cast<int> (
                    std::lround (kCombDelays[i] * scale)))), 0.0f);
            combs[i].index = 0;
        }
        for (size_t i = 0; i < allpasses.size(); ++i)
        {
            allpasses[i].buffer.assign (
                static_cast<size_t> (juce::jmax (1, static_cast<int> (
                    std::lround (kAllpassDelays[i] * scale)))), 0.0f);
            allpasses[i].index = 0;
        }
        reset();
    }

    void reset() noexcept
    {
        for (auto& c : combs)
        {
            std::fill (c.buffer.begin(), c.buffer.end(), 0.0f);
            c.index = 0;
        }
        for (auto& a : allpasses)
        {
            std::fill (a.buffer.begin(), a.buffer.end(), 0.0f);
            a.index = 0;
        }
        lowpassState = 0.0f;
    }

    void setParameters (const ReverbParams& p) noexcept
    {
        params = p;
        // Freeverb's room-size mapping: below ~0.7 the tail dies almost at
        // once, above ~0.98 it never does.
        feedback = juce::jlimit (0.0f, 0.98f, 0.7f + 0.28f * p.size);
        damping = juce::jlimit (0.0f, 0.95f, 0.4f * p.damp);
    }

    bool isActive() const noexcept { return params.enabled && params.level > 1.0e-6f; }

    // Returns the wet signal only. The caller sums it with the dry path, so
    // `level` is a return gain rather than a dry/wet balance -- which is what
    // makes per-oscillator sends meaningful.
    float processSample (float x) noexcept
    {
        if (! isActive())
            return 0.0f;

        float wet = 0.0f;
        for (auto& c : combs)
            wet += c.process (x, feedback);
        wet /= static_cast<float> (combs.size());

        for (auto& a : allpasses)
            wet = a.process (wet, kAllpassGain);

        if (damping > 1.0e-9f)
        {
            lowpassState = (1.0f - damping) * wet + damping * lowpassState;
            wet = lowpassState;
        }

        return wet * params.level;
    }

private:
    // Comb and allpass share one recurrence, b[n] = x[n] + g*b[n-D]; they
    // differ only in which tap becomes the output.
    struct Line
    {
        std::vector<float> buffer;
        int index = 0;

        float advance (float x, float gain) noexcept
        {
            const auto delayed = buffer[static_cast<size_t> (index)];
            buffer[static_cast<size_t> (index)] = x + gain * delayed;
            if (++index >= static_cast<int> (buffer.size()))
                index = 0;
            return delayed;
        }
    };

    struct Comb : Line
    {
        float process (float x, float gain) noexcept { return advance (x, gain); }
    };

    struct Allpass : Line
    {
        float process (float x, float gain) noexcept { return -x + advance (x, gain); }
    };

    ReverbParams params {};
    std::array<Comb, 8> combs;
    std::array<Allpass, 4> allpasses;
    float feedback = 0.84f;
    float damping = 0.2f;
    float lowpassState = 0.0f;
    double sampleRate = 44100.0;
};

} // namespace autosynth
