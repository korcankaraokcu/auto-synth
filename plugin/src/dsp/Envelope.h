#pragma once

#include "ir/Patch.h"
#include <cmath>

namespace autosynth
{

// What an `Adsr` in the patch actually means, as a function of time.
//
// Not a player: nothing here advances a note. Vital plays the envelope; this is
// the definition the fitter shapes one against and the exporter translates, and
// it matches engine/envelope.py, so an envelope fitted here means what it meant
// in the reference implementation the fixtures came from.
class Envelope
{
public:
    // Closed-form value at an arbitrary time. The curve fitter evaluates
    // candidate envelopes at frame times, and duplicating the formula there
    // would let the two drift apart silently.
    static float evaluate (const Adsr& params, double t, double gateTime) noexcept
    {
        const auto a = juce::jmax (params.attack, 1.0e-6f);
        const auto d = juce::jmax (params.decay, 1.0e-6f);
        const auto s = juce::jlimit (0.0f, 1.0f, params.sustain);
        const auto r = juce::jmax (params.release, 1.0e-6f);

        const auto held = [&] (double time)
        {
            const auto tf = static_cast<float> (time);
            if (tf < a)
                return shape (tf / a, params.attackCurve);
            if (tf < a + d)
                return 1.0f + (s - 1.0f) * shape (juce::jlimit (0.0f, 1.0f, (tf - a) / d), params.curve);
            return s;
        };

        if (t < gateTime)
            return juce::jlimit (0.0f, 1.0f, held (t));

        const auto levelAtGate = juce::jlimit (0.0f, 1.0f, held (gateTime));
        const auto progress = juce::jlimit (0.0f, 1.0f, static_cast<float> ((t - gateTime)) / r);
        return juce::jlimit (0.0f, 1.0f, levelAtGate * (1.0f - shape (progress, params.curve)));
    }

    // Curve shaping: 0 is linear, larger bends toward exponential.
    //
    // The attack uses `attackCurve` and the decay and release use `curve`, which
    // is the same function applied to two independent numbers. Sharing one was
    // tried and made both real samples worse: an attack that wants to be concave
    // and a decay that wants to be convex cannot be described by one parameter,
    // and the fit lands between them.
    static float shape (float progress, float curve) noexcept
    {
        if (curve <= 1.0e-6f)
            return progress;
        const auto denom = 1.0f - std::exp (-curve);
        if (std::abs (denom) < 1.0e-12f)
            return progress;
        return (1.0f - std::exp (-curve * progress)) / denom;
    }
};

} // namespace autosynth
