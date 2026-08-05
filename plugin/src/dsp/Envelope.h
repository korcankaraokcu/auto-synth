#pragma once

#include "ir/Patch.h"
#include <cmath>

namespace autosynth
{

// Sample-accurate ADSR matching engine/envelope.py.
//
// The Python version evaluates the whole envelope as a vectorised function of
// time. This one advances one sample at a time, but deliberately computes the
// same closed form from an elapsed-time counter rather than accumulating a
// per-sample increment. Accumulation drifts, and drift here would show up as a
// slow divergence from the reference engine that the conformance test would
// flag without explaining.
class Envelope
{
public:
    void setParameters (const Adsr& adsr) noexcept { params = adsr; }
    void prepare (double sr) noexcept { sampleRate = sr; }

    void noteOn() noexcept
    {
        elapsed = 0.0;
        released = false;
        levelAtRelease = 0.0f;
        releaseElapsed = 0.0;
    }

    void noteOff() noexcept
    {
        if (released)
            return;
        levelAtRelease = heldValue (elapsed);
        released = true;
        releaseElapsed = 0.0;
    }

    bool isFinished() const noexcept
    {
        return released && releaseElapsed >= juce::jmax (params.release, 1.0e-6f);
    }

    float nextSample() noexcept
    {
        const auto value = released ? releasedValue() : heldValue (elapsed);
        const auto step = 1.0 / juce::jmax (sampleRate, 1.0);
        if (released)
            releaseElapsed += step;
        else
            elapsed += step;
        return juce::jlimit (0.0f, 1.0f, value);
    }

    // Closed-form value at an arbitrary time, independent of any playing note.
    // The curve fitter needs to evaluate candidate envelopes at frame times
    // without running a voice, and duplicating the formula there would let the
    // two drift apart silently.
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
                return tf / a;
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

    // Curve shaping: 0 is linear, larger bends toward exponential decay.
    static float shape (float progress, float curve) noexcept
    {
        if (curve <= 1.0e-6f)
            return progress;
        const auto denom = 1.0f - std::exp (-curve);
        if (std::abs (denom) < 1.0e-12f)
            return progress;
        return (1.0f - std::exp (-curve * progress)) / denom;
    }

private:
    float heldValue (double t) const noexcept
    {
        const auto a = juce::jmax (params.attack, 1.0e-6f);
        const auto d = juce::jmax (params.decay, 1.0e-6f);
        const auto s = juce::jlimit (0.0f, 1.0f, params.sustain);
        const auto tf = static_cast<float> (t);

        if (tf < a)
            return tf / a;
        if (tf < a + d)
        {
            const auto progress = juce::jlimit (0.0f, 1.0f, (tf - a) / d);
            return 1.0f + (s - 1.0f) * shape (progress, params.curve);
        }
        return s;
    }

    float releasedValue() const noexcept
    {
        const auto r = juce::jmax (params.release, 1.0e-6f);
        const auto progress = juce::jlimit (0.0f, 1.0f, static_cast<float> (releaseElapsed) / r);
        return levelAtRelease * (1.0f - shape (progress, params.curve));
    }

    Adsr params {};
    double sampleRate = 44100.0;
    double elapsed = 0.0;
    double releaseElapsed = 0.0;
    float levelAtRelease = 0.0f;
    bool released = false;
};

// Bipolar LFO with fade-in and start phase, matching engine/lfo.py.
class LfoOsc
{
public:
    void setParameters (const Lfo& lfo) noexcept { params = lfo; }
    void prepare (double sr) noexcept { sampleRate = sr; }
    void reset() noexcept { elapsed = 0.0; }

    float nextSample() noexcept
    {
        const auto t = static_cast<float> (elapsed);
        const auto phase = params.rateHz * t + params.phase;

        float wave = 0.0f;
        const auto twoPi = juce::MathConstants<float>::twoPi;
        const auto frac = phase - std::floor (phase);
        switch (params.shape)
        {
            case LfoShape::sine:     wave = std::sin (twoPi * phase); break;
            case LfoShape::triangle: wave = 2.0f / juce::MathConstants<float>::pi
                                          * std::asin (juce::jlimit (-1.0f, 1.0f, std::sin (twoPi * phase))); break;
            case LfoShape::saw:      wave = 2.0f * frac - 1.0f; break;
            case LfoShape::square:   wave = std::sin (twoPi * phase) >= 0.0f ? 1.0f : -1.0f; break;
        }

        const auto fade = params.delay > 1.0e-6f
                        ? juce::jlimit (0.0f, 1.0f, t / params.delay)
                        : 1.0f;

        elapsed += 1.0 / juce::jmax (sampleRate, 1.0);
        return wave * params.depth * fade;
    }

private:
    Lfo params {};
    double sampleRate = 44100.0;
    double elapsed = 0.0;
};

} // namespace autosynth
