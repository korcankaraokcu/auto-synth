#pragma once

#include "ir/Patch.h"
#include <vector>

namespace autosynth
{

// Port of autosynth/fit/effects.py -- delay detection.
//
// Delay only. Discrete repeats are literal copies of the signal displaced in
// time, so they show up as clean peaks in the autocorrelation of the loudness
// envelope. Reverb is a dense diffuse tail with no discrete repeats to
// correlate against, and separating it from an instrument's own decay is blind
// dereverberation.
//
// There is a second reason to stop at delay: a reverb tail and a long release
// produce nearly identical loudness curves, so a fitter given both with no
// structure to separate them trades one against the other and gets both wrong.
// Delay has no such twin.
class EffectsFit
{
public:
    static constexpr double kMinDelaySeconds = 0.02;
    static constexpr double kMaxDelaySeconds = 1.0;
    static constexpr double kMinPeakRatio = 0.15;

    struct DelayEstimate
    {
        bool found = false;
        double time = 0.0;
        double feedback = 0.0;
        double mix = 0.0;
        double strength = 0.0;
    };

    // `gateSeconds` enables the check that the repeats outlive the note, which
    // is what separates a delay from vibrato or tremolo -- both make the
    // loudness envelope just as periodic. Pass a negative value when note-off
    // is unknown, and the whole-file evidence stands alone.
    static DelayEstimate detectDelay (const float* samples, int numSamples, double sampleRate,
                                      int hop = 256, double smoothSeconds = 0.01,
                                      double gateSeconds = -1.0);

    // Detected delay as IR parameters, disabled when nothing was found.
    static DelayParams fitDelay (const float* samples, int numSamples, double sampleRate,
                                 int hop = 256, double gateSeconds = -1.0);

    // --- reverb ------------------------------------------------------------
    //
    // Blind dereverberation in general is out of reach, and the header above
    // says so. What is within reach, once note-off is known accurately, is the
    // narrower question: after the note stops, is there a diffuse tail, how
    // long does it take to die, and how loud is it?
    //
    // The reason to do it at all is a measurement. On two library samples the
    // note body fitted to within 4 dB while the tail was out by 27 dB, so the
    // un-modelled reverb was very nearly the whole error.
    //
    // The degeneracy that made this dangerous -- a reverb tail and a long
    // release describe the same curve -- is settled by structure rather than by
    // search. The decay after note-off is treated as two segments: the direct
    // sound's own release owns the fast knee at the start, and the reverb owns
    // the slow exponential that follows. Each is measured on its own segment,
    // so neither is free to absorb the other.
    static constexpr double kMinRt60Seconds = 0.25;
    static constexpr double kMaxRt60Seconds = 12.0;
    // Below this the "tail" is just the release and there is nothing to model.
    static constexpr double kMinTailSeconds = 0.12;
    // A reverb tail decays as a straight line in dB. This is how straight it
    // has to be, as a coefficient of determination.
    static constexpr double kMinDecayFit = 0.55;

    struct ReverbEstimate
    {
        bool found = false;
        double rt60 = 0.0;          // seconds to fall 60 dB
        double releaseSeconds = 0.0; // where the direct sound stops and the tail takes over
        double levelBelowBodyDb = 0.0; // tail level at note-off, relative to the note
        double decayFit = 0.0;      // 1 is a perfectly exponential decay
        double brightnessLossOct = 0.0; // how much the tail darkens as it dies
    };

    static ReverbEstimate detectReverb (const float* samples, int numSamples, double sampleRate,
                                        double gateSeconds, int hop = 256);

    // Detected reverb as IR parameters, disabled when nothing was found.
    static ReverbParams fitReverb (const float* samples, int numSamples, double sampleRate,
                                   double gateSeconds, int hop = 256);

    // Room size that produces a given RT60 in dsp/Reverb.h, and its inverse.
    // Exposed because the recovery test needs to state its expectation in
    // seconds rather than in an opaque 0-1 knob position.
    static double sizeForRt60 (double rt60Seconds);
    static double rt60ForSize (double size);
};

} // namespace autosynth
