#pragma once

#include "ir/Patch.h"
#include <vector>

namespace autosynth
{

// Port of autosynth/fit/envelope.py -- amplitude trajectory to ADSR, plus the
// sustained-vs-one-shot decision.
class EnvelopeFit
{
public:
    struct Gate
    {
        double time = 0.0;
        bool oneShot = true;
    };

    // First index reaching `frac` of the peak -- where the attack has finished.
    //
    // Not argmax. On a sustained tone the envelope is essentially flat, so
    // argmax lands on whatever frame the ripple happened to make highest, which
    // can be anywhere including the very end. That made a constant-amplitude
    // saw report a 1.7-second attack and get classified as a one-shot.
    static int peakIndex (const std::vector<float>& curve, float frac = 0.95f);

    // Find the note-off point, and whether there is one at all.
    //
    // The test is the level *range* over a window, not the slope. A slope
    // threshold was tried first and fails on anything with fast shallow ripple:
    // a saw with 5 Hz vibrato varies only 7% in amplitude but swings the
    // derivative +/-11 dB/s, past any threshold a decay would also clear.
    static Gate detectGate (const std::vector<float>& rms, const std::vector<float>& times,
                            float maxPlateauRangeDb = 2.5f,
                            double minPlateauSeconds = 0.15,
                            double rangeWindowSeconds = 0.25,
                            float levelFloor = 0.1f,
                            double smoothSeconds = 0.05,
                            // Roughly one vibrato period. Flatness is measured
                            // on a copy smoothed this far, so a played note's
                            // natural swing does not disqualify its sustain.
                            double plateauSmoothSeconds = 0.25);

    // With `oneShot` the sound never sustains, so sustain is pinned to zero and
    // release mirrors the decay. Without that pin the fitter reads the tail of a
    // pluck as a low sustain and the patch drones for as long as a key is held.
    // How long the contour takes to reach nine tenths of the level it holds,
    // in seconds. A *measurement* of a signal, not a parameter of a patch --
    // which is the distinction `fitAdsr` used to blur, because its `attack`
    // field once carried this number and then acquired a curve that changed
    // what the same number rendered as. A diagnostic comparing two audio files
    // wants this; the engine wants the parameter.
    static float attackSeconds (const std::vector<float>& curve, const std::vector<float>& times,
                                float floor = 0.05f);

    // The attack parameter whose envelope *measures back* as `measured`.
    //
    // The parameter and the measurement are different quantities and the gap
    // between them is not a constant. `attackSeconds` measures the time to nine
    // tenths of the level the note *holds*, which is after the decay stage has
    // pulled it down; the attack segment's own peak is 1. So the crossing
    // depends on the attack, the decay, the sustain and the curve at once, and
    // there is no closed form worth trusting.
    //
    // Solved numerically against `Envelope::evaluate` instead, which also means
    // this cannot drift from whatever the engine's curve law happens to be. The
    // crossing rises monotonically with the attack, so a bisection is enough.
    static float attackMeasuring (Adsr env, double gateTime, float measured,
                                  double frameRate = 200.0, float floor = 0.05f);

    // The attack curve that best follows the observed rise.
    //
    // Fitted on the attack segment alone, and *before* the attack length is
    // solved from the measured crossing, because the crossing depends on the
    // curve. Getting that order wrong is how the shared-curve version failed:
    // the attack was solved against one shape and then rendered with another.
    // `amplitudeDomain` says whether `observed` is a loudness contour or an
    // already-logarithmic one such as a cutoff trajectory in octaves.
    //
    // It decides which domain the attack curve is fitted in, and it is not a
    // detail. In decibels is right for loudness: a linear ramp's fault is
    // concentrated where it is 10 to 16 dB down but only a few percent away in
    // amplitude, so an amplitude comparison cannot see it. Applied to a cutoff
    // trajectory, which is *already* a log quantity, the same criterion takes
    // the logarithm twice and bends the curve far too hard -- a clarinet's
    // filter came back opening two octaves in the first fraction of its attack,
    // which a listener heard as a burst of noise that then dissolved.
    static float fitAttackCurve (const std::vector<float>& observed,
                                 const std::vector<float>& times, const Adsr& env,
                                 double gateTime, bool amplitudeDomain = true);

    // `amplitudeDomain` is passed through to `fitAttackCurve`; see there.
    static Adsr fitAdsr (const std::vector<float>& curve, const std::vector<float>& times,
                         double gateTime, float floor = 0.05f, bool oneShot = false,
                         bool amplitudeDomain = true);

private:
    static float fitCurve (const std::vector<float>& observed, const std::vector<float>& times,
                           const Adsr& env, double gateTime);
};

} // namespace autosynth
