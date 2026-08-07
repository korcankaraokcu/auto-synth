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
    static Adsr fitAdsr (const std::vector<float>& curve, const std::vector<float>& times,
                         double gateTime, float floor = 0.05f, bool oneShot = false);

private:
    static float fitCurve (const std::vector<float>& observed, const std::vector<float>& times,
                           const Adsr& env, double gateTime);
};

} // namespace autosynth
