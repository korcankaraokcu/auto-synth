#pragma once

#include "ir/Patch.h"
#include <vector>

namespace autosynth
{

// Port of autosynth/fit/modulation.py -- trajectory to LFO.
//
// A trajectory is split into a slow part and an oscillating part; the
// oscillating part becomes an LFO if it is periodic enough to be worth one.
// Three trajectories compete, one per destination the engine supports, and the
// IR carries a single LFO -- so a sound with both vibrato and tremolo can keep
// only one. That is a real limitation, and the point at which a mod matrix
// stops being optional.
class Modulation
{
public:
    static constexpr double kMinRateHz = 0.3;
    static constexpr double kMaxRateHz = 18.0;
    // A single step can fake 2.5 cycles of a square wave; it cannot fake four.
    static constexpr double kMinCycles = 3.5;
    // Measured: genuine 5 Hz vibrato scores 0.83, while the pitch-track step
    // produced by two decaying sources scores 0.23. The old 0.10 admitted the
    // artefact, which rendered as a note jumping a semitone and back.
    // How irregular a trajectory may be and still count as an oscillation.
    //
    // Measured as mean-crossings divided by the number the detected rate
    // implies, so a clean LFO sits at 1.0. This is the discriminator that
    // concentration and correlation could not provide:
    //
    //     synthetic LFO      0.98
    //     violin vibrato     1.7 - 2.5
    //     phantom vibrato    6.5 - 14.6
    //
    // The phantom is not a clean step, which is what made it hard to reject --
    // it is a *jittering* pitch track, crossing its mean far more often than
    // any rate it reports. Real vibrato wanders but still turns over about
    // twice a cycle.
    static constexpr double kMaxOscillationRatio = 3.0;

    // Rates a played instrument's vibrato or tremolo actually occupies.
    //
    // Inside this band the shape requirement is relaxed to
    // `kVibratoCorrelation`, because a human wobble is irregular by nature: a
    // violin's vibrato correlates at 0.27 where a synthetic LFO reaches 0.99.
    //
    // The band is what makes the relaxation safe. The spurious detection that
    // forced the general threshold to stay at 0.40 sat at 17.7 Hz -- nothing
    // plays vibrato at 17.7 Hz, so it never gets the easier test.
    static constexpr double kVibratoMinHz = 3.0;
    static constexpr double kVibratoMaxHz = 9.0;
    static constexpr double kVibratoCorrelation = 0.25;

    // How peaked the modulation spectrum has to be.
    //
    // This is the threshold that currently blocks vibrato detection on real
    // recordings, and lowering it is *not* the answer -- tried and measured. A
    // played violin's vibrato scores 0.07 here, because the rate and depth vary
    // from cycle to cycle and the energy smears across bins. A synthetic LFO
    // scores 0.83. But the phantom vibrato this threshold was raised to kill --
    // a pitch track stepping as one source fades -- scores 0.23, which is
    // *above* the violin. No setting of this number accepts real vibrato and
    // rejects that artefact.
    //
    // Dropping it to 0.05 and letting `kMinCorrelation` decide was tried and
    // immediately produced a second, spurious LFO on a synthetic fixture that
    // has exactly one. Correlation is not sufficient either.
    //
    // `kMaxOscillationRatio` above is that discriminator, so this is now a
    // floor against noise rather than the main gate.
    static constexpr double kMinConcentration = 0.05;
    // How well the trajectory must match an actual LFO shape.
    //
    // Relaxed from 0.45 now that `kMaxOscillationRatio` guards the artefact
    // case independently. A played instrument's modulation is not sinusoidal --
    // a violin's tremolo correlates at 0.42 where a synthetic LFO reaches 0.99 --
    // so the old bar rejected every real recording.
    //
    // 0.40 was chosen by sweeping: at 0.30 nothing changes, and at 0.25 a
    // fixture with one pitch LFO acquires a spurious 17.7 Hz amp LFO. This is
    // the loosest setting at which every synthetic case is still exactly right.
    static constexpr double kMinCorrelation = 0.40;
    // Marginally below the 0.05 it was, because a real violin's vibrato lands
    // exactly on that boundary: 24 cents of wobble over a trajectory whose full
    // span includes the note's own pitch drift. The phantom case sits at 0.02
    // and is unaffected.
    static constexpr double kMinRelativeAmplitude = 0.04;
    // Shorter than it was, because a half-second window leaves too much slow
    // drift behind and the rate search locked onto it: a violin whose vibrato
    // runs at about 6 Hz was reported at 2.3 Hz. At 0.3 s the same trajectory
    // reports 4.0 Hz, which is the wobble rather than the wander.
    static constexpr double kDetrendSeconds = 0.30;

    struct Detected
    {
        bool found = false;
        LfoDest dest = LfoDest::none;
        double rateHz = 0.0;
        double amplitude = 0.0; // in the trajectory's own units
        LfoShape shape = LfoShape::sine;
        double phase = 0.0;
        double delay = 0.0;
        double concentration = 0.0;

        // Why it was rejected, when it was. Detection here is a stack of
        // thresholds and any one of them can veto; without knowing which, a
        // missed vibrato is just silence. Reported so a real recording can be
        // asked what it actually failed on.
        double correlation = 0.0;
        double relativeAmplitude = 0.0;
        // Times the trajectory crosses its own mean, as a fraction of what the
        // detected rate implies. An oscillation crosses constantly; a step
        // crosses once however large it is.
        double oscillationRatio = 0.0;
        const char* rejectedBy = nullptr;
    };

    static Detected analyseTrajectory (const std::vector<float>& trajectory, double dt,
                                       LfoDest dest);

    static Lfo toLfo (const Detected& detected);

    // The three trajectories, each expressed in the unit its destination is
    // modulated in -- cents, linear gain, octaves -- so a detected amplitude
    // converts to `depth` without a second scaling step.
    struct Trajectories
    {
        std::vector<float> pitchCents;
        std::vector<float> ampRelative;
        std::vector<float> centroidOctaves;
        double dt = 0.0;
        bool hasPitch = false;
    };

    static Trajectories extract (const float* samples, int numSamples, double sampleRate,
                                 int hop = 256);

    // Returns an LFO with dest = none when nothing is periodic enough to be
    // worth the parameters.
    static Lfo best (const Trajectories& trajectories);

    // The most convincing modulations, at most one per destination. One per
    // destination on purpose: two LFOs both fitted to pitch would be the
    // detector finding the same wobble twice, not two modulations.
    static std::vector<Lfo> bestSeveral (const Trajectories& trajectories,
                                         int maxCount = kNumLfo);
};

} // namespace autosynth
