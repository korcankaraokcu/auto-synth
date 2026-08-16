#pragma once

namespace autosynth
{

// How much of a sound is *not* one of its harmonics.
//
// This is the "roles before counting" question, narrowed to the one role that
// turned out to be worth separating. Two others were tried and did not pay:
//
//   * Splitting the tracked partials into a steady body and unstable fragments,
//     and grouping only the body. On the recovery harness that was worth
//     nothing at the loosest threshold which kept counting intact -- 62.5%
//     either way -- and actively harmful once a duration test was added, since
//     duration correlates with loudness and the partials it discarded were the
//     quiet second source the count was already missing. Grouping's own guards
//     refuse noise well enough: salience charges for predicted harmonics that
//     are absent, and a later group has to show its own bottom.
//   * That same split as a *measurement*. It reads whatever the analysis window
//     makes it read: at 2048 points a violin measured 72% of its tracked energy
//     in unstable partials against a clarinet's 0.9%, which looks like exactly
//     the right discrimination -- but the fitter picks its window from the
//     fundamental, and at the 1024 points it actually uses both samples read
//     0.03%. A number that moves three orders of magnitude with a window is not
//     measuring the sound.
//
// So the noise is measured where it can be seen directly: in the spectrum,
// between the harmonics. Broadband noise lifts the floor *between* partials; a
// partial that wanders with vibrato does not, because it stays near its own
// harmonic. That is the distinction spectral flatness cannot make -- it reads
// smeared harmonics as noise, and chasing it means adding hiss to imitate
// smearing.
//
// `autosynth_diff` reports the same quantity as "noisiness" through this same
// function: a diagnostic that says a fit is too clean and a fitter deciding how
// much noise to allow have to be measuring the same thing.
class Roles
{
public:
    // Bins either side of a harmonic that count as the harmonic itself. A Hann
    // main lobe is about four bins wide, so this is the partial rather than a
    // guard band around it.
    static constexpr int kHarmonicGuardBins = 2;

    // Fraction of spectral energy sitting away from any harmonic of `f0`.
    // Returns 0 when there is no pitch to measure against.
    static double noiseShare (const float* samples, int numSamples, double sampleRate,
                              double f0Hz, int fftSize = 2048, int hop = 256);

    // The same measurement frame by frame, and the *inter-harmonic energy*
    // rather than its share.
    //
    // A share is the wrong shape for an envelope: it rises as a note dies,
    // because the harmonics fade faster than the room does, so fitting an
    // envelope to it would ask the noise to get louder as the note ends. The
    // energy itself is what a noise envelope has to follow.
};

} // namespace autosynth
