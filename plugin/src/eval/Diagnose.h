#pragma once

#include <juce_core/juce_core.h>

#include <vector>

namespace autosynth
{

// How far a fit is from the recording, on axes a person can name.
//
// A single distance number cannot say why a fit sounds wrong, and can mislead:
// envelope distance once rated a hissing violin above a good clarinet. These
// are the quantities that have agreed with a listener repeatedly, each one
// something a parameter can fix.
//
// It lives in the library rather than in the tool that prints it because two
// callers report it -- the `diff` verb and the UI -- and a diagnostic written
// twice will eventually disagree with itself. Same rule the fitter follows:
// the number shown and the number measured are the same number.
class Diagnose
{
public:
    struct Options
    {
        int hop = 256;
        int fft = 2048;
    };

    // Where the rise is sampled, in milliseconds after the note starts.
    //
    // Spread over the range an attack occupies rather than evenly, because the
    // interesting part is the first tenth of a second: an instrument that
    // chiffs does most of its work there and an envelope that ramps does almost
    // none.
    static constexpr double kRiseMilliseconds[] = { 10.0, 25.0, 50.0, 100.0, 200.0, 400.0 };

    struct Measurements
    {
        double f0 = 0.0;
        double brightnessLog2 = 0.0;      // mean spectral centroid, log2 Hz
        std::vector<double> profile;      // harmonic amplitudes, peak-normalised
        double ampWobbleDb = 0.0;         // rms deviation of the sustained loudness
        double ampWobbleRateHz = 0.0;
        double pitchWobbleCents = 0.0;
        double pitchWobbleRateHz = 0.0;
        double attackSeconds = 0.0;
        double noteOffSeconds = 0.0;
        double noisiness = 0.0;           // energy away from the harmonics
        double peak = 0.0;

        // How far the note falls from its peak to the level it holds, in
        // decibels.
        //
        // Every other axis here is shape-relative -- each signal is normalised
        // by its own peak or its own profile before being compared -- so all of
        // them can read in tolerance while a preset loses two thirds of its
        // level after the attack. That happened: Vital squares the amplitude
        // envelope, so a sustain of 0.508 arrived as 0.258, and the whole
        // diagnostic said the patch was fine while a listener heard the peak
        // stand out as a surge.
        double sustainToPeakDb = 0.0;

        // How much of the note is present a twentieth of a second in, as a
        // fraction of its peak. The other half of the same blindness: attack
        // *time* is the moment a threshold is crossed and says nothing about
        // the shape of the rise, so an onset that fades in and one that arrives
        // can measure the same. A listener called an envelope too slow while
        // the attack read `ok`.
        double onsetAt50ms = 0.0;

        // The shape of the rise, sampled, as a fraction of the loudest frame.
        std::vector<double> rise;

        // How much the harmonic profile moves between the first and last thirds
        // of the note, in dB. A static oscillator scores near zero however
        // wrong its tone is, so this is a separate question from the profile.
        double timbreDriftDb = 0.0;
        std::vector<double> profileEarly, profileLate;

        // The loudness contour, frame by frame, and the seconds per frame.
        // Kept because it is what a plot of the two notes wants to draw, and
        // recomputing it elsewhere would be a second implementation of the
        // shape every envelope axis here is derived from.
        std::vector<float> loudness;
        double secondsPerFrame = 0.0;
    };

    // One named axis, already worded, because a verdict is only useful if it
    // says which way and by how much.
    struct Axis
    {
        juce::String name;
        juce::String target;
        juce::String fit;
        juce::String verdict;
        bool ok = false;
    };

    static Measurements measure (const std::vector<float>& x, double sampleRate,
                                 const Options& options = {});

    // The eleven axes, in the order they are worth reading.
    static std::vector<Axis> compare (const Measurements& target, const Measurements& fit);
};

} // namespace autosynth
