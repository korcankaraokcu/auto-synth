// Why does the fit not sound like the target?
//
// A single distance number cannot answer that, and worse, it misleads: the
// envelope distance called a hissing violin *better* than a good clarinet,
// because it measures loudness contour and is deaf to noise. Listening catches
// what it misses, but "rough" is not something to act on.
//
// This reports the difference on axes a person can name and a parameter can
// fix -- pitch, brightness, the harmonic profile, modulation, envelope, noise,
// tail -- and says which way each one is wrong.
//
// The measuring is `Diagnose`, in the library, because the UI reports the same
// axes and a diagnostic written twice will eventually disagree with itself.
// What stays here is the printing, and the three tables under the axes, which
// are shaped by the terminal rather than by the measurement.
//
// Usage:
//   autosynth diff target.wav fit.wav [--hop 256] [--fft 2048]

#include "eval/Diagnose.h"
#include "fit/Modulation.h"

#include "Cli.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace
{

using autosynth::cli::Args;

void line (const char* name, const juce::String& target, const juce::String& fit,
           const juce::String& verdict)
{
    std::printf ("  %-18s %-23s %-23s %s\n", name, target.toRawUTF8(), fit.toRawUTF8(),
                 verdict.toRawUTF8());
}

} // namespace

int runDiff (const Args& args)
{
    if (args.positional.size() < 2)
    {
        std::fprintf (stderr, "usage: autosynth diff <target.wav> <fit.wav> "
                              "[--hop n] [--fft n]\n");
        return 2;
    }

    autosynth::Diagnose::Options options;
    options.hop = static_cast<int> (args.value ("--hop", 256.0));
    options.fft = static_cast<int> (args.value ("--fft", 2048.0));

    double targetRate = 0.0, fitRate = 0.0;
    const auto target = autosynth::cli::readMono (args.file (0), targetRate);
    const auto fit = autosynth::cli::readMono (args.file (1), fitRate);

    if (target.empty() || fit.empty())
    {
        std::fprintf (stderr, "error: could not read both files\n");
        return 1;
    }

    const auto a = autosynth::Diagnose::measure (target, targetRate, options);
    const auto b = autosynth::Diagnose::measure (fit, fitRate, options);

    std::printf ("\n%s  vs  %s\n\n", args.positional[0].toRawUTF8(),
                 args.positional[1].toRawUTF8());
    std::printf ("  %-18s %-23s %-23s %s\n", "", "target", "fit", "verdict");
    std::printf ("  %s\n", juce::String::repeatedString ("-", 88).toRawUTF8());

    for (const auto& axis : autosynth::Diagnose::compare (a, b))
        line (axis.name.toRawUTF8(), axis.target, axis.fit, axis.verdict);

    // The shape of the rise, which is what a single onset number cannot show.
    //
    // A clarinet reaches 0.44 of its peak inside fifty milliseconds, settles
    // back to 0.38 by a hundred and fifty, and only then swells to full over
    // the next quarter second. No ADSR attack does that at any curve: the shape
    // is monotonic by construction, so a fit can match the chiff or the swell
    // and not both. Printed because the alternative is believing the onset is
    // a knob that was set wrong.
    const auto& riseAt = autosynth::Diagnose::kRiseMilliseconds;
    if (a.rise.size() == std::size (riseAt) && b.rise.size() == std::size (riseAt))
    {
        std::printf ("\n  the rise (fraction of the loudest frame)\n");
        std::printf ("  %-10s", "at");
        for (const auto ms : riseAt)
            std::printf ("%7s", (juce::String ((int) ms) + "ms").toRawUTF8());
        std::printf ("\n  %-10s", "target");
        for (const auto v : a.rise)
            std::printf ("%7.2f", v);
        std::printf ("\n  %-10s", "fit");
        for (const auto v : b.rise)
            std::printf ("%7.2f", v);
        std::printf ("\n");
    }

    // The harmonic profile, which is where "the tone is wrong" actually lives.
    const auto harmonics = std::min<size_t> (8, std::min (a.profile.size(), b.profile.size()));
    if (harmonics > 0)
    {
        std::printf ("\n  harmonic profile (dB relative to the fundamental)\n");
        std::printf ("  %-10s", "harmonic");
        for (size_t k = 0; k < harmonics; ++k)
            std::printf ("%7d", static_cast<int> (k + 1));
        std::printf ("\n  %-10s", "target");
        for (size_t k = 0; k < harmonics; ++k)
            std::printf ("%7.1f", 20.0 * std::log10 (std::max (a.profile[k], 1.0e-4)));
        std::printf ("\n  %-10s", "fit");
        for (size_t k = 0; k < harmonics; ++k)
            std::printf ("%7.1f", 20.0 * std::log10 (std::max (b.profile[k], 1.0e-4)));
        std::printf ("\n  %-10s", "error");
        for (size_t k = 0; k < harmonics; ++k)
            std::printf ("%+7.1f", 20.0 * std::log10 (std::max (b.profile[k], 1.0e-4))
                                 - 20.0 * std::log10 (std::max (a.profile[k], 1.0e-4)));
        std::printf ("\n");
    }

    // Where the timbre drift lives, harmonic by harmonic.
    //
    // The single drift number says the tone moves and says nothing about what
    // moves, which is not enough to act on: four attempts at the wavetable
    // ladder were aimed at a figure whose origin nobody could point to. The
    // deconvolved partials the fitter sees carry 1.8 dB of the clarinet's 4.3,
    // and its filter envelope adds none of the rest, so the remainder has to be
    // somewhere this shows and that did not.
    const auto drifting = juce::jmin ((size_t) 8,
                                      juce::jmin (a.profileEarly.size(), a.profileLate.size()));
    if (drifting > 0 && b.profileEarly.size() >= drifting && b.profileLate.size() >= drifting)
    {
        const auto asDb = [] (double v) { return 20.0 * std::log10 (std::max (v, 1.0e-4)); };

        std::printf ("\n  timbre drift by harmonic (late third minus early third, dB)\n");
        std::printf ("  %-10s", "harmonic");
        for (size_t k = 0; k < drifting; ++k)
            std::printf ("%7d", (int) (k + 1));

        std::printf ("\n  %-10s", "target");
        for (size_t k = 0; k < drifting; ++k)
            std::printf ("%+7.1f", asDb (a.profileLate[k]) - asDb (a.profileEarly[k]));

        std::printf ("\n  %-10s", "fit");
        for (size_t k = 0; k < drifting; ++k)
            std::printf ("%+7.1f", asDb (b.profileLate[k]) - asDb (b.profileEarly[k]));

        std::printf ("\n  %-10s", "early tgt");
        for (size_t k = 0; k < drifting; ++k)
            std::printf ("%7.1f", asDb (a.profileEarly[k]));

        std::printf ("\n  %-10s", "late tgt");
        for (size_t k = 0; k < drifting; ++k)
            std::printf ("%7.1f", asDb (a.profileLate[k]));
        std::printf ("\n");
    }

    // What the modulation detector made of the target, and where it gave up.
    //
    // Detection is a stack of thresholds and any one of them can veto. Without
    // knowing which, a missed vibrato looks exactly like a sample that has
    // none -- which is how a violin's 24 cents of it went unnoticed.
    const auto trajectories = autosynth::Modulation::extract (target.data(), (int) target.size(),
                                                              targetRate, options.hop);
    struct Row { const char* name; const std::vector<float>* data; autosynth::LfoDest dest; };
    const Row rows[] = {
        { "pitch",  &trajectories.pitchCents,      autosynth::LfoDest::pitch  },
        { "amp",    &trajectories.ampRelative,     autosynth::LfoDest::amp    },
        { "cutoff", &trajectories.centroidOctaves, autosynth::LfoDest::cutoff },
    };

    std::printf ("\n  modulation detector on the target\n");
    for (const auto& row : rows)
    {
        const auto d = autosynth::Modulation::analyseTrajectory (*row.data, trajectories.dt,
                                                                 row.dest);
        std::printf ("  %-8s rate %5.2f Hz   concentration %.2f   correlation %.2f   "
                     "rel.amp %.2f   osc %.2f   -> %s\n",
                     row.name, d.rateHz, d.concentration, d.correlation, d.relativeAmplitude,
                     d.oscillationRatio,
                     d.found ? "ACCEPTED" : (d.rejectedBy ? d.rejectedBy : "not analysed"));

        // Whether that rate holds still. A steady LFO keeps one period; a
        // player does not, and the difference is what makes a fit sound
        // mechanical even when its depth is right.
        const auto w = autosynth::Modulation::detectWander (*row.data, trajectories.dt);
        std::printf ("           wander %5.2f Hz   %.2f octaves over %d cycles   -> %s\n",
                     w.rateHz, w.octaves, w.cycles, w.found ? "DRIFTS" : "steady");
    }

    std::printf ("\n");
    return 0;
}
