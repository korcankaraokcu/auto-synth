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
// Usage:
//   autosynth_diff target.wav fit.wav [--hop 256] [--fft 2048]

#include "analysis/Grouping.h"
#include "analysis/Partials.h"
#include "analysis/Stft.h"
#include "analysis/Yin.h"
#include "fit/EnvelopeFit.h"
#include "fit/Modulation.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace
{

struct Args
{
    std::map<juce::String, juce::String> options;
    juce::StringArray positional;

    double value (const char* flag, double fallback) const
    {
        const auto it = options.find (flag);
        if (it == options.end() || it->second.isEmpty())
            return fallback;
        return it->second.getDoubleValue();
    }
};

Args parseArgs (int argc, char* argv[])
{
    Args out;
    juce::StringArray raw;
    for (int i = 1; i < argc; ++i)
        raw.add (juce::String (argv[i]));

    for (int i = 0; i < raw.size(); ++i)
    {
        if (! raw[i].startsWith ("--"))
        {
            out.positional.add (raw[i]);
            continue;
        }
        auto key = raw[i];
        juce::String value;
        if (key.containsChar ('='))
        {
            value = key.fromFirstOccurrenceOf ("=", false, false);
            key = key.upToFirstOccurrenceOf ("=", false, false);
        }
        else if (i + 1 < raw.size() && ! raw[i + 1].startsWith ("--"))
        {
            value = raw[++i];
        }
        out.options[key] = value;
    }
    return out;
}

std::vector<float> readMono (const juce::File& file, double& sampleRateOut)
{
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (file));
    if (reader == nullptr)
        return {};

    const auto numSamples = static_cast<int> (reader->lengthInSamples);
    juce::AudioBuffer<float> buffer (static_cast<int> (reader->numChannels),
                                     juce::jmax (1, numSamples));
    reader->read (&buffer, 0, numSamples, 0, true, true);

    if (buffer.getNumChannels() > 1)
    {
        for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
            buffer.addFrom (0, 0, buffer, ch, 0, numSamples);
        buffer.applyGain (0, 0, numSamples, 1.0f / buffer.getNumChannels());
    }
    sampleRateOut = reader->sampleRate;
    const auto* data = buffer.getReadPointer (0);
    return std::vector<float> (data, data + numSamples);
}

// --- the measurements ------------------------------------------------------

struct Measurements
{
    double f0 = 0.0;
    double brightnessLog2 = 0.0;     // mean spectral centroid, log2 Hz
    std::vector<double> profile;      // harmonic amplitudes, peak-normalised
    double ampWobbleDb = 0.0;         // rms deviation of the sustained loudness
    double ampWobbleRateHz = 0.0;
    double pitchWobbleCents = 0.0;
    double pitchWobbleRateHz = 0.0;
    double attackSeconds = 0.0;
    double noteOffSeconds = 0.0;
    double noisiness = 0.0;           // energy away from the harmonics
    double tailRt60 = 0.0;
    double peak = 0.0;
    // How much the harmonic profile moves between the first and last thirds of
    // the note, in dB. A static oscillator scores near zero however wrong its
    // tone is, so this is a separate question from the profile itself.
    double timbreDriftDb = 0.0;
    std::vector<double> profileEarly, profileLate;
};

// Energy-weighted mean harmonic profile over a frame range, peak-normalised.
std::vector<double> profileOver (const autosynth::HarmonicGroup& g, int from, int to)
{
    std::vector<double> profile (static_cast<size_t> (std::max (g.numHarmonics, 1)), 0.0);
    if (g.numHarmonics <= 0 || to <= from)
        return profile;

    for (int k = 0; k < g.numHarmonics; ++k)
    {
        const auto* row = g.harmonic (k);
        double acc = 0.0;
        for (int t = from; t < to && t < g.numFrames; ++t)
            acc += row[t];
        profile[static_cast<size_t> (k)] = acc / std::max (1, to - from);
    }
    const auto peak = *std::max_element (profile.begin(), profile.end());
    if (peak > 1.0e-12)
        for (auto& v : profile)
            v /= peak;
    return profile;
}

// Rate of a zero-mean signal by counting sign changes. Crude next to a
// periodogram, and enough here: the question is "about how fast", not "at
// exactly what frequency".
double wobbleRate (const std::vector<double>& deviation, double framesPerSecond)
{
    if (deviation.size() < 8)
        return 0.0;

    std::vector<double> smooth (deviation.size());
    for (size_t i = 0; i < deviation.size(); ++i)
    {
        const auto lo = i >= 2 ? i - 2 : 0;
        const auto hi = std::min (deviation.size() - 1, i + 2);
        double acc = 0.0;
        for (auto j = lo; j <= hi; ++j)
            acc += deviation[j];
        smooth[i] = acc / (hi - lo + 1);
    }

    int crossings = 0;
    for (size_t i = 1; i < smooth.size(); ++i)
        if ((smooth[i - 1] < 0.0) != (smooth[i] < 0.0))
            ++crossings;

    const auto seconds = deviation.size() / framesPerSecond;
    return seconds > 0.0 ? crossings / 2.0 / seconds : 0.0;
}

Measurements measure (const std::vector<float>& x, double sampleRate, int hop, int fft)
{
    Measurements m;
    if (x.empty())
        return m;

    const auto n = static_cast<int> (x.size());
    const auto fps = sampleRate / hop;

    for (auto v : x)
        m.peak = std::max (m.peak, std::abs ((double) v));

    double confidence = 0.0;
    autosynth::Yin::estimate (x.data(), n, sampleRate, m.f0, confidence, hop);

    const auto spectrogram = autosynth::Stft::magnitudeSpectrogram (x.data(), n, fft, hop, sampleRate);
    const auto centroid = autosynth::Stft::spectralCentroid (spectrogram);
    {
        double acc = 0.0;
        int counted = 0;
        for (auto c : centroid)
            if (c > 20.0f) { acc += std::log2 (c); ++counted; }
        m.brightnessLog2 = counted > 0 ? acc / counted : 0.0;
    }

    // Energy that is *not* near a harmonic, as a fraction of the whole.
    //
    // Spectral flatness was tried first and is the wrong measure: it cannot
    // tell hiss from smeared harmonics. A real violin's partials wander with
    // the vibrato and are slightly inharmonic, which spreads energy across bins
    // and reads as flat -- the target scored 0.113 where our fit reached only
    // 0.055 at a noise level that was plainly hissy to listen to. Chasing that
    // number would have meant adding hiss to imitate smearing.
    //
    // Measuring away from the harmonics separates them. Broadband noise lifts
    // the floor between partials; a wandering partial does not, because it
    // stays near its own harmonic.
    if (m.f0 > 20.0)
    {
        const auto binHz = sampleRate / fft;
        // Two bins either side: a Hann main lobe is about four wide, so this is
        // the partial itself rather than a guard band around it.
        const auto guard = 2;

        double harmonicEnergy = 0.0, floorEnergy = 0.0;
        for (int t = 0; t < spectrogram.numFrames; ++t)
        {
            const auto* frame = spectrogram.frame (t);
            for (int k = 1; k < spectrogram.numBins; ++k)
            {
                const auto freq = k * binHz;
                const auto nearest = std::max (1.0, std::round (freq / m.f0));
                const auto distanceBins = std::abs (freq - nearest * m.f0) / binHz;
                (distanceBins <= guard ? harmonicEnergy : floorEnergy) += frame[k];
            }
        }
        const auto total = harmonicEnergy + floorEnergy;
        m.noisiness = total > 1.0e-9 ? floorEnergy / total : 0.0;
    }

    // Harmonic profile of the dominant source.
    autosynth::PartialTracker::Options trackOptions;
    trackOptions.fftSize = fft;
    trackOptions.hop = hop;
    const auto partials = autosynth::PartialTracker::track (x.data(), n, sampleRate, trackOptions);
    const auto groups = autosynth::Grouping::group (partials, 3);
    if (! groups.empty())
    {
        const auto& g = groups.front();
        m.profile.assign (static_cast<size_t> (g.numHarmonics), 0.0);
        for (int k = 0; k < g.numHarmonics; ++k)
        {
            const auto* row = g.harmonic (k);
            double acc = 0.0;
            for (int t = 0; t < g.numFrames; ++t)
                acc += row[t];
            m.profile[static_cast<size_t> (k)] = acc / std::max (1, g.numFrames);
        }
        const auto peak = *std::max_element (m.profile.begin(), m.profile.end());
        if (peak > 1.0e-12)
            for (auto& v : m.profile)
                v /= peak;

        // Does the tone move across the note?
        //
        // Measured because a filter envelope cannot answer it: on both library
        // samples the second harmonic swings four to five dB while the third
        // holds, which is not a brightness sweep and no cutoff setting produces
        // it. A fit with a static oscillator scores near zero here however
        // close its average profile is.
        m.profileEarly = profileOver (g, 0, g.numFrames / 3);
        m.profileLate = profileOver (g, 2 * g.numFrames / 3, g.numFrames);

        double drift = 0.0;
        int counted = 0;
        for (size_t k = 0; k < std::min<size_t> (6, m.profileEarly.size()); ++k)
        {
            const auto a = 20.0 * std::log10 (std::max (m.profileEarly[k], 1.0e-4));
            const auto b = 20.0 * std::log10 (std::max (m.profileLate[k], 1.0e-4));
            drift += std::abs (b - a);
            ++counted;
        }
        m.timbreDriftDb = counted > 0 ? drift / counted : 0.0;
    }

    // Envelope: attack and note-off.
    const auto loudness = autosynth::Stft::loudnessEnvelope (x.data(), n, hop);
    std::vector<float> times (loudness.size());
    for (size_t i = 0; i < loudness.size(); ++i)
        times[i] = static_cast<float> (i / fps);

    const auto gate = autosynth::EnvelopeFit::detectGate (loudness, times);
    m.noteOffSeconds = gate.time;
    const auto env = autosynth::EnvelopeFit::fitAdsr (loudness, times, gate.time, 0.05f, gate.oneShot);
    m.attackSeconds = autosynth::EnvelopeFit::attackSeconds (loudness, times);

    // Modulation, measured over the sustained middle so the attack and the
    // release do not read as wobble.
    const auto lo = static_cast<size_t> (juce::jlimit (0.0, gate.time * 0.7, 0.3) * fps);
    const auto hi = static_cast<size_t> (std::min<double> (loudness.size(), gate.time * 0.95 * fps));
    if (hi > lo + 8)
    {
        const auto peak = *std::max_element (loudness.begin(), loudness.end());
        std::vector<double> db;
        for (auto i = lo; i < hi; ++i)
            db.push_back (20.0 * std::log10 (std::max<double> (loudness[i], peak * 1.0e-4) / peak));

        const auto mean = std::accumulate (db.begin(), db.end(), 0.0) / db.size();
        std::vector<double> deviation;
        double sumSquares = 0.0;
        for (auto v : db)
        {
            deviation.push_back (v - mean);
            sumSquares += (v - mean) * (v - mean);
        }
        m.ampWobbleDb = std::sqrt (sumSquares / deviation.size());
        m.ampWobbleRateHz = wobbleRate (deviation, fps);
    }

    const auto pitch = autosynth::Yin::track (x.data(), n, sampleRate, hop);
    {
        std::vector<double> cents;
        std::vector<double> confident;
        for (size_t i = 0; i < pitch.f0.size(); ++i)
            if (pitch.f0[i] > 0.0f && i < pitch.confidence.size() && pitch.confidence[i] > 0.5f)
                confident.push_back (pitch.f0[i]);

        if (confident.size() > 8)
        {
            auto sorted = confident;
            std::sort (sorted.begin(), sorted.end());
            const auto median = sorted[sorted.size() / 2];
            for (auto v : confident)
                cents.push_back (1200.0 * std::log2 (v / median));

            // Detrend before measuring, so this reports *vibrato* rather than
            // vibrato plus drift.
            //
            // Taken raw it read a violin at 24 cents where only about 10 of
            // those oscillate -- the rest is the pitch wandering across the
            // note. That made analysis look like it was under-reading depth by
            // half when it was measuring the right thing and this was not.
            const auto smoothSpan = std::max<size_t> (3, static_cast<size_t> (0.30 * fps));
            std::vector<double> wobble (cents.size());
            for (size_t i = 0; i < cents.size(); ++i)
            {
                const auto lo2 = i >= smoothSpan / 2 ? i - smoothSpan / 2 : 0;
                const auto hi2 = std::min (cents.size() - 1, i + smoothSpan / 2);
                double acc = 0.0;
                for (auto j = lo2; j <= hi2; ++j)
                    acc += cents[j];
                wobble[i] = cents[i] - acc / (hi2 - lo2 + 1);
            }

            auto spread = wobble;
            std::sort (spread.begin(), spread.end());
            const auto lo10 = spread[static_cast<size_t> (0.10 * (spread.size() - 1))];
            const auto hi90 = spread[static_cast<size_t> (0.90 * (spread.size() - 1))];
            m.pitchWobbleCents = hi90 - lo10;
            m.pitchWobbleRateHz = wobbleRate (wobble, fps);
        }
    }

    return m;
}

// juce::String (double, 0) prints "as many digits as needed", not none, which
// turns a tidy column into 23.7975.
juce::String rounded (double v, int decimals = 0)
{
    if (decimals <= 0)
        return juce::String (static_cast<juce::int64> (std::llround (v)));
    return juce::String (v, decimals);
}

void line (const char* name, const juce::String& target, const juce::String& fit,
           const juce::String& verdict)
{
    std::printf ("  %-18s %-23s %-23s %s\n", name, target.toRawUTF8(), fit.toRawUTF8(),
                 verdict.toRawUTF8());
}

// A verdict is only useful if it says which way, and by how much.
juce::String verdictFor (double target, double fit, double tolerance,
                         const char* lowWord, const char* highWord, const char* unit)
{
    const auto delta = fit - target;
    if (std::abs (delta) <= tolerance)
        return "ok";
    const auto magnitude = std::abs (delta);
    return juce::String (delta > 0 ? highWord : lowWord) + " by "
         + (magnitude < 10.0 ? juce::String (magnitude, 2) : rounded (magnitude)) + unit;
}

} // namespace

int main (int argc, char* argv[])
{
    const auto args = parseArgs (argc, argv);
    if (args.positional.size() < 2)
    {
        std::fprintf (stderr, "usage: autosynth_diff <target.wav> <fit.wav> "
                              "[--hop n] [--fft n]\n");
        return 2;
    }

    const auto hop = static_cast<int> (args.value ("--hop", 256.0));
    const auto fft = static_cast<int> (args.value ("--fft", 2048.0));

    const auto cwd = juce::File::getCurrentWorkingDirectory();
    double targetRate = 0.0, fitRate = 0.0;
    const auto target = readMono (cwd.getChildFile (args.positional[0]), targetRate);
    const auto fit = readMono (cwd.getChildFile (args.positional[1]), fitRate);

    if (target.empty() || fit.empty())
    {
        std::fprintf (stderr, "error: could not read both files\n");
        return 1;
    }

    const auto a = measure (target, targetRate, hop, fft);
    const auto b = measure (fit, fitRate, hop, fft);

    std::printf ("\n%s  vs  %s\n\n", args.positional[0].toRawUTF8(),
                 args.positional[1].toRawUTF8());
    std::printf ("  %-18s %-23s %-23s %s\n", "", "target", "fit", "verdict");
    std::printf ("  %s\n", juce::String::repeatedString ("-", 88).toRawUTF8());

    const auto cents = (a.f0 > 0.0 && b.f0 > 0.0) ? 1200.0 * std::log2 (b.f0 / a.f0) : 0.0;
    line ("pitch", juce::String (a.f0, 1) + " Hz", juce::String (b.f0, 1) + " Hz",
          std::abs (cents) < 15.0 ? "ok"
              : juce::String (cents > 0 ? "sharp by " : "flat by ")
                  + rounded (std::abs (cents)) + " cents");

    const auto octaves = b.brightnessLog2 - a.brightnessLog2;
    line ("brightness",
          rounded (std::pow (2.0, a.brightnessLog2)) + " Hz",
          rounded (std::pow (2.0, b.brightnessLog2)) + " Hz",
          std::abs (octaves) < 0.15 ? "ok"
              : juce::String (octaves > 0 ? "bright by " : "dull by ")
                  + juce::String (std::abs (octaves), 2) + " oct");

    line ("noisiness", juce::String (a.noisiness, 3), juce::String (b.noisiness, 3),
          verdictFor (a.noisiness, b.noisiness, 0.02, "cleaner", "hissier", ""));

    line ("amplitude wobble",
          juce::String (a.ampWobbleDb, 1) + " dB @" + juce::String (a.ampWobbleRateHz, 1) + "Hz",
          juce::String (b.ampWobbleDb, 1) + " dB @" + juce::String (b.ampWobbleRateHz, 1) + "Hz",
          verdictFor (a.ampWobbleDb, b.ampWobbleDb, 0.8, "too steady", "too wobbly", " dB"));

    line ("pitch wobble",
          rounded (a.pitchWobbleCents) + " cents @" + juce::String (a.pitchWobbleRateHz, 1) + "Hz",
          rounded (b.pitchWobbleCents) + " cents @" + juce::String (b.pitchWobbleRateHz, 1) + "Hz",
          verdictFor (a.pitchWobbleCents, b.pitchWobbleCents, 8.0, "no vibrato", "over-modulated", " cents"));

    line ("attack", juce::String (a.attackSeconds, 3) + " s", juce::String (b.attackSeconds, 3) + " s",
          verdictFor (a.attackSeconds, b.attackSeconds, 0.05, "faster", "slower", " s"));

    line ("note-off", juce::String (a.noteOffSeconds, 2) + " s", juce::String (b.noteOffSeconds, 2) + " s",
          verdictFor (a.noteOffSeconds, b.noteOffSeconds, 0.1, "early", "late", " s"));

    line ("timbre drift", rounded (a.timbreDriftDb, 1) + " dB", rounded (b.timbreDriftDb, 1) + " dB",
          verdictFor (a.timbreDriftDb, b.timbreDriftDb, 1.0, "too static", "too restless", " dB"));

    line ("peak level", juce::String (a.peak, 3), juce::String (b.peak, 3),
          verdictFor (20.0 * std::log10 (std::max (a.peak, 1.0e-6)),
                      20.0 * std::log10 (std::max (b.peak, 1.0e-6)), 2.0,
                      "quiet", "loud", " dB"));

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

    // What the modulation detector made of the target, and where it gave up.
    //
    // Detection is a stack of thresholds and any one of them can veto. Without
    // knowing which, a missed vibrato looks exactly like a sample that has
    // none -- which is how a violin's 24 cents of it went unnoticed.
    const auto trajectories = autosynth::Modulation::extract (target.data(), (int) target.size(),
                                                              targetRate, hop);
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
