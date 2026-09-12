#include "eval/Diagnose.h"

#include "analysis/Grouping.h"
#include "analysis/Partials.h"
#include "analysis/Stft.h"
#include "analysis/Yin.h"
#include "fit/EnvelopeFit.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace autosynth
{

namespace
{

// Energy-weighted mean harmonic profile over a frame range, peak-normalised.
std::vector<double> profileOver (const HarmonicGroup& g, int from, int to)
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

// juce::String (double, 0) prints "as many digits as needed", not none, which
// turns a tidy column into 23.7975.
juce::String rounded (double v, int decimals = 0)
{
    if (decimals <= 0)
        return juce::String (static_cast<juce::int64> (std::llround (v)));
    return juce::String (v, decimals);
}

// A verdict is only useful if it says which way, and by how much.
Diagnose::Axis axis (const char* name, const juce::String& target, const juce::String& fit,
                     double targetValue, double fitValue, double tolerance,
                     const char* lowWord, const char* highWord, const char* unit)
{
    const auto delta = fitValue - targetValue;
    const auto magnitude = std::abs (delta);

    Diagnose::Axis out;
    out.name = name;
    out.target = target;
    out.fit = fit;
    out.ok = magnitude <= tolerance;
    out.verdict = out.ok
                    ? juce::String ("ok")
                    : juce::String (delta > 0 ? highWord : lowWord) + " by "
                          + (magnitude < 10.0 ? juce::String (magnitude, 2) : rounded (magnitude))
                          + unit;
    return out;
}

} // namespace

Diagnose::Measurements Diagnose::measure (const std::vector<float>& x, double sampleRate,
                                          const Options& options)
{
    Measurements m;
    if (x.empty())
        return m;

    const auto n = static_cast<int> (x.size());
    const auto hop = options.hop;
    const auto fft = options.fft;
    const auto fps = sampleRate / hop;
    m.secondsPerFrame = 1.0 / fps;

    for (auto v : x)
        m.peak = std::max (m.peak, std::abs ((double) v));

    double confidence = 0.0;
    Yin::estimate (x.data(), n, sampleRate, m.f0, confidence, hop);

    const auto spectrogram = Stft::magnitudeSpectrogram (x.data(), n, fft, hop, sampleRate);
    const auto centroid = Stft::spectralCentroid (spectrogram);
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
    PartialTracker::Options trackOptions;
    trackOptions.fftSize = fft;
    trackOptions.hop = hop;
    const auto partials = PartialTracker::track (x.data(), n, sampleRate, trackOptions);
    const auto groups = Grouping::group (partials, 3);
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
    m.loudness = Stft::loudnessEnvelope (x.data(), n, hop);
    const auto& loudness = m.loudness;
    std::vector<float> times (loudness.size());
    for (size_t i = 0; i < loudness.size(); ++i)
        times[i] = static_cast<float> (i / fps);

    const auto gate = EnvelopeFit::detectGate (loudness, times);
    m.noteOffSeconds = gate.time;
    m.attackSeconds = EnvelopeFit::attackSeconds (loudness, times);

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

        // The level the note holds, against the loudest it ever reaches. Taken
        // as a median so one crest cannot stand for the sustain, over the same
        // window the wobble is measured in.
        std::vector<float> sustained (loudness.begin() + (std::ptrdiff_t) lo,
                                      loudness.begin() + (std::ptrdiff_t) hi);
        std::sort (sustained.begin(), sustained.end());
        const auto held = sustained[sustained.size() / 2];
        const auto loudestFrame = *std::max_element (loudness.begin(), loudness.end());
        if (loudestFrame > 1.0e-9f && held > 1.0e-9f)
            m.sustainToPeakDb = 20.0 * std::log10 (held / loudestFrame);
    }

    {
        const auto loudestFrame = *std::max_element (loudness.begin(), loudness.end());
        const auto at = static_cast<size_t> (0.05 * fps);
        if (loudestFrame > 1.0e-9f && at < loudness.size())
            m.onsetAt50ms = loudness[at] / loudestFrame;

        // And the rest of the rise, because one point cannot say what shape it
        // is. An attack that is late at fifty milliseconds and an attack that
        // is the wrong *shape* report the same single number, and only one of
        // them can be fixed by turning a curve up.
        for (const auto ms : kRiseMilliseconds)
        {
            const auto frame = static_cast<size_t> (ms / 1000.0 * fps);
            m.rise.push_back (loudestFrame > 1.0e-9f && frame < loudness.size()
                                  ? loudness[frame] / loudestFrame
                                  : 0.0);
        }
    }

    const auto pitch = Yin::track (x.data(), n, sampleRate, hop);
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

std::vector<Diagnose::Axis> Diagnose::compare (const Measurements& a, const Measurements& b)
{
    std::vector<Axis> axes;

    // Pitch is compared in cents rather than hertz, because fifteen cents is
    // the same error at every note and fifteen hertz is not.
    {
        const auto cents = (a.f0 > 0.0 && b.f0 > 0.0) ? 1200.0 * std::log2 (b.f0 / a.f0) : 0.0;
        Axis pitch;
        pitch.name = "pitch";
        pitch.target = juce::String (a.f0, 1) + " Hz";
        pitch.fit = juce::String (b.f0, 1) + " Hz";
        pitch.ok = std::abs (cents) < 15.0;
        pitch.verdict = pitch.ok ? juce::String ("ok")
                                 : juce::String (cents > 0 ? "sharp by " : "flat by ")
                                       + rounded (std::abs (cents)) + " cents";
        axes.push_back (pitch);
    }

    {
        const auto octaves = b.brightnessLog2 - a.brightnessLog2;
        Axis brightness;
        brightness.name = "brightness";
        brightness.target = rounded (std::pow (2.0, a.brightnessLog2)) + " Hz";
        brightness.fit = rounded (std::pow (2.0, b.brightnessLog2)) + " Hz";
        brightness.ok = std::abs (octaves) < 0.15;
        brightness.verdict = brightness.ok
                               ? juce::String ("ok")
                               : juce::String (octaves > 0 ? "bright by " : "dull by ")
                                     + juce::String (std::abs (octaves), 2) + " oct";
        axes.push_back (brightness);
    }

    axes.push_back (axis ("noisiness",
                          juce::String (a.noisiness, 3), juce::String (b.noisiness, 3),
                          a.noisiness, b.noisiness, 0.02, "cleaner", "hissier", ""));

    axes.push_back (axis ("amplitude wobble",
                          juce::String (a.ampWobbleDb, 1) + " dB @"
                              + juce::String (a.ampWobbleRateHz, 1) + "Hz",
                          juce::String (b.ampWobbleDb, 1) + " dB @"
                              + juce::String (b.ampWobbleRateHz, 1) + "Hz",
                          a.ampWobbleDb, b.ampWobbleDb, 0.8,
                          "too steady", "too wobbly", " dB"));

    axes.push_back (axis ("pitch wobble",
                          rounded (a.pitchWobbleCents) + " cents @"
                              + juce::String (a.pitchWobbleRateHz, 1) + "Hz",
                          rounded (b.pitchWobbleCents) + " cents @"
                              + juce::String (b.pitchWobbleRateHz, 1) + "Hz",
                          a.pitchWobbleCents, b.pitchWobbleCents, 8.0,
                          "no vibrato", "over-modulated", " cents"));

    axes.push_back (axis ("attack",
                          juce::String (a.attackSeconds, 3) + " s",
                          juce::String (b.attackSeconds, 3) + " s",
                          a.attackSeconds, b.attackSeconds, 0.05, "faster", "slower", " s"));

    axes.push_back (axis ("note-off",
                          juce::String (a.noteOffSeconds, 2) + " s",
                          juce::String (b.noteOffSeconds, 2) + " s",
                          a.noteOffSeconds, b.noteOffSeconds, 0.1, "early", "late", " s"));

    axes.push_back (axis ("timbre drift",
                          rounded (a.timbreDriftDb, 1) + " dB",
                          rounded (b.timbreDriftDb, 1) + " dB",
                          a.timbreDriftDb, b.timbreDriftDb, 1.0,
                          "too static", "too restless", " dB"));

    axes.push_back (axis ("sustain vs peak",
                          juce::String (a.sustainToPeakDb, 1) + " dB",
                          juce::String (b.sustainToPeakDb, 1) + " dB",
                          a.sustainToPeakDb, b.sustainToPeakDb, 1.5,
                          "drops further", "drops less", " dB"));

    axes.push_back (axis ("onset at 50 ms",
                          juce::String (a.onsetAt50ms, 2), juce::String (b.onsetAt50ms, 2),
                          a.onsetAt50ms, b.onsetAt50ms, 0.12,
                          "arrives later", "arrives sooner", ""));

    // Compared in decibels though it is printed as a peak, because two levels
    // are a ratio and a difference of amplitudes is not a tolerance anyone can
    // state.
    axes.push_back (axis ("peak level",
                          juce::String (a.peak, 3), juce::String (b.peak, 3),
                          20.0 * std::log10 (std::max (a.peak, 1.0e-6)),
                          20.0 * std::log10 (std::max (b.peak, 1.0e-6)), 2.0,
                          "quiet", "loud", " dB"));

    return axes;
}

} // namespace autosynth
