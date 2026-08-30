// Analysis conformance against the frozen reference report.
//
// Analysis produces no audio to compare, so the reference implementation was
// validated by dumping every intermediate stage and diffing it element-wise.
// Those dumps are checked in here, and this file replays the comparison
// against the surviving C++ chain.
//
// Tolerances are per-stage rather than global, because the stages do not all
// deserve the same trust. STFT framing is arithmetic and must match exactly;
// f0 is a median over confident frames and can legitimately land a cent away;
// a fitted ADSR is the output of a least-squares search over a noisy envelope
// and only its shape is meaningful.

#include "Helpers.h"

#include "analysis/Grouping.h"
#include "analysis/Partials.h"
#include "analysis/Yin.h"
#include "fit/EnvelopeFit.h"
#include "fit/Modulation.h"
#include "fit/PartialFit.h"
#include "fit/WaveformFit.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>

using namespace autotest;

namespace
{

constexpr int kHop = 256;
constexpr int kFft = 2048;

struct AnalysisCase
{
    juce::String name;
    std::vector<float> input;
    juce::var golden;
};

std::vector<AnalysisCase> loadCases()
{
    static std::vector<AnalysisCase> cases;
    if (! cases.empty())
        return cases;

    const auto manifest = readJson (goldenDir().getChildFile ("manifest.json"));
    const auto* entries = manifest.getProperty ("analysis", {}).getArray();
    REQUIRE (entries != nullptr);

    const auto dir = goldenDir().getChildFile ("analysis");
    for (const auto& entry : *entries)
    {
        AnalysisCase c;
        c.name = entry.getProperty ("name", {}).toString();
        c.input = readWav (dir.getChildFile (entry.getProperty ("input", {}).toString()));
        c.golden = readJson (dir.getChildFile (entry.getProperty ("golden", {}).toString()));
        REQUIRE_FALSE (c.input.empty());
        cases.push_back (std::move (c));
    }
    return cases;
}

std::vector<double> asVector (const juce::var& v)
{
    std::vector<double> out;
    if (const auto* arr = v.getArray())
        for (const auto& item : *arr)
            out.push_back (static_cast<double> (item));
    return out;
}

// Mean absolute difference, ignoring length mismatch beyond the shorter one.
double meanAbsDiff (const std::vector<double>& a, const std::vector<float>& b)
{
    const auto n = juce::jmin (a.size(), b.size());
    if (n == 0)
        return 0.0;
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i)
        acc += std::abs (a[i] - static_cast<double> (b[i]));
    return acc / static_cast<double> (n);
}

double relativeDiff (double a, double b)
{
    const auto scale = juce::jmax (1.0e-9, std::abs (a), std::abs (b));
    return std::abs (a - b) / scale;
}

// The fixtures store enums by name, not by index -- deliberately, so that
// inserting a waveform cannot silently rewrite the meaning of every file
// already on disk. These turn a stored name back into its enum value.
const std::array<const char*, 6> kWaveformNames { "sine", "triangle", "saw", "square",
                                                  "pulse", "noise" };
const std::array<const char*, 4> kLfoDestNames { "none", "pitch", "amp", "cutoff" };

template <size_t N>
int enumIndex (const juce::var& stored, const std::array<const char*, N>& names)
{
    const auto text = stored.toString();
    for (size_t i = 0; i < N; ++i)
        if (text == names[i])
            return (int) i;
    return -1; // an unknown name should fail loudly, not read as index 0
}

} // namespace

TEST_CASE ("analysis fixtures are present", "[golden][analysis]")
{
    REQUIRE (loadCases().size() >= 17);
}

TEST_CASE ("STFT framing matches the reference", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        INFO ("case: " << c.name);
        const auto spec = autosynth::Stft::magnitudeSpectrogram (
            c.input.data(), (int) c.input.size(), kFft, kHop, kSampleRate);

        // Framing is pure arithmetic. Anything but exact equality here means
        // the two chains are not even looking at the same windows.
        CHECK (spec.numFrames == (int) c.golden.getProperty ("num_frames", 0));
        CHECK (spec.numBins == (int) c.golden.getProperty ("num_bins", 0));

        const auto times = asVector (c.golden.getProperty ("stft_times", {}));
        CHECK (meanAbsDiff (times, spec.times) < 1.0e-6);
    }
}

TEST_CASE ("STFT magnitudes match the reference", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        INFO ("case: " << c.name);
        const auto spec = autosynth::Stft::magnitudeSpectrogram (
            c.input.data(), (int) c.input.size(), kFft, kHop, kSampleRate);
        if (spec.numFrames == 0)
            continue;

        const auto expected = asVector (c.golden.getProperty ("stft_mid_frame", {}));
        if (expected.empty())
            continue;

        const auto* frame = spec.frame (spec.numFrames / 2);
        std::vector<float> actual (frame, frame + spec.numBins);

        // Scaling and windowing both show up here directly, rather than only
        // through their effect on later stages.
        CHECK (meanAbsDiff (expected, actual) < 1.0e-4);
    }
}

TEST_CASE ("spectral centroid and loudness match the reference", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        INFO ("case: " << c.name);
        const auto spec = autosynth::Stft::magnitudeSpectrogram (
            c.input.data(), (int) c.input.size(), kFft, kHop, kSampleRate);
        const auto centroid = autosynth::Stft::spectralCentroid (spec);
        const auto loudness = autosynth::Stft::loudnessEnvelope (
            c.input.data(), (int) c.input.size(), kHop);

        CHECK (meanAbsDiff (asVector (c.golden.getProperty ("centroid", {})), centroid) < 1.0);
        CHECK (meanAbsDiff (asVector (c.golden.getProperty ("loudness", {})), loudness) < 1.0e-4);
    }
}

TEST_CASE ("f0 estimation matches the reference", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        INFO ("case: " << c.name);
        double f0 = 0.0, confidence = 0.0;
        autosynth::Yin::estimate (c.input.data(), (int) c.input.size(), kSampleRate,
                                  f0, confidence, kHop);

        const auto expectedF0 = static_cast<double> (c.golden.getProperty ("f0", 0.0));
        if (expectedF0 <= 0.0)
        {
            // Unpitched or silent: the reference declined to commit, and so
            // must this chain.
            CHECK (f0 <= 0.0);
            continue;
        }
        // A cent is 0.06%. Allowing 0.5% permits a different frame landing in
        // the median without permitting a different note.
        CHECK (relativeDiff (f0, expectedF0) < 0.005);

        const auto expectedNote = c.golden.getProperty ("note", {}).toString();
        double cents = 0.0;
        CHECK (autosynth::Yin::noteName (f0, cents) == expectedNote);
    }
}

TEST_CASE ("f0 track matches the reference frame by frame", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        INFO ("case: " << c.name);
        const auto track = autosynth::Yin::track (c.input.data(), (int) c.input.size(),
                                                  kSampleRate, kHop);
        const auto expected = asVector (c.golden.getProperty ("f0_track", {}));
        if (expected.empty())
            continue;

        REQUIRE (track.f0.size() == expected.size());

        // Compared as a fraction of confident frames that agree, not as a mean:
        // one octave-doubled frame in an otherwise perfect track would swamp
        // an average and says nothing about the frames either side of it.
        const auto expectedConf = asVector (c.golden.getProperty ("f0_confidence", {}));
        int confident = 0, agreeing = 0;
        for (size_t i = 0; i < expected.size(); ++i)
        {
            if (expected[i] <= 0.0 || (i < expectedConf.size() && expectedConf[i] < 0.5))
                continue;
            ++confident;
            if (relativeDiff (track.f0[i], expected[i]) < 0.02)
                ++agreeing;
        }
        if (confident < 10)
            continue;
        INFO (agreeing << " of " << confident << " confident frames agree");
        CHECK (agreeing >= static_cast<int> (0.95 * confident));
    }
}

TEST_CASE ("partial tracking matches the reference", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        INFO ("case: " << c.name);
        autosynth::PartialTracker::Options options;
        options.fftSize = kFft;
        options.hop = kHop;
        const auto set = autosynth::PartialTracker::track (
            c.input.data(), (int) c.input.size(), kSampleRate, options);

        const auto expected = (int) c.golden.getProperty ("num_partials", 0);
        INFO ("partials: " << set.partials.size() << " expected " << expected);

        // Exact equality. Partial count is a discrete decision made by
        // thresholds both chains share, so a difference means a threshold
        // drifted rather than that a peak was marginal.
        CHECK ((int) set.partials.size() == expected);

        const auto* golden = c.golden.getProperty ("partials", {}).getArray();
        if (golden == nullptr || set.partials.size() != golden->size())
            continue;

        for (size_t i = 0; i < set.partials.size(); ++i)
        {
            const auto& p = set.partials[i];
            const auto& g = (*golden)[(int) i];
            CHECK (relativeDiff (p.meanFreq(), (double) g.getProperty ("mean_freq", 0.0)) < 0.01);
            CHECK (p.length() == (int) g.getProperty ("length", 0));
        }
    }
}

TEST_CASE ("harmonic grouping matches the reference", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        INFO ("case: " << c.name);
        autosynth::PartialTracker::Options options;
        options.fftSize = kFft;
        options.hop = kHop;
        const auto set = autosynth::PartialTracker::track (
            c.input.data(), (int) c.input.size(), kSampleRate, options);
        const auto groups = autosynth::Grouping::group (set, 3);

        const auto* golden = c.golden.getProperty ("groups", {}).getArray();
        REQUIRE (golden != nullptr);
        INFO ("groups: " << groups.size() << " expected " << golden->size());
        CHECK (groups.size() == golden->size());

        for (size_t i = 0; i < juce::jmin (groups.size(), (size_t) golden->size()); ++i)
        {
            const auto& g = groups[i];
            const auto& e = (*golden)[(int) i];
            INFO ("group " << i);
            CHECK (relativeDiff (g.f0, (double) e.getProperty ("f0", 0.0)) < 0.02);
            CHECK ((int) g.partials.size() == (int) e.getProperty ("num_partials", 0));
            CHECK (g.numHarmonics == (int) e.getProperty ("num_harmonics", 0));

            int voices = 1;
            float detune = 0.0f;
            autosynth::Grouping::estimateUnison (g, voices, detune);
            CHECK (voices == (int) e.getProperty ("unison_voices", 1));
        }
    }
}

TEST_CASE ("gate detection and ADSR fitting match the reference", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        INFO ("case: " << c.name);
        const auto loudness = autosynth::Stft::loudnessEnvelope (
            c.input.data(), (int) c.input.size(), kHop);
        std::vector<float> times (loudness.size());
        for (size_t i = 0; i < loudness.size(); ++i)
            times[i] = static_cast<float> (i * kHop / kSampleRate);

        const auto gate = autosynth::EnvelopeFit::detectGate (loudness, times);
        const auto goldenGate = c.golden.getProperty ("gate", {});
        CHECK (gate.oneShot == (bool) goldenGate.getProperty ("one_shot", false));
        CHECK (std::abs (gate.time - (double) goldenGate.getProperty ("time", 0.0)) < 0.05);

        const auto env = autosynth::EnvelopeFit::fitAdsr (loudness, times, gate.time,
                                                          0.05f, gate.oneShot);
        const auto goldenEnv = c.golden.getProperty ("amp_env", {});
        // Absolute tolerances in seconds: a least-squares fit over a noisy
        // envelope is entitled to a few milliseconds, but not to a different
        // shape.
        CHECK (std::abs (env.attack - (double) goldenEnv.getProperty ("attack", 0.0)) < 0.02);
        CHECK (std::abs (env.decay - (double) goldenEnv.getProperty ("decay", 0.0)) < 0.05);
        CHECK (std::abs (env.sustain - (double) goldenEnv.getProperty ("sustain", 0.0)) < 0.1);
        CHECK (std::abs (env.release - (double) goldenEnv.getProperty ("release", 0.0)) < 0.05);
    }
}

TEST_CASE ("waveform matching matches the reference", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        const auto golden = c.golden.getProperty ("waveform_fit", {});
        if (! golden.isObject())
            continue;
        INFO ("case: " << c.name);

        const auto profile = asVector (golden.getProperty ("profile", {}));
        std::vector<float> p (profile.begin(), profile.end());
        if (p.empty())
            continue;

        // Fed the reference's own profile rather than one recomputed here, so
        // this isolates the matcher from every stage that precedes it.
        const auto match = autosynth::WaveformFit::match (p);
        CHECK ((int) match.waveform == enumIndex (golden.getProperty ("waveform", {}),
                                                  kWaveformNames));
        CHECK (std::abs (match.error - (double) golden.getProperty ("error", 0.0)) < 1.0e-3);
    }
}

TEST_CASE ("LFO detection matches the reference", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        INFO ("case: " << c.name);
        const auto trajectories = autosynth::Modulation::extract (
            c.input.data(), (int) c.input.size(), kSampleRate, kHop);
        const auto lfos = autosynth::Modulation::bestSeveral (trajectories);

        const auto* golden = c.golden.getProperty ("lfos", {}).getArray();
        REQUIRE (golden != nullptr);
        INFO ("detected " << lfos.size() << " expected " << golden->size());
        CHECK (lfos.size() == golden->size());

        for (size_t i = 0; i < juce::jmin (lfos.size(), (size_t) golden->size()); ++i)
        {
            const auto& e = (*golden)[(int) i];
            CHECK ((int) lfos[i].dest == enumIndex (e.getProperty ("dest", {}), kLfoDestNames));
            // Rate comes from a periodogram peak; a bin either way is fine,
            // a different rate is not.
            CHECK (relativeDiff (lfos[i].rateHz, (double) e.getProperty ("rate_hz", 0.0)) < 0.1);
        }
    }
}

TEST_CASE ("assembled patch matches the reference", "[golden][analysis]")
{
    for (const auto& c : loadCases())
    {
        INFO ("case: " << c.name);
        // Deliberately without a renderer, so this compares the deterministic
        // half of the fit and only that. Level calibration is closed-loop
        // around Vital, and the reference these fixtures come from closed its
        // loop around a different synth -- holding the two to the same numbers
        // would be asking two synths to agree, not two implementations.
        autosynth::PartialFit::Options options;
        options.hop = kHop;
        const auto fitted = autosynth::PartialFit::fit (
            c.input.data(), (int) c.input.size(), kSampleRate, options);

        const auto golden = c.golden.getProperty ("fitted", {});
        REQUIRE (golden.isObject());

        const auto expectedRoot = (double) golden.getProperty ("root_hz", 0.0);
        if (expectedRoot > 0.0)
            CHECK (relativeDiff (fitted.rootHz, expectedRoot) < 0.02);

        const auto* oscs = golden.getProperty ("oscs", {}).getArray();
        if (oscs == nullptr)
            continue;

        // Structure is compared over oscillators that carry audible weight,
        // not over every oscillator with a non-zero level.
        //
        // The reason is a measured one. Level calibration ends in a
        // non-negative least squares over near-collinear columns -- a third
        // "source" that is really an analysis artefact is barely
        // distinguishable from the two real ones -- and on such a system tiny
        // input differences move the answer. The reference implementation
        // itself is inconsistent at that scale: it solved one marginal
        // oscillator to exactly 0.0 and kept another, of the same magnitude
        // (~0.005, some 46 dB down), at a non-zero level. Demanding that two
        // implementations agree there would be demanding that they share a
        // rounding pattern, not that they agree about the sound.
        //
        // What must agree is the structure a listener could hear.
        const auto significantCount = [] (const auto& patch)
        {
            float loudest = 0.0f;
            for (const auto& o : patch.oscs)
                if (o.enabled)
                    loudest = juce::jmax (loudest, o.level);
            if (loudest <= 0.0f)
                return 0;
            int n = 0;
            for (const auto& o : patch.oscs)
                if (o.enabled && o.level > 0.01f * loudest)
                    ++n;
            return n;
        };

        float goldenLoudest = 0.0f;
        for (int i = 0; i < oscs->size(); ++i)
            if ((bool) (*oscs)[i].getProperty ("enabled", false))
                goldenLoudest = juce::jmax (goldenLoudest,
                                            (float) (double) (*oscs)[i].getProperty ("level", 0.0));

        int goldenSignificant = 0;
        for (int i = 0; i < oscs->size(); ++i)
            if ((bool) (*oscs)[i].getProperty ("enabled", false)
                && (float) (double) (*oscs)[i].getProperty ("level", 0.0) > 0.01f * goldenLoudest)
                ++goldenSignificant;

        INFO ("significant oscs: " << significantCount (fitted)
              << " expected " << goldenSignificant);
        CHECK (significantCount (fitted) == goldenSignificant);

        // The total count may differ by the marginal oscillator described
        // above, but not by more -- a genuine structural disagreement about
        // how many sources are present would show up here.
        CHECK (std::abs (fitted.activeOscCount()
                         - (int) golden.getProperty ("active_oscs", 0)) <= 1);

        for (int i = 0; i < autosynth::kNumOsc && i < oscs->size(); ++i)
        {
            const auto& e = (*oscs)[i];
            const auto level = (float) (double) e.getProperty ("level", 0.0);
            if (! (bool) e.getProperty ("enabled", false) || level <= 0.01f * goldenLoudest)
                continue;

            INFO ("osc " << i);
            CHECK (fitted.oscs[(size_t) i].enabled);
            CHECK ((int) fitted.oscs[(size_t) i].waveform
                   == enumIndex (e.getProperty ("waveform", {}), kWaveformNames));
            CHECK (fitted.oscs[(size_t) i].semitones == (int) e.getProperty ("semitones", 0));
        }
    }
}
