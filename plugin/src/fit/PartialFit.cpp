#include "fit/PartialFit.h"

#include "analysis/Grouping.h"
#include "analysis/Partials.h"
#include "analysis/Roles.h"
#include "analysis/Stft.h"
#include "analysis/Yin.h"
#include "fit/EffectsFit.h"
#include "fit/EnvelopeFit.h"
#include "fit/FilterFit.h"
#include "fit/NdFilters.h"
#include "fit/Modulation.h"
#include "fit/Nmf.h"
#include "fit/Nnls.h"
#include "fit/WaveformFit.h"
#include "fit/WavetableFit.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace autosynth
{
namespace
{

// Smallest power-of-two window that still resolves f0, matching
// analysis.harmonics.window_for_f0. Six bins per harmonic spacing: three was
// tried and measurably degraded waveform recovery, because a Hann main lobe is
// about four bins wide and the peak-picking window then overlaps its
// neighbour's leakage.
int windowForF0 (double f0Hz, double sampleRate, int lo = 1024, int hi = 8192)
{
    if (f0Hz <= 0.0)
        return hi;
    const auto needed = 6.0 * sampleRate / f0Hz;
    auto n = lo;
    while (n < needed && n < hi)
        n *= 2;
    return juce::jlimit (lo, hi, n);
}

// Energy-weighted mean harmonic profile, peak-normalised. Weighted so the quiet
// tail -- where partial amplitudes are least reliable -- does not get an equal
// vote in the waveform decision.
std::vector<float> meanProfile (const std::vector<float>& H, int numHarmonics, int numFrames)
{
    std::vector<float> profile (static_cast<size_t> (juce::jmax (1, numHarmonics)), 1.0f);
    if (numHarmonics <= 0 || numFrames <= 0)
        return profile;

    std::vector<double> frameWeight (static_cast<size_t> (numFrames), 0.0);
    double total = 0.0;
    for (int t = 0; t < numFrames; ++t)
    {
        for (int k = 0; k < numHarmonics; ++k)
            frameWeight[static_cast<size_t> (t)] +=
                H[static_cast<size_t> (k) * numFrames + t];
        total += frameWeight[static_cast<size_t> (t)];
    }
    if (total <= 1.0e-12)
        return profile;

    for (int k = 0; k < numHarmonics; ++k)
    {
        double acc = 0.0;
        for (int t = 0; t < numFrames; ++t)
            acc += H[static_cast<size_t> (k) * numFrames + t]
                 * frameWeight[static_cast<size_t> (t)] / total;
        profile[static_cast<size_t> (k)] = static_cast<float> (acc);
    }

    const auto peak = *std::max_element (profile.begin(), profile.end());
    if (peak > 1.0e-12f)
        for (auto& v : profile)
            v /= peak;
    return profile;
}

std::vector<double> spectralFeatures (const float* samples, int numSamples, double sampleRate)
{
    const auto spectrogram = Stft::magnitudeSpectrogram (samples, numSamples, 1024, 512, sampleRate);
    std::vector<double> out (spectrogram.magnitude.size());
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = spectrogram.magnitude[i];
    return out;
}

// Split one harmonic group into the oscillators it actually contains.
//
// Grouping asks which partials share a fundamental and stops there, so two
// sources an octave apart arrive as one group -- the upper one's partials are
// all harmonics of the lower one's fundamental. That is every remaining
// under-count on the recovery harness.
//
// Inside the group they are still separate *components*: one oscillator makes a
// rank-one matrix, a fixed spectral profile times one envelope, and two make it
// rank two whatever their interval. So the question is asked here, per group,
// where it is answerable -- not globally on a spectrogram, where the octave case
// is hopeless and a fifth is not even sampled.
//
// A component's own fundamental comes out of *which* harmonics it uses. Energy
// only on even harmonics of f0 means the component is really at 2*f0, and its
// third harmonic is the group's sixth. The greatest common divisor of the
// harmonics carrying real weight recovers that, and generalises past octaves to
// twelfths and double octaves, which the harness also gets wrong.
//
// `flat` is the group's harmonic matrix with the filter already divided out,
// and the rank is read from *that* rather than from the raw one. The order
// matters as much here as it does for the waveform: a filter sweep tilts the
// spectrum over time, which is genuinely not rank one, so a single oscillator
// behind a moving filter factorises as two. Measured, that is what it did -- a
// clarinet came back as two oscillators at the same pitch and a violin as
// three, which is a worse description of both than "one".
//
// The components are then used as a *mask* on the original matrix rather than
// as a reconstruction of it, so each sub-group is still in the un-deconvolved
// domain the rest of the fitter expects, and the filter gets divided out of it
// once, downstream, exactly as before.
std::vector<HarmonicGroup> splitByRank (const HarmonicGroup& group,
                                        const std::vector<float>& flat, int maxComponents)
{
    if (maxComponents <= 1 || group.numHarmonics < 4 || group.numFrames < 8
        || flat.size() != group.H.size())
        return { group };

    const auto choice = nmf::selectRank (flat, group.numHarmonics, group.numFrames,
                                         maxComponents);
    if (choice.rank <= 1)
        return { group };

    const auto factors = nmf::factorise (flat, group.numHarmonics, group.numFrames,
                                         choice.rank);
    if (factors.rank != choice.rank)
        return { group };

    const auto rows = static_cast<size_t> (group.numHarmonics);
    const auto cols = static_cast<size_t> (group.numFrames);

    std::vector<HarmonicGroup> out;
    std::vector<int> divisors;
    for (int c = 0; c < choice.rank; ++c)
    {
        const auto* profile = factors.w.data() + static_cast<size_t> (c) * rows;
        const auto* activation = factors.h.data() + static_cast<size_t> (c) * cols;

        // Which harmonics this component actually *owns*.
        //
        // Not merely "has energy at": the factorisation leaks, and it leaks
        // worst onto the low harmonics where the other component is loudest. A
        // source a twelfth up came back with 1.00 on harmonic three and 0.13 on
        // harmonic one, and the greatest common divisor of {1, 3} is 1 -- so
        // the component that was plainly at 3*f0 claimed to be at f0, and the
        // rule below then discarded a correct split. A harmonic counts only
        // where this component is the dominant explanation for it.
        std::vector<float> componentMean (static_cast<size_t> (choice.rank), 0.0f);
        for (int r = 0; r < choice.rank; ++r)
        {
            double acc = 0.0;
            for (size_t t = 0; t < cols; ++t)
                acc += factors.h[static_cast<size_t> (r) * cols + t];
            componentMean[static_cast<size_t> (r)] = static_cast<float> (acc / juce::jmax<size_t> (cols, 1));
        }

        auto divisor = 0;
        for (int k = 0; k < group.numHarmonics; ++k)
        {
            if (profile[static_cast<size_t> (k)] <= 0.25f)
                continue;

            const auto mine = profile[static_cast<size_t> (k)] * componentMean[static_cast<size_t> (c)];
            auto owned = true;
            for (int r = 0; r < choice.rank && owned; ++r)
                if (r != c)
                    owned = mine >= factors.w[static_cast<size_t> (r) * rows + static_cast<size_t> (k)]
                                   * componentMean[static_cast<size_t> (r)];
            if (owned)
                divisor = divisor == 0 ? k + 1 : std::gcd (divisor, k + 1);
        }
        divisor = juce::jmax (1, divisor);

        HarmonicGroup sub;
        sub.f0 = group.f0 * divisor;
        sub.salience = group.salience;
        sub.numFrames = group.numFrames;
        sub.frameRateHz = group.frameRateHz;
        sub.numHarmonics = juce::jmax (1, group.numHarmonics / divisor);
        sub.H.assign (static_cast<size_t> (sub.numHarmonics) * cols, 0.0f);

        for (int j = 0; j < sub.numHarmonics; ++j)
        {
            const auto k = (j + 1) * divisor - 1;
            if (k >= group.numHarmonics)
                break;

            for (size_t t = 0; t < cols; ++t)
            {
                // Soft mask: this component's share of the bin, applied to what
                // was actually measured. Using the rank-one reconstruction
                // instead would hand the rest of the fitter a matrix the
                // factorisation invented, and every envelope and profile read
                // off it afterwards would be a fit to a fit.
                float total = 0.0f;
                for (int r = 0; r < choice.rank; ++r)
                    total += factors.w[static_cast<size_t> (r) * rows + static_cast<size_t> (k)]
                           * factors.h[static_cast<size_t> (r) * cols + t];

                const auto mine = profile[static_cast<size_t> (k)] * activation[t];
                const auto share = total > 1.0e-9f ? mine / total : 0.0f;
                sub.H[static_cast<size_t> (j) * cols + t] =
                    group.H[static_cast<size_t> (k) * cols + t] * share;
            }
        }

        // Partials, so the unison estimator still has beating to read. Only the
        // ones this component's fundamental actually claims.
        for (size_t p = 0; p < group.partials.size(); ++p)
        {
            const auto k = static_cast<int> (std::lround (group.partials[p].meanFreq() / group.f0));
            if (k >= divisor && k % divisor == 0)
            {
                sub.partials.push_back (group.partials[p]);
                sub.harmonicIndices.push_back (k / divisor);
            }
        }

        if (sub.energy() > 0.0f)
        {
            divisors.push_back (divisor);
            out.push_back (std::move (sub));
        }
    }

    // Two sources also have two *envelopes*.
    //
    // Different harmonic sets alone are not enough. A clarinet's spectrum is
    // odd-harmonic dominated, so a factorisation can split it into an "odd" and
    // an "even" component -- and the even one, using harmonics 2, 4, 6, looks
    // exactly like a source an octave up. What gives it away is that the two
    // rise and fall together: they are one instrument, and one instrument has
    // one envelope. Two players do not.
    //
    // Measured, this is the guard that matters on real material: without it a
    // single clarinet note came back as three oscillators.
    constexpr float kMaxEnvelopeCorrelation = 0.90f;
    for (int a = 0; a < choice.rank; ++a)
        for (int b = a + 1; b < choice.rank; ++b)
        {
            const auto* ha = factors.h.data() + static_cast<size_t> (a) * cols;
            const auto* hb = factors.h.data() + static_cast<size_t> (b) * cols;

            double meanA = 0.0, meanB = 0.0;
            for (size_t t = 0; t < cols; ++t)
            {
                meanA += ha[t];
                meanB += hb[t];
            }
            meanA /= static_cast<double> (cols);
            meanB /= static_cast<double> (cols);

            double num = 0.0, da = 0.0, db = 0.0;
            for (size_t t = 0; t < cols; ++t)
            {
                const auto x = ha[t] - meanA, y = hb[t] - meanB;
                num += x * y;
                da += x * x;
                db += y * y;
            }
            const auto denom = std::sqrt (da * db);
            const auto correlation = denom > 1.0e-12 ? num / denom : 1.0;
            if (correlation > kMaxEnvelopeCorrelation)
                return { group };
        }

    // Only a split into *different fundamentals* is a split into sources.
    //
    // Two components sharing one fundamental are not two oscillators, they are
    // one oscillator whose timbre moves -- and that already has a model, the
    // wavetable frames, which describe it in sixteen numbers instead of a whole
    // second oscillator. Without this rule the two models compete for the same
    // evidence and the more expensive one wins by default: a clarinet with 4.3
    // dB of measured drift came back as two oscillators at the same pitch, and
    // a violin as three. It is the same discipline as the release and the
    // reverb -- two mechanisms that explain one observation have to be
    // separated by structure, never by whichever fits marginally better.
    const auto sameFundamental = std::adjacent_find (divisors.begin(), divisors.end(),
                                                     std::not_equal_to<>()) == divisors.end();
    if (sameFundamental)
        return { group };

    return out.size() >= 2 ? out : std::vector<HarmonicGroup> { group };
}

} // namespace

Patch PartialFit::calibrateLevels (Patch patch, const float* target, int numSamples,
                                   double sampleRate, double gateSeconds,
                                   const Renderer& renderer, float noiseCeiling)
{
    std::vector<int> active;
    for (int i = 0; i < kNumOsc; ++i)
        if (patch.oscs[static_cast<size_t> (i)].enabled
            && patch.oscs[static_cast<size_t> (i)].level > 1.0e-4f)
            active.push_back (i);
    if (active.empty())
        return patch;

    const auto duration = numSamples / sampleRate;
    const auto hasNoise = patch.noiseLevel > 1.0e-6f;
    const auto numColumns = static_cast<int> (active.size()) + (hasNoise ? 1 : 0);

    if (! renderer)
        return patch;

    const auto renderSolo = [&] (int oscIndex, bool noiseOnly)
    {
        auto probe = patch;
        probe.masterLevel = 1.0f;
        probe.noiseLevel = 0.0f;
        for (auto& osc : probe.oscs)
        {
            osc.enabled = false;
            osc.level = 0.0f;
        }
        if (noiseOnly)
            probe.noiseLevel = 1.0f;
        else
        {
            probe.oscs[static_cast<size_t> (oscIndex)].enabled = true;
            probe.oscs[static_cast<size_t> (oscIndex)].level = 1.0f;
        }

        const auto rendered = renderer (probe, duration, gateSeconds);
        return spectralFeatures (rendered.data(), (int) rendered.size(), sampleRate);
    };

    std::vector<std::vector<double>> columns;
    columns.reserve (static_cast<size_t> (numColumns));
    for (auto index : active)
        columns.push_back (renderSolo (index, false));
    if (hasNoise)
        columns.push_back (renderSolo (0, true));

    const auto observed = spectralFeatures (target, numSamples, sampleRate);

    size_t rows = observed.size();
    for (const auto& c : columns)
        rows = std::min (rows, c.size());
    if (rows == 0)
        return patch;

    std::vector<double> A (rows * static_cast<size_t> (numColumns));
    for (int j = 0; j < numColumns; ++j)
        std::copy (columns[static_cast<size_t> (j)].begin(),
                   columns[static_cast<size_t> (j)].begin() + static_cast<long> (rows),
                   A.begin() + static_cast<long> (j * rows));

    std::vector<double> b (observed.begin(), observed.begin() + static_cast<long> (rows));
    const auto gains = nnls::solve (A, b, static_cast<int> (rows), numColumns);

    // Scale by the loudest *oscillator*, never by the noise gain. Broadband
    // noise has energy in every bin, so in a magnitude-domain least squares it
    // can absorb an arbitrary amount of a poorly-fitted harmonic sound -- which
    // drove every oscillator level to zero and switched whole patches off.
    double scale = 0.0;
    for (size_t i = 0; i < active.size(); ++i)
        scale = std::max (scale, gains[i]);
    if (scale <= 1.0e-9)
        return patch;

    for (size_t i = 0; i < active.size(); ++i)
        patch.oscs[static_cast<size_t> (active[i])].level =
            static_cast<float> (juce::jlimit (0.0, 1.0, gains[i] / scale));
    if (hasNoise)
        patch.noiseLevel = static_cast<float> (juce::jlimit (0.0, 1.0, gains.back() / scale));
    patch.masterLevel = static_cast<float> (juce::jlimit (0.0, (double) kMaxMasterLevel, scale));

    // A source solved to zero is one the analysis proposed and the mix
    // rejected. The loudest is always kept: "no oscillators" is never a better
    // description of a pitched sound than one oscillator.
    size_t loudest = 0;
    for (size_t i = 1; i < active.size(); ++i)
        if (gains[i] > gains[loudest])
            loudest = i;
    for (size_t i = 0; i < active.size(); ++i)
        if (i != loudest && patch.oscs[static_cast<size_t> (active[i])].level <= 1.0e-3f)
            patch.oscs[static_cast<size_t> (active[i])].enabled = false;

    // Noise is a colouring on a pitched patch, never a co-equal source.
    //
    // The least-squares solve will happily hand the noise column a large gain,
    // because broadband energy fits the low-level content a real recording is
    // full of -- room tone, bow and breath noise, the diffuse part of a reverb.
    // On a violin it came back at 0.18 and the result hissed; refinement then
    // pushed it to 0.41 and the patch was unlistenable, which no distance
    // metric here objected to because filling empty bins genuinely lowers a
    // log-spectral error.
    //
    // The mistake is modelling: our noise is flat, static and untuned, so it
    // stands in badly for content that is shaped and correlated with the note.
    // Until it is a better model, a pitched fit gets a bounded amount of it.
    // A sound that really is broadband takes the other branch in `fit`, which
    // proposes no oscillators at all and is not capped here.
    patch.noiseLevel = juce::jmin (patch.noiseLevel, noiseCeiling);

    return patch;
}

Patch PartialFit::calibrateNoise (Patch patch, const float* target, int numSamples,
                                  double sampleRate, double gateSeconds, float ceiling,
                                  const Renderer& renderer)
{
    if (ceiling <= 1.0e-6f || numSamples <= 0 || patch.rootHz <= 20.0f)
    {
        patch.noiseLevel = 0.0f;
        return patch;
    }

    const auto wanted = Roles::noiseShare (target, numSamples, sampleRate, patch.rootHz);

    // Only when the target is *substantially* noisier, because this measurement
    // cannot tell hiss from smearing.
    //
    // A vibrato'd oscillator spreads energy between its own harmonics, so a
    // perfectly clean synthetic target still measures inter-harmonic energy --
    // and matching that number means adding hiss to imitate a wobble. Measured
    // on the recovery harness, whose targets are generated noise-free, chasing
    // it without this margin cost 0.10 octaves of brightness. A violin reads
    // 0.29 against a fit's 0.10 and clears it easily; smearing does not.
    constexpr double kNoiseFloor = 0.05;
    if (wanted <= kNoiseFloor)
    {
        patch.noiseLevel = 0.0f;
        return patch;
    }

    if (! renderer)
    {
        patch.noiseLevel = 0.0f;
        return patch;
    }

    const auto duration = numSamples / sampleRate;

    const auto renderedShare = [&] (const Patch& candidate)
    {
        const auto rendered = renderer (candidate, duration, gateSeconds);
        return Roles::noiseShare (rendered.data(), (int) rendered.size(),
                                  sampleRate, candidate.rootHz);
    };

    // Start from something audible rather than from whatever the level solve
    // left, so the first measurement is informative even when it left nothing.
    patch.noiseLevel = juce::jlimit (1.0e-3f, ceiling, juce::jmax (patch.noiseLevel, 0.02f));

    for (int pass = 0; pass < 3; ++pass)
    {
        const auto have = renderedShare (patch);
        if (have <= 1.0e-6)
            break;

        // A source's own inharmonic content is already in there, so the noise
        // can only ever add. Asking it to remove any is a sign the fit is
        // hissing for some other reason, and the right answer is silence.
        if (have >= wanted * 0.75)
        {
            patch.noiseLevel = pass == 0 ? 0.0f : patch.noiseLevel;
            break;
        }

        const auto scale = std::sqrt (wanted * (1.0 - have) / juce::jmax (have * (1.0 - wanted), 1.0e-9));
        patch.noiseLevel = juce::jlimit (0.0f, ceiling,
                                         static_cast<float> (patch.noiseLevel * scale));
        if (patch.noiseLevel >= ceiling)
            break;
    }

    return patch;
}

Patch PartialFit::fit (const float* samples, int numSamples, double sampleRate,
                       const Options& options)
{
    Patch patch;
    patch.name = "partialfit";
    for (auto& osc : patch.oscs)
    {
        osc.enabled = false;
        osc.level = 0.0f;
    }

    const auto rms = Stft::loudnessEnvelope (samples, numSamples, options.hop);
    std::vector<float> times (rms.size());
    for (size_t i = 0; i < rms.size(); ++i)
        times[i] = static_cast<float> (i * options.hop / sampleRate);

    double gateTime;
    bool oneShot;
    if (options.gateSeconds >= 0.0)
    {
        gateTime = options.gateSeconds;
        oneShot = false;
    }
    else
    {
        const auto detected = EnvelopeFit::detectGate (rms, times);
        gateTime = detected.time;
        oneShot = detected.oneShot;
    }

    const auto globalEnv = EnvelopeFit::fitAdsr (rms, times, gateTime, 0.05f, oneShot);

    float peak = 0.0f;
    for (int i = 0; i < numSamples; ++i)
        peak = juce::jmax (peak, std::abs (samples[i]));
    // The target's own peak, as a starting guess. It is only a guess: the peak
    // a patch *reaches* is this times whatever the envelope and filter leave,
    // so calibration and refinement both have room above one to make that up.
    patch.masterLevel = juce::jlimit (0.05f, kMaxMasterLevel, peak);

    double f0Hint = 0.0, confidence = 0.0;
    Yin::estimate (samples, numSamples, sampleRate, f0Hint, confidence, options.hop);

    PartialTracker::Options trackOptions;
    trackOptions.hop = options.hop;
    trackOptions.fftSize = windowForF0 (f0Hint > 0.0 ? f0Hint : 100.0, sampleRate);
    const auto partialSet = PartialTracker::track (samples, numSamples, sampleRate, trackOptions);

    const auto groups = Grouping::group (partialSet, options.maxOscillators, options.tolCents);

    if (groups.empty() || f0Hint <= 0.0 || confidence <= 0.0)
    {
        // Nothing periodic to model. This used to set a bare global noise level
        // and switch the filter off, which describes a cymbal and a snare and a
        // breath the same way: flat, static, and lasting exactly as long as the
        // amplitude envelope says.
        //
        // As an *oscillator* it inherits everything an oscillator has. The
        // envelope shapes it, the filter colours it, and both are already
        // fitted by the code below -- so a noise source now gets a measured
        // attack and a measured brightness instead of neither.
        // Silence is not unpitched material, it is nothing, and a noise
        // oscillator at any level is a worse description of it than an empty
        // patch. `masterLevel` has already been clamped to a floor by this
        // point, so the peak is measured again here rather than read back.
        auto loudest = 0.0f;
        for (int i = 0; i < numSamples; ++i)
            loudest = juce::jmax (loudest, std::abs (samples[i]));
        if (loudest < 1.0e-3f)
        {
            patch.noiseLevel = 0.0f;
            patch.filter.type = FilterType::off;
            patch.ampEnv = globalEnv;
            return patch;
        }

        auto& osc = patch.oscs[0];
        osc.enabled = true;
        osc.waveform = Waveform::noise;
        osc.waveformB = Waveform::noise;
        osc.waveMorph = 0.0f;
        osc.level = 1.0f;
        patch.noiseLevel = 0.0f;
        patch.ampEnv = globalEnv;

        // Brightness, from where the energy actually is. A one-pole at the
        // spectral centroid is a crude match to a shaped noise floor, and it is
        // a great deal less crude than no filter at all.
        const auto spectrogram = Stft::magnitudeSpectrogram (samples, numSamples, 2048,
                                                             options.hop, sampleRate);
        const auto centroid = Stft::spectralCentroid (spectrogram);
        double sum = 0.0;
        int counted = 0;
        for (const auto c : centroid)
            if (c > 20.0f)
            {
                sum += c;
                ++counted;
            }

        patch.filter.type = FilterType::lowpass;
        patch.filter.cutoffHz = counted > 0
                              ? juce::jlimit (30.0f, 18000.0f,
                                              static_cast<float> (sum / counted) * 2.0f)
                              : 18000.0f;
        patch.filter.resonance = FilterFit::kDefaultQ;
        patch.filter.envAmount = 0.0f;
        return patch;
    }

    // Root is the lowest source, so every other oscillator transposes upward
    // and stays inside the +/-24 semitone range the IR allows.
    auto root = groups.front().f0;
    for (const auto& g : groups)
        root = std::min (root, g.f0);
    patch.rootHz = static_cast<float> (options.noteHz > 0.0 ? options.noteHz : root);

    // --- filter, from the dominant source ---------------------------------
    auto dominant = groups.front();
    for (const auto& g : groups)
        if (g.energy() > dominant.energy())
            dominant = g;

    const auto profile = meanProfile (dominant.H, dominant.numHarmonics, dominant.numFrames);
    const auto anchor = WaveformFit::matchWithCutoff (profile, dominant.f0, sampleRate);
    const auto trajectory = FilterFit::estimateCutoffTrajectory (
        dominant.H, dominant.numHarmonics, dominant.numFrames, dominant.f0, sampleRate);
    const auto soundingFrames = gateTime > 0.0
                                  ? (int) std::ceil (gateTime * sampleRate / options.hop)
                                  : 0;
    const auto split = FilterFit::trajectoryToEnv (trajectory.cutoffHz, anchor.cutoffHz,
                                                   0.25f, soundingFrames);

    patch.filter.type = FilterType::lowpass;
    patch.filter.cutoffHz = juce::jlimit (30.0f, 18000.0f, split.baseCutoffHz);
    patch.filter.resonance = FilterFit::kDefaultQ;
    patch.filter.envAmount = juce::jlimit (0.0f, 4.0f, split.envAmountOctaves);
    if (split.envAmountOctaves > 0.0f)
    {
        std::vector<float> shapeTimes (split.shape.size());
        for (size_t i = 0; i < shapeTimes.size(); ++i)
            shapeTimes[i] = i < partialSet.times.size() ? partialSet.times[i]
                                                        : static_cast<float> (i * options.hop / sampleRate);
        // Smoothed hard before the envelope is fitted to it.
        //
        // A loudness contour wobbles with the vibrato; a cutoff trajectory does
        // that and is an estimate on top, and the estimate is the noisier half.
        // Frame to frame the clarinet's runs 1180, 1292, 1731, 1505, 2498,
        // 3108, 4124, 2128, 3975, 1076, 2866 Hz. An ADSR fitted to that chases
        // whichever spike happens to be highest and then falls off it: the
        // violin came back with a two second sweep to fifteen kilohertz and a
        // sustain of 0.02, which is a filter that opens across the whole note
        // and then shuts, and is nothing a bowed note does.
        //
        // A third of a second is about a vibrato period and a good deal longer
        // than the spikes. Only the shape is smoothed -- the base cutoff and
        // the sweep depth are already settled by `trajectoryToEnv`, which has
        // its own defence against outliers.
        auto envShape = split.shape;
        const auto shapeSpan = juce::jmax (1, (int) std::lround (0.33 * sampleRate / options.hop));
        if ((int) envShape.size() > shapeSpan * 2)
            envShape = nd::uniformFilter1d (envShape, shapeSpan);

        // Not an amplitude contour: a cutoff trajectory is already measured in
        // octaves, so fitting its attack curve in decibels takes the logarithm
        // twice and bends it far too hard.
        patch.filter.env = EnvelopeFit::fitAdsr (envShape, shapeTimes, gateTime, 0.05f,
                                                 oneShot, false);
    }


    // --- sources -> oscillators --------------------------------------------
    auto ordered = groups;
    std::sort (ordered.begin(), ordered.end(),
               [] (const HarmonicGroup& a, const HarmonicGroup& b) { return a.energy() > b.energy(); });

    // Groups first, then the oscillators inside each of them. A group that
    // holds two sources spends two of the budget; the loudest group is split
    // first, because if the budget runs out it should run out on the quietest
    // thing in the sound.
    std::vector<HarmonicGroup> sources;
    for (const auto& g : ordered)
    {
        const auto remaining = options.maxOscillators - static_cast<int> (sources.size());
        if (remaining <= 0)
            break;
        // Flattened here, for the rank decision only. The sub-groups come back
        // in the original domain and are deconvolved downstream like any other.
        const auto flatForRank = FilterFit::deconvolve (g.H, g.numHarmonics, g.numFrames,
                                                        trajectory.cutoffHz, g.f0);
        for (auto& part : splitByRank (g, flatForRank, remaining))
        {
            sources.push_back (std::move (part));
            if (static_cast<int> (sources.size()) >= options.maxOscillators)
                break;
        }
    }
    if (sources.empty())
        sources = ordered;

    const auto multi = sources.size() > 1;
    const auto count = juce::jmin (static_cast<int> (sources.size()), options.maxOscillators);

    for (int i = 0; i < count; ++i)
    {
        const auto& g = sources[static_cast<size_t> (i)];
        auto& osc = patch.oscs[static_cast<size_t> (i)];
        osc.enabled = true;
        osc.level = 1.0f;

        const auto interval = 12.0 * std::log2 (g.f0 / patch.rootHz);
        const auto semis = juce::jlimit (-24, 24, static_cast<int> (std::lround (interval)));
        osc.semitones = semis;
        // The remainder is real detuning, not rounding error.
        osc.cents = static_cast<float> (juce::jlimit (-50.0, 50.0, (interval - semis) * 100.0));

        // Waveform is read after removing the filter, so a bright source behind
        // a closed filter is not mistaken for a dull one.
        const auto flat = FilterFit::deconvolve (g.H, g.numHarmonics, g.numFrames,
                                                 trajectory.cutoffHz, g.f0);
        // Blend of two shapes rather than the single best one. Five waveforms
        // are a coarse net for a real instrument's harmonic profile, and the
        // blend turns them into a continuum at the cost of one parameter that
        // the IR, the engine, the editor and refinement all already supported
        // but that nothing had ever set.
        const auto blend = WaveformFit::matchBlend (meanProfile (flat, g.numHarmonics, g.numFrames));
        osc.waveform = blend.waveform;
        osc.waveformB = blend.waveformB;
        osc.waveMorph = blend.morph;
        osc.pulseWidth = blend.pulseWidth;

        // Unison is visible only because partials are tracked individually.
        Grouping::estimateUnison (g, osc.unisonVoices, osc.unisonDetune);

        std::vector<float> groupTimes (static_cast<size_t> (g.numFrames));
        for (size_t j = 0; j < groupTimes.size(); ++j)
            groupTimes[j] = j < partialSet.times.size() ? partialSet.times[j]
                                                        : static_cast<float> (j * options.hop / sampleRate);

        // Harmonic frames, from the same filter-flattened array the waveform
        // came from. Fitted after the filter is already frozen, because a
        // free-form spectrum and a cutoff explain the same signal and whichever
        // is fitted second gets only the residue -- which is the one that should
        // have the freedom.
        //
        // Keeps the blend underneath it either way: the frames can be switched
        // off in the editor and what is left is still a described sound, not
        // silence.
        const auto share = dominant.energy() > 0.0f ? g.energy() / dominant.energy() : 0.0f;
        const auto table = WavetableFit::fit (flat.data(), g.numHarmonics, g.numFrames,
                                              groupTimes, static_cast<float> (gateTime), oneShot,
                                              blend, share);
        if (table.useCustomFrames)
        {
            osc.numFrames = table.numFrames;
            for (int f = 0; f < table.numFrames; ++f)
            {
                auto& frame = osc.frames[static_cast<size_t> (f)];
                frame.custom = true;
                frame.harmonics = table.frames[static_cast<size_t> (f)];
            }
            osc.framePosition = table.position;
            osc.framePositionEnvAmount = table.envAmount;
            osc.framePositionEnv = table.env;
        }

        if (multi)
        {
            osc.envEnabled = true;
            std::vector<float> total (static_cast<size_t> (g.numFrames), 0.0f);
            for (int t = 0; t < g.numFrames; ++t)
                for (int k = 0; k < g.numHarmonics; ++k)
                    total[static_cast<size_t> (t)] += g.harmonic (k)[t];

            osc.env = EnvelopeFit::fitAdsr (total, groupTimes, gateTime, 0.05f, oneShot);
        }
    }

    if (multi)
    {
        // The patch-wide envelope becomes a pass-through supplying only the
        // note-off release, so per-oscillator envelopes are not multiplied by a
        // second shape they did not account for.
        patch.ampEnv = { 0.001f, 0.005f, 1.0f, globalEnv.release, globalEnv.curve };
    }
    else
    {
        patch.ampEnv = globalEnv;
    }

    // The IR carries one LFO, so vibrato, tremolo and filter wobble compete and
    // only the most convincingly periodic survives.
    // Two slots now, so vibrato and tremolo can coexist instead of competing.
    // Filled most-convincing-first; unused slots stay routed to none.
    const auto detected = Modulation::bestSeveral (
        Modulation::extract (samples, numSamples, sampleRate, options.hop));
    for (size_t i = 0; i < detected.size() && i < patch.lfos.size(); ++i)
        patch.lfos[i] = detected[i];

    patch.delay = EffectsFit::fitDelay (samples, numSamples, sampleRate, options.hop, gateTime);

    // Reverb, and the release it was hiding inside.
    //
    // These two are near-degenerate -- a tail and a long release draw the same
    // curve -- so they are separated structurally rather than left for the
    // optimiser to trade off. `detectReverb` measures the slow exponential on
    // the later part of the decay, where the direct sound has gone, and reports
    // where the handover happened. That handover is the release; everything
    // after it belongs to the room.
    //
    // Without this the release absorbs the whole tail, which measured *better*
    // on whole-file distance than a correct fit did, because a long release is
    // the only way the patch could account for a room at all.
    const auto reverb = EffectsFit::detectReverb (samples, numSamples, sampleRate,
                                                  gateTime, options.hop);
    patch.reverb = EffectsFit::fitReverb (samples, numSamples, sampleRate,
                                          gateTime, options.hop);

    // Shorten the release only when a reverb is actually taking the tail over.
    //
    // These are two different conditions: `detectReverb` can find a decay that
    // `fitReverb` then declines to model, because the return works out too
    // quiet to be worth switching on. Cutting the release on the first
    // condition rather than the second threw the tail away with nothing to
    // replace it, and cost about a decibel of loudness accuracy on every seed
    // of the harness -- small, consistent, and easy to mistake for noise.
    if (reverb.found && patch.reverb.enabled)
    {
        // Hand the tail to the reverb by cutting the release back to the direct
        // sound's own decay. Left alone it would model the room twice.
        const auto release = static_cast<float> (juce::jlimit (5.0e-3, 4.0,
                                                               reverb.releaseSeconds));
        patch.ampEnv.release = juce::jmin (patch.ampEnv.release, release);
        for (auto& osc : patch.oscs)
            if (osc.envEnabled)
                osc.env.release = juce::jmin (osc.env.release, release);

    }

    double claimed = 0.0;
    for (const auto& g : groups)
        claimed += g.energy();
    const auto totalPartialEnergy = juce::jmax (partialSet.totalEnergy(), 1.0e-12f);
    const auto residual = 1.0 - claimed / totalPartialEnergy;
    patch.noiseLevel = static_cast<float> (juce::jlimit (0.0, 1.0, residual)) * 0.5f;

    // How much noise this material has actually got, rather than how much a
    // pitched patch is allowed in general. Measured between the harmonics of
    // the fundamental that was just fitted, which is the same measurement
    // `autosynth_diff` prints as noisiness.
    const auto share = Roles::noiseShare (samples, numSamples, sampleRate, patch.rootHz,
                                          trackOptions.fftSize, options.hop);
    const auto ceiling = kMaxPitchedNoise
                       * static_cast<float> (juce::jlimit (0.0, 1.0, share / kFullNoiseShare));
    auto calibrated = calibrateLevels (patch, samples, numSamples, sampleRate, gateTime,
                                       options.renderer, ceiling);
    return calibrateNoise (std::move (calibrated), samples, numSamples, sampleRate, gateTime,
                           static_cast<float> (ceiling), options.renderer);
}

} // namespace autosynth
