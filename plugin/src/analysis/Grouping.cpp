#include "analysis/Grouping.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace autosynth
{
namespace
{

// numpy's median: for an even count it averages the middle pair.
double medianOf (std::vector<double> values)
{
    if (values.empty())
        return 0.0;
    std::sort (values.begin(), values.end());
    const auto mid = values.size() / 2;
    if (values.size() % 2 == 1)
        return values[mid];
    return 0.5 * (values[mid - 1] + values[mid]);
}

std::vector<double> candidateFundamentals (const std::vector<Partial>& partials)
{
    std::vector<double> seen;
    for (const auto& p : partials)
    {
        const auto f = static_cast<double> (p.meanFreq());
        for (int k = 1; k <= 8; ++k)
        {
            const auto candidate = f / k;
            if (candidate >= Grouping::kMinF0 && candidate <= Grouping::kMaxF0)
                seen.push_back (candidate);
        }
    }
    if (seen.empty())
        return {};

    std::sort (seen.begin(), seen.end());
    std::vector<double> merged { seen.front() };
    for (size_t i = 1; i < seen.size(); ++i)
        if (1200.0 * std::log2 (seen[i] / merged.back()) > 20.0)
            merged.push_back (seen[i]);
    return merged;
}

void assign (double f0, const std::vector<double>& freqs, double tolCents,
             std::vector<bool>& matchedOut, std::vector<int>& harmonicOut)
{
    matchedOut.assign (freqs.size(), false);
    harmonicOut.assign (freqs.size(), 1);
    for (size_t i = 0; i < freqs.size(); ++i)
    {
        auto k = static_cast<int> (std::lround (freqs[i] / f0));
        k = juce::jlimit (1, Grouping::kMaxHarmonic, k);
        harmonicOut[i] = k;
        const auto cents = 1200.0 * std::abs (std::log2 ((freqs[i] + 1.0e-9) / (k * f0)));
        matchedOut[i] = (cents <= tolCents) && (freqs[i] >= f0 * 0.5);
    }
}

int greatestCommonDivisor (int a, int b)
{
    while (b != 0)
    {
        const auto t = b;
        b = a % b;
        a = t;
    }
    return a;
}

double salience (double f0, const std::vector<double>& freqs,
                 const std::vector<double>& energies, double tolCents)
{
    std::vector<bool> matched;
    std::vector<int> harmonics;
    assign (f0, freqs, tolCents, matched, harmonics);

    double explained = 0.0;
    std::set<int> found;
    std::map<int, double> harmonicEnergy;
    bool any = false;
    for (size_t i = 0; i < freqs.size(); ++i)
    {
        if (! matched[i])
            continue;
        any = true;
        explained += energies[i];
        found.insert (harmonics[i]);
        harmonicEnergy[harmonics[i]] += energies[i];
    }
    if (! any)
        return 0.0;

    // The presence guard. A candidate an octave too low predicts a full set of
    // odd harmonics, and not finding them is one way to disqualify it.
    constexpr int kCheck = 6;
    int missing = 0;
    for (int h = 1; h <= kCheck; ++h)
        if (found.find (h) == found.end())
            ++missing;
    auto score = explained * (1.0 - 0.8 * missing / kCheck);

    // The energy guard, which is the one that survives contact with real
    // recordings.
    //
    // Presence alone is not enough. A real sample carries room tone, bow or
    // breath noise and reverb rumble, and once a few hundred partials are being
    // tracked there is nearly always *something* within the tolerance of any
    // predicted frequency -- so every harmonic reads as "present" and the guard
    // above never fires. Measured on a violin note, the true fundamental at 877
    // Hz and the octave-below candidate at 439 Hz both showed harmonics 1 to 8
    // present, and the wrong one won.
    //
    // What separates them is where the energy actually is. If every harmonic
    // carrying real weight is a multiple of m, then the true fundamental is m
    // times higher and this candidate is a subharmonic of it. For that violin,
    // the 439 Hz candidate had 0.5% of its energy on odd harmonics; for the
    // clarinet's 226 Hz candidate, 0.1%. Genuine fundamentals measured 58.6%
    // and 71.4%.
    //
    // Taking the divisor rather than testing odd-versus-even generalises the
    // same idea to f/3 and f/4, and it is safe for instruments whose character
    // *is* an unusual harmonic balance: a clarinet's odd-harmonic dominance
    // gives a divisor of 1, because harmonic 1 is itself odd.
    double significant = 0.0;
    for (const auto& entry : harmonicEnergy)
        significant = juce::jmax (significant, entry.second);

    if (significant > 0.0)
    {
        auto divisor = 0;
        for (const auto& entry : harmonicEnergy)
            if (entry.second >= 0.03 * explained)
                divisor = greatestCommonDivisor (divisor, entry.first);

        if (divisor > 1)
            score /= static_cast<double> (divisor) * divisor;
    }

    return score;
}

// Fit the fundamental to the partials it claimed.
//
// Candidates are generated coarsely -- every tracked partial divided by 1..8,
// then merged at 20 cents -- and matching tolerates 50 cents. A candidate tens
// of cents away therefore claims exactly the right partials and wins on
// salience without ever being corrected, which left a clarinet whose true
// fundamental was 441.5 Hz described as 430.9: flat by 42 cents, and audibly
// out of tune when played back.
//
// Given the claimed partials and their harmonic numbers, the fundamental is a
// weighted least-squares fit, minimising sum of e·(f - k·f0)². Energy weighting
// keeps a handful of weak, badly-tracked upper partials from dragging it.
double refineFundamental (double f0, const std::vector<double>& freqs,
                          const std::vector<double>& energies, double tolCents)
{
    for (int iteration = 0; iteration < 4; ++iteration)
    {
        std::vector<bool> matched;
        std::vector<int> harmonics;
        assign (f0, freqs, tolCents, matched, harmonics);

        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < freqs.size(); ++i)
        {
            if (! matched[i])
                continue;
            const auto k = static_cast<double> (harmonics[i]);
            num += energies[i] * k * freqs[i];
            den += energies[i] * k * k;
        }
        if (den <= 0.0)
            break;

        const auto next = num / den;
        if (next < Grouping::kMinF0 || next > Grouping::kMaxF0)
            break;

        const auto movedCents = std::abs (1200.0 * std::log2 (next / f0));
        f0 = next;
        if (movedCents < 0.5)
            break; // converged
    }
    return f0;
}

HarmonicGroup buildGroup (double f0, const std::vector<Partial>& claimed,
                          const std::vector<int>& harmonics, int numFrames, double salienceValue,
                          double frameRateHz)
{
    HarmonicGroup group;
    group.f0 = f0;
    group.salience = salienceValue;
    group.partials = claimed;
    group.numFrames = numFrames;
    group.frameRateHz = frameRateHz;

    group.numHarmonics = 1;
    for (auto k : harmonics)
        group.numHarmonics = juce::jmax (group.numHarmonics, k);

    group.H.assign (static_cast<size_t> (group.numHarmonics) * static_cast<size_t> (numFrames), 0.0f);

    std::set<int> unique;
    for (size_t i = 0; i < claimed.size(); ++i)
    {
        const auto k = harmonics[i];
        unique.insert (k);
        auto* row = group.H.data() + static_cast<size_t> (k - 1) * static_cast<size_t> (numFrames);
        const auto& p = claimed[i];
        for (size_t j = 0; j < p.frames.size(); ++j)
        {
            const auto frame = p.frames[j];
            if (frame >= 0 && frame < numFrames)
                row[frame] += p.amps[j];
        }
    }
    group.harmonicIndices.assign (unique.begin(), unique.end());
    return group;
}

} // namespace

float HarmonicGroup::energy() const noexcept
{
    return static_cast<float> (std::accumulate (H.begin(), H.end(), 0.0));
}

std::vector<HarmonicGroup> Grouping::group (const PartialSet& set, int maxGroups,
                                            double tolCents, double minEnergyFraction)
{
    std::vector<HarmonicGroup> groups;
    auto pool = set.partials;
    if (pool.empty())
        return groups;

    double totalEnergy = 0.0;
    for (const auto& p : pool)
        totalEnergy += p.energy();
    if (totalEnergy <= 1.0e-12)
        return groups;

    for (int iteration = 0; iteration < maxGroups; ++iteration)
    {
        if (pool.empty())
            break;

        double remaining = 0.0;
        for (const auto& p : pool)
            remaining += p.energy();
        if (remaining < minEnergyFraction * totalEnergy)
            break;

        std::vector<double> freqs, energies;
        freqs.reserve (pool.size());
        energies.reserve (pool.size());
        for (const auto& p : pool)
        {
            freqs.push_back (p.meanFreq());
            energies.push_back (p.energy());
        }

        const auto candidates = candidateFundamentals (pool);
        if (candidates.empty())
            break;

        double bestScore = -1.0;
        size_t bestIndex = 0;
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            const auto score = salience (candidates[i], freqs, energies, tolCents);
            if (score > bestScore)
            {
                bestScore = score;
                bestIndex = i;
            }
        }
        if (bestScore <= 0.0)
            break;

        // Selection is coarse; this lands the value.
        const auto f0 = refineFundamental (candidates[bestIndex], freqs, energies, tolCents);
        std::vector<bool> matched;
        std::vector<int> harmonics;
        assign (f0, freqs, tolCents, matched, harmonics);

        std::vector<Partial> claimed;
        std::vector<int> claimedHarmonics;
        std::vector<Partial> leftover;
        for (size_t i = 0; i < pool.size(); ++i)
        {
            if (matched[i])
            {
                claimed.push_back (pool[i]);
                claimedHarmonics.push_back (harmonics[i]);
            }
            else
            {
                leftover.push_back (pool[i]);
            }
        }
        if (claimed.empty())
            break;

        groups.push_back (buildGroup (f0, claimed, claimedHarmonics, set.numFrames(), bestScore,
                                      set.hop > 0 ? set.sampleRate / set.hop : 0.0));
        pool = std::move (leftover);
    }

    return groups;
}

std::map<int, std::vector<float>> Grouping::harmonicClusters (const HarmonicGroup& group)
{
    std::map<int, std::vector<float>> clusters;
    for (const auto& p : group.partials)
    {
        const auto mean = static_cast<double> (p.meanFreq());
        const auto k = juce::jmax (1, static_cast<int> (std::lround (mean / group.f0)));
        const auto cents = 1200.0 * std::log2 (mean / (k * group.f0));
        clusters[k].push_back (static_cast<float> (cents));
    }
    return clusters;
}

std::vector<float> Grouping::mergeClose (std::vector<float> cents, float tol)
{
    if (cents.empty())
        return {};
    std::sort (cents.begin(), cents.end());

    std::vector<std::vector<float>> groups { { cents.front() } };
    for (size_t i = 1; i < cents.size(); ++i)
    {
        if (cents[i] - groups.back().back() <= tol)
            groups.back().push_back (cents[i]);
        else
            groups.push_back ({ cents[i] });
    }

    std::vector<float> means;
    means.reserve (groups.size());
    for (const auto& g : groups)
        means.push_back (static_cast<float> (
            std::accumulate (g.begin(), g.end(), 0.0) / static_cast<double> (g.size())));
    return means;
}

namespace
{

// --- unison by beating -----------------------------------------------------
//
// Spectral clustering can only see unison once the voices resolve into separate
// peaks, which needs a detune wide enough to beat the analysis resolution. Below
// that they merge into one partial -- and the estimator reports one voice, which
// is why narrow unison was systematically under-counted.
//
// But two voices d cents apart do not stop existing when they stop resolving.
// They amplitude-modulate at their difference frequency, and that beating is
// plainly measurable long before the peaks separate. At harmonic k of a group
// whose fundamental is f0:
//
//     beat rate = k * f0 * (2^(d/1200) - 1)
//
// The useful part is the factor of k. A beat rate *grows with harmonic number*,
// proportionally, because the frequency gap between two detuned partials widens
// as you go up the series. Nothing else in this signal chain does that:
//
//   * tremolo modulates every harmonic at the same rate,
//   * vibrato likewise,
//   * a decaying envelope is not periodic at all.
//
// So the discriminator is not "is there periodicity in the harmonic envelopes"
// -- that would fire on every LFO in the test set. It is "does the periodicity
// rate rise in proportion to k". That single constraint is what separates
// unison from modulation, and it is why this is worth doing at all.

constexpr double kMinBeatRateHz = 1.0;
constexpr double kMaxBeatRateHz = 45.0;
constexpr double kMinCycles = 2.5;
// A beat is a deep modulation -- two equal voices cancel completely at the
// trough. Requiring real depth keeps low-level analysis wobble out.
constexpr double kMinRelativeDepth = 0.12;
constexpr double kMinPeakStrength = 0.35;
constexpr int kMinAgreeingHarmonics = 3;
// Detunes outside this are either inaudible or better described as two separate
// oscillators, which the grouping stage already handles.
constexpr double kMinDetuneCents = 1.5;
constexpr double kMaxDetuneCents = 60.0;

struct HarmonicBeat
{
    int harmonic = 0;
    double rateHz = 0.0;
    double strength = 0.0;
};

// Relative fluctuation of one harmonic's amplitude envelope: x / smooth(x) - 1.
//
// Dividing rather than subtracting is deliberate. It removes the note's own
// amplitude envelope -- which is far larger than the beating riding on it --
// and leaves a signal whose size means the same thing at the attack and in the
// tail, so one threshold works across the whole note.
std::vector<double> relativeFluctuation (const float* envelope, int numFrames, int smoothFrames)
{
    std::vector<double> out (static_cast<size_t> (numFrames), 0.0);
    if (numFrames <= 0)
        return out;

    const auto half = juce::jmax (1, smoothFrames / 2);
    for (int t = 0; t < numFrames; ++t)
    {
        const auto lo = juce::jmax (0, t - half);
        const auto hi = juce::jmin (numFrames - 1, t + half);
        double acc = 0.0;
        for (int i = lo; i <= hi; ++i)
            acc += envelope[i];
        const auto mean = acc / (hi - lo + 1);
        out[static_cast<size_t> (t)] = mean > 1.0e-9 ? envelope[t] / mean - 1.0 : 0.0;
    }
    return out;
}

// Strongest periodicity in a fluctuation signal, by autocorrelation.
// Returns false when nothing stands out enough to be called a beat.
bool dominantRate (const std::vector<double>& x, double frameRateHz,
                   double& rateOut, double& strengthOut)
{
    const auto n = static_cast<int> (x.size());
    if (n < 8 || frameRateHz <= 0.0)
        return false;

    double energy = 0.0;
    for (auto v : x)
        energy += v * v;
    if (energy <= 1.0e-12)
        return false;

    const auto rms = std::sqrt (energy / n);
    if (rms < kMinRelativeDepth)
        return false; // present, but too shallow to be two voices cancelling

    const auto minLag = juce::jmax (2, static_cast<int> (std::floor (frameRateHz / kMaxBeatRateHz)));
    // Enough of the window must be covered for a "period" to mean anything.
    const auto maxLag = juce::jmin (static_cast<int> (n / kMinCycles),
                                    static_cast<int> (std::ceil (frameRateHz / kMinBeatRateHz)));
    if (maxLag <= minLag)
        return false;

    std::vector<double> correlation (static_cast<size_t> (maxLag + 1), 0.0);
    auto best = 0.0;
    for (int lag = minLag; lag <= maxLag; ++lag)
    {
        double acc = 0.0;
        for (int t = 0; t + lag < n; ++t)
            acc += x[static_cast<size_t> (t)] * x[static_cast<size_t> (t + lag)];
        correlation[static_cast<size_t> (lag)] = acc / energy;
        best = juce::jmax (best, correlation[static_cast<size_t> (lag)]);
    }

    if (best < kMinPeakStrength)
        return false;

    // Take the *first* strong peak, not the tallest.
    //
    // Autocorrelation of a periodic signal peaks at every multiple of the
    // period, and the tallest of those is not reliably the first -- picking the
    // global maximum reported half the true beat rate often enough to wreck the
    // proportional fit, which is the classic octave error in period detection.
    // The first peak within a whisker of the tallest is the period itself.
    constexpr auto kPeakTolerance = 0.85;
    auto chosen = -1;
    for (int lag = minLag + 1; lag < maxLag; ++lag)
    {
        const auto v = correlation[static_cast<size_t> (lag)];
        if (v >= kPeakTolerance * best
            && v >= correlation[static_cast<size_t> (lag - 1)]
            && v >= correlation[static_cast<size_t> (lag + 1)])
        {
            chosen = lag;
            break;
        }
    }
    if (chosen < 0)
        return false;

    rateOut = frameRateHz / chosen;
    strengthOut = correlation[static_cast<size_t> (chosen)];
    return true;
}

double medianValue (std::vector<double> values)
{
    if (values.empty())
        return 0.0;
    const auto mid = values.size() / 2;
    std::nth_element (values.begin(), values.begin() + (long) mid, values.end());
    return values[mid];
}

// Decide between "rate rises with k" and "rate is the same for every k", and
// report how strongly the proportional story wins.
//
// Least squares was tried first and is the wrong tool: a handful of harmonics
// whose period detection went astray drags the line badly, and R-squared then
// condemns a set of rates that are mostly right. Counting inliers around a
// *median* slope is unbothered by a minority of bad harmonics, which is the
// normal case here -- high harmonics are weak and their envelopes noisy.
//
// The comparison against a constant model is the explicit tremolo guard. An
// amplitude LFO gives one rate at every harmonic; that fits a constant
// perfectly and a proportional line not at all, so it is rejected by losing
// this comparison rather than by any threshold tuned to it.
bool fitProportional (const std::vector<HarmonicBeat>& beats, double& slopeOut, double& fitOut)
{
    if (beats.size() < 2)
        return false;

    std::vector<double> perHarmonic;
    std::vector<double> rates;
    perHarmonic.reserve (beats.size());
    rates.reserve (beats.size());
    for (const auto& b : beats)
    {
        if (b.harmonic > 0)
            perHarmonic.push_back (b.rateHz / b.harmonic);
        rates.push_back (b.rateHz);
    }

    const auto slope = medianValue (perHarmonic);
    if (slope <= 0.0)
        return false;

    constexpr auto kTolerance = 0.25;
    const auto constantRate = medianValue (rates);

    int proportionalInliers = 0, constantInliers = 0;
    for (const auto& b : beats)
    {
        if (std::abs (b.rateHz - slope * b.harmonic) <= kTolerance * slope * b.harmonic)
            ++proportionalInliers;
        if (constantRate > 0.0 && std::abs (b.rateHz - constantRate) <= kTolerance * constantRate)
            ++constantInliers;
    }

    if (proportionalInliers <= constantInliers)
        return false; // a single rate explains it at least as well: not unison

    slopeOut = slope;
    fitOut = static_cast<double> (proportionalInliers) / beats.size();
    return true;
}

} // namespace

Grouping::UnisonBeating Grouping::detectUnisonBeating (const HarmonicGroup& group,
                                                       int minHarmonic)
{
    UnisonBeating result;
    if (group.frameRateHz <= 0.0 || group.numFrames < 16 || group.f0 <= 0.0)
        return result;

    // Roughly 60 ms of smoothing: long enough to leave the beating alone, short
    // enough to follow the note's own envelope.
    const auto smoothFrames = juce::jmax (3, static_cast<int> (0.06 * group.frameRateHz));

    std::vector<HarmonicBeat> beats;
    for (int k = minHarmonic; k <= group.numHarmonics; ++k)
    {
        const auto* envelope = group.harmonic (k - 1);

        double peak = 0.0;
        for (int t = 0; t < group.numFrames; ++t)
            peak = juce::jmax (peak, static_cast<double> (envelope[t]));
        if (peak <= 1.0e-6)
            continue;

        const auto fluctuation = relativeFluctuation (envelope, group.numFrames, smoothFrames);
        double rate = 0.0, strength = 0.0;
        if (dominantRate (fluctuation, group.frameRateHz, rate, strength))
            beats.push_back ({ k, rate, strength });
    }

    if (static_cast<int> (beats.size()) < kMinAgreeingHarmonics)
        return result;

    double slope = 0.0, fit = 0.0;
    if (! fitProportional (beats, slope, fit))
        return result;

    // slope = f0 * (2^(d/1200) - 1)
    const auto ratio = 1.0 + slope / group.f0;
    if (ratio <= 1.0)
        return result;

    const auto detune = 1200.0 * std::log2 (ratio);
    if (detune < kMinDetuneCents || detune > kMaxDetuneCents)
        return result;

    result.detuneCents = detune;
    result.beatRateAtFundamental = slope;
    result.proportionalFit = fit;
    result.harmonicsAgreeing = static_cast<int> (beats.size());

    // Most of the harmonics that showed a beat must agree with one proportional
    // line. A bare majority is not enough: the constant-model comparison has
    // already been passed, so this is guarding against a scatter of unrelated
    // rates that happens to lean proportional.
    result.found = fit >= 0.6 && result.harmonicsAgreeing >= kMinAgreeingHarmonics;
    return result;
}

void Grouping::estimateUnison (const HarmonicGroup& group, int& voicesOut, float& detuneOut,
                               int maxVoices, int minHarmonic)
{
    voicesOut = 1;
    detuneOut = 0.0f;

    // Only partials that persist count as voices.
    //
    // Unison voices are *simultaneous*: two detuned oscillators both sound for
    // the whole note. Track fragments are *sequential* -- vibrato walks a
    // partial's frequency until the tracker gives up and starts a new one, and
    // room tone throws up short-lived peaks that get rounded onto whichever
    // harmonic is nearest.
    //
    // Counting those as voices is why a solo violin came back as three
    // oscillators fifty cents apart. Measured on one: harmonic 1 had 189
    // partials assigned to it, of which exactly one lasted longer than a tenth
    // of the note and the rest were room noise sitting two octaves away.
    //
    // Requiring real duration is what separates the two, and it costs nothing
    // on synthesised unison, where every voice runs the length of the note.
    const auto minLength = static_cast<int> (0.4 * group.numFrames);

    std::map<int, std::vector<float>> clusters;
    for (const auto& p : group.partials)
    {
        if (p.length() < minLength)
            continue;
        const auto mean = static_cast<double> (p.meanFreq());
        const auto k = juce::jmax (1, static_cast<int> (std::lround (mean / group.f0)));
        const auto cents = 1200.0 * std::log2 (mean / (k * group.f0));
        clusters[k].push_back (static_cast<float> (cents));
    }

    std::vector<double> counts, spreads;
    for (const auto& entry : clusters)
    {
        if (entry.first < minHarmonic || entry.second.empty())
            continue;
        const auto merged = mergeClose (entry.second);
        counts.push_back (juce::jmin (static_cast<int> (merged.size()), maxVoices));
        if (merged.size() > 1)
            spreads.push_back (merged.back() - merged.front());
    }
    const auto resolved = counts.empty() ? 1 : static_cast<int> (medianOf (counts));
    if (resolved > 1 && ! spreads.empty())
    {
        // The voices resolved into separate peaks; that is the more informative
        // measurement, since it counts them rather than inferring a minimum.
        voicesOut = juce::jlimit (1, maxVoices, resolved);
        detuneOut = static_cast<float> (juce::jlimit (0.0, 50.0, medianOf (spreads)));
        return;
    }

    // Nothing resolved. Before concluding there is one voice, check whether the
    // harmonics are beating -- that is what narrow unison looks like when the
    // peaks have merged.
    const auto beating = detectUnisonBeating (group, minHarmonic);
    if (! beating.found)
        return;

    // Two is the *minimum* number of voices that explains a beat, and it is
    // what is reported. Recovering the true count from the envelope shape is
    // possible in principle -- more voices sharpen the peaks of the sum -- but
    // it is not attempted here, and claiming a number the measurement does not
    // support would be worse than admitting to a lower bound.
    voicesOut = juce::jlimit (2, maxVoices, 2);
    detuneOut = static_cast<float> (juce::jlimit (0.0, 50.0, beating.detuneCents));
}

double Grouping::detuneCents (const HarmonicGroup& group)
{
    if (group.partials.empty())
        return 0.0;

    std::vector<double> deviations;
    deviations.reserve (group.partials.size());
    for (const auto& p : group.partials)
    {
        const auto mean = static_cast<double> (p.meanFreq());
        const auto k = juce::jmax (1, static_cast<int> (std::lround (mean / group.f0)));
        deviations.push_back (1200.0 * std::log2 (mean / (k * group.f0)));
    }
    if (deviations.size() < 2)
        return 0.0;

    const auto mean = std::accumulate (deviations.begin(), deviations.end(), 0.0)
                    / static_cast<double> (deviations.size());
    double variance = 0.0;
    for (auto d : deviations)
        variance += (d - mean) * (d - mean);
    return std::sqrt (variance / static_cast<double> (deviations.size()));
}

} // namespace autosynth
