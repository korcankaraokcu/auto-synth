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

double salience (double f0, const std::vector<double>& freqs,
                 const std::vector<double>& energies, double tolCents)
{
    std::vector<bool> matched;
    std::vector<int> harmonics;
    assign (f0, freqs, tolCents, matched, harmonics);

    double explained = 0.0;
    std::set<int> found;
    bool any = false;
    for (size_t i = 0; i < freqs.size(); ++i)
    {
        if (! matched[i])
            continue;
        any = true;
        explained += energies[i];
        found.insert (harmonics[i]);
    }
    if (! any)
        return 0.0;

    // The sub-harmonic guard. A candidate an octave too low predicts a full set
    // of odd harmonics; finding none of them is what disqualifies it.
    constexpr int kCheck = 6;
    int missing = 0;
    for (int h = 1; h <= kCheck; ++h)
        if (found.find (h) == found.end())
            ++missing;

    return explained * (1.0 - 0.8 * missing / kCheck);
}

HarmonicGroup buildGroup (double f0, const std::vector<Partial>& claimed,
                          const std::vector<int>& harmonics, int numFrames, double salienceValue)
{
    HarmonicGroup group;
    group.f0 = f0;
    group.salience = salienceValue;
    group.partials = claimed;
    group.numFrames = numFrames;

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

        const auto f0 = candidates[bestIndex];
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

        groups.push_back (buildGroup (f0, claimed, claimedHarmonics, set.numFrames(), bestScore));
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

void Grouping::estimateUnison (const HarmonicGroup& group, int& voicesOut, float& detuneOut,
                               int maxVoices, int minHarmonic)
{
    voicesOut = 1;
    detuneOut = 0.0f;

    const auto clusters = harmonicClusters (group);
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
    if (counts.empty())
        return;

    const auto voices = static_cast<int> (medianOf (counts));
    if (voices <= 1 || spreads.empty())
        return;

    voicesOut = juce::jlimit (1, maxVoices, voices);
    detuneOut = static_cast<float> (juce::jlimit (0.0, 50.0, medianOf (spreads)));
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
