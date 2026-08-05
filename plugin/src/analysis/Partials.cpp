#include "analysis/Partials.h"

#include "analysis/Stft.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace autosynth
{

float Partial::meanFreq() const noexcept
{
    double weighted = 0.0, total = 0.0;
    for (size_t i = 0; i < freqs.size(); ++i)
    {
        const auto w = static_cast<double> (amps[i]) + 1.0e-12;
        weighted += freqs[i] * w;
        total += w;
    }
    return static_cast<float> (weighted / juce::jmax (total, 1.0e-12));
}

float Partial::energy() const noexcept
{
    return static_cast<float> (std::accumulate (amps.begin(), amps.end(), 0.0));
}

float Partial::peakAmp() const noexcept
{
    return amps.empty() ? 0.0f : *std::max_element (amps.begin(), amps.end());
}

float PartialSet::totalEnergy() const noexcept
{
    double sum = 0.0;
    for (const auto& p : partials)
        sum += p.energy();
    return static_cast<float> (sum);
}

void PartialTracker::findPeaks (const float* magnitude, int numBins, double sampleRate,
                                int fftSize, double floorDb, int maxPeaks,
                                std::vector<float>& freqsOut, std::vector<float>& ampsOut)
{
    freqsOut.clear();
    ampsOut.clear();
    if (numBins < 3)
        return;

    const auto peak = *std::max_element (magnitude, magnitude + numBins);
    if (peak <= 1.0e-12f)
        return;

    const auto threshold = peak * static_cast<float> (std::pow (10.0, floorDb / 20.0));

    std::vector<int> indices;
    for (int b = 1; b < numBins - 1; ++b)
        if (magnitude[b] > magnitude[b - 1] && magnitude[b] >= magnitude[b + 1]
            && magnitude[b] > threshold)
            indices.push_back (b);

    if (indices.empty())
        return;

    if (static_cast<int> (indices.size()) > maxPeaks)
    {
        // Keep the loudest, then restore ascending bin order -- the tracker
        // does not care, but matching the reference exactly keeps the
        // conformance comparison meaningful.
        std::partial_sort (indices.begin(), indices.begin() + maxPeaks, indices.end(),
                           [magnitude] (int a, int b) { return magnitude[a] > magnitude[b]; });
        indices.resize (static_cast<size_t> (maxPeaks));
        std::sort (indices.begin(), indices.end());
    }

    for (auto b : indices)
    {
        const auto toDb = [magnitude] (int i)
        {
            return 20.0 * std::log10 (static_cast<double> (magnitude[i]) + 1.0e-12);
        };
        const auto alpha = toDb (b - 1);
        const auto beta = toDb (b);
        const auto gamma = toDb (b + 1);

        const auto denom = alpha - 2.0 * beta + gamma;
        auto shift = 0.0;
        if (std::abs (denom) > 1.0e-12)
            shift = juce::jlimit (-0.5, 0.5, 0.5 * (alpha - gamma) / denom);

        freqsOut.push_back (static_cast<float> ((b + shift) * sampleRate / fftSize));
        const auto ampDb = beta - 0.25 * (alpha - gamma) * shift;
        ampsOut.push_back (static_cast<float> (std::pow (10.0, ampDb / 20.0)));
    }
}

namespace
{

struct LiveTrack
{
    std::vector<int> frames;
    std::vector<float> freqs;
    std::vector<float> amps;
    int sleep = 0;
};

// Greedy nearest-frequency matching. Greedy rather than optimal (Hungarian)
// because the cost matrix is tiny and nearly diagonal in practice: partials
// move slowly relative to their spacing, so the two agree almost always.
std::map<int, int> linkFrame (const std::vector<LiveTrack>& live,
                              const std::vector<float>& peakFreqs,
                              double tolCents)
{
    std::map<int, int> pairs;
    if (live.empty() || peakFreqs.empty())
        return pairs;

    const auto numTracks = static_cast<int> (live.size());
    const auto numPeaks = static_cast<int> (peakFreqs.size());

    std::vector<double> cost (static_cast<size_t> (numTracks) * numPeaks,
                              std::numeric_limits<double>::infinity());
    for (int t = 0; t < numTracks; ++t)
    {
        const auto trackFreq = static_cast<double> (live[static_cast<size_t> (t)].freqs.back());
        for (int p = 0; p < numPeaks; ++p)
        {
            const auto cents = 1200.0 * std::abs (
                std::log2 ((peakFreqs[static_cast<size_t> (p)] + 1.0e-9) / (trackFreq + 1.0e-9)));
            if (cents <= tolCents)
                cost[static_cast<size_t> (t) * numPeaks + p] = cents;
        }
    }

    while (true)
    {
        double best = std::numeric_limits<double>::infinity();
        int bestT = -1, bestP = -1;
        for (int t = 0; t < numTracks; ++t)
            for (int p = 0; p < numPeaks; ++p)
            {
                const auto c = cost[static_cast<size_t> (t) * numPeaks + p];
                if (c < best)
                {
                    best = c;
                    bestT = t;
                    bestP = p;
                }
            }

        if (bestT < 0 || ! std::isfinite (best))
            break;

        pairs[bestT] = bestP;
        for (int p = 0; p < numPeaks; ++p)
            cost[static_cast<size_t> (bestT) * numPeaks + p] = std::numeric_limits<double>::infinity();
        for (int t = 0; t < numTracks; ++t)
            cost[static_cast<size_t> (t) * numPeaks + bestP] = std::numeric_limits<double>::infinity();
    }
    return pairs;
}

Partial finish (const LiveTrack& track)
{
    Partial p;
    p.frames = track.frames;
    p.freqs = track.freqs;
    p.amps = track.amps;
    return p;
}

} // namespace

PartialSet PartialTracker::track (const float* samples, int numSamples, double sampleRate,
                                  const Options& options)
{
    PartialSet result;
    result.sampleRate = sampleRate;
    result.fftSize = options.fftSize;
    result.hop = options.hop;

    const auto spectrogram = Stft::magnitudeSpectrogram (samples, numSamples,
                                                         options.fftSize, options.hop, sampleRate);
    result.times = spectrogram.times;

    std::vector<Partial> finished;
    std::vector<LiveTrack> live;
    std::vector<float> freqs, amps;

    for (int f = 0; f < spectrogram.numFrames; ++f)
    {
        findPeaks (spectrogram.frame (f), spectrogram.numBins, sampleRate, options.fftSize,
                   options.floorDb, options.maxPeaks, freqs, amps);

        const auto pairs = linkFrame (live, freqs, options.tolCents);

        std::vector<bool> peakMatched (freqs.size(), false);
        for (const auto& pair : pairs)
            peakMatched[static_cast<size_t> (pair.second)] = true;

        for (int t = 0; t < static_cast<int> (live.size()); ++t)
        {
            auto& track = live[static_cast<size_t> (t)];
            const auto it = pairs.find (t);
            if (it != pairs.end())
            {
                const auto p = static_cast<size_t> (it->second);
                track.frames.push_back (f);
                track.freqs.push_back (freqs[p]);
                track.amps.push_back (amps[p]);
                track.sleep = 0;
            }
            else
            {
                ++track.sleep;
            }
        }

        for (size_t p = 0; p < freqs.size(); ++p)
        {
            if (peakMatched[p])
                continue;
            LiveTrack born;
            born.frames.push_back (f);
            born.freqs.push_back (freqs[p]);
            born.amps.push_back (amps[p]);
            live.push_back (std::move (born));
        }

        std::vector<LiveTrack> stillLive;
        stillLive.reserve (live.size());
        for (auto& track : live)
        {
            if (track.sleep > options.maxSleep)
                finished.push_back (finish (track));
            else
                stillLive.push_back (std::move (track));
        }
        live = std::move (stillLive);
    }

    for (const auto& track : live)
        finished.push_back (finish (track));

    for (auto& p : finished)
        if (p.length() >= options.minFrames)
            result.partials.push_back (std::move (p));

    return result;
}

} // namespace autosynth
