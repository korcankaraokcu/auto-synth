#include "fit/FilterFit.h"

#include "dsp/FilterResponse.h"
#include "fit/NdFilters.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace autosynth
{
namespace
{
inline float at (const std::vector<float>& H, int numFrames, int k, int t) noexcept
{
    return H[static_cast<size_t> (k) * static_cast<size_t> (numFrames) + static_cast<size_t> (t)];
}
} // namespace

FilterFit::Trajectory FilterFit::estimateCutoffTrajectory (const std::vector<float>& H,
                                                           int numHarmonics, int numFrames,
                                                           double f0Hz, double sampleRate,
                                                           float q, FilterType type,
                                                           int numIterations, int smoothFrames)
{
    Trajectory result;
    result.cutoffHz.assign (static_cast<size_t> (numFrames), 0.0f);
    result.source.assign (static_cast<size_t> (numHarmonics), 1.0f);
    if (numHarmonics <= 0 || numFrames <= 0)
        return result;

    // geomspace(max(f0, 30), max(f0 * 1.2, sr * 0.45), 40)
    const auto lo = std::log (std::max (f0Hz, 30.0));
    const auto hi = std::log (std::max (f0Hz * 1.2, sampleRate * 0.45));
    std::vector<double> cutoffs (kNumCutoffGrid);
    for (int c = 0; c < kNumCutoffGrid; ++c)
    {
        const auto t = static_cast<double> (c) / (kNumCutoffGrid - 1);
        cutoffs[static_cast<size_t> (c)] = std::exp (lo + t * (hi - lo));
    }

    // resp[c][k]
    std::vector<float> resp (static_cast<size_t> (kNumCutoffGrid) * static_cast<size_t> (numHarmonics));
    for (int c = 0; c < kNumCutoffGrid; ++c)
        for (int k = 0; k < numHarmonics; ++k)
            resp[static_cast<size_t> (c) * numHarmonics + k] = std::max (
                analogMagnitude (type, static_cast<float> ((k + 1) * f0Hz),
                                 static_cast<float> (cutoffs[static_cast<size_t> (c)]), q),
                1.0e-6f);

    std::vector<float> logH (H.size());
    for (size_t i = 0; i < H.size(); ++i)
        logH[i] = static_cast<float> (std::log10 (H[i] + 1.0e-6));

    std::vector<double> energy (static_cast<size_t> (numFrames), 0.0);
    double peakEnergy = 0.0;
    for (int t = 0; t < numFrames; ++t)
    {
        for (int k = 0; k < numHarmonics; ++k)
            energy[static_cast<size_t> (t)] += at (H, numFrames, k, t);
        peakEnergy = std::max (peakEnergy, energy[static_cast<size_t> (t)]);
    }

    std::vector<bool> active (static_cast<size_t> (numFrames), false);
    bool anyActive = false;
    const auto energyFloor = std::max (1.0e-9, 0.01 * peakEnergy);
    for (int t = 0; t < numFrames; ++t)
    {
        active[static_cast<size_t> (t)] = energy[static_cast<size_t> (t)] > energyFloor;
        anyActive = anyActive || active[static_cast<size_t> (t)];
    }

    if (! anyActive)
    {
        std::fill (result.cutoffHz.begin(), result.cutoffHz.end(),
                   static_cast<float> (cutoffs.back()));
        return result;
    }

    const auto peakH = H.empty() ? 0.0f : *std::max_element (H.begin(), H.end());

    std::vector<int> index (static_cast<size_t> (numFrames), kNumCutoffGrid - 1);
    std::vector<double> source (static_cast<size_t> (numHarmonics), 1.0);

    for (int iteration = 0; iteration < numIterations; ++iteration)
    {
        // --- source spectrum given the current filter ---------------------
        double weightSum = 0.0;
        std::fill (source.begin(), source.end(), 0.0);
        for (int t = 0; t < numFrames; ++t)
        {
            if (! active[static_cast<size_t> (t)])
                continue;
            const auto w = energy[static_cast<size_t> (t)];
            weightSum += w;
            const auto* r = resp.data() + static_cast<size_t> (index[static_cast<size_t> (t)]) * numHarmonics;
            for (int k = 0; k < numHarmonics; ++k)
                source[static_cast<size_t> (k)] += (at (H, numFrames, k, t) / r[k]) * w;
        }
        for (auto& s : source)
            s /= (weightSum + 1.0e-12);

        const auto peakSource = *std::max_element (source.begin(), source.end());
        if (peakSource > 1.0e-12)
            for (auto& s : source)
                s /= peakSource;
        else
            std::fill (source.begin(), source.end(), 1.0);

        // --- filter given the current source spectrum ----------------------
        // Per-(frame, cutoff) optimal log-gain in closed form, so overall level
        // does not leak into the cutoff estimate.
        for (int t = 0; t < numFrames; ++t)
        {
            auto best = std::numeric_limits<double>::infinity();
            auto bestIndex = index[static_cast<size_t> (t)];

            for (int c = 0; c < kNumCutoffGrid; ++c)
            {
                const auto* r = resp.data() + static_cast<size_t> (c) * numHarmonics;
                double weighted = 0.0, weightTotal = 0.0;
                for (int k = 0; k < numHarmonics; ++k)
                {
                    const auto w = at (H, numFrames, k, t) / (peakH + 1.0e-12f) + 1.0e-3f;
                    const auto model = std::log10 (source[static_cast<size_t> (k)] * r[k] + 1.0e-6);
                    weighted += w * (at (logH, numFrames, k, t) - model);
                    weightTotal += w;
                }
                const auto gain = weighted / std::max (weightTotal, 1.0e-12);

                double error = 0.0;
                for (int k = 0; k < numHarmonics; ++k)
                {
                    const auto w = at (H, numFrames, k, t) / (peakH + 1.0e-12f) + 1.0e-3f;
                    const auto model = std::log10 (source[static_cast<size_t> (k)] * r[k] + 1.0e-6);
                    const auto residual = at (logH, numFrames, k, t) - model - gain;
                    error += w * residual * residual;
                }

                if (error < best)
                {
                    best = error;
                    bestIndex = c;
                }
            }
            index[static_cast<size_t> (t)] = bestIndex;
        }
    }

    for (int t = 0; t < numFrames; ++t)
        result.cutoffHz[static_cast<size_t> (t)] =
            static_cast<float> (cutoffs[static_cast<size_t> (index[static_cast<size_t> (t)])]);

    // Hold the last confident value through silent frames rather than letting
    // them swing the trajectory.
    std::vector<int> valid;
    for (int t = 0; t < numFrames; ++t)
        if (active[static_cast<size_t> (t)])
            valid.push_back (t);

    if (! valid.empty() && static_cast<int> (valid.size()) < numFrames)
    {
        std::vector<float> interpolated (static_cast<size_t> (numFrames));
        for (int t = 0; t < numFrames; ++t)
        {
            if (t <= valid.front())
                interpolated[static_cast<size_t> (t)] = result.cutoffHz[static_cast<size_t> (valid.front())];
            else if (t >= valid.back())
                interpolated[static_cast<size_t> (t)] = result.cutoffHz[static_cast<size_t> (valid.back())];
            else
            {
                const auto upper = std::lower_bound (valid.begin(), valid.end(), t);
                const auto hiIndex = *upper;
                const auto loIndex = *(upper - 1);
                const auto span = static_cast<float> (hiIndex - loIndex);
                const auto frac = span > 0.0f ? (t - loIndex) / span : 0.0f;
                const auto a = result.cutoffHz[static_cast<size_t> (loIndex)];
                const auto b = result.cutoffHz[static_cast<size_t> (hiIndex)];
                interpolated[static_cast<size_t> (t)] = a + (b - a) * frac;
            }
        }
        result.cutoffHz = std::move (interpolated);
    }

    if (smoothFrames > 1 && numFrames > smoothFrames)
    {
        // Median first to drop single-frame outliers, then mean to smooth. A
        // cutoff trajectory is an envelope, not a per-frame decision.
        result.cutoffHz = nd::medianFilter1d (result.cutoffHz, smoothFrames);
        result.cutoffHz = nd::uniformFilter1d (result.cutoffHz, smoothFrames);
    }

    for (int k = 0; k < numHarmonics; ++k)
        result.source[static_cast<size_t> (k)] = static_cast<float> (source[static_cast<size_t> (k)]);
    return result;
}

std::vector<float> FilterFit::deconvolve (const std::vector<float>& H,
                                          int numHarmonics, int numFrames,
                                          const std::vector<float>& cutoffHz, double f0Hz,
                                          float q, FilterType type)
{
    std::vector<float> out (H.size(), 0.0f);
    for (int k = 0; k < numHarmonics; ++k)
    {
        const auto freq = static_cast<float> ((k + 1) * f0Hz);
        for (int t = 0; t < numFrames; ++t)
        {
            const auto cutoff = t < static_cast<int> (cutoffHz.size())
                              ? cutoffHz[static_cast<size_t> (t)]
                              : cutoffHz.empty() ? 1.0f : cutoffHz.back();
            const auto response = std::max (analogMagnitude (type, freq, cutoff, q), 1.0e-4f);
            out[static_cast<size_t> (k) * numFrames + t] = at (H, numFrames, k, t) / response;
        }
    }
    return out;
}

FilterFit::EnvSplit FilterFit::trajectoryToEnv (const std::vector<float>& cutoffHz,
                                                double anchorHz, float minOctaves)
{
    EnvSplit split;
    split.baseCutoffHz = static_cast<float> (anchorHz);
    split.shape.assign (cutoffHz.size(), 0.0f);
    if (cutoffHz.empty())
        return split;

    std::vector<double> u (cutoffHz.size());
    double mean = 0.0;
    for (size_t i = 0; i < cutoffHz.size(); ++i)
    {
        u[i] = std::log2 (std::max (cutoffHz[i], 20.0f));
        mean += u[i];
    }
    mean /= static_cast<double> (u.size());

    auto lo = std::numeric_limits<double>::infinity();
    auto hi = -std::numeric_limits<double>::infinity();
    for (auto& value : u)
    {
        value -= mean;
        lo = std::min (lo, value);
        hi = std::max (hi, value);
    }

    const auto span = hi - lo;
    if (span < minOctaves)
        return split;

    split.baseCutoffHz = static_cast<float> (anchorHz * std::pow (2.0, lo));
    split.envAmountOctaves = static_cast<float> (span);
    for (size_t i = 0; i < u.size(); ++i)
        split.shape[i] = static_cast<float> ((u[i] - lo) / span);
    return split;
}

} // namespace autosynth
