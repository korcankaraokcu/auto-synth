#include "fit/Nmf.h"

#include <algorithm>
#include <cmath>
#include <juce_core/juce_core.h>

namespace autosynth
{
namespace nmf
{
namespace
{

constexpr float kEps = 1.0e-9f;

double frobenius (const std::vector<float>& v) noexcept
{
    double acc = 0.0;
    for (const auto x : v)
        acc += static_cast<double> (x) * x;
    return std::sqrt (acc);
}

} // namespace

Factorisation factorise (const std::vector<float>& v, int numRows, int numCols, int rank,
                         int iterations, unsigned seed)
{
    Factorisation out;
    if (numRows <= 0 || numCols <= 0 || rank <= 0
        || static_cast<int> (v.size()) < numRows * numCols)
        return out;

    out.rank = rank;
    const auto r = static_cast<size_t> (rank);
    const auto rows = static_cast<size_t> (numRows);
    const auto cols = static_cast<size_t> (numCols);

    // Seeded rather than random. Multiplicative updates only find a local
    // optimum, so an unseeded start makes the *oscillator count* depend on a
    // clock -- and a fitter whose structural decisions are not reproducible
    // cannot be measured against ground truth at all.
    juce::Random rng (static_cast<juce::int64> (seed));
    const auto mean = [&v]
    {
        double acc = 0.0;
        for (const auto x : v)
            acc += x;
        return v.empty() ? 1.0 : std::max (acc / v.size(), 1.0e-6);
    }();

    out.w.assign (rows * r, 0.0f);
    out.h.assign (r * cols, 0.0f);
    for (auto& x : out.w)
        x = static_cast<float> (mean * (0.5 + rng.nextDouble()));
    for (auto& x : out.h)
        x = static_cast<float> (0.5 + rng.nextDouble());

    std::vector<float> wh (rows * cols, 0.0f);
    std::vector<float> numerator, denominator;

    const auto reconstruct = [&]
    {
        for (size_t i = 0; i < rows; ++i)
            for (size_t t = 0; t < cols; ++t)
            {
                float acc = 0.0f;
                for (size_t c = 0; c < r; ++c)
                    acc += out.w[c * rows + i] * out.h[c * cols + t];
                wh[i * cols + t] = acc;
            }
    };

    for (int it = 0; it < iterations; ++it)
    {
        reconstruct();

        // H <- H * (W^T V) / (W^T W H)
        numerator.assign (r * cols, 0.0f);
        denominator.assign (r * cols, 0.0f);
        for (size_t c = 0; c < r; ++c)
            for (size_t t = 0; t < cols; ++t)
            {
                float num = 0.0f, den = 0.0f;
                for (size_t i = 0; i < rows; ++i)
                {
                    const auto wic = out.w[c * rows + i];
                    num += wic * v[i * cols + t];
                    den += wic * wh[i * cols + t];
                }
                numerator[c * cols + t] = num;
                denominator[c * cols + t] = den;
            }
        for (size_t i = 0; i < out.h.size(); ++i)
            out.h[i] *= numerator[i] / std::max (denominator[i], kEps);

        reconstruct();

        // W <- W * (V H^T) / (W H H^T)
        numerator.assign (rows * r, 0.0f);
        denominator.assign (rows * r, 0.0f);
        for (size_t c = 0; c < r; ++c)
            for (size_t i = 0; i < rows; ++i)
            {
                float num = 0.0f, den = 0.0f;
                for (size_t t = 0; t < cols; ++t)
                {
                    const auto hct = out.h[c * cols + t];
                    num += v[i * cols + t] * hct;
                    den += wh[i * cols + t] * hct;
                }
                numerator[c * rows + i] = num;
                denominator[c * rows + i] = den;
            }
        for (size_t i = 0; i < out.w.size(); ++i)
            out.w[i] *= numerator[i] / std::max (denominator[i], kEps);
    }

    reconstruct();
    std::vector<float> residual (rows * cols, 0.0f);
    for (size_t i = 0; i < residual.size(); ++i)
        residual[i] = v[i] - wh[i];

    const auto scale = frobenius (v);
    out.error = scale > 1.0e-12 ? frobenius (residual) / scale : 0.0;

    // Scale each component so its profile peaks at one and its activation
    // carries the level. Without this the split between W and H is arbitrary,
    // and the profile is about to be read as a harmonic spectrum.
    for (size_t c = 0; c < r; ++c)
    {
        float peak = 0.0f;
        for (size_t i = 0; i < rows; ++i)
            peak = std::max (peak, out.w[c * rows + i]);
        if (peak <= kEps)
            continue;
        for (size_t i = 0; i < rows; ++i)
            out.w[c * rows + i] /= peak;
        for (size_t t = 0; t < cols; ++t)
            out.h[c * cols + t] *= peak;
    }

    return out;
}

RankChoice selectRank (const std::vector<float>& v, int numRows, int numCols, int maxRank,
                       double mustImproveBy, double mustExplain, double rankOneFailing)
{
    RankChoice choice;
    if (numRows <= 0 || numCols <= 0 || maxRank <= 0)
        return choice;

    const auto limit = juce::jmin (maxRank, numRows);
    for (int r = 1; r <= limit; ++r)
        choice.errorByRank.push_back (factorise (v, numRows, numCols, r).error);

    choice.rank = 1;
    if (choice.errorByRank.empty())
        return choice;

    const auto rankOneBad = choice.errorByRank.front() >= rankOneFailing;
    for (int r = 2; r <= limit; ++r)
    {
        const auto previous = choice.errorByRank[static_cast<size_t> (choice.rank - 1)];
        const auto here = choice.errorByRank[static_cast<size_t> (r - 1)];
        if (here <= previous * mustImproveBy && (here <= mustExplain || rankOneBad))
            choice.rank = r;
    }
    return choice;
}

} // namespace nmf

} // namespace autosynth
