#include "fit/CmaEs.h"

#include <limits>
#include <numeric>

namespace autosynth
{
namespace
{
inline double& el (std::vector<double>& M, int n, int i, int j) noexcept
{
    return M[static_cast<size_t> (i) * n + j];
}
inline double el (const std::vector<double>& M, int n, int i, int j) noexcept
{
    return M[static_cast<size_t> (i) * n + j];
}
} // namespace

void CmaEs::jacobiEigen (std::vector<double> A, int n,
                         std::vector<double>& eigenvalues, std::vector<double>& V,
                         int maxSweeps)
{
    V.assign (static_cast<size_t> (n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        el (V, n, i, i) = 1.0;

    for (int sweep = 0; sweep < maxSweeps; ++sweep)
    {
        double off = 0.0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
                off += el (A, n, p, q) * el (A, n, p, q);
        if (off < 1.0e-24)
            break;

        for (int p = 0; p < n; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                const auto apq = el (A, n, p, q);
                if (std::abs (apq) < 1.0e-18)
                    continue;

                const auto theta = (el (A, n, q, q) - el (A, n, p, p)) / (2.0 * apq);
                const auto sign = theta >= 0.0 ? 1.0 : -1.0;
                const auto t = sign / (std::abs (theta) + std::sqrt (theta * theta + 1.0));
                const auto c = 1.0 / std::sqrt (t * t + 1.0);
                const auto s = t * c;

                for (int k = 0; k < n; ++k)
                {
                    const auto akp = el (A, n, k, p);
                    const auto akq = el (A, n, k, q);
                    el (A, n, k, p) = c * akp - s * akq;
                    el (A, n, k, q) = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k)
                {
                    const auto apk = el (A, n, p, k);
                    const auto aqk = el (A, n, q, k);
                    el (A, n, p, k) = c * apk - s * aqk;
                    el (A, n, q, k) = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k)
                {
                    const auto vkp = el (V, n, k, p);
                    const auto vkq = el (V, n, k, q);
                    el (V, n, k, p) = c * vkp - s * vkq;
                    el (V, n, k, q) = s * vkp + c * vkq;
                }
            }
        }
    }

    eigenvalues.assign (static_cast<size_t> (n), 0.0);
    for (int i = 0; i < n; ++i)
        eigenvalues[static_cast<size_t> (i)] = el (A, n, i, i);
}

CmaEs::Result CmaEs::minimise (const std::vector<double>& x0, const Options& options,
                               const Objective& objective)
{
    Result result;
    const auto n = static_cast<int> (x0.size());
    result.best = x0;
    if (n < 2)
        return result;

    const auto lambda = std::max (4, options.populationSize);
    const auto mu = lambda / 2;

    std::vector<double> weights (static_cast<size_t> (mu));
    for (int i = 0; i < mu; ++i)
        weights[static_cast<size_t> (i)] = std::log (mu + 0.5) - std::log (static_cast<double> (i + 1));
    const auto weightSum = std::accumulate (weights.begin(), weights.end(), 0.0);
    for (auto& w : weights)
        w /= weightSum;

    double squaredSum = 0.0;
    for (auto w : weights)
        squaredSum += w * w;
    const auto mueff = 1.0 / squaredSum;

    const auto nd = static_cast<double> (n);
    const auto cc = (4.0 + mueff / nd) / (nd + 4.0 + 2.0 * mueff / nd);
    const auto cs = (mueff + 2.0) / (nd + mueff + 5.0);
    const auto c1 = 2.0 / ((nd + 1.3) * (nd + 1.3) + mueff);
    const auto cmu = std::min (1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff)
                                             / ((nd + 2.0) * (nd + 2.0) + mueff));
    const auto damps = 1.0 + 2.0 * std::max (0.0, std::sqrt ((mueff - 1.0) / (nd + 1.0)) - 1.0) + cs;
    const auto chiN = std::sqrt (nd) * (1.0 - 1.0 / (4.0 * nd) + 1.0 / (21.0 * nd * nd));

    std::vector<double> xmean = x0;
    auto sigma = options.sigma;

    std::vector<double> pc (static_cast<size_t> (n), 0.0);
    std::vector<double> ps (static_cast<size_t> (n), 0.0);
    std::vector<double> C (static_cast<size_t> (n) * n, 0.0);
    std::vector<double> B (static_cast<size_t> (n) * n, 0.0);
    std::vector<double> D (static_cast<size_t> (n), 1.0);
    std::vector<double> invSqrtC (static_cast<size_t> (n) * n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        el (C, n, i, i) = 1.0;
        el (B, n, i, i) = 1.0;
        el (invSqrtC, n, i, i) = 1.0;
    }

    std::mt19937 rng (options.seed);
    std::normal_distribution<double> gauss (0.0, 1.0);

    const auto clamp01 = [] (std::vector<double> v)
    {
        for (auto& x : v)
            x = std::min (std::max (x, 0.0), 1.0);
        return v;
    };

    result.initialValue = objective ({ clamp01 (x0) }).front();
    result.bestValue = result.initialValue;
    result.evaluations = 1;
    if (! std::isfinite (result.initialValue))
        return result;

    auto eigenEval = 0;

    while (result.evaluations < options.maxEvaluations)
    {
        std::vector<std::vector<double>> population;
        std::vector<std::vector<double>> raw;
        population.reserve (static_cast<size_t> (lambda));
        raw.reserve (static_cast<size_t> (lambda));

        for (int k = 0; k < lambda; ++k)
        {
            std::vector<double> z (static_cast<size_t> (n));
            for (auto& v : z)
                v = gauss (rng);

            std::vector<double> candidate (static_cast<size_t> (n), 0.0);
            for (int i = 0; i < n; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                    sum += el (B, n, i, j) * D[static_cast<size_t> (j)] * z[static_cast<size_t> (j)];
                candidate[static_cast<size_t> (i)] = xmean[static_cast<size_t> (i)] + sigma * sum;
            }
            raw.push_back (candidate);
            population.push_back (clamp01 (candidate));
        }

        const auto values = objective (population);
        result.evaluations += lambda;

        std::vector<int> order (static_cast<size_t> (lambda));
        std::iota (order.begin(), order.end(), 0);
        std::sort (order.begin(), order.end(),
                   [&values] (int a, int b) { return values[static_cast<size_t> (a)]
                                                   < values[static_cast<size_t> (b)]; });

        if (values[static_cast<size_t> (order.front())] < result.bestValue)
        {
            result.bestValue = values[static_cast<size_t> (order.front())];
            result.best = population[static_cast<size_t> (order.front())];
        }

        const auto xold = xmean;
        std::fill (xmean.begin(), xmean.end(), 0.0);
        for (int i = 0; i < mu; ++i)
        {
            const auto& candidate = raw[static_cast<size_t> (order[static_cast<size_t> (i)])];
            for (int j = 0; j < n; ++j)
                xmean[static_cast<size_t> (j)] += weights[static_cast<size_t> (i)]
                                                * candidate[static_cast<size_t> (j)];
        }

        std::vector<double> diff (static_cast<size_t> (n));
        for (int j = 0; j < n; ++j)
            diff[static_cast<size_t> (j)] = (xmean[static_cast<size_t> (j)]
                                             - xold[static_cast<size_t> (j)]) / sigma;

        std::vector<double> invDiff (static_cast<size_t> (n), 0.0);
        for (int i = 0; i < n; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < n; ++j)
                sum += el (invSqrtC, n, i, j) * diff[static_cast<size_t> (j)];
            invDiff[static_cast<size_t> (i)] = sum;
        }

        for (int j = 0; j < n; ++j)
            ps[static_cast<size_t> (j)] = (1.0 - cs) * ps[static_cast<size_t> (j)]
                                        + std::sqrt (cs * (2.0 - cs) * mueff) * invDiff[static_cast<size_t> (j)];

        double psNorm = 0.0;
        for (auto v : ps)
            psNorm += v * v;
        psNorm = std::sqrt (psNorm);

        const auto generation = static_cast<double> (result.evaluations) / lambda;
        const auto hsig = psNorm / std::sqrt (1.0 - std::pow (1.0 - cs, 2.0 * generation)) / chiN
                        < 1.4 + 2.0 / (nd + 1.0);

        for (int j = 0; j < n; ++j)
            pc[static_cast<size_t> (j)] = (1.0 - cc) * pc[static_cast<size_t> (j)]
                                        + (hsig ? std::sqrt (cc * (2.0 - cc) * mueff) : 0.0)
                                              * diff[static_cast<size_t> (j)];

        const auto decay = 1.0 - c1 - cmu;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                auto value = decay * el (C, n, i, j)
                           + c1 * (pc[static_cast<size_t> (i)] * pc[static_cast<size_t> (j)]
                                   + (hsig ? 0.0 : cc * (2.0 - cc)) * el (C, n, i, j));
                for (int k = 0; k < mu; ++k)
                {
                    const auto& candidate = raw[static_cast<size_t> (order[static_cast<size_t> (k)])];
                    const auto di = (candidate[static_cast<size_t> (i)] - xold[static_cast<size_t> (i)]) / sigma;
                    const auto dj = (candidate[static_cast<size_t> (j)] - xold[static_cast<size_t> (j)]) / sigma;
                    value += cmu * weights[static_cast<size_t> (k)] * di * dj;
                }
                el (C, n, i, j) = value;
            }

        sigma *= std::exp ((cs / damps) * (psNorm / chiN - 1.0));
        sigma = std::min (std::max (sigma, 1.0e-8), 1.0);

        // Re-decomposing every generation would dominate the cost; the standard
        // schedule amortises it against the population size.
        if (result.evaluations - eigenEval > lambda / (c1 + cmu) / nd / 10.0)
        {
            eigenEval = result.evaluations;

            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j)
                {
                    const auto avg = 0.5 * (el (C, n, i, j) + el (C, n, j, i));
                    el (C, n, i, j) = avg;
                    el (C, n, j, i) = avg;
                }

            std::vector<double> eigenvalues;
            jacobiEigen (C, n, eigenvalues, B);
            for (int i = 0; i < n; ++i)
                D[static_cast<size_t> (i)] = std::sqrt (std::max (eigenvalues[static_cast<size_t> (i)], 1.0e-20));

            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < n; ++k)
                        sum += el (B, n, i, k) * (1.0 / D[static_cast<size_t> (k)]) * el (B, n, j, k);
                    el (invSqrtC, n, i, j) = sum;
                }
        }
    }

    result.improved = result.bestValue < result.initialValue;
    if (! result.improved)
        result.best = x0;
    return result;
}

} // namespace autosynth
