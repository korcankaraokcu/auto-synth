#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace autosynth
{

// Non-negative least squares (Lawson-Hanson active set), replacing
// scipy.optimize.nnls.
//
// Used to solve oscillator and noise levels: everything upstream of the filter
// is linear in those gains, so rendering each source in isolation and solving
// for the mix is exact rather than approximate. Non-negativity is the
// physically meaningful constraint -- a level cannot be negative, and allowing
// it would let sources cancel in ways the engine can never reproduce.
//
// The normal equations are used for the inner least-squares solve. That is
// usually a red flag for conditioning, but here the design matrix has at most
// four columns (three oscillators plus noise) against tens of thousands of
// rows, so A^T A is a 4x4 that is well away from singular.
namespace nnls
{

// `A` is column-major: column j occupies [j*rows, (j+1)*rows).
inline std::vector<double> solve (const std::vector<double>& A, const std::vector<double>& b,
                                  int rows, int cols, int maxIterations = 0, double tol = 1e-10)
{
    std::vector<double> x (static_cast<size_t> (cols), 0.0);
    if (rows <= 0 || cols <= 0)
        return x;
    if (maxIterations <= 0)
        maxIterations = 3 * cols;

    const auto column = [&A, rows] (int j) { return A.data() + static_cast<size_t> (j) * rows; };

    // Gram matrix and right-hand side, formed once.
    std::vector<double> gram (static_cast<size_t> (cols) * cols, 0.0);
    std::vector<double> atb (static_cast<size_t> (cols), 0.0);
    for (int i = 0; i < cols; ++i)
    {
        const auto* ci = column (i);
        for (int j = i; j < cols; ++j)
        {
            const auto* cj = column (j);
            double sum = 0.0;
            for (int r = 0; r < rows; ++r)
                sum += ci[r] * cj[r];
            gram[static_cast<size_t> (i) * cols + j] = sum;
            gram[static_cast<size_t> (j) * cols + i] = sum;
        }
        double sum = 0.0;
        for (int r = 0; r < rows; ++r)
            sum += ci[r] * b[static_cast<size_t> (r)];
        atb[static_cast<size_t> (i)] = sum;
    }

    std::vector<bool> passive (static_cast<size_t> (cols), false);

    // Least squares restricted to the passive set, via Cholesky on its Gram
    // submatrix. Returns false if the submatrix is not positive definite, which
    // means the columns are collinear and the set cannot be expanded.
    const auto solvePassive = [&] (std::vector<double>& s) -> bool
    {
        std::vector<int> index;
        for (int j = 0; j < cols; ++j)
            if (passive[static_cast<size_t> (j)])
                index.push_back (j);

        std::fill (s.begin(), s.end(), 0.0);
        const auto n = static_cast<int> (index.size());
        if (n == 0)
            return true;

        std::vector<double> L (static_cast<size_t> (n) * n, 0.0);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                auto sum = gram[static_cast<size_t> (index[i]) * cols + index[j]];
                for (int k = 0; k < j; ++k)
                    sum -= L[static_cast<size_t> (i) * n + k] * L[static_cast<size_t> (j) * n + k];

                if (i == j)
                {
                    if (sum <= 1e-12)
                        return false;
                    L[static_cast<size_t> (i) * n + j] = std::sqrt (sum);
                }
                else
                {
                    L[static_cast<size_t> (i) * n + j] = sum / L[static_cast<size_t> (j) * n + j];
                }
            }
        }

        std::vector<double> y (static_cast<size_t> (n), 0.0);
        for (int i = 0; i < n; ++i)
        {
            auto sum = atb[static_cast<size_t> (index[i])];
            for (int k = 0; k < i; ++k)
                sum -= L[static_cast<size_t> (i) * n + k] * y[static_cast<size_t> (k)];
            y[static_cast<size_t> (i)] = sum / L[static_cast<size_t> (i) * n + i];
        }
        for (int i = n - 1; i >= 0; --i)
        {
            auto sum = y[static_cast<size_t> (i)];
            for (int k = i + 1; k < n; ++k)
                sum -= L[static_cast<size_t> (k) * n + i] * s[static_cast<size_t> (index[k])];
            s[static_cast<size_t> (index[i])] = sum / L[static_cast<size_t> (i) * n + i];
        }
        return true;
    };

    std::vector<double> w (static_cast<size_t> (cols), 0.0);
    std::vector<double> s (static_cast<size_t> (cols), 0.0);

    for (int iteration = 0; iteration < maxIterations * 10; ++iteration)
    {
        // w = A^T (b - A x)
        for (int j = 0; j < cols; ++j)
        {
            auto sum = atb[static_cast<size_t> (j)];
            for (int k = 0; k < cols; ++k)
                sum -= gram[static_cast<size_t> (j) * cols + k] * x[static_cast<size_t> (k)];
            w[static_cast<size_t> (j)] = sum;
        }

        int best = -1;
        double bestValue = tol;
        for (int j = 0; j < cols; ++j)
            if (! passive[static_cast<size_t> (j)] && w[static_cast<size_t> (j)] > bestValue)
            {
                bestValue = w[static_cast<size_t> (j)];
                best = j;
            }
        if (best < 0)
            break;

        passive[static_cast<size_t> (best)] = true;
        if (! solvePassive (s))
        {
            passive[static_cast<size_t> (best)] = false;
            break;
        }

        int guard = 0;
        while (guard++ < maxIterations * 10)
        {
            double minValue = 0.0;
            bool anyNegative = false;
            for (int j = 0; j < cols; ++j)
                if (passive[static_cast<size_t> (j)] && s[static_cast<size_t> (j)] <= 0.0)
                {
                    anyNegative = true;
                    minValue = std::min (minValue, s[static_cast<size_t> (j)]);
                }
            if (! anyNegative)
                break;

            auto alpha = 1.0;
            for (int j = 0; j < cols; ++j)
            {
                if (! passive[static_cast<size_t> (j)] || s[static_cast<size_t> (j)] > 0.0)
                    continue;
                const auto denom = x[static_cast<size_t> (j)] - s[static_cast<size_t> (j)];
                if (denom > 1e-15)
                    alpha = std::min (alpha, x[static_cast<size_t> (j)] / denom);
            }

            for (int j = 0; j < cols; ++j)
                x[static_cast<size_t> (j)] += alpha * (s[static_cast<size_t> (j)] - x[static_cast<size_t> (j)]);

            for (int j = 0; j < cols; ++j)
                if (passive[static_cast<size_t> (j)] && x[static_cast<size_t> (j)] <= 1e-12)
                {
                    passive[static_cast<size_t> (j)] = false;
                    x[static_cast<size_t> (j)] = 0.0;
                }

            if (! solvePassive (s))
                break;
        }

        for (int j = 0; j < cols; ++j)
            x[static_cast<size_t> (j)] = passive[static_cast<size_t> (j)]
                                       ? std::max (s[static_cast<size_t> (j)], 0.0)
                                       : 0.0;
    }

    return x;
}

} // namespace nnls
} // namespace autosynth
