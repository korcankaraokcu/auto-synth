#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

namespace autosynth
{

// CMA-ES, replacing pycma.
//
// Chosen over gradient descent because the landscape is not convex and much of
// the parameter space is discrete -- and because Instrumental (arXiv 2603.15905)
// measured CMA-ES winning on exactly this problem.
//
// The eigendecomposition uses cyclic Jacobi rather than pulling in Eigen. For a
// symmetric positive-definite covariance of a few dozen dimensions Jacobi is
// accurate, unconditionally convergent, and about sixty lines -- a whole
// dependency for one call would have been the worse trade.
//
// **This cannot be bit-compatible with pycma.** Different RNG, different bound
// handling, different eigensolver ordering. Its conformance tests are therefore
// behavioural: it must improve on its starting point, must never return
// something worse, and must reach a comparable loss.
class CmaEs
{
public:
    struct Options
    {
        int populationSize = 16;
        double sigma = 0.12;
        int maxEvaluations = 192;
        unsigned seed = 1;
    };

    // `objective` scores a whole generation at once, so callers can render a
    // population in a single batched pass.
    using Objective = std::function<std::vector<double> (const std::vector<std::vector<double>>&)>;

    struct Result
    {
        std::vector<double> best;
        double bestValue = 0.0;
        double initialValue = 0.0;
        int evaluations = 0;
        bool improved = false;
    };

    static Result minimise (const std::vector<double>& x0, const Options& options,
                            const Objective& objective);

    // Cyclic Jacobi eigendecomposition of a symmetric matrix, in place.
    // `A` is row-major n*n; on return `eigenvalues` and `V` (columns are
    // eigenvectors) describe A = V diag(eigenvalues) V^T.
    static void jacobiEigen (std::vector<double> A, int n,
                             std::vector<double>& eigenvalues, std::vector<double>& V,
                             int maxSweeps = 60);
};

} // namespace autosynth
