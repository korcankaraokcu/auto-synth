#pragma once

#include <algorithm>
#include <vector>

namespace autosynth
{

// Ports of the three `scipy.ndimage` filters the fitting code uses, with
// `mode="nearest"` (edge replication).
//
// The window convention is copied exactly rather than approximated: for size N
// the window for output i spans inputs [i - N/2, i - N/2 + N - 1]. For odd N
// that is symmetric; for even N it leans one sample earlier. Getting this wrong
// shifts the gate-detection plateau by a frame or two, which does not look like
// a bug -- it looks like a slightly different note-off time.
namespace nd
{

inline int clampIndex (int i, int size) noexcept
{
    return std::min (std::max (i, 0), size - 1);
}

template <typename Reduce>
std::vector<float> filter1d (const std::vector<float>& x, int size, Reduce reduce, float init)
{
    const auto n = static_cast<int> (x.size());
    std::vector<float> out (x.size(), 0.0f);
    if (n == 0)
        return out;
    size = std::max (1, size);
    const auto lo = -(size / 2);

    for (int i = 0; i < n; ++i)
    {
        auto acc = init;
        for (int k = 0; k < size; ++k)
            acc = reduce (acc, x[static_cast<size_t> (clampIndex (i + lo + k, n))]);
        out[static_cast<size_t> (i)] = acc;
    }
    return out;
}

inline std::vector<float> uniformFilter1d (const std::vector<float>& x, int size)
{
    const auto n = static_cast<int> (x.size());
    std::vector<float> out (x.size(), 0.0f);
    if (n == 0)
        return out;
    size = std::max (1, size);
    const auto lo = -(size / 2);

    for (int i = 0; i < n; ++i)
    {
        double sum = 0.0;
        for (int k = 0; k < size; ++k)
            sum += x[static_cast<size_t> (clampIndex (i + lo + k, n))];
        out[static_cast<size_t> (i)] = static_cast<float> (sum / size);
    }
    return out;
}

inline std::vector<float> maximumFilter1d (const std::vector<float>& x, int size)
{
    return filter1d (x, size, [] (float a, float b) { return std::max (a, b); },
                     -std::numeric_limits<float>::infinity());
}

inline std::vector<float> minimumFilter1d (const std::vector<float>& x, int size)
{
    return filter1d (x, size, [] (float a, float b) { return std::min (a, b); },
                     std::numeric_limits<float>::infinity());
}

inline std::vector<float> medianFilter1d (const std::vector<float>& x, int size)
{
    const auto n = static_cast<int> (x.size());
    std::vector<float> out (x.size(), 0.0f);
    if (n == 0)
        return out;
    size = std::max (1, size);
    const auto lo = -(size / 2);

    std::vector<float> window (static_cast<size_t> (size));
    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < size; ++k)
            window[static_cast<size_t> (k)] = x[static_cast<size_t> (clampIndex (i + lo + k, n))];
        std::nth_element (window.begin(), window.begin() + size / 2, window.end());
        out[static_cast<size_t> (i)] = window[static_cast<size_t> (size / 2)];
    }
    return out;
}

} // namespace nd
} // namespace autosynth
