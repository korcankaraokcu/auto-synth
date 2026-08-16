// Rank recovery on matrices whose rank we chose.
//
// This is the layer that decides whether a harmonic group holds one oscillator
// or two, so what it has to get right is the *count*, not the reconstruction.
// The tests are built the way the real matrices are: a spectral profile times
// an envelope, summed.

#include "Helpers.h"

#include "fit/Nmf.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autotest;

namespace
{

// One component: `profile` over harmonics, `envelope` over frames.
void addComponent (std::vector<float>& v, int numRows, int numCols,
                   const std::vector<float>& profile, const std::vector<float>& envelope)
{
    for (int k = 0; k < numRows; ++k)
        for (int t = 0; t < numCols; ++t)
            v[(size_t) (k * numCols + t)] +=
                profile[(size_t) k % profile.size()] * envelope[(size_t) t % envelope.size()];
}

std::vector<float> ramp (int n, float from, float to)
{
    std::vector<float> out ((size_t) n);
    for (int i = 0; i < n; ++i)
        out[(size_t) i] = from + (to - from) * (n > 1 ? (float) i / (n - 1) : 0.0f);
    return out;
}

} // namespace

TEST_CASE ("one oscillator factorises as rank one", "[nmf]")
{
    constexpr int rows = 16, cols = 120;
    std::vector<float> v ((size_t) (rows * cols), 0.0f);

    std::vector<float> saw ((size_t) rows);
    for (int k = 0; k < rows; ++k)
        saw[(size_t) k] = 1.0f / (k + 1);
    addComponent (v, rows, cols, saw, ramp (cols, 1.0f, 0.2f));

    const auto choice = autosynth::nmf::selectRank (v, rows, cols, 3);
    INFO ("errors: " << choice.errorByRank[0] << " " << choice.errorByRank[1]);
    CHECK (choice.rank == 1);
    CHECK (choice.errorByRank[0] < 0.02);
}

TEST_CASE ("an octave pair factorises as rank two", "[nmf]")
{
    // The case grouping cannot split: a source an octave up puts all of its
    // energy on the even harmonics of the one below, so both arrive as a single
    // harmonic group. They are still two components, because their envelopes
    // differ -- and if they did not, they would be one oscillator with a
    // different waveform, which is a true statement about the sound.
    constexpr int rows = 16, cols = 120;
    std::vector<float> v ((size_t) (rows * cols), 0.0f);

    std::vector<float> low ((size_t) rows, 0.0f), octave ((size_t) rows, 0.0f);
    for (int k = 0; k < rows; ++k)
    {
        low[(size_t) k] = 1.0f / (k + 1);
        if ((k + 1) % 2 == 0)
            octave[(size_t) k] = 1.0f / ((k + 1) / 2);
    }

    addComponent (v, rows, cols, low, ramp (cols, 1.0f, 0.3f));
    addComponent (v, rows, cols, octave, ramp (cols, 0.2f, 1.0f));

    const auto choice = autosynth::nmf::selectRank (v, rows, cols, 3);
    INFO ("errors: " << choice.errorByRank[0] << " " << choice.errorByRank[1]
                     << " " << choice.errorByRank[2]);
    CHECK (choice.rank == 2);
}

TEST_CASE ("two sources sharing one envelope are one component", "[nmf]")
{
    // Deliberately *not* rank two. Two spectra that rise and fall together are
    // indistinguishable from one oscillator whose waveform is their sum -- that
    // is not a limitation of the factorisation, it is what the signal contains,
    // and a fitter that reported two here would be inventing evidence.
    constexpr int rows = 16, cols = 120;
    std::vector<float> v ((size_t) (rows * cols), 0.0f);

    std::vector<float> a ((size_t) rows, 0.0f), b ((size_t) rows, 0.0f);
    for (int k = 0; k < rows; ++k)
    {
        a[(size_t) k] = 1.0f / (k + 1);
        if ((k + 1) % 2 == 0)
            b[(size_t) k] = 1.0f / ((k + 1) / 2);
    }

    const auto shared = ramp (cols, 1.0f, 0.4f);
    addComponent (v, rows, cols, a, shared);
    addComponent (v, rows, cols, b, shared);

    const auto choice = autosynth::nmf::selectRank (v, rows, cols, 3);
    CHECK (choice.rank == 1);
}

TEST_CASE ("the factorisation is reproducible", "[nmf]")
{
    // The oscillator count must not depend on a clock. Multiplicative updates
    // find a local optimum, so an unseeded start would make the structure of
    // the patch vary run to run -- and nothing downstream could be measured.
    constexpr int rows = 12, cols = 80;
    std::vector<float> v ((size_t) (rows * cols), 0.0f);
    addComponent (v, rows, cols, ramp (rows, 1.0f, 0.1f), ramp (cols, 0.3f, 1.0f));

    const auto first = autosynth::nmf::factorise (v, rows, cols, 2);
    const auto second = autosynth::nmf::factorise (v, rows, cols, 2);
    REQUIRE (first.w.size() == second.w.size());
    for (size_t i = 0; i < first.w.size(); ++i)
        CHECK (first.w[i] == Catch::Approx (second.w[i]));
    CHECK (first.error == Catch::Approx (second.error));
}

TEST_CASE ("each component's profile is peak-normalised", "[nmf]")
{
    // Because it is about to be read as a harmonic spectrum by the waveform
    // matcher, which normalises anyway -- but a profile that carries the level
    // makes every intermediate print unreadable.
    constexpr int rows = 10, cols = 60;
    std::vector<float> v ((size_t) (rows * cols), 0.0f);
    addComponent (v, rows, cols, ramp (rows, 1.0f, 0.2f), ramp (cols, 1.0f, 0.5f));

    const auto f = autosynth::nmf::factorise (v, rows, cols, 1);
    float peak = 0.0f;
    for (int k = 0; k < rows; ++k)
        peak = juce::jmax (peak, f.w[(size_t) k]);
    CHECK (peak == Catch::Approx (1.0f).margin (1.0e-4));
}
