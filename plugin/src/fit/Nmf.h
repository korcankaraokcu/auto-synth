#pragma once

#include <vector>

namespace autosynth
{

// Non-negative matrix factorisation, and the rank selection built on it.
//
// The job is one question: does this harmonic group hold *one* oscillator or
// several? Grouping answers "which partials share a fundamental", and stops
// there. But two oscillators an octave apart share every harmonic index the
// lower one has, so grouping hands them over as a single source and the count
// comes back one short. Measured on the recovery harness, that is every
// remaining under-count: `7 and 19`, `-8 and 17`, `-3 and 6`.
//
// What separates them inside the group is that they are *different columns*. A
// harmonic amplitude matrix H[k, t] built from one oscillator is rank one -- a
// fixed spectral profile times a single envelope -- and two oscillators make it
// rank two, whatever their interval. Factorising and reading the rank is
// therefore the right question asked in the right place: per group, after
// grouping, rather than globally on a spectrogram where the octave problem is
// hopeless and a fifth is not even sampled.
//
// Multiplicative updates (Lee and Seung) on the Frobenius objective. Chosen
// over anything fancier because the matrices are tiny -- at most 32 harmonics
// by a few hundred frames -- and because the update is four lines that cannot
// wander outside the non-negative orthant, which matters when the result is
// going to be read as "an oscillator's spectrum".
namespace nmf
{

struct Factorisation
{
    // W is [numRows * rank], column-major by component: component r's spectral
    // profile occupies [r * numRows, (r + 1) * numRows).
    std::vector<float> w;
    // H is [rank * numCols], row-major by component: component r's activation
    // over time occupies [r * numCols, (r + 1) * numCols).
    std::vector<float> h;
    int rank = 0;

    // Relative reconstruction error, ||V - WH|| / ||V||. Comparable across
    // ranks, which is what makes the selection below a comparison rather than
    // a threshold.
    double error = 1.0;
};

// `v` is row-major: row i occupies [i * numCols, (i + 1) * numCols).
Factorisation factorise (const std::vector<float>& v, int numRows, int numCols, int rank,
                         int iterations = 200, unsigned seed = 0x9e3779b9u);

// The smallest rank that explains the matrix.
//
// Not the rank that minimises error -- that is always `maxRank`, because every
// extra component can only fit more. A component has to *earn* itself by
// reducing the error by a real margin, which is the same parsimony rule the
// wavetable ladder and the waveform blend both use, and for the same reason:
// the deliverable is a patch a person can read, and an oscillator that buys
// nothing is a line in that patch which means nothing.
struct RankChoice
{
    int rank = 1;
    std::vector<double> errorByRank;   // index 0 is rank 1
};

// A ratio alone is meaningless: a real oscillator's harmonics are never
// *exactly* rank one -- the tracker adds noise, the note is slightly
// inharmonic, its timbre drifts -- so a second component always buys something,
// and taken greedily it split eight of twelve single-oscillator targets in two.
//
// So there are two kinds of absolute evidence, and either will do, because they catch
// different cases and each on its own gets one of them wrong.
//
// `mustExplain`: the higher rank reconstructs the matrix almost exactly. Two
// sources are two components with nothing left over -- the synthetic octave
// pair reaches 2e-7 -- while one drifting source is never fully explained by a
// second component, only chipped at.
//
// `rankOneFailing`: rank one is so far off that whatever is there is not one
// oscillator, however imperfectly two describe it. This is what catches real
// recordings of two instruments, where neither rank reconstructs cleanly.
//
// Requiring *both* refuses the clean synthetic octave whenever rank one happens
// to score tolerably. Requiring neither splits every real solo note in three.
RankChoice selectRank (const std::vector<float>& v, int numRows, int numCols, int maxRank,
                       double mustImproveBy = 0.45, double mustExplain = 0.05,
                       double rankOneFailing = 0.40);

} // namespace nmf

} // namespace autosynth
