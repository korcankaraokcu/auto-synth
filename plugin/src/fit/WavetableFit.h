#pragma once

#include "fit/WaveformFit.h"

#include <array>
#include <vector>

namespace autosynth
{

// Fits an oscillator's harmonic frames from a source's tracked partials.
//
// This exists because of a measurement, not a wish for more knobs. A played
// note's harmonic balance *moves*: across the clarinet the second harmonic
// swings 4.3 dB while the third barely shifts. That is not brightness, so no
// cutoff setting reproduces it, and a fixed waveform cannot either -- fits with
// one measured 1.2 dB of drift against the target's 4.3.
//
// The degeneracy to be careful about is that a free-form spectrum and a filter
// explain the same signal, so letting both float finds a bright table behind a
// closed filter as readily as the truth. The order fixes it: the filter is
// frozen first, from the average profile, and the frames are fitted to what is
// left over. Nothing here searches the cutoff.
namespace WavetableFit
{

// How many frames a fit will ever spend. The format carries sixteen so a
// person can build one by hand; the fitter uses at most three, because three is
// what the evidence supports -- a played note's timbre trajectory is smooth,
// and beyond a start, a middle and an end the extra frames are fitting the
// analysis noise that the smoothing below exists to remove.
constexpr int kFittedFrames = 3;

struct Result
{
    // Whether the frames below should be *drawn* into the patch. False means
    // the oscillator keeps its generated frame, which is not "no wavetable" --
    // there is always a wavetable -- it is a table whose one frame is still the
    // waveform it was named as.
    bool useCustomFrames = false;
    int numFrames = 1;
    std::array<std::array<float, Oscillator::kFrameHarmonics>, kFittedFrames> frames {};
    float position = 0.0f;
    float envAmount = 0.0f;
    Adsr env { 0.05f, 0.4f, 0.6f, 0.2f, 0.0f };

    // All three rungs of the ladder, measured the same way, so they are
    // directly comparable and the decision can be read back out of them.
    double blendError = 0.0;    // the classic waveform blend
    double staticError = 0.0;   // one table, the whole-note mean profile
    double frameError = 0.0;    // three tables and a sweeping position

    // How far the tone travels, in decibels, measured the same way
    // `autosynth_diff` measures it.
    double driftDb = 0.0;
};

// `H` is harmonic-major (`H[k * numFrames + t]`), already deconvolved from the
// filter -- the same array the waveform match is given. `times` is one time per
// frame, in seconds.
// `blend` is the classic-waveform fit this would replace, and is what the table
// has to beat before it is worth having. `energyShare` is this source's energy
// as a fraction of the loudest source's.
Result fit (const float* H, int numHarmonics, int numFrames,
            const std::vector<float>& times, float gateSeconds, bool oneShot,
            const WaveformFit::Blend& blend, double energyShare);

} // namespace WavetableFit

} // namespace autosynth
