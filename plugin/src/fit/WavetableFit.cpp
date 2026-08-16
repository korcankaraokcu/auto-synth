#include "fit/WavetableFit.h"

#include "fit/WaveformFit.h"

#include <algorithm>
#include <cmath>
#include <juce_core/juce_core.h>

namespace autosynth
{
namespace
{

constexpr auto kNumFrames = WavetableFit::kFittedFrames;
constexpr auto kNumHarmonics = Oscillator::kFrameHarmonics;

// Frames below this share of the loudest frame's energy are not timbre, they
// are the attack transient and the tail. Reading a profile there measures the
// noise floor and the room.
constexpr auto kSoundingFloor = 0.1f;

// Half-width of the smoothing applied before either model is scored. A vibrato
// slower than about 3 Hz is not vibrato, so a sixth of a second either side
// removes the modulation without flattening a timbre sweep, which happens over
// the whole note rather than over a fraction of it.
constexpr auto kSmoothSeconds = 0.15f;

// A ladder of three models, each rung costing more parameters than the last:
//
//   1. a blend of two classic shapes  -- what the oscillator already does
//   2. one sixteen-harmonic table     -- fixes the *shape*
//   3. three tables and a position    -- fixes the shape *moving*
//
// A rung is only taken if it beats the one below by a margin, which is the same
// rule the waveform blend applies to itself. It matters more here: a table is
// the least legible thing in the patch, and the project's output is meant to be
// a patch a person can read and edit, not a spectrum dump with a play button.
constexpr auto kTableMustBeatBlendBy = 0.75;

// With the absolute half that the blend's own parsimony rule needed for the
// same reason: a ratio is meaningless once the simpler model already fits. A
// clean saw scores about a thousandth here, and a table beat that too -- by
// fitting the third decimal place of a profile that was already right. Every
// golden tone came back as a wavetable, and several changed oscillator count
// because the level solve was then working against a different source. The
// same number the waveform blend uses, on the same scale.
constexpr auto kBlendGoodEnough = 0.05;

// And only for a source loud enough for its spectrum to be worth believing.
//
// A quiet layer's harmonic profile is the least reliable measurement in the
// analysis: it is read on top of a louder sound, and partial tracking hands a
// shared partial to whichever source is stronger. Spending the most expressive
// model exactly where the evidence is weakest is backwards, and it showed --
// every golden tone grew a spurious second oscillator carrying a table fitted
// to tracking noise, and because that oscillator then rendered differently the
// level solve disabled it, changing the source count of fits that had nothing
// to do with wavetables.
constexpr auto kMinEnergyShare = 0.25;
constexpr auto kFramesMustBeatTableBy = 0.80;

// And a floor in decibels for the third rung, stated in the same units and
// measured the same way as the timbre drift `autosynth_diff` reports -- the
// diagnostic that says a fit is too static and the fitter that decides to fix
// it have to be measuring the same quantity, or one will keep reporting a
// problem the other has decided does not exist.
//
// Set from what the third rung is actually worth. Forced on regardless, the
// clarinet's rendered drift went from 2.9 dB short of its target to 1.0 dB
// over it and its harmonic profile improved slightly; the violin's landed in
// the same place either way, so it pays thirty-two numbers for nothing. Their
// measured movement is 1.8 dB and 1.4 dB, and the threshold sits between them.
constexpr auto kMinDriftDb = 1.5;

// Mean absolute change in the first few harmonics between two profiles, in
// decibels. Peak-normalised first, so this is a statement about balance rather
// than about level.
constexpr int kDriftHarmonics = 6;

double driftDb (const std::array<float, kNumHarmonics>& a,
                const std::array<float, kNumHarmonics>& b)
{
    const auto peakA = *std::max_element (a.begin(), a.end());
    const auto peakB = *std::max_element (b.begin(), b.end());
    if (peakA <= 0.0f || peakB <= 0.0f)
        return 0.0;

    double total = 0.0;
    for (int k = 0; k < kDriftHarmonics && k < kNumHarmonics; ++k)
    {
        const auto x = 20.0 * std::log10 (std::max (a[static_cast<size_t> (k)] / peakA, 1.0e-4f));
        const auto y = 20.0 * std::log10 (std::max (b[static_cast<size_t> (k)] / peakB, 1.0e-4f));
        total += std::abs (y - x);
    }
    return total / kDriftHarmonics;
}

// Energy-weighted mean over [from, to). Weighted so a loud frame counts for
// more than a quiet one, which is the same rule the whole-note profile uses.
std::vector<float> meanOver (const float* H, int numHarmonics, int numFrames,
                             const std::vector<float>& energy, int from, int to)
{
    const auto n = static_cast<size_t> (juce::jmin (numHarmonics, kNumHarmonics));
    std::vector<double> acc (n, 0.0);
    double weightSum = 0.0;

    for (int t = from; t < to; ++t)
    {
        const auto w = static_cast<double> (energy[static_cast<size_t> (t)]);
        if (w <= 0.0)
            continue;
        weightSum += w;
        for (size_t k = 0; k < n; ++k)
            acc[k] += w * H[k * static_cast<size_t> (numFrames) + static_cast<size_t> (t)];
    }

    std::vector<float> out (n, 0.0f);
    if (weightSum <= 0.0)
        return out;
    for (size_t k = 0; k < n; ++k)
        out[k] = static_cast<float> (acc[k] / weightSum);
    return out;
}

// Constant energy across frames, so moving the position changes the tone and
// not the volume -- the amp envelope already owns the loudness, and a frame
// carrying its own would be counted twice.
void unitEnergy (std::vector<float>& profile)
{
    double sum = 0.0;
    for (const auto v : profile)
        sum += static_cast<double> (v) * v;
    const auto norm = std::sqrt (sum);
    if (norm > 1.0e-12)
        for (auto& v : profile)
            v = static_cast<float> (v / norm);
}

} // namespace

WavetableFit::Result WavetableFit::fit (const float* H, int numHarmonics, int numFrames,
                                        const std::vector<float>& times, float gateSeconds,
                                        bool oneShot, const WaveformFit::Blend& blend,
                                        double energyShare)
{
    Result result;
    if (H == nullptr || numHarmonics <= 0 || numFrames < kNumFrames * 2
        || energyShare < kMinEnergyShare)
        return result;

    std::vector<float> energy (static_cast<size_t> (numFrames), 0.0f);
    for (int t = 0; t < numFrames; ++t)
    {
        double sum = 0.0;
        for (int k = 0; k < numHarmonics; ++k)
            sum += H[static_cast<size_t> (k) * static_cast<size_t> (numFrames)
                     + static_cast<size_t> (t)];
        energy[static_cast<size_t> (t)] = static_cast<float> (sum);
    }

    const auto peak = *std::max_element (energy.begin(), energy.end());
    if (peak <= 0.0f)
        return result;

    auto first = 0;
    while (first < numFrames && energy[static_cast<size_t> (first)] < peak * kSoundingFloor)
        ++first;
    auto last = numFrames - 1;
    while (last > first && energy[static_cast<size_t> (last)] < peak * kSoundingFloor)
        --last;

    // Stop at note-off unless the sound is one-shot. After the key is released
    // the harmonics are decaying at their own rates and the room is mixed in;
    // the shape measured there is the release, not the instrument's timbre. The
    // position holds at the last frame through the tail, which is what a held
    // sustain does anyway.
    if (! oneShot && gateSeconds > 0.0f)
    {
        auto gateFrame = last;
        for (size_t i = 0; i < times.size() && static_cast<int> (i) < numFrames; ++i)
        {
            if (times[i] >= gateSeconds)
            {
                gateFrame = juce::jmax (first, static_cast<int> (i) - 1);
                break;
            }
        }
        last = juce::jmin (last, gateFrame);
    }

    const auto span = last - first + 1;
    if (span < kNumFrames * 2)
        return result;

    // Half-window for the smoothing below, in analysis frames. Wide enough to
    // average a vibrato cycle away at any rate the modulation fitter accepts,
    // and capped so it can never span a sixth of the note and flatten a real
    // sweep along with the wobble.
    const auto frameSeconds = times.size() > 1 ? juce::jmax (1.0e-4f, times[1] - times[0]) : 0.006f;
    const auto smoothHalf = juce::jlimit (4, juce::jmax (4, span / 6),
                                          static_cast<int> (kSmoothSeconds / frameSeconds));

    // --- the three candidate frames -----------------------------------------
    std::array<std::array<float, kNumHarmonics>, kNumFrames> moving {};
    for (int f = 0; f < kNumFrames; ++f)
    {
        const auto from = first + (span * f) / kNumFrames;
        const auto to = first + (span * (f + 1)) / kNumFrames;
        auto profile = meanOver (H, numHarmonics, numFrames, energy, from, juce::jmax (to, from + 1));
        unitEnergy (profile);
        for (size_t k = 0; k < profile.size(); ++k)
            moving[static_cast<size_t> (f)][k] = profile[k];
    }

    auto staticProfile = meanOver (H, numHarmonics, numFrames, energy, first, last + 1);
    unitEnergy (staticProfile);

    // --- score all three rungs the same way ---------------------------------
    //
    // Every model is compared against each sounding frame's own profile,
    // weighted by that frame's energy, so the numbers are directly comparable
    // and the decision below is a comparison rather than three thresholds.
    const auto blendProfile = WaveformFit::profileFor (blend, static_cast<int> (staticProfile.size()));

    std::vector<float> interpolated (staticProfile.size(), 0.0f);
    double blendTotal = 0.0, staticTotal = 0.0, frameTotal = 0.0, weightTotal = 0.0;

    for (int t = first; t <= last; ++t)
    {
        const auto weight = static_cast<double> (energy[static_cast<size_t> (t)]);
        if (weight <= 0.0)
            continue;
        // Scored against the *slow* profile, not the raw one.
        //
        // Frame by frame the measured balance jumps by 10 dB and more: vibrato
        // slides every partial across the analysis bins, so a harmonic's
        // reading wobbles at the vibrato rate. That is modulation, which the
        // LFOs already carry, and it is common to both models -- left in, it
        // drowns the slow movement the frames exist to describe, and the two
        // scores came out within a percent of each other however far the tone
        // actually travelled.
        const auto observed = meanOver (H, numHarmonics, numFrames, energy,
                                        juce::jmax (first, t - smoothHalf),
                                        juce::jmin (last + 1, t + smoothHalf + 1));

        // Same arithmetic the oscillator does. All frames are built in sine
        // phase, so a crossfade of two tables is exactly a crossfade of their
        // harmonic amplitudes -- the fitter's model and the engine's output are
        // the same thing, not an approximation of each other.
        const auto position = span > 1
                            ? static_cast<float> (t - first) / static_cast<float> (span - 1)
                            : 0.0f;
        const auto scaled = position * static_cast<float> (kNumFrames - 1);
        const auto lower = juce::jlimit (0, kNumFrames - 2, static_cast<int> (scaled));
        const auto morph = juce::jlimit (0.0f, 1.0f, scaled - static_cast<float> (lower));
        for (size_t k = 0; k < interpolated.size(); ++k)
            interpolated[k] = moving[static_cast<size_t> (lower)][k] * (1.0f - morph)
                            + moving[static_cast<size_t> (lower + 1)][k] * morph;

        blendTotal += weight * WaveformFit::profileError (observed, blendProfile);
        staticTotal += weight * WaveformFit::profileError (observed, staticProfile);
        frameTotal += weight * WaveformFit::profileError (observed, interpolated);
        weightTotal += weight;
    }

    if (weightTotal <= 0.0)
        return result;

    result.blendError = blendTotal / weightTotal;
    result.staticError = staticTotal / weightTotal;
    result.frameError = frameTotal / weightTotal;
    result.driftDb = driftDb (moving.front(), moving.back());

    // Rung 1 -> 2. A table exists only if the classic shapes are genuinely
    // failing to describe the spectrum. Below this the oscillator keeps the
    // waveform it was given, which is a thing a person can read.
    if (result.blendError < kBlendGoodEnough
        || result.staticError > result.blendError * kTableMustBeatBlendBy)
        return result;

    result.useCustomFrames = true;
    result.numFrames = 1;
    for (size_t k = 0; k < staticProfile.size() && k < kNumHarmonics; ++k)
        for (auto& frame : result.frames)
            frame[k] = staticProfile[k];

    // Rung 2 -> 3. Three frames and a sweeping position, only if the tone
    // actually travels far enough to hear and the movement is worth the extra
    // thirty-two numbers. Otherwise all three frames stay identical, which is
    // one table with the position control doing nothing.
    if (result.driftDb < kMinDriftDb
        || result.frameError > result.staticError * kFramesMustBeatTableBy)
        return result;

    result.frames = moving;
    result.numFrames = kNumFrames;

    // Position sweeps 0 to 1 across the sounding part of the note and then
    // holds. An attack-only envelope is that ramp: there is no separate "move
    // the table" modulator in the IR, and adding one would be a second way to
    // say what an envelope already says.
    const auto startTime = first < static_cast<int> (times.size()) ? times[static_cast<size_t> (first)] : 0.0f;
    const auto endTime = last < static_cast<int> (times.size()) ? times[static_cast<size_t> (last)] : startTime;
    const auto sweep = juce::jmax (0.01f, endTime - startTime);

    result.position = 0.0f;
    result.envAmount = 1.0f;
    // Sustain 1 and the release unused: the engine never releases this
    // envelope, so the position holds wherever the note ended.
    result.env = { sweep, 0.001f, 1.0f, 0.1f, 0.0f };
    return result;
}

} // namespace autosynth
