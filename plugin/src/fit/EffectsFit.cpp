#include "fit/EffectsFit.h"

#include "analysis/Stft.h"
#include "dsp/Reverb.h"
#include "fit/NdFilters.h"

#include <algorithm>
#include <cmath>
#include <juce_dsp/juce_dsp.h>
#include <numeric>

namespace autosynth
{
namespace
{

// Autocorrelation of a zero-mean signal, normalised so lag 0 is 1.
std::vector<float> normalisedAutocorrelation (const std::vector<float>& input)
{
    const auto n = static_cast<int> (input.size());
    std::vector<float> out (input.size(), 0.0f);
    if (n < 2)
        return out;

    const auto mean = std::accumulate (input.begin(), input.end(), 0.0)
                    / static_cast<double> (n);

    auto size = 2;
    while (size < 2 * n)
        size <<= 1;
    const auto order = static_cast<int> (std::log2 (static_cast<double> (size)));

    std::vector<float> buffer (static_cast<size_t> (size) * 2, 0.0f);
    for (int i = 0; i < n; ++i)
        buffer[static_cast<size_t> (i)] = static_cast<float> (input[static_cast<size_t> (i)] - mean);

    juce::dsp::FFT fft (order);
    fft.performRealOnlyForwardTransform (buffer.data(), true);

    // Power spectrum, then back: X * conj(X) is real, so the imaginary part is
    // zeroed rather than computed.
    const auto numBins = size / 2 + 1;
    for (int b = 0; b < numBins; ++b)
    {
        const auto re = buffer[static_cast<size_t> (b) * 2];
        const auto im = buffer[static_cast<size_t> (b) * 2 + 1];
        buffer[static_cast<size_t> (b) * 2] = re * re + im * im;
        buffer[static_cast<size_t> (b) * 2 + 1] = 0.0f;
    }
    fft.performRealOnlyInverseTransform (buffer.data());

    const auto zeroLag = buffer[0];
    if (zeroLag <= 1.0e-12f)
        return out;

    for (int i = 0; i < n; ++i)
        out[static_cast<size_t> (i)] = buffer[static_cast<size_t> (i)] / zeroLag;
    return out;
}

} // namespace

EffectsFit::DelayEstimate EffectsFit::detectDelay (const float* samples, int numSamples,
                                                   double sampleRate, int hop,
                                                   double smoothSeconds, double gateSeconds)
{
    DelayEstimate estimate;

    auto envelope = Stft::loudnessEnvelope (samples, numSamples, hop);
    if (envelope.size() < 32)
        return estimate;

    const auto dt = hop / sampleRate;
    const auto span = std::max (1, static_cast<int> (std::lround (smoothSeconds / dt)));
    if (span > 1)
        envelope = nd::uniformFilter1d (envelope, span);

    const auto correlation = normalisedAutocorrelation (envelope);
    const auto size = static_cast<int> (correlation.size());
    const auto lo = std::max (1, static_cast<int> (std::lround (kMinDelaySeconds / dt)));
    const auto hi = std::min (size - 1, static_cast<int> (std::lround (kMaxDelaySeconds / dt)));
    if (hi <= lo)
        return estimate;

    // Skip the main lobe. An envelope is smooth, so its autocorrelation
    // descends slowly from 1.0 and is still ~0.9 several frames later -- taking
    // the maximum over a lag band just reports the left edge of that slope,
    // which "detects" a ~20 ms delay in everything including dry material.
    auto i = 1;
    while (i < size - 1 && correlation[static_cast<size_t> (i)]
                            <= correlation[static_cast<size_t> (i - 1)])
        ++i;
    const auto start = std::max (i, lo);
    if (start >= hi)
        return estimate;

    // And require a genuine local maximum. A single decaying note has a
    // monotonic autocorrelation with no interior peak, which is the rejection
    // wanted.
    auto peak = -1;
    auto strength = 0.0;
    for (int j = start + 1; j < hi; ++j)
    {
        const auto c = static_cast<double> (correlation[static_cast<size_t> (j)]);
        if (c > correlation[static_cast<size_t> (j - 1)]
            && c >= correlation[static_cast<size_t> (j + 1)] && c > strength)
        {
            peak = j;
            strength = c;
        }
    }

    if (peak < 0 || strength < kMinPeakRatio)
        return estimate;

    // Confirm the repeats outlive the note.
    //
    // A delay copies the signal, so its echoes carry on after note-off. Vibrato
    // and tremolo make the loudness envelope every bit as periodic, but only
    // while the note is sounding -- and that is enough to fool an
    // autocorrelation taken over the whole file. A violin with 4.5 Hz vibrato
    // was fitted with a 0.22 s delay at 0.84 feedback, which rings at under
    // 7 dB per second and left the patch droning for the whole tail. The delay
    // time was exactly the vibrato period.
    //
    // Looking for the same period *after* note-off separates them, because a
    // played note's decay is smooth and has no interior peak there.
    if (gateSeconds >= 0.0)
    {
        const auto gateIndex = static_cast<int> (std::lround (gateSeconds / dt));
        const auto available = static_cast<int> (envelope.size()) - gateIndex;

        // Two periods is the least that can show a repeat. With less than that
        // there is nothing to confirm against, and the estimate stands or falls
        // on the whole-file evidence alone.
        if (gateIndex > 0 && available >= 2 * peak)
        {
            const std::vector<float> tail (envelope.begin() + gateIndex, envelope.end());
            const auto tailCorrelation = normalisedAutocorrelation (tail);
            const auto window = std::max (1, peak / 4);

            auto confirmed = false;
            for (auto j = std::max (1, peak - window);
                 j <= std::min (static_cast<int> (tailCorrelation.size()) - 2, peak + window); ++j)
            {
                const auto c = tailCorrelation[static_cast<size_t> (j)];
                if (c > tailCorrelation[static_cast<size_t> (j - 1)]
                    && c >= tailCorrelation[static_cast<size_t> (j + 1)]
                    && c > kMinPeakRatio)
                    confirmed = true;
            }

            if (! confirmed)
                return estimate;
        }
    }

    // Feedback from how much the second repeat retains relative to the first.
    const auto second = 2 * peak;
    auto ratio = 0.0;
    if (second < size)
        ratio = juce::jlimit (0.0, 1.0,
                              correlation[static_cast<size_t> (second)] / std::max (strength, 1.0e-9));

    estimate.found = true;
    estimate.time = juce::jlimit (kMinDelaySeconds, kMaxDelaySeconds, peak * dt);
    // A proxy, not the coefficient itself; CMA-ES refines it, so a rough
    // mapping is enough to start from.
    estimate.feedback = juce::jlimit (0.0, 0.85, std::sqrt (ratio));
    estimate.mix = juce::jlimit (0.0, 1.0, strength);
    estimate.strength = strength;
    return estimate;
}

// The conversion itself now lives on Reverb, because it is a fact about that
// reverb rather than about fitting, and the Vital exporter needs it too. What
// stays here is the fitter's own view of an unbounded tail: a size that never
// decays is reported as the longest RT60 this fitter will consider.
double EffectsFit::rt60ForSize (double size)
{
    const auto rt60 = Reverb::rt60ForSize (size);
    return rt60 > 0.0 ? rt60 : kMaxRt60Seconds;
}

double EffectsFit::sizeForRt60 (double rt60Seconds)
{
    return Reverb::sizeForRt60 (rt60Seconds);
}

EffectsFit::ReverbEstimate EffectsFit::detectReverb (const float* samples, int numSamples,
                                                     double sampleRate, double gateSeconds,
                                                     int hop)
{
    ReverbEstimate estimate;
    if (numSamples <= 0 || sampleRate <= 0.0 || hop <= 0)
        return estimate;

    const auto envelope = Stft::loudnessEnvelope (samples, numSamples, hop);
    if (envelope.size() < 8)
        return estimate;

    const auto dt = hop / sampleRate;
    const auto peak = *std::max_element (envelope.begin(), envelope.end());
    if (peak <= 1.0e-9f)
        return estimate;

    std::vector<double> db (envelope.size());
    for (size_t i = 0; i < envelope.size(); ++i)
        db[i] = 20.0 * std::log10 (envelope[i] / peak + 1.0e-9);

    // Level of the note itself, for reporting how far below it the tail sits.
    const auto gateIndex = juce::jlimit (1, static_cast<int> (db.size()) - 1,
                                         static_cast<int> (std::lround (gateSeconds / dt)));
    std::vector<double> body (db.begin(), db.begin() + gateIndex);
    std::sort (body.begin(), body.end());
    const auto bodyDb = body.empty() ? 0.0 : body[body.size() / 2];

    // The tail ends where it reaches the recording's own noise floor, and the
    // floor has to be measured rather than assumed.
    //
    // A fixed -60 dB threshold reads a quiet room's hiss as part of the tail.
    // On a violin whose reverb actually died within a quarter of a second, that
    // put two thirds of the fitted region inside the noise floor, and the slope
    // of hiss is shallow -- so the estimate came back with a two-second RT60
    // for a room that had none, and a tail that then rang for the rest of the
    // file. The floor is the median of the last tenth of the recording, and
    // only what stands clear of it is treated as reverb.
    const auto floorFrom = static_cast<int> (db.size() * 9 / 10);
    std::vector<double> quiet (db.begin() + floorFrom, db.end());
    std::sort (quiet.begin(), quiet.end());
    const auto noiseFloorDb = quiet.empty() ? -80.0 : quiet[quiet.size() / 2];

    constexpr auto kClearOfFloorDb = 6.0;
    const auto floorDb = noiseFloorDb + kClearOfFloorDb;

    auto tailEnd = static_cast<int> (db.size());
    while (tailEnd > gateIndex && db[static_cast<size_t> (tailEnd - 1)] < floorDb)
        --tailEnd;

    const auto tailFrames = tailEnd - gateIndex;
    if (tailFrames * dt < kMinTailSeconds || tailFrames < 6)
        return estimate; // nothing after the note but silence

    // The reverb slope is measured on the *later* part of the tail, where the
    // direct sound has already gone. Measuring it from note-off would mix the
    // instrument's own release into the estimate -- which is precisely the
    // trade that makes a fitter get both wrong.
    const auto slopeStart = gateIndex + tailFrames / 3;
    if (tailEnd - slopeStart < 4)
        return estimate;

    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    auto n = 0;
    for (int i = slopeStart; i < tailEnd; ++i)
    {
        const auto x = (i - gateIndex) * dt;
        const auto y = db[static_cast<size_t> (i)];
        sx += x; sy += y; sxx += x * x; sxy += x * y;
        ++n;
    }
    const auto denom = n * sxx - sx * sx;
    if (std::abs (denom) < 1.0e-12)
        return estimate;

    const auto slope = (n * sxy - sx * sy) / denom;      // dB per second
    const auto intercept = (sy - slope * sx) / n;        // dB at note-off
    if (slope >= -1.0e-6)
        return estimate; // not decaying

    // How exponential it actually was. A diffuse tail is a straight line in dB;
    // a lumpy one is something else and should not be called reverb.
    double residual = 0.0, total = 0.0;
    const auto meanY = sy / n;
    for (int i = slopeStart; i < tailEnd; ++i)
    {
        const auto x = (i - gateIndex) * dt;
        const auto predicted = slope * x + intercept;
        const auto y = db[static_cast<size_t> (i)];
        residual += (y - predicted) * (y - predicted);
        total += (y - meanY) * (y - meanY);
    }
    estimate.decayFit = total > 1.0e-12 ? 1.0 - residual / total : 0.0;

    estimate.rt60 = -60.0 / slope;
    estimate.levelBelowBodyDb = intercept - bodyDb;

    // Where the direct sound hands over: the first frame after note-off that
    // has fallen to the reverb line. Everything before it is release.
    estimate.releaseSeconds = tailFrames * dt;
    for (int i = gateIndex; i < tailEnd; ++i)
    {
        const auto x = (i - gateIndex) * dt;
        if (db[static_cast<size_t> (i)] <= slope * x + intercept + 3.0)
        {
            estimate.releaseSeconds = std::max (x, 5.0e-3);
            break;
        }
    }

    // How much the tail darkens as it dies, which is what `damp` controls.
    const auto spectrogram = Stft::magnitudeSpectrogram (samples, numSamples, 2048, hop, sampleRate);
    const auto centroid = Stft::spectralCentroid (spectrogram);
    if (static_cast<int> (centroid.size()) > tailEnd && tailEnd > slopeStart)
    {
        const auto early = centroid[static_cast<size_t> (juce::jmin (gateIndex + 1,
                                                                     (int) centroid.size() - 1))];
        const auto late = centroid[static_cast<size_t> (tailEnd - 1)];
        if (early > 20.0f && late > 20.0f)
            estimate.brightnessLossOct = std::log2 (early / late);
    }

    estimate.found = estimate.rt60 >= kMinRt60Seconds
                  && estimate.rt60 <= kMaxRt60Seconds
                  && estimate.decayFit >= kMinDecayFit;
    return estimate;
}

ReverbParams EffectsFit::fitReverb (const float* samples, int numSamples, double sampleRate,
                                    double gateSeconds, int hop)
{
    ReverbParams params;
    const auto estimate = detectReverb (samples, numSamples, sampleRate, gateSeconds, hop);
    if (! estimate.found)
    {
        params.enabled = false;
        params.level = 0.0f;
        return params;
    }

    params.enabled = true;
    params.size = static_cast<float> (sizeForRt60 (estimate.rt60));

    // A tail that loses an octave of brightness over its life is heavily
    // damped; one that keeps it is not.
    params.damp = static_cast<float> (juce::jlimit (0.0, 1.0, estimate.brightnessLossOct));

    // Return gain, corrected for how much the reverb amplifies on its own.
    //
    // `level` is not the wet/dry ratio. A comb with feedback f settles at a
    // gain of 1/(1-f) while a note is held, and at the sizes a real room
    // implies that is a factor of eight or nine -- so a level naively set to
    // the measured tail-to-note ratio came out some 28 dB too loud, and the
    // fitted violin rang at -17 dB where the recording was at -42.
    //
    // Dividing by the steady-state gain gives the return that actually lands
    // the tail where it was measured. It is still only a starting point:
    // `reverb.*` is in refinement scope whenever the reverb is on, so CMA-ES
    // polishes it against the audio. Analysis owns the structural call -- that
    // there is a room at all -- and refinement owns the value.
    const auto feedback = juce::jlimit (0.0, 0.98, 0.7 + 0.28 * (double) params.size);
    const auto steadyStateGain = 1.0 / juce::jmax (0.02, 1.0 - feedback);
    const auto tailRatio = std::pow (10.0, estimate.levelBelowBodyDb / 20.0);

    params.level = static_cast<float> (juce::jlimit (0.0, 1.0, tailRatio / steadyStateGain));
    if (params.level < 1.0e-3f)
    {
        // Too quiet to be worth a reverb at all; leave it off rather than ship
        // a patch with an inaudible effect switched on.
        params.enabled = false;
        params.level = 0.0f;
    }
    return params;
}

DelayParams EffectsFit::fitDelay (const float* samples, int numSamples, double sampleRate,
                                  int hop, double gateSeconds)
{
    DelayParams params;
    const auto estimate = detectDelay (samples, numSamples, sampleRate, hop, 0.01, gateSeconds);
    if (! estimate.found)
    {
        params.enabled = false;
        params.mix = 0.0f;
        return params;
    }

    params.enabled = true;
    params.time = static_cast<float> (estimate.time);
    params.feedback = static_cast<float> (estimate.feedback);
    params.mix = static_cast<float> (estimate.mix);
    return params;
}

} // namespace autosynth
