#include "fit/WaveformFit.h"

#include "dsp/FilterResponse.h"
#include "dsp/Tables.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace autosynth
{

float WaveformFit::pulseWidthAt (int index) noexcept
{
    return 0.10f + 0.35f * static_cast<float> (index) / static_cast<float> (kNumPulseWidths - 1);
}

std::vector<float> WaveformFit::normalise (std::vector<float> v)
{
    if (v.empty())
        return v;
    const auto peak = *std::max_element (v.begin(), v.end());
    if (peak > 1.0e-12f)
        for (auto& x : v)
            x /= peak;
    return v;
}

double WaveformFit::profileError (const std::vector<float>& observed,
                                  const std::vector<float>& model)
{
    const auto obs = normalise (observed);
    const auto mdl = normalise (model);
    const auto n = std::min (obs.size(), mdl.size());

    double total = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const auto weight = obs[i] + 1.0e-3f; // loud harmonics carry more evidence
        const auto diff = std::log10 (obs[i] + 1.0e-4)
                        - std::log10 (mdl[i] + 1.0e-4);
        total += weight * diff * diff;
    }
    return total;
}

namespace
{

struct Candidate
{
    Waveform waveform;
    float pulseWidth;
    std::vector<float> amplitudes;
};

std::vector<Candidate> buildCandidates (int numHarmonics)
{
    std::vector<Candidate> out;
    const Waveform all[] = { Waveform::sine, Waveform::triangle, Waveform::saw,
                             Waveform::square, Waveform::pulse };

    for (auto waveform : all)
    {
        const auto numWidths = (waveform == Waveform::pulse) ? WaveformFit::kNumPulseWidths : 1;
        for (int w = 0; w < numWidths; ++w)
        {
            const auto pulseWidth = (waveform == Waveform::pulse)
                                  ? WaveformFit::pulseWidthAt (w)
                                  : 0.5f;
            std::vector<float> amps;
            bool cosinePhase = false;
            WaveTables::harmonicAmplitudes (waveform, numHarmonics, pulseWidth, amps, cosinePhase);
            for (auto& a : amps)
                a = std::abs (a);
            out.push_back ({ waveform, pulseWidth, std::move (amps) });
        }
    }
    return out;
}

} // namespace

WaveformFit::Match WaveformFit::match (const std::vector<float>& profile)
{
    Match best;
    best.error = std::numeric_limits<double>::infinity();
    if (profile.empty())
        return best;

    for (const auto& candidate : buildCandidates (static_cast<int> (profile.size())))
    {
        const auto error = profileError (profile, candidate.amplitudes);
        if (error < best.error)
        {
            best.error = error;
            best.waveform = candidate.waveform;
            best.pulseWidth = candidate.pulseWidth;
        }
    }
    return best;
}

WaveformFit::Match WaveformFit::matchWithCutoff (const std::vector<float>& profile, double f0Hz,
                                                 double sampleRate, float q, int numGrid)
{
    Match best;
    best.error = std::numeric_limits<double>::infinity();
    if (profile.empty())
        return best;

    const auto numHarmonics = static_cast<int> (profile.size());
    std::vector<float> freqs (static_cast<size_t> (numHarmonics));
    for (int k = 0; k < numHarmonics; ++k)
        freqs[static_cast<size_t> (k)] = static_cast<float> ((k + 1) * f0Hz);

    // geomspace(max(f0 * 1.05, 30), min(18000, sr * 0.45), numGrid)
    const auto lo = std::log (std::max (f0Hz * 1.05, 30.0));
    const auto hi = std::log (std::min (18000.0, sampleRate * 0.45));
    best.cutoffHz = static_cast<float> (std::exp (hi));

    std::vector<float> model (static_cast<size_t> (numHarmonics));

    for (const auto& candidate : buildCandidates (numHarmonics))
    {
        for (int g = 0; g < numGrid; ++g)
        {
            const auto t = (numGrid > 1) ? static_cast<double> (g) / (numGrid - 1) : 0.0;
            const auto cutoff = static_cast<float> (std::exp (lo + t * (hi - lo)));

            for (int k = 0; k < numHarmonics; ++k)
                model[static_cast<size_t> (k)] =
                    candidate.amplitudes[static_cast<size_t> (k)]
                    * analogMagnitude (FilterType::lowpass, freqs[static_cast<size_t> (k)], cutoff, q);

            const auto error = profileError (profile, model);
            if (error < best.error)
            {
                best.error = error;
                best.waveform = candidate.waveform;
                best.pulseWidth = candidate.pulseWidth;
                best.cutoffHz = cutoff;
            }
        }
    }
    return best;
}

} // namespace autosynth
