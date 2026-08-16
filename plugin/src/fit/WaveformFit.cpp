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
    // Noise is deliberately absent. Its profile is flat, and a flat profile is
    // close enough to a badly-measured one that the matcher would reach for it
    // whenever the analysis was struggling -- describing a pitched note as
    // noise, which is the least useful thing a patch can say. Only the branch
    // that has already decided a sound is unpitched selects it.
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

WaveformFit::Blend WaveformFit::matchBlend (const std::vector<float>& profile, int numMorphSteps)
{
    Blend best;
    best.error = std::numeric_limits<double>::infinity();
    if (profile.empty())
        return best;

    const auto candidates = buildCandidates (static_cast<int> (profile.size()));
    const auto steps = juce::jmax (2, numMorphSteps);

    std::vector<float> blended (profile.size());
    for (size_t a = 0; a < candidates.size(); ++a)
    {
        for (size_t b = 0; b < candidates.size(); ++b)
        {
            // The oscillator has one pulse-width control, but it only affects a
            // pulse table -- every other shape ignores it. So a pair is
            // playable unless *both* are pulses wanting different widths.
            //
            // Requiring the widths to match outright excluded saw-against-
            // narrow-pulse, which is the single most useful blend available: a
            // narrow pulse is the only shape here that puts a strong peak on a
            // low harmonic, and pairing it with a saw is how a formant gets
            // approximated at all.
            const auto pulseA = candidates[a].waveform == Waveform::pulse;
            const auto pulseB = candidates[b].waveform == Waveform::pulse;
            if (pulseA && pulseB && candidates[a].pulseWidth != candidates[b].pulseWidth)
                continue;
            if (! pulseA && ! pulseB && b < a)
                continue; // blends of two non-pulse shapes are symmetric

            const auto pulseWidth = pulseA ? candidates[a].pulseWidth
                                  : pulseB ? candidates[b].pulseWidth
                                           : 0.5f;

            const auto sameShape = candidates[a].waveform == candidates[b].waveform;
            for (int s = 0; s < steps; ++s)
            {
                const auto morph = static_cast<float> (s) / (steps - 1);
                // A blend of a shape with itself is the shape; searching the
                // morph there just wastes evaluations on identical points.
                if (sameShape && s > 0)
                    break;

                for (size_t k = 0; k < profile.size(); ++k)
                    blended[k] = candidates[a].amplitudes[k] * (1.0f - morph)
                               + candidates[b].amplitudes[k] * morph;

                const auto error = profileError (profile, blended);
                if (error < best.error)
                {
                    best.error = error;
                    best.waveform = candidates[a].waveform;
                    best.waveformB = candidates[b].waveform;
                    best.morph = sameShape ? 0.0f : morph;
                    best.pulseWidth = pulseWidth;
                }
            }
        }
    }

    // Parsimony: keep the single waveform unless the blend genuinely earns the
    // parameter.
    //
    // A blend has more freedom and will almost always fit a little better, so
    // taken greedily it renames things for nothing -- a plain saw with vibrato
    // came back as "triangle, morphed towards a narrow pulse", which is a worse
    // *description* of the same sound. The objective here is not minimum error,
    // it is minimum error for the parameters spent, and a patch a person cannot
    // read has failed at the only thing this project is for.
    //
    // A fifth off the error is the bar, and it is deliberately conservative.
    // At a tenth the violin blended too -- genuinely helping its second
    // harmonic -- but so did a plain saw with vibrato, because vibrato smears
    // the measured profile and the extra freedom fits the smearing as readily
    // as a real formant. Under-using the blend costs a little accuracy on one
    // sample; over-using it costs legibility on every patch.
    // Two conditions, and the absolute one is the important half.
    //
    // A purely relative threshold is meaningless once the single waveform
    // already fits: a vibrato'd saw measured 1.00 0.50 0.33 0.24 0.19 with a
    // match error of 0.016 -- textbook -- and a blend still beat that by well
    // over the relative bar, because a fifth of almost nothing is almost
    // nothing. It described a saw as "triangle morphed 88% toward saw", fitting
    // the third decimal place of a profile that was already right.
    //
    // So the single fit has to be *bad enough to be worth fixing* before a
    // blend is considered at all. Below this the shape is already identified
    // and the remaining error is measurement noise, not timbre.
    constexpr auto kSingleGoodEnough = 0.05;
    constexpr auto kBlendMustBeatSingleBy = 0.80;

    const auto single = match (profile);
    if (single.error < kSingleGoodEnough
        || best.error > single.error * kBlendMustBeatSingleBy)
    {
        best.waveform = single.waveform;
        best.waveformB = single.waveform;
        best.morph = 0.0f;
        best.pulseWidth = single.pulseWidth;
        best.error = single.error;
        return best;
    }

    // Collapse the degenerate ends.
    //
    // A morph of 0 or 1 *is* a single waveform, and leaving it expressed as a
    // blend makes the patch harder to read for no gain -- a plain saw came back
    // as "sine, morphed fully to saw", which is the same sound described
    // confusingly. It also made pure-waveform fixtures fail for a cosmetic
    // reason.
    if (best.morph >= 1.0f - 1.0e-6f)
    {
        best.waveform = best.waveformB;
        best.morph = 0.0f;
    }
    if (best.morph <= 1.0e-6f)
    {
        best.morph = 0.0f;
        best.waveformB = best.waveform;
    }
    return best;
}

std::vector<float> WaveformFit::profileFor (const Blend& blend, int numHarmonics)
{
    return WaveTables::blendedHarmonics (blend.waveform, blend.waveformB, blend.morph,
                                         blend.pulseWidth, numHarmonics);
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
