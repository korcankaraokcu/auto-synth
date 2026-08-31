// Fitting behaviour, checked by round trip: render a patch whose parameters we
// chose, fit it back, and ask whether the fitter recovered what was put in.
//
// This is the ground-truth harness the project has always leaned on, and it
// has one structural blind spot worth stating plainly: it only ever tests
// sounds the parameterisation can already make, so it can measure precision but
// never the modelling gap. A real sample tells you *that* something is wrong; this
// tells you *why*.

#include "Helpers.h"

#include "dsp/Envelope.h"
#include "fit/CmaEs.h"
#include "fit/EffectsFit.h"
#include "fit/EnvelopeFit.h"
#include "fit/FilterFit.h"
#include "fit/Modulation.h"
#include "fit/Nnls.h"
#include "fit/PartialFit.h"
#include "fit/Refine.h"

#include <algorithm>
#include "fit/WaveformFit.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autotest;
using namespace autosynth;

namespace
{

constexpr int kHop = 256;

std::vector<float> renderFor (const Patch& patch, double duration = 1.0)
{
    return render (patch, patch.rootHz, duration, duration);
}

Patch fitOf (const std::vector<float>& x)
{
    PartialFit::Options options;
    options.hop = kHop;
    options.renderer = renderer();
    return PartialFit::fit (x.data(), (int) x.size(), kSampleRate, options);
}

} // namespace

// --- waveform --------------------------------------------------------------

TEST_CASE ("waveform matching recovers the shape from its harmonic profile", "[fit]")
{
    // Built from the known series rather than from analysis output, so a
    // failure here is the matcher's and not the tracker's.
    struct Case { Waveform waveform; };
    const int n = 16;

    SECTION ("saw")
    {
        std::vector<float> profile ((size_t) n);
        for (int k = 0; k < n; ++k)
            profile[(size_t) k] = 1.0f / (float) (k + 1);
        CHECK (WaveformFit::match (profile).waveform == Waveform::saw);
    }

    SECTION ("square")
    {
        std::vector<float> profile ((size_t) n, 0.0f);
        for (int k = 0; k < n; k += 2)
            profile[(size_t) k] = 1.0f / (float) (k + 1);
        CHECK (WaveformFit::match (profile).waveform == Waveform::square);
    }

    SECTION ("sine")
    {
        std::vector<float> profile ((size_t) n, 0.0f);
        profile[0] = 1.0f;
        CHECK (WaveformFit::match (profile).waveform == Waveform::sine);
    }
}

TEST_CASE ("profile error is zero for an exact match", "[fit]")
{
    std::vector<float> profile { 1.0f, 0.5f, 0.333f, 0.25f };
    CHECK (WaveformFit::profileError (profile, profile) == Catch::Approx (0.0).margin (1.0e-6));
}

TEST_CASE ("joint waveform and cutoff search beats the plain match on a filtered saw", "[fit]")
{
    // A filtered saw looks like a triangle to a plain profile match, because
    // both roll off. Absolute cutoff is not identifiable from the profile
    // alone -- it is blind deconvolution -- so the cutoff has to be anchored
    // by searching for it jointly with the waveform.
    const int n = 24;
    std::vector<float> profile ((size_t) n);
    for (int k = 0; k < n; ++k)
    {
        const auto harmonic = (float) (k + 1);
        const auto rolloff = 1.0f / std::sqrt (1.0f + std::pow (harmonic * 220.0f / 1200.0f, 4.0f));
        profile[(size_t) k] = rolloff / harmonic;
    }
    const auto peak = *std::max_element (profile.begin(), profile.end());
    for (auto& v : profile)
        v /= peak;

    const auto joint = WaveformFit::matchWithCutoff (profile, 220.0, kSampleRate);
    CHECK (joint.waveform == Waveform::saw);
    CHECK (joint.cutoffHz > 300.0f);
    CHECK (joint.cutoffHz < 5000.0f);
}

// --- envelope --------------------------------------------------------------

TEST_CASE ("a sustained note is detected as sustained", "[fit]")
{
    auto patch = simplePatch();
    patch.ampEnv = { 0.02f, 0.1f, 0.8f, 0.1f, 0.0f };
    const auto x = render (patch, 220.0, 1.0, 0.7);

    const auto loud = Stft::loudnessEnvelope (x.data(), (int) x.size(), kHop);
    std::vector<float> times (loud.size());
    for (size_t i = 0; i < loud.size(); ++i)
        times[i] = (float) (i * kHop / kSampleRate);

    const auto gate = EnvelopeFit::detectGate (loud, times);
    CHECK_FALSE (gate.oneShot);
    CHECK (gate.time > 0.4);
}

TEST_CASE ("a plucked note is detected as one-shot", "[fit]")
{
    auto patch = simplePatch();
    patch.ampEnv = { 0.001f, 0.25f, 0.0f, 0.05f, 1.0f };
    const auto x = render (patch, 220.0, 1.0, 0.9);

    const auto loud = Stft::loudnessEnvelope (x.data(), (int) x.size(), kHop);
    std::vector<float> times (loud.size());
    for (size_t i = 0; i < loud.size(); ++i)
        times[i] = (float) (i * kHop / kSampleRate);

    CHECK (EnvelopeFit::detectGate (loud, times).oneShot);
}

TEST_CASE ("ADSR fitting orders decays correctly", "[fit]")
{
    // Ordering, not absolute recovery. The fit runs on an RMS envelope whose
    // analysis window smears the knee, so a decay time comes back systematically
    // long -- 0.4 s reads as roughly 0.8. Asserting an absolute value would be
    // asserting the smearing, and would break the moment the window changed.
    // What must hold, and what the fitter is actually for, is that a faster
    // decay fits a shorter one.
    const auto fitDecayFor = [] (float decay)
    {
        auto patch = simplePatch();
        patch.ampEnv = { 0.005f, decay, 0.3f, 0.1f, 0.0f };
        const auto x = render (patch, 220.0, 1.0, 0.7);

        const auto loud = Stft::loudnessEnvelope (x.data(), (int) x.size(), kHop);
        std::vector<float> times (loud.size());
        for (size_t i = 0; i < loud.size(); ++i)
            times[i] = (float) (i * kHop / kSampleRate);

        const auto gate = EnvelopeFit::detectGate (loud, times);
        return EnvelopeFit::fitAdsr (loud, times, gate.time, 0.05f, gate.oneShot);
    };

    const auto fast = fitDecayFor (0.05f);
    const auto slow = fitDecayFor (0.6f);
    INFO ("fast " << fast.decay << "  slow " << slow.decay);
    CHECK (fast.decay < slow.decay);

    // The sustain level is directly observable, unlike the decay knee, so it
    // is held to a real tolerance -- but only where there is a plateau to
    // observe. The 0.6 s decay barely reaches sustain before the 0.7 s gate
    // closes, and the fitter correctly declines to report a level it never saw.
    CHECK (fast.sustain == Catch::Approx (0.3f).margin (0.25));
}

TEST_CASE ("the envelope curve is fitted rather than defaulted", "[fit]")
{
    auto plucked = simplePatch();
    plucked.ampEnv = { 0.001f, 0.5f, 0.0f, 0.05f, 1.0f };
    const auto x = render (plucked, 220.0, 1.0, 0.9);

    const auto loud = Stft::loudnessEnvelope (x.data(), (int) x.size(), kHop);
    std::vector<float> times (loud.size());
    for (size_t i = 0; i < loud.size(); ++i)
        times[i] = (float) (i * kHop / kSampleRate);

    const auto gate = EnvelopeFit::detectGate (loud, times);
    const auto env = EnvelopeFit::fitAdsr (loud, times, gate.time, 0.05f, gate.oneShot);
    // A linear-only fit mis-stated this decay by 5-7 dB. The curve must move.
    CHECK (env.curve > 0.1f);
}

// --- modulation ------------------------------------------------------------

TEST_CASE ("an LFO is detected on the destination it was applied to", "[fit]")
{
    struct Case { LfoDest dest; float cutoff; };
    for (auto c : { Case { LfoDest::pitch, 0.0f },
                    Case { LfoDest::amp, 0.0f },
                    Case { LfoDest::cutoff, 1200.0f } })
    {
        auto patch = simplePatch (Waveform::saw);
        patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
        if (c.cutoff > 0.0f)
        {
            patch.filter.type = FilterType::lowpass;
            patch.filter.cutoffHz = c.cutoff;
        }
        patch.lfos[0] = { LfoShape::sine, c.dest, 5.0f, 0.6f, 0.0f, 0.0f };

        const auto x = render (patch, 220.0, 2.0, 2.0);
        const auto lfos = Modulation::bestSeveral (
            Modulation::extract (x.data(), (int) x.size(), kSampleRate, kHop));

        INFO ("dest " << (int) c.dest);
        REQUIRE_FALSE (lfos.empty());
        bool found = false;
        for (const auto& l : lfos)
            if (l.dest == c.dest && std::abs (l.rateHz - 5.0f) < 1.0f)
                found = true;
        CHECK (found);
    }
}

TEST_CASE ("an unmodulated signal reports no LFO", "[fit]")
{
    // The false-positive case that mattered in practice: a pitch track that
    // steps as one source fades correlates with a square wave, and a permissive
    // threshold turned that into a phantom vibrato that was audible in the
    // plugin. Real vibrato measured 0.83 concentration, the artefact 0.23.
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
    const auto x = render (patch, 220.0, 2.0, 2.0);

    const auto lfos = Modulation::bestSeveral (
        Modulation::extract (x.data(), (int) x.size(), kSampleRate, kHop));
    for (const auto& l : lfos)
    {
        INFO ("spurious dest " << (int) l.dest << " rate " << l.rateHz);
        CHECK (l.depth < 0.1f);
    }
}

TEST_CASE ("two modulations are reported on different destinations", "[fit]")
{
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
    patch.lfos[0] = { LfoShape::sine, LfoDest::pitch, 5.5f, 0.6f, 0.0f, 0.0f };
    patch.lfos[1] = { LfoShape::sine, LfoDest::amp, 3.0f, 0.6f, 0.0f, 0.0f };

    const auto x = render (patch, 220.0, 2.0, 2.0);
    const auto lfos = Modulation::bestSeveral (
        Modulation::extract (x.data(), (int) x.size(), kSampleRate, kHop));

    CHECK (lfos.size() <= (size_t) kNumLfo);
    // One slot per destination: two entries must not both target the same one.
    for (size_t i = 1; i < lfos.size(); ++i)
        CHECK (lfos[i].dest != lfos[i - 1].dest);
}

// --- delay -----------------------------------------------------------------

TEST_CASE ("delay time is recovered from the loudness autocorrelation", "[fit]")
{
    for (float time : { 0.15f, 0.3f })
    {
        auto patch = simplePatch (Waveform::saw);
        patch.ampEnv = { 0.001f, 0.05f, 0.0f, 0.01f, 0.0f };
        patch.delay = { true, time, 0.6f, 0.9f };

        const auto x = render (patch, 220.0, 2.0, 0.1);
        const auto estimate = EffectsFit::detectDelay (x.data(), (int) x.size(),
                                                       kSampleRate, kHop);
        INFO ("time " << time << " detected " << estimate.time);
        CHECK (estimate.time == Catch::Approx ((double) time).margin (0.03));
    }
}

TEST_CASE ("vibrato is not mistaken for a delay", "[fit]")
{
    // The false positive that cost the most. A delay repeats the *signal*, so
    // its echoes outlive the note; vibrato and tremolo make the loudness
    // envelope every bit as periodic but stop when the note does. Correlating
    // over the whole file cannot tell them apart, and a violin with 4.5 Hz
    // vibrato was fitted with a 0.22 s delay at 0.84 feedback -- ringing at
    // under 7 dB per second, which left the patch droning through the entire
    // tail. The delay time was exactly the vibrato period.
    for (auto dest : { LfoDest::pitch, LfoDest::amp })
    {
        auto patch = simplePatch (Waveform::saw);
        patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.05f, 0.0f };
        patch.lfos[0] = { LfoShape::sine, dest, 4.5f, 0.7f, 0.0f, 0.0f };

        const auto x = render (patch, 220.0, 2.5, 1.8);
        const auto estimate = EffectsFit::detectDelay (x.data(), (int) x.size(), kSampleRate,
                                                       kHop, 0.01, 1.8);
        INFO ("dest " << (int) dest << " reported time " << estimate.time);
        CHECK_FALSE (estimate.found);
    }
}

// --- reverb ----------------------------------------------------------------

TEST_CASE ("reverb round-trips through detection", "[fit]")
{
    // Render a known room, measure it back. RT60 is the thing being recovered;
    // `size` is an opaque knob position, so the check is stated in seconds.
    for (float size : { 0.4f, 0.75f })
    {
        auto patch = simplePatch (Waveform::saw);
        patch.ampEnv = { 0.005f, 0.05f, 0.8f, 0.05f, 0.0f };
        patch.reverb = { true, size, 0.0f, 0.6f };

        const auto x = render (patch, 220.0, 3.0, 1.2);
        const auto estimate = EffectsFit::detectReverb (x.data(), (int) x.size(),
                                                        kSampleRate, 1.2, kHop);

        const auto expected = EffectsFit::rt60ForSize (size);
        INFO ("size " << size << " expected RT60 " << expected
              << " measured " << estimate.rt60 << " fit " << estimate.decayFit);
        REQUIRE (estimate.found);
        // Within a third. The tail is measured through the note's own release
        // and a finite noise floor, so this is an estimate, not a readout.
        CHECK (estimate.rt60 > expected * 0.66);
        CHECK (estimate.rt60 < expected * 1.5);
    }
}

TEST_CASE ("a dry note has no reverb to find", "[fit]")
{
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.05f, 0.8f, 0.05f, 0.0f };
    patch.reverb.enabled = false;

    const auto x = render (patch, 220.0, 3.0, 1.2);
    const auto estimate = EffectsFit::detectReverb (x.data(), (int) x.size(),
                                                    kSampleRate, 1.2, kHop);
    INFO ("rt60 " << estimate.rt60 << " fit " << estimate.decayFit);
    CHECK_FALSE (estimate.found);
}

TEST_CASE ("a bigger room gives a longer tail", "[fit]")
{
    const auto rt60For = [] (float size)
    {
        auto patch = simplePatch (Waveform::saw);
        patch.ampEnv = { 0.005f, 0.05f, 0.8f, 0.05f, 0.0f };
        patch.reverb = { true, size, 0.0f, 0.6f };
        const auto x = render (patch, 220.0, 3.0, 1.2);
        return EffectsFit::detectReverb (x.data(), (int) x.size(), kSampleRate, 1.2, kHop).rt60;
    };

    CHECK (rt60For (0.8f) > rt60For (0.35f));
}

TEST_CASE ("the RT60 mapping inverts itself", "[fit]")
{
    for (double size : { 0.1, 0.4, 0.7, 0.9 })
    {
        const auto rt60 = EffectsFit::rt60ForSize (size);
        INFO ("size " << size << " -> " << rt60 << " s");
        CHECK (EffectsFit::sizeForRt60 (rt60) == Catch::Approx (size).margin (0.02));
    }
}

TEST_CASE ("fitting a reverbed note keeps the release off the tail", "[fit]")
{
    // The degeneracy, guarded. Given a note whose tail is all room, the release
    // must stay short and the reverb must carry it -- otherwise the release
    // absorbs the room, which measures *better* on whole-file distance while
    // being the wrong description.
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.05f, 0.8f, 0.08f, 0.0f };
    patch.reverb = { true, 0.7f, 0.2f, 0.5f };

    const auto target = render (patch, 220.0, 3.0, 1.2);

    PartialFit::Options options;
    options.hop = kHop;
    options.renderer = renderer();
    const auto fitted = PartialFit::fit (target.data(), (int) target.size(), kSampleRate, options);

    INFO ("reverb " << fitted.reverb.enabled << " size " << fitted.reverb.size
          << " level " << fitted.reverb.level << " release " << fitted.ampEnv.release);
    CHECK (fitted.reverb.enabled);
    CHECK (fitted.ampEnv.release < 1.0f);
}

// --- NNLS ------------------------------------------------------------------

TEST_CASE ("NNLS solves an exactly determined non-negative system", "[fit]")
{
    // A = I, b = [1, 2] -> x = [1, 2].
    std::vector<double> A { 1.0, 0.0, 0.0, 1.0 }; // column-major
    std::vector<double> b { 1.0, 2.0 };
    const auto x = nnls::solve (A, b, 2, 2);
    REQUIRE (x.size() == 2);
    CHECK (x[0] == Catch::Approx (1.0).margin (1.0e-6));
    CHECK (x[1] == Catch::Approx (2.0).margin (1.0e-6));
}

TEST_CASE ("NNLS never returns a negative coefficient", "[fit]")
{
    // The unconstrained solution here is negative; the constraint is the point.
    // A level cannot be negative, and allowing one would let the fitter cancel
    // oscillators against each other instead of modelling the sound.
    std::vector<double> A { 1.0, 0.0, 0.0, 1.0 };
    std::vector<double> b { -3.0, 2.0 };
    const auto x = nnls::solve (A, b, 2, 2);
    REQUIRE (x.size() == 2);
    CHECK (x[0] >= 0.0);
    CHECK (x[1] == Catch::Approx (2.0).margin (1.0e-6));
}

// --- CMA-ES ----------------------------------------------------------------

TEST_CASE ("CMA-ES minimises a quadratic bowl", "[fit]")
{
    CmaEs::Options options;
    options.maxEvaluations = 400;
    options.populationSize = 12;

    // The objective is evaluated on a whole population at once, which is what
    // lets the real one share a single noise realisation across the batch --
    // common random numbers, so candidates are compared on equal terms.
    const std::vector<double> x0 { 0.8, 0.2, 0.6 };
    const auto result = CmaEs::minimise (x0, options,
        [] (const std::vector<std::vector<double>>& population)
        {
            std::vector<double> losses;
            losses.reserve (population.size());
            for (const auto& candidate : population)
            {
                double acc = 0.0;
                for (auto v : candidate)
                    acc += (v - 0.5) * (v - 0.5);
                losses.push_back (acc);
            }
            return losses;
        });

    CHECK (result.bestValue < 1.0e-3);
    for (auto v : result.best)
        CHECK (v == Catch::Approx (0.5).margin (0.05));
}

TEST_CASE ("the Jacobi eigensolver is sane", "[fit]")
{
    // Hand-written rather than pulled from Eigen, so it gets its own check:
    // a diagonal matrix must come back with its own diagonal as eigenvalues.
    std::vector<double> A { 2.0, 0.0, 0.0,
                            0.0, 3.0, 0.0,
                            0.0, 0.0, 5.0 };
    std::vector<double> values, vectors;
    CmaEs::jacobiEigen (A, 3, values, vectors);

    REQUIRE (values.size() == 3);
    std::sort (values.begin(), values.end());
    CHECK (values[0] == Catch::Approx (2.0).margin (1.0e-8));
    CHECK (values[1] == Catch::Approx (3.0).margin (1.0e-8));
    CHECK (values[2] == Catch::Approx (5.0).margin (1.0e-8));
}

// --- end to end ------------------------------------------------------------

TEST_CASE ("a single saw is fitted back to one oscillator", "[fit]")
{
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
    const auto fitted = fitOf (renderFor (patch));

    CHECK (fitted.activeOscCount() >= 1);
    CHECK (fitted.rootHz == Catch::Approx (220.0f).epsilon (0.05));
}

TEST_CASE ("silence is fitted without crashing or inventing content", "[fit]")
{
    const std::vector<float> x ((size_t) kSampleRate, 0.0f);
    const auto fitted = fitOf (x);
    CHECK (peakOf (render (fitted)) < 0.2f);
}

TEST_CASE ("refinement improves the objective and never regresses", "[fit]")
{
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
    const auto target = renderFor (patch);

    const auto fitted = fitOf (target);
    Refine::Options options;
    options.maxEvaluations = 96;
    options.renderer = renderer();
    const auto result = Refine::run (fitted, target.data(), (int) target.size(),
                                     kSampleRate, options);

    INFO ("initial " << result.initialLoss << " final " << result.finalLoss);
    // The best candidate is kept explicitly, so a worse result is not merely
    // unlikely -- it is a bug.
    CHECK (result.finalLoss <= result.initialLoss + 1.0e-9);
}

TEST_CASE ("a fit without a renderer is visibly unfinished", "[fit]")
{
    // The guard for a closed loop silently becoming no loop.
    //
    // Level calibration and noise calibration both render, so both do nothing
    // without a synth to render through. A caller that forgets to pass one gets
    // a patch back that looks like a fit the whole way -- oscillators, envelope,
    // filter, all populated -- and is wrong in exactly the two places nothing
    // else checks. `autosynth_vital --fit` forgot, and shipped a clarinet four
    // decibels quiet with its noise bed at zero.
    //
    // So the two states have to be distinguishable, and this says how: no
    // renderer, no noise.
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
    patch.noiseLevel = 0.3f;
    const auto target = render (patch, 220.0, 2.0, 2.0);

    PartialFit::Options bare;
    bare.hop = kHop;
    const auto unfinished = PartialFit::fit (target.data(), (int) target.size(),
                                             kSampleRate, bare);
    CHECK (unfinished.noiseLevel == 0.0f);

    PartialFit::Options full;
    full.hop = kHop;
    full.renderer = renderer();
    const auto fitted = PartialFit::fit (target.data(), (int) target.size(),
                                         kSampleRate, full);
    INFO ("noise " << fitted.noiseLevel);
    CHECK (fitted.noiseLevel > 0.0f);
}

TEST_CASE ("the objective can see a patch that is simply quiet", "[fit]")
{
    // The anchor the loss was missing, and the reason the level wandered.
    //
    // `loudness` is a mean over every frame, so a tail 20 dB too loud outweighs
    // a note body 3 dB too quiet, and the cheapest way for an optimiser to
    // reduce it is to turn the whole patch down. `spectral` pulls the other
    // way. Between them the level a listener hears was whatever the two biases
    // cancelled to, which is how one clarinet fitted from three seeds came back
    // 6.9 dB quiet, correct, and correct again.
    //
    // This term is a median over the note alone, so the tail cannot reach it
    // and the attack cannot either.
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.01f, 0.1f, 0.7f, 0.15f, 0.0f };
    patch.masterLevel = 0.6f;

    const auto target = render (patch, 220.0, 2.0, 1.4);

    Refine::Options options;
    options.gateSeconds = 1.4;
    options.renderer = renderer();

    const auto matched = Refine::measure (patch, target.data(), (int) target.size(),
                                          kSampleRate, options);
    INFO ("matched level error " << matched.level << " dB");
    CHECK (matched.level < 0.5);

    // Half the gain is 6 dB down, and the term reads it as such rather than
    // normalising it away.
    auto quiet = patch;
    quiet.masterLevel = 0.3f;
    const auto quieter = Refine::measure (quiet, target.data(), (int) target.size(),
                                          kSampleRate, options);
    INFO ("6 dB down reads as " << quieter.level << " dB");
    CHECK (quieter.level == Catch::Approx (6.0).margin (1.5));

    // And the same the other way, so it is a distance rather than a penalty on
    // being quiet.
    auto loud = patch;
    loud.masterLevel = 1.2f;
    const auto louder = Refine::measure (loud, target.data(), (int) target.size(),
                                         kSampleRate, options);
    INFO ("6 dB up reads as " << louder.level << " dB");
    CHECK (louder.level == Catch::Approx (6.0).margin (1.5));
}

TEST_CASE ("the master is not clamped to one", "[fit]")
{
    // Every axis but one in the diagnostic is shape-relative, so a fit that is
    // uniformly quiet reports as fine on all of them. That is what happened: a
    // violin needing a master of 1.24 was clamped at 1.0 and came back seven
    // decibels down with ten axes saying ok. Freed, it lands within two.
    //
    // The clamp was wrong because everything between an oscillator and the
    // output attenuates, so the level a recording was made at is routinely more
    // than a patch reaches with its master at one. The ceiling is what the
    // preset can carry, which `the master ceiling is the one the preset can
    // carry` pins from the export side.
    //
    // Asserted where the clamp actually was rather than through a fit. A fitted
    // master cannot be predicted from the truth's: refinement is free to choose
    // a structure with different losses below the master, so the value that
    // reproduces one target legitimately differs between two patches that
    // sound alike. The violin is the end-to-end evidence and it is in
    // CONTRIBUTING.md, with numbers.
    const auto specs = Refine::continuousSpecs();
    const auto master = std::find_if (specs.begin(), specs.end(),
                                      [] (const Refine::ParamSpec& s) { return s.path == "master_level"; });
    REQUIRE (master != specs.end());
    INFO ("search range " << master->lo << " to " << master->hi);
    CHECK (master->hi == Catch::Approx (autosynth::kMaxMasterLevel));
    CHECK (master->hi > 1.0);

    // And analysis is allowed to ask for it, which is the other half: a level
    // solve that saturates at one hands refinement a starting point it cannot
    // reason its way out of.
    auto patch = simplePatch (Waveform::saw);
    patch.masterLevel = 2.5f;
    juce::String error;
    const auto back = Patch::fromJsonString (patch.toJson(), &error);
    REQUIRE (error.isEmpty());
    CHECK (back.masterLevel == Catch::Approx (2.5f));
}

TEST_CASE ("refinement finds note-off for itself", "[fit]")
{
    // The default used to be "hold the note for the whole file", which is wrong
    // for anything that stops before the end -- every real recording. Candidates
    // were rendered still sounding while the target had long gone quiet, so the
    // only way to fit was to mangle the envelope into faking a release it was
    // never allowed to perform, and refinement made library samples three to
    // five times *worse*. The plugin refines by default, so what it produced
    // was worse than the raw analysis it started from.
    //
    // The harness never caught it because it is the one caller that passed the
    // gate explicitly.
    //
    // Stated as the decision rather than as its consequence. The first version
    // compared the render from a detected gate against the render from a
    // held-open one and required the first to be closer. That held while the
    // renderer was ours and does not hold through Vital: given ninety-six
    // evaluations its envelope fakes the decay well enough that the two land
    // within half a decibel of each other, either way round from run to run.
    // What that measures is how much optimiser budget a mistake can be papered
    // over with, not whether the mistake was made.
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.01f, 0.1f, 0.7f, 0.15f, 0.0f };
    // A note that stops early and leaves a long quiet tail -- the shape of any
    // real recording, and where holding the note open is ruinous rather than
    // merely inaccurate.
    const auto target = render (patch, 220.0, 3.0, 0.6);

    Refine::Options options;
    options.maxEvaluations = 8;   // the gate is decided before the search runs
    options.renderer = renderer();
    const auto result = Refine::run (fitOf (target), target.data(), (int) target.size(),
                                     kSampleRate, options);

    INFO ("detected gate " << result.gateSeconds << " s, true 0.6");
    CHECK (result.gateSeconds == Catch::Approx (0.6).margin (0.1));

    // And a caller that says where the note ends is believed.
    options.gateSeconds = 3.0;
    CHECK (Refine::run (fitOf (target), target.data(), (int) target.size(),
                        kSampleRate, options).gateSeconds == Catch::Approx (3.0));
}

TEST_CASE ("refinement does not quietly turn the patch down", "[fit]")
{
    // The loudness term used to normalise each signal by its own peak, which
    // made it blind to level: a candidate 10 dB too quiet scored exactly as
    // well as one that matched. Nothing else anchored level either -- the
    // spectral log-term prefers a quiet render, since a fit carries more energy
    // than the target in the many near-silent bins. Refinement reliably dropped
    // real patches 8 to 11 dB below their source.
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.01f, 0.1f, 0.7f, 0.15f, 0.0f };
    const auto target = render (patch, 220.0, 2.0, 1.2);

    const auto fitted = fitOf (target);
    Refine::Options options;
    options.maxEvaluations = 96;
    options.renderer = renderer();
    const auto refined = Refine::run (fitted, target.data(), (int) target.size(),
                                      kSampleRate, options);

    const auto before = peakOf (render (fitted, 220.0, 2.0, 1.2));
    const auto after = peakOf (render (refined.patch, 220.0, 2.0, 1.2));
    const auto targetPeak = peakOf (target);

    INFO ("target " << targetPeak << "  before " << before << "  after " << after);
    REQUIRE (targetPeak > 0.01f);
    // Within 6 dB of the target's level. Refinement may trade a little level
    // for spectral accuracy; it may not walk away from it.
    CHECK (after > targetPeak * 0.5f);
}

TEST_CASE ("refinement never invents oscillators", "[fit]")
{
    // Refinement cannot add structure: an oscillator analysis left disabled or
    // at zero level is not in scope, so nothing CMA-ES does can bring it back.
    // It *can* retire one, by driving a level it does own to zero -- that is an
    // ordinary continuous move, not a structural edit through the back door.
    auto patch = simplePatch (Waveform::saw);
    patch.ampEnv = { 0.005f, 0.01f, 1.0f, 0.01f, 0.0f };
    const auto target = renderFor (patch);

    const auto fitted = fitOf (target);
    Refine::Options options;
    options.maxEvaluations = 96;
    options.renderer = renderer();
    const auto result = Refine::run (fitted, target.data(), (int) target.size(),
                                     kSampleRate, options);

    CHECK (result.patch.activeOscCount() <= fitted.activeOscCount());
    CHECK (result.patch.activeOscCount() >= 1);
}

TEST_CASE ("refinement scope excludes what analysis owns", "[fit]")
{
    // The split the whole design rests on: analysis makes the discrete and
    // structural decisions, refinement moves the continuous values. A waveform
    // or an oscillator count in this list would mean CMA-ES could undo a
    // decision it has no way to reason about.
    auto patch = simplePatch (Waveform::saw);
    patch.reverb = { true, 0.5f, 0.5f, 0.5f };
    const auto scope = Refine::scopeFor (patch);
    REQUIRE_FALSE (scope.empty());

    for (const auto& path : scope)
    {
        INFO (path);
        CHECK (path.find ("waveform") == std::string::npos);
        CHECK (path.find ("enabled") == std::string::npos);
        CHECK (path.find ("semitones") == std::string::npos);
        CHECK (path.find (".type") == std::string::npos);
    }
}

TEST_CASE ("refinement scope covers the reverb only when it is on", "[fit]")
{
    auto off = simplePatch (Waveform::saw);
    off.reverb.enabled = false;
    auto on = off;
    on.reverb = { true, 0.5f, 0.5f, 0.5f };

    const auto scopeOff = Refine::scopeFor (off);
    const auto scopeOn = Refine::scopeFor (on);

    const auto mentionsReverb = [] (const std::vector<std::string>& scope)
    {
        for (const auto& p : scope)
            if (p.rfind ("reverb.", 0) == 0)
                return true;
        return false;
    };

    // Dead parameters inflate the dimensionality CMA-ES has to cover, and its
    // sample efficiency falls off with dimension.
    CHECK_FALSE (mentionsReverb (scopeOff));
    CHECK (mentionsReverb (scopeOn));
}

// The envelope the fitter writes has to measure back as what it measured.
//
// `fitAdsr` returns a *parameter* and `attackSeconds` returns a *measurement*,
// and they are only the same number when the curve is linear. Before the two
// were separated, a violin measured at 0.401 s was written as an attack of
// 0.401 with a curve of 3 and rendered as 0.277 -- and the diagnostic then
// reported an attack error the analysis had not made.
TEST_CASE ("a fitted attack renders as the attack that was measured", "[envelope]")
{
    for (const float wanted : { 0.05f, 0.2f, 0.5f })
    {
        INFO ("wanted " << wanted);

        // A contour that ramps linearly to full over `wanted`, then holds.
        constexpr double fps = 200.0;
        constexpr double gate = 2.0;
        std::vector<float> curve, times;
        for (int i = 0; i < static_cast<int> (fps * 2.5); ++i)
        {
            const auto t = i / fps;
            times.push_back (static_cast<float> (t));
            curve.push_back (t < wanted ? static_cast<float> (t / wanted) : 1.0f);
        }

        // The definition, pinned: onset (5% of full) to nine tenths of full.
        // On a linear ramp that is 85% of the ramp's length.
        const auto measured = autosynth::EnvelopeFit::attackSeconds (curve, times);
        CHECK (measured == Catch::Approx (0.85 * wanted).margin (0.02));

        // Render the fitted envelope over the same grid and measure it the same
        // way. Any disagreement here is the fitter writing one thing and the
        // engine playing another.
        const auto env = autosynth::EnvelopeFit::fitAdsr (curve, times, gate, 0.05f, false);
        std::vector<float> rendered (times.size());
        for (size_t i = 0; i < times.size(); ++i)
            rendered[i] = autosynth::Envelope::evaluate (env, times[i], gate);

        const auto back = autosynth::EnvelopeFit::attackSeconds (rendered, times);
        CHECK (back == Catch::Approx (measured).margin (0.05));
    }
}

// The same invariant, held against a curve the optimiser chose rather than the
// one the fit did.
//
// This is what the first version got wrong. It assumed the crossing was a fixed
// fraction of the attack segment, which is only true when the decay does not
// pull the level down underneath it: `attackSeconds` measures nine tenths of
// the level the note *holds*, not of the attack's own peak. With a decay to a
// sustain of 0.45 the two are far apart, and a violin whose envelope should
// have measured 0.40 s rendered at 0.29.
TEST_CASE ("the attack solve survives a change of curve", "[envelope]")
{
    constexpr double fps = 200.0;
    constexpr double gate = 2.0;
    std::vector<float> times (static_cast<size_t> (fps * 2.5));
    for (size_t i = 0; i < times.size(); ++i)
        times[i] = static_cast<float> (i / fps);

    for (const float curve : { 0.0f, 1.5f, 3.0f })
        for (const float sustain : { 1.0f, 0.45f })
        {
            INFO ("curve " << curve << " sustain " << sustain);

            autosynth::Adsr env { 0.2f, 0.15f, sustain, 0.2f, curve };
            constexpr float wanted = 0.35f;
            env.attack = autosynth::EnvelopeFit::attackMeasuring (env, gate, wanted);

            std::vector<float> rendered (times.size());
            for (size_t i = 0; i < times.size(); ++i)
                rendered[i] = autosynth::Envelope::evaluate (env, times[i], gate);

            CHECK (autosynth::EnvelopeFit::attackSeconds (rendered, times)
                       == Catch::Approx (wanted).margin (0.02));
        }
}
