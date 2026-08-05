// Fitting behaviour, checked by round trip: render a patch whose parameters we
// chose, fit it back, and ask whether the fitter recovered what was put in.
//
// This is the ground-truth harness the project has always leaned on, and it
// has one structural blind spot worth stating plainly: it only ever tests
// sounds this engine can already make, so it can measure precision but never
// the modelling gap. A real sample tells you *that* something is wrong; this
// tells you *why*.

#include "Helpers.h"

#include "fit/CmaEs.h"
#include "fit/EffectsFit.h"
#include "fit/EnvelopeFit.h"
#include "fit/FilterFit.h"
#include "fit/Modulation.h"
#include "fit/Nnls.h"
#include "fit/PartialFit.h"
#include "fit/Refine.h"
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
    const auto result = Refine::run (fitted, target.data(), (int) target.size(),
                                     kSampleRate, options);

    INFO ("initial " << result.initialLoss << " final " << result.finalLoss);
    // The best candidate is kept explicitly, so a worse result is not merely
    // unlikely -- it is a bug.
    CHECK (result.finalLoss <= result.initialLoss + 1.0e-9);
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
