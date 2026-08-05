// Engine behaviour, stated as properties rather than as sample comparisons.
//
// The golden fixtures already pin the exact numbers. What these add is the
// question the fixtures cannot answer: does each control still *do* the thing
// it is named after? A parameter that has quietly become a no-op still matches
// its own frozen output if that output was frozen after it broke -- and three
// such no-ops did once survive in Voice::render precisely because every test
// compared audio against audio and none asserted that a knob changed anything.

#include "Helpers.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autotest;
using namespace autosynth;

TEST_CASE ("every waveform produces audio", "[engine]")
{
    for (auto wf : { Waveform::sine, Waveform::triangle, Waveform::saw,
                     Waveform::square, Waveform::pulse })
    {
        INFO ("waveform " << (int) wf);
        const auto out = render (simplePatch (wf));
        CHECK (peakOf (out) > 0.1f);
    }
}

TEST_CASE ("waveforms differ in brightness as expected", "[engine]")
{
    // A sine has one partial, a saw has all of them. If these ever converge,
    // the band-limited table build has collapsed.
    const auto sine = meanCentroidHz (render (simplePatch (Waveform::sine)));
    const auto saw = meanCentroidHz (render (simplePatch (Waveform::saw)));
    const auto square = meanCentroidHz (render (simplePatch (Waveform::square)));

    CHECK (sine < saw);
    CHECK (sine < square);
    CHECK (sine < 400.0);
}

TEST_CASE ("pitch tracks the requested note", "[engine]")
{
    for (double hz : { 110.0, 220.0, 440.0 })
    {
        auto patch = simplePatch (Waveform::sine);
        patch.rootHz = (float) hz;
        const auto out = render (patch);
        // A sine's centroid is its fundamental.
        INFO ("requested " << hz);
        CHECK (meanCentroidHz (out) == Catch::Approx (hz).epsilon (0.15));
    }
}

TEST_CASE ("the amplitude envelope shapes the note", "[engine]")
{
    auto slow = simplePatch();
    slow.ampEnv = { 0.4f, 0.1f, 0.8f, 0.1f, 0.0f };
    auto fast = simplePatch();
    fast.ampEnv = { 0.001f, 0.1f, 0.8f, 0.1f, 0.0f };

    const auto slowOut = render (slow);
    const auto fastOut = render (fast);

    // Early in the note the slow attack must still be climbing.
    const auto window = (size_t) (0.05 * kSampleRate);
    double slowEarly = 0.0, fastEarly = 0.0;
    for (size_t i = 0; i < window; ++i)
    {
        slowEarly += std::abs (slowOut[i]);
        fastEarly += std::abs (fastOut[i]);
    }
    CHECK (slowEarly < 0.5 * fastEarly);
}

TEST_CASE ("envelope curve changes the decay shape", "[engine]")
{
    // Not cosmetic: a linear-only envelope mis-stated a plucked decay by
    // 5-7 dB and then hit digital silence while the real tail was still
    // audible. The curve parameter exists because of that measurement.
    auto linear = simplePatch();
    linear.ampEnv = { 0.001f, 0.5f, 0.0f, 0.05f, 0.0f };
    auto curved = linear;
    curved.ampEnv.curve = 1.0f;

    const auto a = render (linear);
    const auto b = render (curved);
    CHECK (loudnessDistanceDb (a, b) > 1.0);
}

TEST_CASE ("a lowpass filter removes high frequencies", "[engine]")
{
    auto open = simplePatch (Waveform::saw);
    auto closed = open;
    closed.filter.type = FilterType::lowpass;
    closed.filter.cutoffHz = 400.0f;
    closed.filter.resonance = 0.3f;

    const auto openOut = render (open);
    const auto closedOut = render (closed);

    CHECK (meanCentroidHz (closedOut) < 0.6 * meanCentroidHz (openOut));
    CHECK (bandEnergy (closedOut, 3000.0, 10000.0)
           < 0.3 * bandEnergy (openOut, 3000.0, 10000.0));
}

TEST_CASE ("a highpass filter removes low frequencies", "[engine]")
{
    auto open = simplePatch (Waveform::saw);
    auto hp = open;
    hp.filter.type = FilterType::highpass;
    hp.filter.cutoffHz = 2000.0f;

    const auto openOut = render (open);
    const auto hpOut = render (hp);

    CHECK (meanCentroidHz (hpOut) > meanCentroidHz (openOut));
    CHECK (bandEnergy (hpOut, 100.0, 400.0) < 0.5 * bandEnergy (openOut, 100.0, 400.0));
}

TEST_CASE ("the filter envelope sweeps the cutoff", "[engine]")
{
    auto patch = simplePatch (Waveform::saw);
    patch.filter.type = FilterType::lowpass;
    patch.filter.cutoffHz = 400.0f;
    patch.filter.envAmount = 3.0f;
    patch.filter.env = { 0.001f, 0.5f, 0.0f, 0.1f, 0.0f };

    const auto out = render (patch);
    auto flat = patch;
    flat.filter.envAmount = 0.0f;
    const auto flatOut = render (flat);

    // With the envelope opening the filter, the note must start brighter.
    CHECK (meanCentroidHz (out) > meanCentroidHz (flatOut));
}

TEST_CASE ("unison thickens without acting as a gain control", "[engine]")
{
    // Equal-power normalisation. Without it, five voices would be five times
    // louder and `unison_voices` would be degenerate with `level` -- two knobs
    // for one effect, which is exactly the ambiguity a fitter cannot resolve.
    //
    // Stated as an energy ratio rather than a loudness distance: detuned voices
    // beat against each other, so the envelope genuinely moves, and a dB bound
    // would be measuring the beating rather than the normalisation.
    const auto rmsOf = [] (const std::vector<float>& x)
    {
        double acc = 0.0;
        for (auto v : x)
            acc += (double) v * v;
        return std::sqrt (acc / juce::jmax<size_t> (1, x.size()));
    };

    auto single = simplePatch (Waveform::saw);
    auto wide = single;
    wide.oscs[0].unisonVoices = 5;
    wide.oscs[0].unisonDetune = 25.0f;

    const auto a = rmsOf (render (single));
    const auto b = rmsOf (render (wide));

    INFO ("single " << a << "  unison " << b << "  ratio " << b / a);
    CHECK (b > 0.1 * a);
    // A gain would put this at 5. Equal-power keeps it near 1.
    CHECK (b < 2.0 * a);
}

TEST_CASE ("delay produces audible repeats after the note ends", "[engine]")
{
    auto dry = simplePatch (Waveform::saw);
    dry.ampEnv = { 0.001f, 0.06f, 0.0f, 0.01f, 0.0f };
    auto wet = dry;
    wet.delay = { true, 0.25f, 0.5f, 0.8f };

    const auto dryOut = render (dry);
    const auto wetOut = render (wet);

    const auto tailStart = (size_t) (0.5 * kSampleRate);
    double dryTail = 0.0, wetTail = 0.0;
    for (size_t i = tailStart; i < dryOut.size(); ++i)
    {
        dryTail += std::abs (dryOut[i]);
        wetTail += std::abs (wetOut[i]);
    }
    CHECK (wetTail > 5.0 * dryTail);
}

TEST_CASE ("a disabled delay is an exact no-op", "[engine]")
{
    auto patch = simplePatch (Waveform::saw);
    auto off = patch;
    off.delay = { false, 0.25f, 0.5f, 0.8f };

    const auto a = render (patch);
    const auto b = render (off);
    REQUIRE (a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        REQUIRE (a[i] == Catch::Approx (b[i]));
}

TEST_CASE ("reverb produces a tail the dry signal does not have", "[engine]")
{
    auto dry = simplePatch (Waveform::saw);
    dry.ampEnv = { 0.001f, 0.06f, 0.0f, 0.01f, 0.0f };
    auto wet = dry;
    wet.reverb = { true, 0.8f, 0.4f, 0.8f };
    wet.oscs[0].reverbSend = 1.0f;

    const auto dryOut = render (dry);
    const auto wetOut = render (wet);

    const auto tailStart = (size_t) (0.5 * kSampleRate);
    double dryTail = 0.0, wetTail = 0.0;
    for (size_t i = tailStart; i < dryOut.size(); ++i)
    {
        dryTail += std::abs (dryOut[i]);
        wetTail += std::abs (wetOut[i]);
    }
    INFO ("dry tail " << dryTail << "  wet tail " << wetTail);
    CHECK (wetTail > 5.0 * dryTail);
}

TEST_CASE ("a zero reverb send is an exact no-op", "[engine]")
{
    // The send is what makes the reverb per-oscillator. If it leaked, an
    // oscillator turned fully dry would still be audible in the return.
    auto patch = simplePatch (Waveform::saw);
    patch.reverb = { true, 0.8f, 0.4f, 0.8f };
    patch.oscs[0].reverbSend = 0.0f;

    auto noVerb = patch;
    noVerb.reverb.enabled = false;

    const auto a = render (patch);
    const auto b = render (noVerb);
    REQUIRE (a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        REQUIRE (a[i] == Catch::Approx (b[i]));
}

TEST_CASE ("reverb size lengthens the tail", "[engine]")
{
    const auto tailEnergy = [] (float size)
    {
        auto p = simplePatch (Waveform::saw);
        p.ampEnv = { 0.001f, 0.06f, 0.0f, 0.01f, 0.0f };
        p.reverb = { true, size, 0.0f, 0.8f };
        p.oscs[0].reverbSend = 1.0f;
        const auto out = render (p);
        double acc = 0.0;
        for (size_t i = (size_t) (0.6 * kSampleRate); i < out.size(); ++i)
            acc += std::abs (out[i]);
        return acc;
    };

    CHECK (tailEnergy (0.9f) > tailEnergy (0.2f));
}

TEST_CASE ("reverb damping darkens the tail", "[engine]")
{
    const auto brightness = [] (float damp)
    {
        auto p = simplePatch (Waveform::saw);
        p.ampEnv = { 0.001f, 0.06f, 0.0f, 0.01f, 0.0f };
        p.reverb = { true, 0.8f, damp, 0.8f };
        p.oscs[0].reverbSend = 1.0f;
        return meanCentroidHz (render (p));
    };

    CHECK (brightness (1.0f) < brightness (0.0f));
}

TEST_CASE ("each LFO destination changes the sound", "[engine]")
{
    auto base = simplePatch (Waveform::saw);
    base.filter.type = FilterType::lowpass;
    base.filter.cutoffHz = 1500.0f;
    const auto plain = render (base);

    for (auto dest : { LfoDest::pitch, LfoDest::amp, LfoDest::cutoff })
    {
        auto patch = base;
        patch.lfos[0] = { LfoShape::sine, dest, 5.0f, 0.5f, 0.0f, 0.0f };
        const auto out = render (patch);
        INFO ("dest " << (int) dest);
        // Something must differ, or the routing is not connected.
        double worst = 0.0;
        for (size_t i = 0; i < out.size(); ++i)
            worst = juce::jmax (worst, (double) std::abs (out[i] - plain[i]));
        CHECK (worst > 1.0e-3);
    }
}

TEST_CASE ("both LFO slots act at once", "[engine]")
{
    // The reason there are two. With one slot, vibrato and tremolo compete and
    // only one survives -- a sound with both cannot be represented at all.
    auto base = simplePatch (Waveform::saw);

    auto pitchOnly = base;
    pitchOnly.lfos[0] = { LfoShape::sine, LfoDest::pitch, 5.5f, 0.4f, 0.0f, 0.0f };

    auto both = pitchOnly;
    both.lfos[1] = { LfoShape::sine, LfoDest::amp, 3.0f, 0.5f, 0.0f, 0.0f };

    const auto a = render (pitchOnly);
    const auto b = render (both);
    CHECK (loudnessDistanceDb (a, b) > 0.5);
}

TEST_CASE ("per-oscillator filters shape oscillators independently", "[engine]")
{
    // The test that was missing when the C++ voice prepared each oscillator's
    // filter, reset it, advanced its envelope, and never passed a sample
    // through it. Comparing filtered against open *within one patch* is what
    // makes that catchable: a global filter cannot produce this signal.
    auto patch = simplePatch (Waveform::saw);
    patch.oscs[1].enabled = true;
    patch.oscs[1].level = 1.0f;
    patch.oscs[1].semitones = 7;

    auto filtered = patch;
    filtered.oscs[0].filterEnabled = true;
    filtered.oscs[0].filter.type = FilterType::lowpass;
    filtered.oscs[0].filter.cutoffHz = 400.0f;
    filtered.oscs[0].filter.resonance = 0.2f;
    filtered.oscs[0].filter.envAmount = 0.0f;

    const auto open = render (patch);
    const auto shaped = render (filtered);

    INFO ("open " << meanCentroidHz (open) << "  shaped " << meanCentroidHz (shaped));
    CHECK (meanCentroidHz (shaped) < meanCentroidHz (open));
    CHECK (centroidDistanceOctaves (shaped, open) > 0.15);
}

TEST_CASE ("the monitor mask silences oscillators", "[engine]")
{
    // Solo/mute is monitoring state, not a patch parameter, so nothing outside
    // the plugin could reach it -- which is how it sat stored-but-unapplied
    // with the editor's buttons doing nothing at all.
    auto patch = simplePatch (Waveform::saw);
    patch.oscs[1].enabled = true;
    patch.oscs[1].level = 1.0f;
    patch.oscs[1].semitones = 7;

    const std::array<float, kNumOsc> all { 1.0f, 1.0f, 1.0f };
    const std::array<float, kNumOsc> first { 1.0f, 0.0f, 1.0f };
    const std::array<float, kNumOsc> none { 0.0f, 0.0f, 0.0f };

    const auto both = render (patch, 0.0, kDuration, kGate, &all);
    const auto only0 = render (patch, 0.0, kDuration, kGate, &first);
    const auto silent = render (patch, 0.0, kDuration, kGate, &none);

    CHECK (peakOf (both) > 1.0e-3f);
    CHECK (peakOf (only0) > 1.0e-3f);

    // The assertion that pins the bug: with the mask unapplied this came back
    // at full level.
    CHECK (peakOf (silent) < 1.0e-6f);

    // Muting the fifth must darken what remains. Compared spectrally rather
    // than by total energy, because the render peak-limits and a quieter mix
    // can carry the larger sample sum.
    CHECK (centroidDistanceOctaves (only0, both) > 0.05);
}

TEST_CASE ("wave morph blends between two tables", "[engine]")
{
    auto a = simplePatch (Waveform::saw);
    a.oscs[0].waveformB = Waveform::square;
    a.oscs[0].waveMorph = 0.0f;

    auto mid = a;
    mid.oscs[0].waveMorph = 0.5f;

    auto b = a;
    b.oscs[0].waveMorph = 1.0f;

    const auto sawOut = render (a);
    const auto midOut = render (mid);
    const auto sqOut = render (b);

    // Full morph must equal the plain square, and the midpoint must be neither.
    const auto plainSquare = render (simplePatch (Waveform::square));
    CHECK (loudnessDistanceDb (sqOut, plainSquare) < 0.5);
    CHECK (loudnessDistanceDb (midOut, sawOut) > 0.0);
}

TEST_CASE ("master level scales the output", "[engine]")
{
    auto loud = simplePatch();
    loud.masterLevel = 0.8f;
    auto quiet = loud;
    quiet.masterLevel = 0.2f;

    CHECK (peakOf (render (quiet)) < peakOf (render (loud)));
}

TEST_CASE ("a silent patch renders silence", "[engine]")
{
    auto patch = simplePatch();
    for (auto& osc : patch.oscs)
    {
        osc.enabled = false;
        osc.level = 0.0f;
    }
    patch.noiseLevel = 0.0f;
    CHECK (peakOf (render (patch)) < 1.0e-6f);
}

TEST_CASE ("rendering is deterministic", "[engine]")
{
    const auto patch = simplePatch (Waveform::saw);
    const auto a = render (patch);
    const auto b = render (patch);
    REQUIRE (a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        REQUIRE (a[i] == b[i]);
}

TEST_CASE ("rendering is deterministic with noise enabled", "[engine]")
{
    // The case the test above could not see. It used a patch with no noise, so
    // it passed throughout the period when the noise generator was seeded from
    // the system clock and every render was a different signal.
    //
    // That was not a cosmetic problem. Level calibration builds a least-squares
    // column by rendering the noise source, so fitted levels drifted between
    // runs and a marginal oscillator appeared or vanished at random -- which
    // showed up as an intermittently failing conformance test rather than as
    // anything obviously to do with noise.
    auto patch = simplePatch (Waveform::saw);
    patch.noiseLevel = 0.4f;

    const auto a = render (patch);
    const auto b = render (patch);
    REQUIRE (a.size() == b.size());
    REQUIRE (peakOf (a) > 0.1f);
    for (size_t i = 0; i < a.size(); ++i)
        REQUIRE (a[i] == b[i]);
}

TEST_CASE ("different notes get different noise", "[engine]")
{
    // Determinism must not go so far as to give every voice the same noise
    // burst, or a chord would sum correlated noise and sound like one loud
    // source rather than several.
    auto patch = simplePatch (Waveform::saw);
    patch.noiseLevel = 0.4f;

    const auto low = render (patch, 110.0);
    const auto high = render (patch, 220.0);
    REQUIRE (low.size() == high.size());

    bool differs = false;
    for (size_t i = 0; i < low.size(); ++i)
        if (low[i] != high[i])
            differs = true;
    CHECK (differs);
}
