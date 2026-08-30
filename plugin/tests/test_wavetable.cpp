// Wavetable oscillators: the engine side, the fitter's parsimony ladder, and
// the round trip through the patch format.
//
// The engine assertions are phrased against the *classic* tables wherever they
// can be. A frame holding a saw's harmonic series has to sound like a saw, or
// the table path and the fixed path have quietly become two different
// oscillators and the fitter's choice between them stops meaning anything.

#include "Helpers.h"

#include "dsp/Tables.h"
#include "fit/WaveformFit.h"
#include "fit/WavetableFit.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace autotest;
using autosynth::Oscillator;
using autosynth::Patch;
using autosynth::Waveform;
using autosynth::WaveTables;
namespace WavetableFit = autosynth::WavetableFit;

namespace
{

using Harmonics = std::array<float, Oscillator::kFrameHarmonics>;
using Frames = std::array<Oscillator::Frame, Oscillator::kMaxFrames>;

// The harmonic series of one of the fixed shapes, as a frame.
Harmonics frameFor (Waveform waveform, float pulseWidth = 0.5f)
{
    const auto amps = WaveTables::blendedHarmonics (waveform, waveform, 0.0f, pulseWidth,
                                                    Oscillator::kFrameHarmonics);
    Harmonics out {};
    for (size_t k = 0; k < out.size(); ++k)
        out[k] = amps[k];
    return out;
}

// A drawn table: every frame custom, so none of them fall back to the shape.
Patch tablePatch (const std::vector<Harmonics>& drawn, float position, float envAmount = 0.0f)
{
    auto patch = simplePatch (Waveform::saw);
    auto& osc = patch.oscs[0];
    osc.numFrames = static_cast<int> (drawn.size());
    for (size_t f = 0; f < drawn.size(); ++f)
    {
        osc.frames[f].custom = true;
        osc.frames[f].harmonics = drawn[f];
    }
    osc.framePosition = position;
    osc.framePositionEnvAmount = envAmount;
    return patch;
}

// A synthetic harmonic matrix: `numHarmonics` by `numFrames`, harmonic-major,
// interpolating from `a` to `b` across the note at a constant total level.
std::vector<float> movingMatrix (const std::vector<float>& a, const std::vector<float>& b,
                                 int numFrames)
{
    const auto numHarmonics = static_cast<int> (a.size());
    std::vector<float> H (static_cast<size_t> (numHarmonics * numFrames), 0.0f);
    for (int t = 0; t < numFrames; ++t)
    {
        const auto u = numFrames > 1 ? static_cast<float> (t) / (numFrames - 1) : 0.0f;
        for (int k = 0; k < numHarmonics; ++k)
            H[static_cast<size_t> (k) * numFrames + t] =
                a[static_cast<size_t> (k)] * (1.0f - u) + b[static_cast<size_t> (k)] * u;
    }
    return H;
}

std::vector<float> frameTimes (int numFrames, float step = 0.01f)
{
    std::vector<float> times (static_cast<size_t> (numFrames));
    for (size_t i = 0; i < times.size(); ++i)
        times[i] = static_cast<float> (i) * step;
    return times;
}

} // namespace

TEST_CASE ("a frame holding a saw's harmonics sounds like a saw", "[wavetable]")
{
    const std::vector<Harmonics> frames { frameFor (Waveform::saw) };

    // Played high enough that Nyquist, not the frame length, is what limits the
    // harmonics: in this octave both are cut to sixteen, so they are the same
    // shape rather than merely a similar one. Lower down the generated saw
    // carries seventy-five harmonics and the frame sixteen, and the frame is
    // legitimately duller -- comparing them there would only measure that a
    // frame holds sixteen numbers, which is a design choice, not a bug.
    constexpr auto highNoteHz = 1500.0;
    const auto table = render (tablePatch (frames, 0.0f), highNoteHz);
    const auto classic = render (simplePatch (Waveform::saw), highNoteHz);

    CHECK (centroidDistanceOctaves (table, classic) < 0.02);

    // The same shape, not the same level. Every frame is peak-normalised on the
    // way into the preset and a saw's peak grows with the harmonics written
    // into it, so the drawn frame arrives about a decibel hotter than the
    // generated one even where both are then band limited to sixteen. That is
    // the normalisation showing, not the shape disagreeing.
    CHECK (loudnessDistanceDb (table, classic) < 1.5);
}

TEST_CASE ("frame position crossfades between frames", "[wavetable]")
{
    const std::vector<Harmonics> frames { frameFor (Waveform::sine),
                                          frameFor (Waveform::triangle),
                                          frameFor (Waveform::saw) };

    const auto dull = render (tablePatch (frames, 0.0f));
    const auto middle = render (tablePatch (frames, 0.5f));
    const auto bright = render (tablePatch (frames, 1.0f));

    const auto a = meanCentroidHz (dull);
    const auto b = meanCentroidHz (middle);
    const auto c = meanCentroidHz (bright);
    CHECK (a < b);
    CHECK (b < c);
}

TEST_CASE ("the frame envelope moves the position over the note", "[wavetable]")
{
    const std::vector<Harmonics> frames { frameFor (Waveform::sine),
                                          frameFor (Waveform::triangle),
                                          frameFor (Waveform::saw) };

    // Position 0 at note-on, swept to 1 over the note, exactly as the fitter
    // writes it.
    auto patch = tablePatch (frames, 0.0f, 1.0f);
    patch.oscs[0].framePositionEnv = { 0.6f, 0.001f, 1.0f, 0.1f, 0.0f };
    const auto swept = render (patch);

    const auto half = swept.size() / 2;
    const std::vector<float> early (swept.begin(), swept.begin() + static_cast<long> (half));
    const std::vector<float> late (swept.begin() + static_cast<long> (half), swept.end());
    CHECK (meanCentroidHz (early) < meanCentroidHz (late));
}

TEST_CASE ("frames survive a patch round trip", "[wavetable]")
{
    const std::vector<Harmonics> frames { frameFor (Waveform::sine),
                                          frameFor (Waveform::square),
                                          frameFor (Waveform::pulse, 0.2f) };

    auto patch = tablePatch (frames, 0.25f, -0.5f);
    patch.oscs[0].framePositionEnv = { 0.7f, 0.02f, 0.9f, 0.3f, 0.0f };

    juce::String error;
    const auto restored = Patch::fromJsonString (patch.toJson(), &error);
    REQUIRE (error.isEmpty());

    const auto& out = restored.oscs[0];
    CHECK (out.numFrames == 3);
    CHECK (out.framePosition == Catch::Approx (0.25f));
    CHECK (out.framePositionEnvAmount == Catch::Approx (-0.5f));
    CHECK (out.framePositionEnv.attack == Catch::Approx (0.7f));
    for (size_t f = 0; f < frames.size(); ++f)
    {
        CHECK (out.frames[f].custom);
        for (size_t k = 0; k < frames[f].size(); ++k)
            CHECK (out.frames[f].harmonics[k] == Catch::Approx (frames[f][k]).margin (1.0e-6));
    }
}

TEST_CASE ("a table is not spent on a sound a waveform already describes", "[wavetable]")
{
    // A clean saw, unmoving. The blend fits it, so no table should appear --
    // this is the case that used to slip through on a purely relative margin.
    std::vector<float> saw (Oscillator::kFrameHarmonics);
    for (size_t k = 0; k < saw.size(); ++k)
        saw[k] = 1.0f / static_cast<float> (k + 1);

    const auto numFrames = 200;
    const auto H = movingMatrix (saw, saw, numFrames);
    const std::vector<float> profile (saw.begin(), saw.end());

    const auto result = WavetableFit::fit (H.data(), Oscillator::kFrameHarmonics, numFrames,
                                           frameTimes (numFrames), 10.0f, false,
                                           autosynth::WaveformFit::matchBlend (profile), 1.0);
    CHECK_FALSE (result.useCustomFrames);
}

TEST_CASE ("a spectrum no waveform reaches earns a table", "[wavetable]")
{
    // A peak on the third harmonic with a cliff after it -- the shape none of
    // the five can make, and the reason the feature exists.
    std::vector<float> odd (Oscillator::kFrameHarmonics, 0.0f);
    odd[0] = 0.5f;
    odd[1] = 0.1f;
    odd[2] = 1.0f;
    odd[3] = 0.05f;
    for (size_t k = 4; k < odd.size(); ++k)
        odd[k] = 0.02f;

    const auto numFrames = 200;
    const auto H = movingMatrix (odd, odd, numFrames);
    const std::vector<float> profile (odd.begin(), odd.end());

    const auto result = WavetableFit::fit (H.data(), Oscillator::kFrameHarmonics, numFrames,
                                           frameTimes (numFrames), 10.0f, false,
                                           autosynth::WaveformFit::matchBlend (profile), 1.0);
    REQUIRE (result.useCustomFrames);

    // One frame, not three: nothing moved.
    CHECK (result.numFrames == 1);
    CHECK (result.envAmount == Catch::Approx (0.0f));
}

TEST_CASE ("a tone that travels earns three frames", "[wavetable]")
{
    std::vector<float> early (Oscillator::kFrameHarmonics, 0.0f);
    std::vector<float> late (Oscillator::kFrameHarmonics, 0.0f);
    early[0] = 1.0f; early[1] = 0.05f; early[2] = 0.8f;
    late[0] = 1.0f;  late[1] = 0.9f;   late[2] = 0.1f;
    for (size_t k = 3; k < early.size(); ++k)
    {
        early[k] = 0.02f;
        late[k] = 0.02f;
    }

    const auto numFrames = 200;
    const auto H = movingMatrix (early, late, numFrames);
    std::vector<float> mean (Oscillator::kFrameHarmonics);
    for (size_t k = 0; k < mean.size(); ++k)
        mean[k] = 0.5f * (early[k] + late[k]);

    const auto result = WavetableFit::fit (H.data(), Oscillator::kFrameHarmonics, numFrames,
                                           frameTimes (numFrames), 10.0f, false,
                                           autosynth::WaveformFit::matchBlend (mean), 1.0);
    REQUIRE (result.useCustomFrames);
    CHECK (result.numFrames == autosynth::WavetableFit::kFittedFrames);
    CHECK (result.envAmount == Catch::Approx (1.0f));
    CHECK (result.frames.front() != result.frames.back());

    // Fitted in the order the note was played, not reversed.
    CHECK (result.frames.front()[2] > result.frames.front()[1]);
    CHECK (result.frames.back()[1] > result.frames.back()[2]);
}

TEST_CASE ("a quiet layer is not given a table", "[wavetable]")
{
    std::vector<float> odd (Oscillator::kFrameHarmonics, 0.02f);
    odd[0] = 0.5f;
    odd[2] = 1.0f;

    const auto numFrames = 200;
    const auto H = movingMatrix (odd, odd, numFrames);
    const std::vector<float> profile (odd.begin(), odd.end());
    const auto blend = autosynth::WaveformFit::matchBlend (profile);

    CHECK (WavetableFit::fit (H.data(), Oscillator::kFrameHarmonics, numFrames,
                              frameTimes (numFrames), 10.0f, false, blend, 1.0).useCustomFrames);
    CHECK_FALSE (WavetableFit::fit (H.data(), Oscillator::kFrameHarmonics, numFrames,
                                    frameTimes (numFrames), 10.0f, false, blend, 0.05).useCustomFrames);
}

TEST_CASE ("an undrawn frame is the waveform itself, at full bandwidth", "[wavetable]")
{
    // The whole reason a frame stays generated until it is touched: a saw is a
    // saw, not its first sixteen harmonics. At 220 Hz that is seventy-five
    // harmonics against sixteen, and the difference is plainly audible -- so
    // this compares a default oscillator, whose single frame nobody has drawn
    // on, against the same frame written out as sixteen numbers.
    //
    // It is a test of the exporter as much as of the format. The first version
    // wrote every frame as sixteen harmonics, generated ones included, and the
    // two renders below came back identical to the last decimal -- the whole
    // band above the sixteenth harmonic thrown away with nothing to say so.
    auto patch = simplePatch (Waveform::saw);
    CHECK (patch.oscs[0].numFrames == 1);
    CHECK_FALSE (patch.oscs[0].frames[0].custom);

    const auto plain = render (patch);

    auto drawn = patch;
    drawn.oscs[0].frames[0].custom = true;
    drawn.oscs[0].frames[0].harmonics = frameFor (Waveform::saw);
    const auto sixteen = render (drawn);

    CHECK (meanCentroidHz (plain) > meanCentroidHz (sixteen) * 2.0);
}

