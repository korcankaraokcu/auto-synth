#include "fit/EnvelopeFit.h"

#include "dsp/Envelope.h"
#include "fit/NdFilters.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace autosynth
{

// How much the amplitude contour is smoothed before flatness, full level and
// the decay are judged on it -- roughly one vibrato period. Named because three
// separate measurements depend on being smoothed by the *same* amount, and a
// decay cannot be shorter than what this can resolve.
constexpr double kContourSmoothSeconds = 0.25;

// How slowly an inferred trajectory is assumed to fall, in seconds per unit of
// drop. A quarter second for a full-range fall is slow enough that no step
// survives and fast enough to leave a real sweep its shape.
constexpr float kMinFallSecondsPerUnit = 0.25f;
namespace
{
const float kCurveCandidates[] = { 0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f };

double meanDiff (const std::vector<float>& times)
{
    if (times.size() < 2)
        return 1.0e-3;
    double sum = 0.0;
    for (size_t i = 1; i < times.size(); ++i)
        sum += times[i] - times[i - 1];
    const auto dt = sum / static_cast<double> (times.size() - 1);
    return dt > 0.0 ? dt : 1.0e-3;
}
} // namespace

int EnvelopeFit::peakIndex (const std::vector<float>& curve, float frac)
{
    if (curve.empty())
        return 0;
    const auto peak = *std::max_element (curve.begin(), curve.end());
    if (peak <= 1.0e-12f)
        return 0;
    for (size_t i = 0; i < curve.size(); ++i)
        if (curve[i] >= frac * peak)
            return static_cast<int> (i);
    return 0;
}

EnvelopeFit::Gate EnvelopeFit::detectGate (const std::vector<float>& rms,
                                           const std::vector<float>& times,
                                           float maxPlateauRangeDb, double minPlateauSeconds,
                                           double rangeWindowSeconds, float levelFloor,
                                           double smoothSeconds, double plateauSmoothSeconds)
{
    Gate gate;
    if (rms.size() < 4 || times.size() < 4)
    {
        gate.time = times.empty() ? 0.0 : times.back();
        gate.oneShot = true;
        return gate;
    }

    const auto peak = *std::max_element (rms.begin(), rms.end());
    if (peak <= 1.0e-9f)
    {
        gate.time = times.back();
        gate.oneShot = true;
        return gate;
    }

    std::vector<float> norm (rms.size());
    std::vector<float> db (rms.size());
    for (size_t i = 0; i < rms.size(); ++i)
    {
        norm[i] = rms[i] / peak;
        db[i] = static_cast<float> (20.0 * std::log10 (norm[i] + 1.0e-6));
    }

    const auto dt = meanDiff (times);

    // Smooth before measuring: frame-to-frame RMS ripple of even 1% reads as
    // ~16 dB/s across a 5 ms hop.
    const auto span = std::max (1, static_cast<int> (std::lround (smoothSeconds / dt)));
    if (span > 1)
        db = nd::uniformFilter1d (db, span);

    // Flatness is judged on a *more* heavily smoothed copy, over roughly one
    // vibrato period.
    //
    // A played note is not a flat note. A violin's sustain swings some 14 dB at
    // vibrato rate, so against a 2.5 dB flatness threshold no plateau was ever
    // found and a four-second sustained bow was classified one-shot with its
    // gate at the very end of the file -- which then fitted a 1.2 s attack and
    // a tail forty times too loud. Synthetic test tones hold a dead-flat
    // sustain, so nothing in the suite could show this.
    //
    // Averaging over a vibrato period does not weaken the pluck test that the
    // window below exists for: smoothing flattens an *oscillation* but leaves a
    // monotonic decay's slope untouched, so a falling envelope still spans its
    // full range and still reads as one-shot.
    auto flatness = db;
    const auto plateauSpan = std::max (1, static_cast<int> (std::lround (plateauSmoothSeconds / dt)));
    if (plateauSpan > 1)
        flatness = nd::uniformFilter1d (flatness, plateauSpan);

    // The range window is longer than the plateau-length requirement on
    // purpose. A slow pluck (-17 dB/s) spans only 2.6 dB over 150 ms, close
    // enough to a vibrato tone's 0.6 dB ripple that no threshold separates
    // them; over 250 ms the same pluck spans 4.4 dB and the gap is comfortable.
    const auto window = std::max (3, static_cast<int> (std::lround (rangeWindowSeconds / dt)));
    const auto upper = nd::maximumFilter1d (flatness, window);
    const auto lower = nd::minimumFilter1d (flatness, window);

    const auto peakI = peakIndex (norm);

    int bestLength = 0, bestEnd = -1, run = 0;
    for (size_t i = 0; i < db.size(); ++i)
    {
        const auto spread = upper[i] - lower[i];
        const auto flat = (static_cast<int> (i) > peakI)
                       && (spread < maxPlateauRangeDb)
                       && (norm[i] > levelFloor);
        run = flat ? run + 1 : 0;
        if (run > bestLength)
        {
            bestLength = run;
            bestEnd = static_cast<int> (i);
        }
    }

    if (bestLength * dt >= minPlateauSeconds && bestEnd >= 0)
    {
        gate.oneShot = false;

        // The flat run says the note *is* sustained. It does not say where the
        // note ends, and using its end for that is fragile: on a real played
        // note the sustain wobbles, so the longest flat stretch finishes
        // wherever the wobble happened to be briefly calm. A clarinet whose
        // note plainly stops at 3.0 s was gated at 1.38 s that way -- and at
        // 2.62 s before the smoothing above, which is the same failure with a
        // different arbitrary answer.
        //
        // Note-off is where the envelope leaves the sustain region for good, so
        // find it by walking back from the end: the last frame still within
        // 6 dB of the level actually being held.
        const auto runStart = std::max (0, bestEnd - bestLength + 1);
        std::vector<float> plateau (db.begin() + runStart, db.begin() + bestEnd + 1);
        std::sort (plateau.begin(), plateau.end());
        const auto sustainDb = plateau[plateau.size() / 2];

        auto release = static_cast<int> (db.size()) - 1;
        while (release > bestEnd && db[static_cast<size_t> (release)] < sustainDb - 6.0f)
            --release;

        gate.time = times[static_cast<size_t> (release)];
    }
    else
    {
        gate.time = times.back();
        gate.oneShot = true;
    }
    return gate;
}

float EnvelopeFit::fitAttackCurve (const std::vector<float>& observed,
                                   const std::vector<float>& times, const Adsr& env,
                                   double gateTime, bool amplitudeDomain)
{
    if (observed.size() < 4 || env.attack <= 1.0e-4f)
        return 0.0f;

    const auto dt = meanDiff (times);
    const auto peak = *std::max_element (observed.begin(), observed.end());
    if (peak <= 1.0e-9f)
        return 0.0f;

    // Only the rise, and compared in decibels, because that is where the fault
    // lives. A linear ramp's error is concentrated in the first third of the
    // attack, where it is 10 to 16 dB below the target but only a few percent
    // away in amplitude -- so an amplitude comparison cannot see it, and picks
    // linear every time.
    const auto last = juce::jlimit<size_t> (2, observed.size() - 1,
                                            static_cast<size_t> (env.attack / juce::jmax (dt, 1.0e-9)));

    auto best = std::numeric_limits<double>::infinity();
    auto bestCurve = 0.0f;
    for (auto candidate : kCurveCandidates)
    {
        auto trial = env;
        trial.attackCurve = candidate;

        double error = 0.0;
        for (size_t i = 0; i <= last; ++i)
        {
            const auto value = Envelope::evaluate (trial, static_cast<double> (i) * dt, gateTime);
            if (amplitudeDomain)
            {
                const auto modelDb = 20.0 * std::log10 (std::max (static_cast<double> (value), 1.0e-4));
                const auto targetDb = 20.0 * std::log10 (std::max (observed[i] / peak, 1.0e-4f));
                error += std::abs (modelDb - targetDb);
            }
            else
            {
                error += std::abs (static_cast<double> (value) - observed[i] / peak);
            }
        }
        if (error < best)
        {
            best = error;
            bestCurve = candidate;
        }
    }
    return bestCurve;
}

float EnvelopeFit::fitCurve (const std::vector<float>& observed, const std::vector<float>& times,
                             const Adsr& env, double gateTime)
{
    if (observed.size() < 4)
        return 0.0f;
    const auto dt = meanDiff (times);
    const auto peak = *std::max_element (observed.begin(), observed.end());
    if (peak <= 1.0e-9f)
        return 0.0f;

    std::vector<float> target (observed.size());
    for (size_t i = 0; i < observed.size(); ++i)
        target[i] = std::max (-80.0f,
                              static_cast<float> (20.0 * std::log10 (observed[i] / peak + 1.0e-12)));

    auto best = std::numeric_limits<double>::infinity();
    auto bestCurve = 0.0f;

    for (auto candidate : kCurveCandidates)
    {
        auto trial = env;
        trial.curve = candidate;

        double error = 0.0;
        for (size_t i = 0; i < observed.size(); ++i)
        {
            const auto value = Envelope::evaluate (trial, static_cast<double> (i) * dt, gateTime);
            const auto modelDb = std::max (-80.0f,
                                           static_cast<float> (20.0 * std::log10 (value + 1.0e-12)));
            error += std::abs (modelDb - target[i]);
        }
        error /= static_cast<double> (observed.size());

        if (error < best)
        {
            best = error;
            bestCurve = candidate;
        }
    }
    return bestCurve;
}

float EnvelopeFit::attackMeasuring (Adsr env, double gateTime, float measured,
                                    double frameRate, float floor)
{
    const auto span = juce::jmax (gateTime, static_cast<double> (measured) * 2.0 + 0.5);
    const auto count = juce::jlimit (16, 4000, static_cast<int> (std::lround (span * frameRate)));

    std::vector<float> times (static_cast<size_t> (count));
    for (int i = 0; i < count; ++i)
        times[static_cast<size_t> (i)] = static_cast<float> (i / frameRate);

    std::vector<float> rendered (times.size());
    const auto crossingFor = [&] (float attack)
    {
        env.attack = attack;
        for (size_t i = 0; i < times.size(); ++i)
            rendered[i] = Envelope::evaluate (env, times[i], gateTime);
        return attackSeconds (rendered, times, floor);
    };

    auto lo = 0.001f, hi = 2.0f;
    for (int i = 0; i < 24; ++i)
    {
        const auto mid = 0.5f * (lo + hi);
        (crossingFor (mid) < measured ? lo : hi) = mid;
    }
    return 0.5f * (lo + hi);
}

float EnvelopeFit::attackSeconds (const std::vector<float>& rawCurve,
                                  const std::vector<float>& times, float floor)
{
    if (rawCurve.size() < 4 || times.size() < 4)
        return 0.0f;

    const auto peak = *std::max_element (rawCurve.begin(), rawCurve.end());
    if (peak <= 1.0e-9f)
        return 0.0f;

    std::vector<float> curve (rawCurve.size());
    for (size_t i = 0; i < rawCurve.size(); ++i)
        curve[i] = rawCurve[i] / peak;

    // "Full level" judged on a smoothed copy, so a vibrato crest a second into
    // the note cannot stand in for the level the note actually holds; the
    // crossing itself is still found on the unsmoothed curve, so a genuinely
    // fast attack keeps its timing.
    auto contour = curve;
    const auto span = std::max (1, static_cast<int> (std::lround (kContourSmoothSeconds / meanDiff (times))));
    if (span > 1 && static_cast<int> (contour.size()) > span)
        contour = nd::uniformFilter1d (contour, span);
    const auto fullLevel = *std::max_element (contour.begin(), contour.end());

    const auto firstAbove = [&curve] (float level)
    {
        for (size_t i = 0; i < curve.size(); ++i)
            if (curve[i] > level)
                return static_cast<int> (i);
        return 0;
    };

    const auto onsetI = firstAbove (floor);
    const auto peakI = firstAbove (0.9f * fullLevel);
    return std::max (times[static_cast<size_t> (peakI)] - times[static_cast<size_t> (onsetI)],
                     1.0e-3f);
}

Adsr EnvelopeFit::fitAdsr (const std::vector<float>& rawCurve, const std::vector<float>& times,
                           double gateTime, float floor, bool oneShot, bool amplitudeDomain)
{
    if (rawCurve.size() < 4 || times.size() < 4)
        return { 0.01f, 0.1f, 0.5f, 0.1f, 0.0f };

    // Full level is the loudest the note *holds*, not the loudest single frame.
    //
    // Normalising by the raw maximum is what puts a step in the envelope, and
    // the step is structural rather than a measurement error: an ADSR attacks
    // to one and then decays, so if one is a vibrato crest the note reaches
    // once, the fit has to spend a decay getting back down to the level it
    // actually sustains. On the violin that came out as a 0.61 s rise to full
    // followed by an 8 ms collapse to 0.60, and on its filter envelope a 2 s
    // sweep to 15 kHz followed by a 29 ms slam back to 4 kHz -- both heard,
    // correctly, as unlike anything a bowed note does.
    //
    // Smoothing over about a vibrato period first and normalising by *that*
    // maximum makes one mean the level the note reaches and keeps. Crests then
    // sit a little above one, which no measurement below minds, and a note that
    // rises and holds fits an attack and a sustain near one with nothing in
    // between. An earlier fix took the same view of the decay alone; the step
    // survived it because the peak it decays *from* was still a crest.
    //
    // The absolute scale this gives up is not information: oscillator levels
    // are solved against the target afterwards, so the envelope only has to
    // carry the shape.
    const auto dtForPeak = meanDiff (times);
    const auto contourSpan = std::max (1, static_cast<int> (std::lround (kContourSmoothSeconds / dtForPeak)));

    auto smoothedRaw = rawCurve;
    if (contourSpan > 1 && static_cast<int> (smoothedRaw.size()) > contourSpan)
        smoothedRaw = nd::uniformFilter1d (smoothedRaw, contourSpan);

    const auto peak = *std::max_element (smoothedRaw.begin(), smoothedRaw.end());
    if (peak <= 1.0e-9f)
        return { 0.01f, 0.1f, 0.5f, 0.1f, 0.0f };

    std::vector<float> curve (rawCurve.size());
    for (size_t i = 0; i < rawCurve.size(); ++i)
        curve[i] = rawCurve[i] / peak;

    // What counts as "full level", judged on a smoothed copy.
    //
    // The attack ends when the note first gets loud, and that was being taken
    // as the first frame reaching 95% of the raw maximum. On a modulated note
    // the raw maximum is a vibrato crest that can land a second or more into
    // the sustain, so the attack swallowed the whole first half of the note: a
    // violin fitted with a 1.16 s amplitude attack, and a clarinet with a
    // 1.55 s *filter* attack at full envelope amount -- which opened the filter
    // slowly across the note and made it swell by 12 dB before settling.
    //
    // Smoothing over roughly one vibrato period gives a level the note actually
    // sustains at rather than one crest of it. The *crossing* is still found on
    // the unsmoothed curve, so a genuinely fast attack keeps its timing.
    auto contour = curve;
    if (contourSpan > 1 && static_cast<int> (contour.size()) > contourSpan)
        contour = nd::uniformFilter1d (contour, contourSpan);
    const auto fullLevel = *std::max_element (contour.begin(), contour.end());

    const auto firstAbove = [&curve] (float level)
    {
        for (size_t i = 0; i < curve.size(); ++i)
            if (curve[i] > level)
                return static_cast<int> (i);
        return 0;
    };

    if (oneShot)
    {
        // 0.9 of the level the note actually holds, not of a single crest.
        const auto peakI = firstAbove (0.9f * fullLevel);
        const auto onsetI = firstAbove (floor);
        const auto attack = std::max (times[static_cast<size_t> (peakI)]
                                          - times[static_cast<size_t> (onsetI)], 1.0e-3f);

        // Measure the decay down to -60 dB, not to the -26 dB onset floor. For
        // a one-shot the decay *is* the whole sound, and ending it at 5% of
        // peak cuts off a tail that is still plainly audible.
        auto decay = std::max (times.back() - times[static_cast<size_t> (peakI)], 5.0e-3f);
        for (size_t i = static_cast<size_t> (peakI); i < curve.size(); ++i)
        {
            if (curve[i] <= 0.001f)
            {
                decay = std::max (times[i] - times[static_cast<size_t> (peakI)], 5.0e-3f);
                break;
            }
        }

        Adsr env;
        env.attack = juce::jlimit (0.001f, 2.0f, attack);
        env.decay = juce::jlimit (0.005f, 4.0f, decay);
        env.sustain = 0.0f;
        env.release = juce::jlimit (0.005f, 4.0f, decay);
        env.curve = fitCurve (curve, times, env, times.back());
        env.attackCurve = fitAttackCurve (curve, times, env, times.back(), amplitudeDomain);
        env.attack = juce::jlimit (0.001f, 2.0f,
                                   attackMeasuring (env, times.back(), env.attack));
        return env;
    }

    auto gateI = 0;
    while (gateI < static_cast<int> (times.size()) && times[static_cast<size_t> (gateI)] < gateTime)
        ++gateI;
    gateI = juce::jlimit (2, static_cast<int> (curve.size()) - 1, gateI);

    const auto onsetI = firstAbove (floor);
    std::vector<float> held (curve.begin(), curve.begin() + gateI);

    // Full level is reached when *either* the raw curve or the smoothed contour
    // says so, whichever comes first.
    //
    // The raw crossing on its own is what a fast attack needs, and on its own
    // it is also how the attack ran late and the decay came out instant. The
    // level being crossed is the contour's maximum, so on a noisy shape the raw
    // curve can fail to exceed it until some late spike -- by which time the
    // contour has already fallen past the point the decay is measured to, the
    // decay loop breaks on its first step, and the envelope becomes a long
    // climb and a cliff. On the violin's filter that read as a two second
    // build and a drop with nothing between them, which is not in the
    // recording.
    //
    // Taking the earlier of the two keeps a genuinely fast attack -- its raw
    // crossing is early by definition -- and stops a slow one being defined by
    // a spike.
    auto contourPeakI = gateI - 1;
    for (int i = 0; i < gateI; ++i)
        if (contour[static_cast<size_t> (i)] >= 0.9f * fullLevel)
        {
            contourPeakI = i;
            break;
        }

    auto peakI = std::min (firstAbove (0.9f * fullLevel), contourPeakI);
    if (peakI >= gateI)
        peakI = peakIndex (held); // never reached it before note-off; fall back
    const auto attack = std::max (times[static_cast<size_t> (peakI)]
                                      - times[static_cast<size_t> (onsetI)], 1.0e-3f);

    // Sustain: the level actually held just before note-off.
    const auto tailLo = std::max (peakI + 1, static_cast<int> (gateI * 0.8));
    float sustain;
    if (tailLo < gateI)
    {
        std::vector<float> tail (held.begin() + tailLo, held.begin() + gateI);
        std::sort (tail.begin(), tail.end());
        const auto mid = tail.size() / 2;
        sustain = (tail.size() % 2 == 1) ? tail[mid] : 0.5f * (tail[mid - 1] + tail[mid]);
    }
    else
    {
        sustain = held.back();
    }
    sustain = juce::jlimit (0.0f, 1.0f, sustain);

    // Decay: peak -> within 10% of the sustain level, measured on the smoothed
    // contour rather than the raw one.
    //
    // On the raw curve the first vibrato trough after the attack peak crosses
    // the target almost immediately, and the decay comes back as one or two
    // frames. That is not a fast decay, it is a crossing on a wobbling curve --
    // and the envelope it produces has a hard corner in it: a violin was fitted
    // with a 0.60 s attack to full level followed by a 7 ms collapse to 46%,
    // which is a 6.8 dB step no bowed note ever makes and which listeners
    // described as the sound "vacuuming back" half a second in.
    //
    // The attack keeps its raw-curve crossing, because a genuinely fast attack
    // must not be smoothed away. Only the decay is judged on the contour, which
    // is the same copy `fullLevel` is taken from.
    //
    const auto target = sustain + 0.1f * (1.0f - sustain);
    auto decay = std::max (times[static_cast<size_t> (gateI - 1)]
                               - times[static_cast<size_t> (peakI)], 5.0e-3f);
    for (int i = peakI; i < gateI; ++i)
    {
        if (contour[static_cast<size_t> (i)] <= target)
        {
            decay = std::max (times[static_cast<size_t> (i)]
                                  - times[static_cast<size_t> (peakI)], 5.0e-3f);
            break;
        }
    }

    // A trajectory that is itself an estimate does not get to assert a step.
    //
    // A loudness contour is measured; a cutoff trajectory is inferred, frame by
    // frame, from a deconvolution that has no unique answer -- and it is noisy
    // enough that a one-frame cliff in it says more about the estimator than
    // about the instrument. Believing one gave the violin a filter that climbed
    // for two seconds and then dropped to 47% with nothing in between, which is
    // audible and is not in the recording.
    //
    // So in that domain a fall is given a floor proportional to its size: a
    // full-range drop takes at least half a second, a half-range drop half of
    // that. Nothing here stops a *measured* envelope decaying as fast as it
    // likes, because amplitude contours keep their own answer.
    if (! amplitudeDomain)
    {
        const auto drop = juce::jlimit (0.0f, 1.0f, 1.0f - sustain);
        decay = std::max (decay, kMinFallSecondsPerUnit * drop);
    }

    // Release: note-off -> effectively silent.
    const auto levelAtGate = curve[static_cast<size_t> (gateI - 1)];
    auto release = 5.0e-3f;
    if (gateI < static_cast<int> (curve.size()) && levelAtGate > 1.0e-6f)
    {
        release = std::max (times.back() - times[static_cast<size_t> (gateI)], 5.0e-3f);
        for (size_t i = static_cast<size_t> (gateI); i < curve.size(); ++i)
        {
            if (curve[i] <= floor * levelAtGate)
            {
                release = std::max (times[i] - times[static_cast<size_t> (gateI)], 5.0e-3f);
                break;
            }
        }
    }

    Adsr env;
    env.attack = juce::jlimit (0.001f, 2.0f, attack);
    env.decay = juce::jlimit (0.005f, 4.0f, decay);
    env.sustain = sustain;
    env.release = juce::jlimit (0.005f, 4.0f, release);
    env.curve = fitCurve (curve, times, env, gateTime);
    env.attackCurve = fitAttackCurve (curve, times, env, gateTime, amplitudeDomain);
    env.attack = juce::jlimit (0.001f, 2.0f, attackMeasuring (env, gateTime, env.attack));
    return env;
}

} // namespace autosynth
