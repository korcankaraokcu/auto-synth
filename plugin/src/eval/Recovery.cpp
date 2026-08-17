#include "eval/Recovery.h"

#include "analysis/Stft.h"
#include "dsp/Voice.h"
#include "fit/PartialFit.h"
#include "fit/Refine.h"

#include <algorithm>
#include <cmath>

namespace autosynth
{
namespace
{

constexpr int kHop = 256;

// Every render in the harness goes through here, target and candidate and
// control alike, so that swapping the synth swaps all of them together. Scoring
// a Vital-rendered target against an engine-rendered control would measure the
// difference between the two synths and call it a fitting error.
std::vector<float> renderPatch (const Patch& patch, double sampleRate,
                                double duration, double gate,
                                const Recovery::Renderer& renderer = {})
{
    std::vector<float> out;

    if (renderer)
    {
        out = renderer (patch, duration, gate);
    }
    else
    {
        Engine engine;
        engine.prepare (sampleRate, 512);
        engine.setPatch (patch);

        juce::AudioBuffer<float> buffer;
        engine.renderOffline (buffer, patch.rootHz, duration, gate);

        const auto* data = buffer.getReadPointer (0);
        out.assign (data, data + buffer.getNumSamples());
    }

    float peak = 0.0f;
    for (auto v : out)
        peak = juce::jmax (peak, std::abs (v));
    if (peak > 1.0f)
        for (auto& v : out)
            v /= peak;
    return out;
}

float peakOf (const std::vector<float>& x)
{
    float peak = 0.0f;
    for (auto v : x)
        peak = juce::jmax (peak, std::abs (v));
    return peak;
}

// Log-magnitude L1 distance over several window sizes at once.
//
// One window cannot see both a click and a slow beat: a short window resolves
// the transient and smears the partials, a long one does the reverse. Summing
// scales is what stops the objective preferring whichever error the chosen
// window happens to be blind to.
double spectralDistance (const std::vector<float>& a, const std::vector<float>& b,
                         double sampleRate)
{
    double total = 0.0;
    int used = 0;
    for (int fft : { 512, 1024, 2048 })
    {
        const auto sa = Stft::magnitudeSpectrogram (a.data(), (int) a.size(), fft, fft / 4, sampleRate);
        const auto sb = Stft::magnitudeSpectrogram (b.data(), (int) b.size(), fft, fft / 4, sampleRate);
        const auto frames = juce::jmin (sa.numFrames, sb.numFrames);
        if (frames <= 0 || sa.numBins != sb.numBins)
            continue;

        double acc = 0.0;
        for (int t = 0; t < frames; ++t)
        {
            const auto* fa = sa.frame (t);
            const auto* fb = sb.frame (t);
            for (int k = 0; k < sa.numBins; ++k)
                acc += std::abs (std::log (fa[k] + 1.0e-5) - std::log (fb[k] + 1.0e-5));
        }
        total += acc / (frames * sa.numBins);
        ++used;
    }
    return used > 0 ? total / used : 0.0;
}

// Mean absolute difference of loudness envelopes in dB, floored 80 dB below
// each signal's own peak.
//
// The floor is not cosmetic. Without it, any comparison against a decayed tail
// is a comparison against digital silence and runs to ~140 dB, which swamps
// every real difference sharing the same average.
double loudnessDistanceDb (const std::vector<float>& a, const std::vector<float>& b)
{
    const auto la = Stft::loudnessEnvelope (a.data(), (int) a.size(), kHop);
    const auto lb = Stft::loudnessEnvelope (b.data(), (int) b.size(), kHop);
    const auto n = juce::jmin (la.size(), lb.size());
    if (n == 0)
        return 0.0;

    const auto floorOf = [] (const std::vector<float>& v)
    {
        float peak = 0.0f;
        for (auto s : v)
            peak = juce::jmax (peak, s);
        return juce::jmax (1.0e-10f, peak * 1.0e-4f);
    };
    const auto fa = floorOf (la), fb = floorOf (lb);

    double acc = 0.0;
    for (size_t i = 0; i < n; ++i)
        acc += std::abs (20.0 * std::log10 (juce::jmax (la[i], fa))
                       - 20.0 * std::log10 (juce::jmax (lb[i], fb)));
    return acc / (double) n;
}

double centroidDistanceOctaves (const std::vector<float>& a, const std::vector<float>& b,
                                double sampleRate)
{
    const auto sa = Stft::magnitudeSpectrogram (a.data(), (int) a.size(), 2048, kHop, sampleRate);
    const auto sb = Stft::magnitudeSpectrogram (b.data(), (int) b.size(), 2048, kHop, sampleRate);
    const auto ca = Stft::spectralCentroid (sa);
    const auto cb = Stft::spectralCentroid (sb);
    const auto n = juce::jmin (ca.size(), cb.size());

    double acc = 0.0;
    int counted = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (ca[i] < 20.0f || cb[i] < 20.0f)
            continue; // silence has no meaningful brightness
        acc += std::abs (std::log2 (ca[i] / cb[i]));
        ++counted;
    }
    return counted > 0 ? acc / counted : 0.0;
}

template <typename Enum>
Enum pick (juce::Random& rng, int count)
{
    return static_cast<Enum> (rng.nextInt (count));
}

// Does this parameter affect what the target actually sounds like?
//
// Scoring one that does not is worse than useless. A detune setting on an
// oscillator with a single unison voice changes nothing audible, so the fitter
// has no way to recover it and no reason to -- and averaging that in produces
// an error near 0.5 (the expected distance between two independent uniforms)
// which is indistinguishable from a real failure.
//
// The first version of this harness scored every continuous parameter for every
// trial, and the worst-recovered list came back with almost everything pinned
// near 0.5. That looked like a fitter that recovered nothing. Most of it was
// parameters that were switched off in the target.
//
// The rule is identifiability: score a parameter only where the target's own
// settings give it an audible effect.
bool affectsAudio (const Patch& patch, const std::string& path)
{
    const auto endsWith = [&path] (const char* suffix)
    {
        const std::string s (suffix);
        return path.size() >= s.size()
            && path.compare (path.size() - s.size(), s.size(), s) == 0;
    };
    const auto startsWith = [&path] (const char* prefix)
    {
        const std::string s (prefix);
        return path.rfind (s, 0) == 0;
    };

    // Noise is forced to zero when generating targets -- a realisation cannot be
    // recovered, only a level, and letting it into a spectral distance would
    // measure the noise rather than the fit. So it is never identifiable here.
    if (path == "noise_level")
        return false;

    if (startsWith ("oscs."))
    {
        const auto index = path[5] - '0';
        if (index < 0 || index >= kNumOsc)
            return false;
        const auto& osc = patch.oscs[(size_t) index];

        if (! osc.enabled || osc.level <= 1.0e-4f)
            return false;

        // Pulse width shapes only the pulse table.
        if (endsWith (".pulse_width"))
            return osc.waveform == Waveform::pulse
                || (osc.waveMorph > 0.01f && osc.waveformB == Waveform::pulse);

        // Detune needs something to detune against.
        if (endsWith (".unison_detune"))
            return osc.unisonVoices > 1;

        // Morphing towards the same waveform is a no-op.
        if (endsWith (".wave_morph"))
            return osc.waveformB != osc.waveform;

        if (path.find (".env.") != std::string::npos && path.find (".filter.") == std::string::npos)
            return osc.envEnabled;

        if (path.find (".filter.") != std::string::npos)
        {
            if (! osc.filterEnabled || osc.filter.type == FilterType::off)
                return false;
            // The filter envelope only matters if it is routed anywhere.
            if (path.find (".filter.env.") != std::string::npos)
                return std::abs (osc.filter.envAmount) > 0.01f;
            return true;
        }

        if (endsWith (".reverb_send"))
            return patch.reverb.enabled && patch.reverb.level > 1.0e-3f;

        return true; // cents, level
    }

    if (startsWith ("filter."))
    {
        if (patch.filter.type == FilterType::off)
            return false;
        if (startsWith ("filter.env."))
            return std::abs (patch.filter.envAmount) > 0.01f;
        return true;
    }

    if (startsWith ("lfos."))
    {
        const auto index = path[5] - '0';
        if (index < 0 || index >= kNumLfo)
            return false;
        const auto& lfo = patch.lfos[(size_t) index];
        return lfo.dest != LfoDest::none && lfo.depth > 1.0e-3f;
    }

    if (startsWith ("delay."))
        return patch.delay.enabled && patch.delay.mix > 1.0e-3f;

    if (startsWith ("reverb."))
    {
        if (! patch.reverb.enabled || patch.reverb.level <= 1.0e-3f)
            return false;
        // A reverb nothing is sent to is inaudible however it is set.
        for (const auto& osc : patch.oscs)
            if (osc.enabled && osc.level > 1.0e-4f && osc.reverbSend > 1.0e-3f)
                return true;
        return false;
    }

    return true; // amp_env.*, master_level
}

double oscFrequency (const Patch& patch, int index)
{
    const auto& osc = patch.oscs[(size_t) index];
    return patch.rootHz * std::pow (2.0, (osc.semitones + osc.cents / 100.0) / 12.0);
}

// Pair up the truth's oscillators with the fitted patch's, by pitch.
//
// Index-to-index comparison is wrong and was quietly corrupting every
// per-oscillator number. The fitter emits sources in salience order; a random
// target's oscillators are in no order at all. Slot 0 on one side has no reason
// to be the same voice as slot 0 on the other, so most comparisons were between
// unrelated oscillators and the resulting errors were closer to noise than to
// measurement.
//
// Matching on sounding frequency is the natural key: it is what makes two
// oscillators "the same one" to a listener, and it is recovered well enough
// (79% within a semitone) to match on.
std::array<int, kNumOsc> matchOscillators (const Patch& truth, const Patch& fitted)
{
    std::array<int, kNumOsc> match {};
    match.fill (-1);

    std::array<bool, kNumOsc> taken {};
    taken.fill (false);

    // Loudest first, so the most significant voice gets the best partner when
    // two truth oscillators compete for one fitted one.
    std::vector<int> order;
    for (int i = 0; i < kNumOsc; ++i)
        if (truth.oscs[(size_t) i].enabled && truth.oscs[(size_t) i].level > 1.0e-4f)
            order.push_back (i);
    std::sort (order.begin(), order.end(), [&truth] (int a, int b)
    {
        return truth.oscs[(size_t) a].level > truth.oscs[(size_t) b].level;
    });

    for (auto t : order)
    {
        const auto want = oscFrequency (truth, t);
        auto bestCents = 0.0;
        auto best = -1;
        for (int f = 0; f < kNumOsc; ++f)
        {
            if (taken[(size_t) f])
                continue;
            const auto& osc = fitted.oscs[(size_t) f];
            if (! osc.enabled || osc.level <= 1.0e-4f)
                continue;

            const auto cents = std::abs (1200.0 * std::log2 (oscFrequency (fitted, f) / want));
            if (best < 0 || cents < bestCents)
            {
                bestCents = cents;
                best = f;
            }
        }

        // Half a semitone. Beyond that they are not the same voice, and pairing
        // them would invent a comparison rather than find one.
        if (best >= 0 && bestCents <= 50.0)
        {
            match[(size_t) t] = best;
            taken[(size_t) best] = true;
        }
    }
    return match;
}

// The lowest pitch the patch actually sounds at.
//
// Not `rootHz`, which is playback metadata: a patch transposes its oscillators
// away from it by up to two octaves, so the pitch a listener -- or a pitch
// tracker -- hears is the lowest transposed oscillator, not the nominal root.
// Scoring `rootHz` directly asks the fitter to return a number it was never
// meant to produce, and reads as a catastrophic failure when nothing is wrong.
double soundingFundamental (const Patch& patch)
{
    double lowest = 0.0;
    for (const auto& osc : patch.oscs)
    {
        if (! osc.enabled || osc.level <= 1.0e-4f)
            continue;
        const auto hz = patch.rootHz
                      * std::pow (2.0, (osc.semitones + osc.cents / 100.0) / 12.0);
        if (lowest <= 0.0 || hz < lowest)
            lowest = hz;
    }
    return lowest > 0.0 ? lowest : patch.rootHz;
}

} // namespace

Patch Recovery::randomPatch (juce::Random& rng)
{
    Patch patch;
    patch.name = "random";

    // Continuous parameters come from the fitter's own ranges, so the harness
    // never poses a target outside the space the fitter searches.
    for (const auto& spec : Refine::continuousSpecs())
    {
        const auto x = rng.nextDouble();
        const auto value = spec.logarithmic
            ? std::exp (std::log (spec.lo) + x * (std::log (spec.hi) - std::log (spec.lo)))
            : spec.lo + x * (spec.hi - spec.lo);
        Refine::setParameterValue (patch, spec.path, value);
    }

    // Discrete parameters, which the fitter decides structurally rather than
    // searching, and which are therefore the interesting thing to score.
    for (int i = 0; i < kNumOsc; ++i)
    {
        auto& osc = patch.oscs[(size_t) i];
        osc.enabled = rng.nextBool();
        osc.waveform = pick<Waveform> (rng, 5);
        osc.waveformB = pick<Waveform> (rng, 5);
        osc.semitones = rng.nextInt ({ -24, 25 });
        osc.unisonVoices = rng.nextInt ({ 1, kMaxUnison + 1 });
        osc.envEnabled = rng.nextBool();
        osc.filterEnabled = rng.nextBool();
        osc.filter.type = pick<FilterType> (rng, 4);
    }

    // At least one oscillator must sound, or the target is silence and the
    // trial measures nothing.
    if (patch.activeOscCount() == 0)
    {
        auto& osc = patch.oscs[0];
        osc.enabled = true;
        osc.level = juce::jmax (0.3f, osc.level);
    }

    patch.filter.type = pick<FilterType> (rng, 4);
    // Only the four *routable* destinations. `lfoRate` and `lfoDepth` point at
    // another LFO rather than at the sound, so a target that used one would be
    // posing a question about modulator chaining rather than about recovery.
    for (auto& lfo : patch.lfos)
    {
        lfo.shape = pick<LfoShape> (rng, 4);
        lfo.dest = pick<LfoDest> (rng, 4);
    }
    patch.delay.enabled = rng.nextBool();
    patch.reverb.enabled = rng.nextBool();

    // Noise is excluded from the target. The fitter cannot recover a noise
    // realisation, only its level, and letting it dominate a spectral distance
    // would measure the noise rather than the fit.
    patch.noiseLevel = 0.0f;

    // Master level is kept audible; a target at -40 dB scores nothing useful.
    patch.masterLevel = 0.4f + 0.6f * rng.nextFloat();

    // A musical root, log-uniform, so root recovery is a real test rather than
    // a constant the fitter could hard-code.
    patch.rootHz = (float) std::exp (std::log (110.0) + rng.nextDouble() * std::log (4.0));
    return patch;
}

Recovery::Scores Recovery::score (const std::vector<float>& candidate,
                                  const std::vector<float>& target,
                                  double sampleRate)
{
    Scores s;
    s.spectral = spectralDistance (candidate, target, sampleRate);
    s.loudnessDb = loudnessDistanceDb (candidate, target);
    s.centroidOctaves = centroidDistanceOctaves (candidate, target, sampleRate);
    return s;
}

Recovery::Summary Recovery::run (const Options& options)
{
    Summary summary;
    juce::Random rng ((juce::int64) options.seed + 1);

    std::map<std::string, double> errorTotals;
    std::map<std::string, int> errorCounts;
    const auto specs = Refine::continuousSpecs();

    int attempts = 0;
    const auto maxAttempts = options.trials * 20;

    while (summary.trials < options.trials && attempts < maxAttempts)
    {
        ++attempts;

        Trial trial;
        trial.truth = randomPatch (rng);
        const auto target = renderPatch (trial.truth, options.sampleRate,
                                         options.duration, options.gate, options.renderer);
        if (peakOf (target) < options.minPeak)
            continue; // inaudible target; resample rather than score noise

        PartialFit::Options fitOptions;
        fitOptions.hop = kHop;
        trial.fitted = PartialFit::fit (target.data(), (int) target.size(),
                                        options.sampleRate, fitOptions);

        if (options.refine)
        {
            Refine::Options refineOptions;
            refineOptions.maxEvaluations = options.refineEvaluations;
            refineOptions.gateSeconds = options.gate;
            refineOptions.renderer = options.renderer;
            trial.fitted = Refine::run (trial.fitted, target.data(), (int) target.size(),
                                        options.sampleRate, refineOptions).patch;
        }

        // The fitted patch is rendered at the pitch it decided on, exactly as
        // the plugin would play it.
        const auto fittedAudio = renderPatch (trial.fitted, options.sampleRate,
                                              options.duration, options.gate,
                                              options.renderer);
        trial.fit = score (fittedAudio, target, options.sampleRate);

        // The control: a default patch, played at the true pitch. It is the
        // reference that makes the fitted numbers mean something.
        Patch control;
        control.rootHz = trial.truth.rootHz;
        const auto controlAudio = renderPatch (control, options.sampleRate,
                                               options.duration, options.gate,
                                               options.renderer);
        trial.control = score (controlAudio, target, options.sampleRate);

        trial.truthReverb = trial.truth.reverb.enabled && trial.truth.reverb.level > 1.0e-3f;
        trial.fittedReverb = trial.fitted.reverb.enabled && trial.fitted.reverb.level > 1.0e-3f;
        trial.truthDelay = trial.truth.delay.enabled && trial.truth.delay.mix > 1.0e-3f;
        trial.fittedDelay = trial.fitted.delay.enabled && trial.fitted.delay.mix > 1.0e-3f;

        const auto tally = [] (Decision& d, bool truth, bool fitted)
        {
            if (truth && fitted)        ++d.truePositive;
            else if (! truth && fitted) ++d.falsePositive;
            else if (truth)             ++d.falseNegative;
            else                        ++d.trueNegative;
        };
        tally (summary.reverb, trial.truthReverb, trial.fittedReverb);
        tally (summary.delay, trial.truthDelay, trial.fittedDelay);

        trial.truthOscCount = trial.truth.activeOscCount();
        trial.fittedOscCount = trial.fitted.activeOscCount();
        trial.oscCountExact = trial.truthOscCount == trial.fittedOscCount;

        const auto truthHz = soundingFundamental (trial.truth);
        const auto fittedHz = soundingFundamental (trial.fitted);
        trial.rootWithinSemitone =
            truthHz > 0.0 && fittedHz > 0.0
            && std::abs (std::log2 (fittedHz / truthHz)) < (1.0 / 12.0);

        // Per-parameter error, over parameters that were audible in the target
        // and belong to an oscillator the fitter also proposed. Comparing the
        // cutoff of an oscillator that does not exist on one side is not a
        // measurement of anything.
        const auto match = matchOscillators (trial.truth, trial.fitted);

        for (const auto& spec : specs)
        {
            if (! affectsAudio (trial.truth, spec.path))
                continue;

            // Per-oscillator parameters are read from the *matched* slot on the
            // fitted side, not the same index.
            auto fittedPath = spec.path;
            if (spec.path.rfind ("oscs.", 0) == 0)
            {
                const auto index = spec.path[5] - '0';
                if (index < 0 || index >= kNumOsc)
                    continue;
                const auto partner = match[(size_t) index];
                if (partner < 0)
                    continue; // the fitter never proposed this voice
                fittedPath[5] = static_cast<char> ('0' + partner);
            }

            const auto truthValue = Refine::parameterValue (trial.truth, spec.path);
            const auto fitValue = Refine::parameterValue (trial.fitted, fittedPath);

            double error = 0.0;
            if (spec.logarithmic && truthValue > 0.0 && fitValue > 0.0)
            {
                const auto span = std::log (spec.hi) - std::log (spec.lo);
                error = span > 0.0
                    ? std::abs (std::log (truthValue) - std::log (fitValue)) / span : 0.0;
            }
            else
            {
                const auto span = spec.hi - spec.lo;
                error = span > 0.0 ? std::abs (truthValue - fitValue) / span : 0.0;
            }

            errorTotals[spec.path] += error;
            errorCounts[spec.path] += 1;
        }

        summary.fit.spectral += trial.fit.spectral;
        summary.fit.loudnessDb += trial.fit.loudnessDb;
        summary.fit.centroidOctaves += trial.fit.centroidOctaves;
        summary.control.spectral += trial.control.spectral;
        summary.control.loudnessDb += trial.control.loudnessDb;
        summary.control.centroidOctaves += trial.control.centroidOctaves;
        summary.oscCountAccuracy += trial.oscCountExact ? 1.0 : 0.0;
        summary.rootAccuracy += trial.rootWithinSemitone ? 1.0 : 0.0;

        ++summary.trials;
        summary.byTrial.push_back (std::move (trial));
    }

    if (summary.trials > 0)
    {
        const auto n = (double) summary.trials;
        summary.fit.spectral /= n;
        summary.fit.loudnessDb /= n;
        summary.fit.centroidOctaves /= n;
        summary.control.spectral /= n;
        summary.control.loudnessDb /= n;
        summary.control.centroidOctaves /= n;
        summary.oscCountAccuracy /= n;
        summary.rootAccuracy /= n;
    }

    for (const auto& entry : errorTotals)
    {
        const auto count = errorCounts[entry.first];
        if (count > 0)
            summary.parameterError[entry.first] = entry.second / count;
    }

    return summary;
}

juce::String Recovery::toJson (const Summary& summary)
{
    auto* root = new juce::DynamicObject();
    root->setProperty ("trials", summary.trials);
    root->setProperty ("osc_count_accuracy", summary.oscCountAccuracy);
    root->setProperty ("root_accuracy", summary.rootAccuracy);

    const auto decisionVar = [] (const Decision& d)
    {
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("true_positive", d.truePositive);
        obj->setProperty ("false_positive", d.falsePositive);
        obj->setProperty ("true_negative", d.trueNegative);
        obj->setProperty ("false_negative", d.falseNegative);
        obj->setProperty ("accuracy", d.accuracy());
        return juce::var (obj);
    };
    root->setProperty ("reverb_decision", decisionVar (summary.reverb));
    root->setProperty ("delay_decision", decisionVar (summary.delay));

    const auto scores = [] (const Scores& s)
    {
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("spectral", s.spectral);
        obj->setProperty ("loudness_db", s.loudnessDb);
        obj->setProperty ("centroid_oct", s.centroidOctaves);
        return juce::var (obj);
    };
    root->setProperty ("fit", scores (summary.fit));
    root->setProperty ("control", scores (summary.control));

    auto* errors = new juce::DynamicObject();
    for (const auto& entry : summary.parameterError)
        errors->setProperty (juce::String (entry.first), entry.second);
    root->setProperty ("parameter_error", juce::var (errors));

    return juce::JSON::toString (juce::var (root), false);
}

juce::String Recovery::toText (const Summary& summary)
{
    juce::String out;
    out << "trials: " << summary.trials << "\n\n";
    out << juce::String ("metric").paddedRight (' ', 16)
        << juce::String ("fit").paddedLeft (' ', 10)
        << juce::String ("control").paddedLeft (' ', 10) << "\n";

    const auto row = [&out] (const char* name, double fit, double control)
    {
        out << juce::String (name).paddedRight (' ', 16)
            << juce::String (fit, 3).paddedLeft (' ', 10)
            << juce::String (control, 3).paddedLeft (' ', 10) << "\n";
    };
    row ("spectral", summary.fit.spectral, summary.control.spectral);
    row ("loudness_db", summary.fit.loudnessDb, summary.control.loudnessDb);
    row ("centroid_oct", summary.fit.centroidOctaves, summary.control.centroidOctaves);

    out << "\noscillator count exact: "
        << juce::String (100.0 * summary.oscCountAccuracy, 1) << "%\n";
    // Where the misses go, not just how many there are. "50% exact" says
    // nothing about whether the fitter invents oscillators or misses them, and
    // those are opposite bugs with opposite fixes.
    {
        std::map<std::pair<int, int>, int> confusion;
        int worst = 1;
        for (const auto& t : summary.byTrial)
        {
            confusion[{ t.truthOscCount, t.fittedOscCount }] += 1;
            worst = juce::jmax (worst, t.truthOscCount, t.fittedOscCount);
        }

        out << "  truth \\ fitted";
        for (int f = 1; f <= worst; ++f)
            out << juce::String (f).paddedLeft (' ', 5);
        out << "\n";
        for (int t = 1; t <= worst; ++t)
        {
            out << juce::String (t).paddedLeft (' ', 15);
            for (int f = 1; f <= worst; ++f)
            {
                const auto it = confusion.find ({ t, f });
                out << juce::String (it == confusion.end() ? 0 : it->second).paddedLeft (' ', 5);
            }
            out << "\n";
        }
    }

    // Every miscount, described the way the analysis would have to see it: the
    // interval between the sources and how loud the quieter one is. Grouping
    // proposes the count, and what it can propose depends almost entirely on
    // those two numbers -- an octave-apart layer shares every harmonic index
    // with the one below it, and a quiet layer's partials are the first to go
    // missing.
    {
        bool any = false;
        for (const auto& t : summary.byTrial)
        {
            if (t.oscCountExact)
                continue;
            if (! any)
            {
                out << "\n  miscounts (truth -> fitted, sources as semitones from the root):\n";
                any = true;
            }

            std::vector<std::pair<int, float>> sources;
            float loudest = 0.0f;
            for (const auto& o : t.truth.oscs)
                if (o.enabled && o.level > 1.0e-4f)
                {
                    sources.push_back ({ o.semitones, o.level });
                    loudest = juce::jmax (loudest, o.level);
                }
            std::sort (sources.begin(), sources.end());

            out << "    " << t.truthOscCount << " -> " << t.fittedOscCount << "   ";
            for (const auto& [semis, level] : sources)
                out << juce::String (semis) << " (" << juce::String (level / juce::jmax (1.0e-6f, loudest), 2)
                    << ")  ";
            out << "\n";
        }
    }

    // Noise the fitter invented. Targets are generated noise-free, so every one
    // of these is an error -- and one no distance metric objects to, because
    // broadband energy lowers a log-spectral error wherever the harmonic fit is
    // imperfect. It has to be reported separately or it is invisible.
    {
        double total = 0.0;
        int audible = 0;
        for (const auto& t : summary.byTrial)
        {
            total += t.fitted.noiseLevel;
            if (t.fitted.noiseLevel > 0.01f)
                ++audible;
        }
        const auto n = juce::jmax (1, static_cast<int> (summary.byTrial.size()));
        out << "invented noise: mean " << juce::String (total / n, 3)
            << ", audible in " << audible << " of " << n
            << " noise-free targets\n";
    }

    out << "root within a semitone: "
        << juce::String (100.0 * summary.rootAccuracy, 1) << "%\n";

    const auto decision = [&out] (const char* name, const Decision& d)
    {
        out << juce::String (name).paddedRight (' ', 10)
            << " accuracy " << juce::String (100.0 * d.accuracy(), 1) << "%"
            << "   (hit " << d.truePositive
            << ", false alarm " << d.falsePositive
            << ", missed " << d.falseNegative
            << ", correctly off " << d.trueNegative << ")\n";
    };
    out << "\nstructural decisions:\n";
    decision ("  reverb", summary.reverb);
    decision ("  delay", summary.delay);

    // Worst-recovered parameters first: this is the to-do list, and the harness
    // reports it honestly. Parameters the fitter does not model at all come out
    // at the top.
    std::vector<std::pair<std::string, double>> ranked (summary.parameterError.begin(),
                                                        summary.parameterError.end());
    std::sort (ranked.begin(), ranked.end(),
               [] (const auto& a, const auto& b) { return a.second > b.second; });

    out << "\nworst-recovered parameters (mean error, normalised by range):\n";
    for (size_t i = 0; i < ranked.size() && i < 12; ++i)
        out << "  " << juce::String (ranked[i].first).paddedRight (' ', 30)
            << juce::String (ranked[i].second, 3) << "\n";

    return out;
}

} // namespace autosynth
