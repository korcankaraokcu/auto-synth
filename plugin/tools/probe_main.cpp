// Analysis probe: WAV in, intermediate results out as JSON.
//
// This is the conformance harness for the analysis port. The engine port could
// be validated by rendering audio and comparing spectra; analysis has no audio
// output, so instead each stage's intermediates are dumped and compared against
// the Python implementation element-wise.
//
// Without this the port would be validated only at the end, by whether the
// final patch looked reasonable -- which is exactly the "debugging two ports at
// once" problem that porting the engine first was meant to avoid.
//
//   autosynth_probe input.wav [--hop 256] [--fft 2048]

#include "analysis/Grouping.h"
#include "fit/EnvelopeFit.h"
#include "fit/FilterFit.h"
#include "fit/Modulation.h"
#include "fit/PartialFit.h"
#include "fit/Refine.h"
#include "fit/WaveformFit.h"
#include "analysis/Partials.h"
#include "analysis/Roles.h"
#include "analysis/Stft.h"
#include "analysis/Yin.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <algorithm>
#include <map>
#include <vector>

namespace
{

struct Args
{
    std::map<juce::String, juce::String> options;
    juce::StringArray positional;

    double value (const char* flag, double fallback) const
    {
        const auto it = options.find (flag);
        if (it == options.end() || it->second.isEmpty())
            return fallback;
        return it->second.getDoubleValue();
    }
};

Args parseArgs (int argc, char* argv[])
{
    Args out;
    std::vector<juce::String> raw;
    for (int i = 1; i < argc; ++i)
        raw.emplace_back (juce::CharPointer_UTF8 (argv[i]));

    for (size_t i = 0; i < raw.size(); ++i)
    {
        if (! raw[i].startsWith ("--"))
        {
            out.positional.add (raw[i]);
            continue;
        }
        auto key = raw[i];
        juce::String value;
        if (key.containsChar ('='))
        {
            value = key.fromFirstOccurrenceOf ("=", false, false);
            key = key.upToFirstOccurrenceOf ("=", false, false);
        }
        else if (i + 1 < raw.size() && ! raw[i + 1].startsWith ("--"))
        {
            value = raw[++i];
        }
        out.options[key] = value;
    }
    return out;
}

juce::var toVar (const std::vector<float>& values)
{
    juce::Array<juce::var> array;
    array.ensureStorageAllocated (static_cast<int> (values.size()));
    for (auto v : values)
        array.add (std::isfinite (v) ? static_cast<double> (v) : 0.0);
    return array;
}

} // namespace

int main (int argc, char* argv[])
{
    const auto args = parseArgs (argc, argv);
    if (args.positional.isEmpty())
    {
        std::fprintf (stderr, "usage: autosynth_probe <input.wav> [--hop n] [--fft n] "
                              "[--refine 1] [--patch out.json]\n");
        return 2;
    }

    const juce::File input (juce::File::getCurrentWorkingDirectory()
                                .getChildFile (args.positional[0]));
    if (! input.existsAsFile())
    {
        std::fprintf (stderr, "error: no such file: %s\n", input.getFullPathName().toRawUTF8());
        return 1;
    }

    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (input));
    if (reader == nullptr)
    {
        std::fprintf (stderr, "error: cannot read %s\n", input.getFullPathName().toRawUTF8());
        return 1;
    }

    const auto numSamples = static_cast<int> (reader->lengthInSamples);
    juce::AudioBuffer<float> buffer (static_cast<int> (reader->numChannels), numSamples);
    reader->read (&buffer, 0, numSamples, 0, true, true);

    // Downmix to mono, matching audio.read_wav.
    if (buffer.getNumChannels() > 1)
    {
        for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
            buffer.addFrom (0, 0, buffer, ch, 0, numSamples);
        buffer.applyGain (0, 0, numSamples, 1.0f / buffer.getNumChannels());
    }

    const auto sampleRate = reader->sampleRate;
    const auto hop = static_cast<int> (args.value ("--hop", 256.0));
    const auto fftSize = static_cast<int> (args.value ("--fft", 2048.0));
    const auto* samples = buffer.getReadPointer (0);

    const auto spectrogram = autosynth::Stft::magnitudeSpectrogram (samples, numSamples,
                                                                    fftSize, hop, sampleRate);
    const auto centroid = autosynth::Stft::spectralCentroid (spectrogram);
    const auto loudness = autosynth::Stft::loudnessEnvelope (samples, numSamples, hop);
    const auto pitch = autosynth::Yin::track (samples, numSamples, sampleRate, hop);

    double f0 = 0.0, confidence = 0.0;
    autosynth::Yin::estimate (samples, numSamples, sampleRate, f0, confidence, hop);
    double cents = 0.0;
    const auto note = autosynth::Yin::noteName (f0, cents);

    // One frame of raw magnitude, so a scaling or windowing mismatch is visible
    // directly rather than only through its effect on later stages.
    std::vector<float> midFrame;
    if (spectrogram.numFrames > 0)
    {
        const auto* frame = spectrogram.frame (spectrogram.numFrames / 2);
        midFrame.assign (frame, frame + spectrogram.numBins);
    }

    // Partial tracking and grouping, using the same defaults as the Python side.
    autosynth::PartialTracker::Options trackOptions;
    trackOptions.fftSize = fftSize;
    trackOptions.hop = hop;
    const auto partialSet = autosynth::PartialTracker::track (samples, numSamples,
                                                              sampleRate, trackOptions);
    auto* rolesObj = new juce::DynamicObject();
    rolesObj->setProperty ("noise_share",
                           autosynth::Roles::noiseShare (samples, numSamples, sampleRate, f0,
                                                         fftSize, hop));
    const juce::var rolesVar (rolesObj);

    const auto groups = autosynth::Grouping::group (partialSet, 3);

    juce::Array<juce::var> partialArray;
    for (const auto& p : partialSet.partials)
    {
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("mean_freq", p.meanFreq());
        obj->setProperty ("energy", p.energy());
        obj->setProperty ("length", p.length());
        obj->setProperty ("start_frame", p.frames.empty() ? 0 : p.frames.front());
        partialArray.add (juce::var (obj));
    }

    juce::Array<juce::var> groupArray;
    for (const auto& g : groups)
    {
        int voices = 1;
        float detune = 0.0f;
        autosynth::Grouping::estimateUnison (g, voices, detune);

        auto* obj = new juce::DynamicObject();
        obj->setProperty ("f0", g.f0);
        obj->setProperty ("salience", g.salience);
        obj->setProperty ("energy", g.energy());
        obj->setProperty ("num_partials", static_cast<int> (g.partials.size()));
        obj->setProperty ("num_harmonics", g.numHarmonics);
        obj->setProperty ("unison_voices", voices);
        obj->setProperty ("unison_detune", detune);
        obj->setProperty ("detune_cents", autosynth::Grouping::detuneCents (g));
        groupArray.add (juce::var (obj));
    }

    // Envelope + waveform fitting, so their conformance is checked on the same
    // inputs the rest of the pipeline sees.
    std::vector<float> loudTimes (loudness.size());
    for (size_t i = 0; i < loudness.size(); ++i)
        loudTimes[i] = static_cast<float> (i * hop / sampleRate);

    const auto gate = autosynth::EnvelopeFit::detectGate (loudness, loudTimes);
    const auto ampEnv = autosynth::EnvelopeFit::fitAdsr (loudness, loudTimes,
                                                         gate.time, 0.05f, gate.oneShot);

    auto* gateObj = new juce::DynamicObject();
    gateObj->setProperty ("time", gate.time);
    gateObj->setProperty ("one_shot", gate.oneShot);

    auto* envObj = new juce::DynamicObject();
    envObj->setProperty ("attack", ampEnv.attack);
    envObj->setProperty ("decay", ampEnv.decay);
    envObj->setProperty ("sustain", ampEnv.sustain);
    envObj->setProperty ("release", ampEnv.release);
    envObj->setProperty ("curve", ampEnv.curve);

    juce::var waveformVar;
    if (! groups.empty() && f0 > 0.0)
    {
        // Energy-weighted mean harmonic profile of the dominant group.
        const auto& g = groups.front();
        std::vector<float> profile (static_cast<size_t> (g.numHarmonics), 0.0f);
        double totalWeight = 0.0;
        std::vector<double> frameWeight (static_cast<size_t> (g.numFrames), 0.0);
        for (int t = 0; t < g.numFrames; ++t)
        {
            for (int k = 0; k < g.numHarmonics; ++k)
                frameWeight[static_cast<size_t> (t)] += g.harmonic (k)[t];
            totalWeight += frameWeight[static_cast<size_t> (t)];
        }
        if (totalWeight > 1.0e-12)
        {
            for (int k = 0; k < g.numHarmonics; ++k)
            {
                double acc = 0.0;
                for (int t = 0; t < g.numFrames; ++t)
                    acc += g.harmonic (k)[t] * frameWeight[static_cast<size_t> (t)] / totalWeight;
                profile[static_cast<size_t> (k)] = static_cast<float> (acc);
            }
            const auto peakProfile = *std::max_element (profile.begin(), profile.end());
            if (peakProfile > 1.0e-12f)
                for (auto& v : profile)
                    v /= peakProfile;
        }

        const auto plain = autosynth::WaveformFit::match (profile);
        const auto joint = autosynth::WaveformFit::matchWithCutoff (profile, g.f0, sampleRate);

        auto* obj = new juce::DynamicObject();
        obj->setProperty ("waveform", static_cast<int> (plain.waveform));
        obj->setProperty ("pulse_width", plain.pulseWidth);
        obj->setProperty ("error", plain.error);
        obj->setProperty ("joint_waveform", static_cast<int> (joint.waveform));
        obj->setProperty ("joint_pulse_width", joint.pulseWidth);
        obj->setProperty ("joint_cutoff", joint.cutoffHz);
        obj->setProperty ("profile", toVar (profile));
        waveformVar = juce::var (obj);
    }

    // Filter trajectory on the dominant group, and LFO detection on the whole
    // signal -- both need the stages above to already agree.
    juce::var filterVar;
    if (! groups.empty())
    {
        const auto& g = groups.front();
        const auto trajectory = autosynth::FilterFit::estimateCutoffTrajectory (
            g.H, g.numHarmonics, g.numFrames, g.f0, sampleRate);
        const auto split = autosynth::FilterFit::trajectoryToEnv (trajectory.cutoffHz, 1000.0);

        auto* obj = new juce::DynamicObject();
        obj->setProperty ("cutoff_traj", toVar (trajectory.cutoffHz));
        obj->setProperty ("source", toVar (trajectory.source));
        obj->setProperty ("base_cutoff", split.baseCutoffHz);
        obj->setProperty ("env_amount", split.envAmountOctaves);
        filterVar = juce::var (obj);
    }

    // Full assembled patch. This is what the plugin will produce on a WAV drop.
    autosynth::PartialFit::Options fitOptions;
    fitOptions.hop = hop;
    const auto fitted = autosynth::PartialFit::fit (samples, numSamples, sampleRate, fitOptions);

    // Refinement, only when asked: it costs ~8s per second of audio.
    juce::var refineVar;
    auto finalPatch = fitted;
    if (args.value ("--refine", 0.0) > 0.0)
    {
        autosynth::Refine::Options refineOptions;
        refineOptions.maxEvaluations = static_cast<int> (args.value ("--refine-evals", 192.0));
        const auto refined = autosynth::Refine::run (fitted, samples, numSamples,
                                                     sampleRate, refineOptions);
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("initial_loss", refined.initialLoss);
        obj->setProperty ("final_loss", refined.finalLoss);
        obj->setProperty ("evaluations", refined.evaluations);
        obj->setProperty ("improved", refined.improved);
        obj->setProperty ("cutoff_hz", refined.patch.filter.cutoffHz);
        obj->setProperty ("master_level", refined.patch.masterLevel);
        obj->setProperty ("active_oscs", refined.patch.activeOscCount());
        refineVar = juce::var (obj);
        finalPatch = refined.patch;
    }

    // Write the fitted patch out, so a real sample can be taken end to end from
    // the command line: probe it, render the patch, compare the two files. The
    // plugin could already do this by drag and drop, but not in a form that can
    // be measured or scripted.
    if (const auto it = args.options.find ("--patch"); it != args.options.end()
        && it->second.isNotEmpty())
    {
        const juce::File out (juce::File::getCurrentWorkingDirectory()
                                  .getChildFile (it->second));
        if (! out.replaceWithText (finalPatch.toJson()))
            std::fprintf (stderr, "warning: could not write %s\n",
                          out.getFullPathName().toRawUTF8());
    }


    const auto trajectories = autosynth::Modulation::extract (samples, numSamples, sampleRate, hop);
    const auto lfo = autosynth::Modulation::best (trajectories);
    auto* lfoObj = new juce::DynamicObject();
    lfoObj->setProperty ("dest", static_cast<int> (lfo.dest));
    lfoObj->setProperty ("shape", static_cast<int> (lfo.shape));
    lfoObj->setProperty ("rate_hz", lfo.rateHz);
    lfoObj->setProperty ("depth", lfo.depth);
    lfoObj->setProperty ("delay", lfo.delay);
    lfoObj->setProperty ("phase", lfo.phase);

    juce::Array<juce::var> lfoList;
    for (const auto& l : autosynth::Modulation::bestSeveral (trajectories))
    {
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("dest", static_cast<int> (l.dest));
        obj->setProperty ("shape", static_cast<int> (l.shape));
        obj->setProperty ("rate_hz", l.rateHz);
        obj->setProperty ("depth", l.depth);
        lfoList.add (juce::var (obj));
    }

    juce::Array<juce::var> fittedOscs;
    for (const auto& osc : fitted.oscs)
    {
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("enabled", osc.enabled);
        obj->setProperty ("waveform", static_cast<int> (osc.waveform));
        obj->setProperty ("semitones", osc.semitones);
        obj->setProperty ("cents", osc.cents);
        obj->setProperty ("level", osc.level);
        obj->setProperty ("unison_voices", osc.unisonVoices);
        obj->setProperty ("unison_detune", osc.unisonDetune);
        fittedOscs.add (juce::var (obj));
    }
    auto* fittedObj = new juce::DynamicObject();
    fittedObj->setProperty ("oscs", fittedOscs);
    fittedObj->setProperty ("root_hz", fitted.rootHz);
    fittedObj->setProperty ("cutoff_hz", fitted.filter.cutoffHz);
    fittedObj->setProperty ("env_amount", fitted.filter.envAmount);
    fittedObj->setProperty ("noise_level", fitted.noiseLevel);
    fittedObj->setProperty ("master_level", fitted.masterLevel);
    fittedObj->setProperty ("active_oscs", fitted.activeOscCount());

    auto* root = new juce::DynamicObject();
    root->setProperty ("roles", rolesVar);
    root->setProperty ("fitted", juce::var (fittedObj));
    root->setProperty ("refine", refineVar);
    root->setProperty ("filter_fit", filterVar);
    root->setProperty ("lfo", juce::var (lfoObj));
    root->setProperty ("lfos", lfoList);
    root->setProperty ("gate", juce::var (gateObj));
    root->setProperty ("amp_env", juce::var (envObj));
    root->setProperty ("waveform_fit", waveformVar);
    root->setProperty ("num_partials", static_cast<int> (partialSet.partials.size()));
    root->setProperty ("partials", partialArray);
    root->setProperty ("groups", groupArray);
    root->setProperty ("sample_rate", sampleRate);
    root->setProperty ("num_samples", numSamples);
    root->setProperty ("hop", hop);
    root->setProperty ("fft_size", fftSize);
    root->setProperty ("num_frames", spectrogram.numFrames);
    root->setProperty ("num_bins", spectrogram.numBins);
    root->setProperty ("f0", f0);
    root->setProperty ("confidence", confidence);
    root->setProperty ("note", note);
    root->setProperty ("cents", cents);
    root->setProperty ("stft_mid_frame", toVar (midFrame));
    root->setProperty ("stft_times", toVar (spectrogram.times));
    root->setProperty ("centroid", toVar (centroid));
    root->setProperty ("loudness", toVar (loudness));
    root->setProperty ("f0_track", toVar (pitch.f0));
    root->setProperty ("f0_confidence", toVar (pitch.confidence));

    std::printf ("%s\n", juce::JSON::toString (juce::var (root), true).toRawUTF8());
    return 0;
}
