// Renders, fits and evaluates through Vital itself.
//
// This is the only thing in the project that makes a sound. The exporter's
// output was once checkable only by reading it -- the preset loaded, the
// parameters were in range, and whether it *sounded* like the fit was a
// question for a listener -- and "the export sounds a bit different" was the
// one report no measurement here could confirm or deny.
//
// Hosting the installed VST3 closes that without a submodule and without a
// build dependency. Vital's plugin state chunk *is* the preset JSON -- its
// getStateInformation calls LoadSave::stateToJson, the same function that
// writes a .vital file -- so the exported text can be handed to the plugin
// directly, which tests the exporter end to end rather than a re-encoding of
// it. `src/vital/VitalHost.h` is the hosting; this file is the command line
// around it.
//
// It renders the version that is installed, which is the version the presets
// will actually be opened in. That is a feature rather than a compromise: a
// submodule would render the public source drop, which is not necessarily the
// same engine.
//
//   autosynth_vital patch.json out.wav [--note 220] [--dur 2.0] [--gate 1.5]
//                   [--sr 48000] [--plugin path/to/Vital.vst3] [--preset out.vital]
//
// How close the result is to the recording is autosynth_diff's job:
//   autosynth_vital fitted.json out.wav --fit samples/violin.wav
//   autosynth_diff  samples/violin.wav  out.wav

#include "eval/Recovery.h"
#include "fit/PartialFit.h"
#include "fit/Refine.h"
#include "ir/Patch.h"
#include "ir/VitalExport.h"
#include "vital/VitalHost.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
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

    juce::String text (const char* flag, const juce::String& fallback = {}) const
    {
        const auto it = options.find (flag);
        return it == options.end() ? fallback : it->second;
    }
};

// Hand-parsed because juce::ArgumentList treats an option's value as a separate
// positional argument.
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

std::vector<float> readMono (const juce::File& file, double& sampleRateOut)
{
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (file));
    if (reader == nullptr)
        return {};

    const auto numSamples = (int) reader->lengthInSamples;
    juce::AudioBuffer<float> buffer ((int) reader->numChannels, juce::jmax (1, numSamples));
    reader->read (&buffer, 0, numSamples, 0, true, true);

    if (buffer.getNumChannels() > 1)
    {
        for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
            buffer.addFrom (0, 0, buffer, ch, 0, numSamples);
        buffer.applyGain (0, 0, numSamples, 1.0f / buffer.getNumChannels());
    }

    sampleRateOut = reader->sampleRate;
    const auto* data = buffer.getReadPointer (0);
    return std::vector<float> (data, data + numSamples);
}

// The inverse, so --dump-state reports the preset the plugin is actually
// holding rather than the container it arrived in.
juce::String unwrapVst3State (const juce::MemoryBlock& raw)
{
    if (auto xml = juce::AudioProcessor::getXmlFromBinary (raw.getData(), (int) raw.getSize()))
        if (auto* component = xml->getChildByName ("IComponent"))
        {
            juce::MemoryBlock inner;
            if (inner.fromBase64Encoding (component->getAllSubText()))
            {
                // Vital writes its state with VST2 compatibility, so the JSON
                // sits inside an FXP chunk with binary either side of it and a
                // null byte early enough that reading the block as text returns
                // four characters. Cutting to the outermost braces recovers it.
                const auto* bytes = static_cast<const char*> (inner.getData());
                const auto size = (int) inner.getSize();

                int start = 0;
                while (start < size && bytes[start] != '{')
                    ++start;
                int end = size - 1;
                while (end > start && bytes[end] != '}')
                    --end;

                if (start < end)
                    return juce::String::fromUTF8 (bytes + start, end - start + 1);
            }
        }

    return {};
}

} // namespace

int main (int argc, char* argv[])
{
    // Hosting needs a message manager: the VST3 format scans and instantiates
    // on it, and asserts without one.
    juce::ScopedJuceInitialiser_GUI juceInit;

    const auto args = parseArgs (argc, argv);
    const auto& positional = args.positional;

    const auto evaluating = args.options.count ("--eval") > 0
                         || args.options.count ("--sweep") > 0;

    if (positional.size() < 2 && ! evaluating)
    {
        std::fprintf (stderr,
                      "usage: autosynth_vital <patch.json> <out.wav> "
                      "[--note hz] [--dur s] [--gate s] [--sr rate] "
                      "[--plugin Vital.vst3] [--preset out.vital]\n"
                      "       autosynth_vital <patch-out.json> <out.wav> "
                      "--fit <target.wav> [--refine-evals n] [--seed n]\n"
                      "       autosynth_vital --eval [--trials n] [--seed n]\n"
                      "       autosynth_vital <patch.json> <out.wav> --sweep\n"
                      "       autosynth_vital <patch.json> <out.wav> --check-repeatable\n");
        return 2;
    }

    const auto cwd = juce::File::getCurrentWorkingDirectory();
    const auto patchFile = cwd.getChildFile (positional.size() > 0 ? positional[0] : "patch.json");
    const auto outFile = cwd.getChildFile (positional.size() > 1 ? positional[1] : "out.wav");

    // With --fit the first argument is an output rather than an input: the
    // target is fitted with Vital doing the rendering, and the patch it settles
    // on is written there.
    const auto fitPath = args.text ("--fit");
    const auto fitting = fitPath.isNotEmpty();

    std::vector<float> target;
    auto sampleRate = args.value ("--sr", 48000.0);
    autosynth::Patch patch;

    if (fitting)
    {
        double targetRate = 0.0;
        target = readMono (cwd.getChildFile (fitPath), targetRate);
        if (target.empty())
        {
            std::fprintf (stderr, "error: cannot read %s\n", fitPath.toRawUTF8());
            return 1;
        }
        // The target's own rate, so the loss compares like with like.
        sampleRate = targetRate;
    }
    else if (! evaluating)
    {
        juce::String error;
        patch = autosynth::Patch::fromFile (patchFile, &error);
        if (error.isNotEmpty())
        {
            std::fprintf (stderr, "error: %s\n", error.toRawUTF8());
            return 1;
        }
    }

    const auto duration = args.value ("--dur", 2.0);
    const auto gate = args.value ("--gate", 1.5);

    autosynth::VitalHost host;
    juce::String hostError;
    if (! host.open (sampleRate, hostError, args.text ("--plugin")))
    {
        std::fprintf (stderr, "error: %s\n", hostError.toRawUTF8());
        return 1;
    }

    auto* plugin = host.plugin();

    const auto loadJson = [&] (const juce::String& json) { host.loadJson (json); };
    const auto loadPreset = [&] (const autosynth::Patch& p) { host.loadPreset (p); };

    // Every parameter the host can see, as the plug-in currently holds them.
    const auto snapshot = [&]
    {
        const auto& parameters = plugin->getParameters();
        std::vector<float> values ((size_t) parameters.size());
        for (int i = 0; i < parameters.size(); ++i)
            values[(size_t) i] = parameters[i]->getValue();
        return values;
    };

    const auto noteFor = [] (double hz) { return autosynth::VitalHost::noteFor (hz); };

    const auto renderLoaded = [&] (double noteHz, double dur, double gateSeconds)
    {
        return host.renderLoaded (noteHz, dur, gateSeconds);
    };

    const auto renderPatch = [&] (const autosynth::Patch& p, double noteHz,
                                  double dur, double gateSeconds)
    {
        return host.render (p, noteHz, dur, gateSeconds);
    };

    // The renderer, in the form everything closed-loop takes it, built once.
    //
    // Once because handing it to some callers and not others is a silent
    // failure rather than a loud one: analysis skips its level and noise
    // calibration without a renderer, so a fit missing it comes back with the
    // levels the factorisation happened to leave and no noise at all, and looks
    // like a fit the whole way.
    const autosynth::Renderer renderer = [&] (const autosynth::Patch& p,
                                              double dur, double gateSeconds)
    {
        return renderPatch (p, p.rootHz, dur, gateSeconds);
    };

    // Which searchable parameters actually move the sound Vital makes.
    //
    // The unit test guarding this can only ask whether a parameter changes the
    // preset *text*. That catches a parameter nobody writes, and misses the
    // worse case: one written into a control that turns out to be inert,
    // because it is on a page Vital ignores, or scaled to nothing, or routed
    // somewhere the signal never reaches. Both look identical from this side of
    // the boundary and only one of them can be heard.
    //
    // So this moves each parameter to the far end of its declared range, plays
    // the result, and reports the spectral distance from the unmoved patch.
    // Anything near zero is carried in name only.
    if (args.options.count ("--sweep") > 0)
    {
        autosynth::Patch probe;
        probe.rootHz = 220.0f;
        probe.noiseLevel = 0.3f;
        probe.reverb = { true, 0.5f, 0.5f, 0.3f };
        probe.delay = { true, 0.25f, 0.35f, 0.4f };
        probe.filter.type = autosynth::FilterType::lowpass;
        probe.filter.cutoffHz = 2000.0f;
        probe.filter.envAmount = 1.0f;

        for (int i = 0; i < autosynth::kNumOsc; ++i)
        {
            auto& osc = probe.oscs[(size_t) i];
            osc.enabled = true;
            osc.level = 0.5f;
            osc.unisonVoices = 3;
            osc.unisonDetune = 12.0f;
            osc.waveform = autosynth::Waveform::pulse;
            osc.waveformB = autosynth::Waveform::square;
            osc.waveMorph = 0.4f;
            osc.pulseWidth = 0.35f;
            osc.numFrames = 3;
            osc.framePositionEnvAmount = 0.5f;
            osc.envEnabled = true;
        }

        probe.lfos[0].dest = autosynth::LfoDest::pitch;
        probe.lfos[0].depth = 0.4f;
        probe.lfos[1].dest = autosynth::LfoDest::cutoff;
        probe.lfos[1].depth = 0.4f;

        const auto base = renderPatch (probe, probe.rootHz, duration, gate);
        const auto specs = autosynth::Refine::continuousSpecs();

        std::vector<std::pair<double, std::string>> rows;
        for (const auto& path : autosynth::Refine::scopeFor (probe))
        {
            const auto spec = std::find_if (specs.begin(), specs.end(),
                                            [&path] (const auto& s) { return s.path == path; });
            if (spec == specs.end())
                continue;

            const auto current = autosynth::Refine::parameterValue (probe, path);
            const auto moved = std::abs (current - spec->lo) > std::abs (current - spec->hi)
                                 ? spec->lo : spec->hi;

            auto altered = probe;
            autosynth::Refine::setParameterValue (altered, path, moved);

            const auto audio = renderPatch (altered, altered.rootHz, duration, gate);
            const auto scored = autosynth::Recovery::score (audio, base, sampleRate);
            rows.emplace_back (scored.spectral, path);
        }

        std::sort (rows.begin(), rows.end());
        std::printf ("\nhow far each searchable parameter moves Vital output\n");
        std::printf ("  spectral distance from the unmoved patch; near zero is carried in name only\n\n");
        for (const auto& row : rows)
            std::printf ("  %-34s %.4f%s\n", row.second.c_str(), row.first,
                         row.first < 0.02 ? "   <- inert" : "");
        std::printf ("\n");
        return 0;
    }

    // Does one patch render the same way twice?
    //
    // It did not, and nothing about the failure was visible from the audio: two
    // renders of one preset differed by 0.072 at the sample on a peak of 0.43,
    // alternating on every other render. In the terms the objective is built
    // from that was a spectral distance of 0.057 and two decibels of loudness,
    // against a fit whose own loudness error is about four and a half -- so
    // nearly half of that term was noise, and had been through every comparison
    // made before it was found.
    //
    // The cause was Vital's reverb chorus free-running at a quarter of a hertz,
    // sampled every two seconds; the exporter now switches it off. What remains
    // is a random LFO where a patch has one, which is random by construction.
    //
    // Kept as a standing check because the failure is silent: an objective that
    // returns a different number for the same patch reads as a search that
    // cannot converge, and there is nothing in a rendered note that says so.
    if (args.options.count ("--check-repeatable") > 0)
    {
        // Both of these begin from the same state: every render settles the
        // plug-in first, and the very first one of a process is discarded --
        // which is the state every evaluation of a fit begins from, and
        // therefore the one worth checking.
        const auto first = renderPatch (patch, patch.rootHz, duration, gate);
        const auto second = renderPatch (patch, patch.rootHz, duration, gate);

        auto worst = 0.0;
        for (size_t i = 0; i < juce::jmin (first.size(), second.size()); ++i)
            worst = juce::jmax (worst, (double) std::abs (first[i] - second[i]));

        const auto self = autosynth::Recovery::score (first, second, sampleRate);
        std::printf ("one patch rendered twice:\n"
                     "  worst sample  %.6f\n"
                     "  spectral      %.4f\n"
                     "  loudness      %.2f dB\n"
                     "  centroid      %.3f oct\n",
                     worst, self.spectral, self.loudnessDb, self.centroidOctaves);
        return self.loudnessDb < 1.0 ? 0 : 1;
    }


    // The recovery harness, with Vital as the synth being measured.
    //
    // Random patches rendered *by Vital* are the targets, and the control and
    // every refinement candidate are rendered there too. For a project whose
    // deliverable is a preset for someone else's synth, that is the question
    // worth answering.
    //
    // Read against the control of the same run rather than against another
    // run's fitted scores: the targets are a different random draw each time,
    // so the absolute numbers are not comparable and the margin over the
    // control is.
    if (evaluating)
    {
        autosynth::Recovery::Options evalOptions;
        evalOptions.trials = (int) args.value ("--trials", 12.0);
        evalOptions.seed = (unsigned) args.value ("--seed", 0.0);
        evalOptions.sampleRate = sampleRate;
        evalOptions.refineEvaluations = (int) args.value ("--refine-evals", 192.0);
        evalOptions.refine = args.value ("--no-refine", 0.0) == 0.0;
        evalOptions.renderer = renderer;

        const auto started = juce::Time::getMillisecondCounterHiRes();
        const auto summary = autosynth::Recovery::run (evalOptions);
        std::printf ("\n%s", autosynth::Recovery::toText (summary).toRawUTF8());
        std::printf ("  rendered through Vital, %.1f s for %d trials\n\n",
                     (juce::Time::getMillisecondCounterHiRes() - started) / 1000.0,
                     summary.trials);
        return summary.trials > 0 ? 0 : 1;
    }

    // Fitting, with Vital as the renderer.
    //
    // This is the point of the whole exercise: analysis decides the structure,
    // and refinement then lands the values against the synth that will actually
    // play them. Nothing has to be found and hand-corrected in the exporter to
    // account for a second synth's differences, because the optimiser is
    // measuring the real output.
    if (fitting)
    {
        autosynth::PartialFit::Options fitOptions;
        fitOptions.gateSeconds = gate;
        fitOptions.renderer = renderer;
        patch = autosynth::PartialFit::fit (target.data(), (int) target.size(),
                                            sampleRate, fitOptions);

        autosynth::Refine::Options refineOptions;
        refineOptions.maxEvaluations = (int) args.value ("--refine-evals", 192.0);
        refineOptions.gateSeconds = gate;
        refineOptions.renderer = renderer;

        // Exposed because one fit is one sample of a search, not the answer.
        // The objective has seven terms and CMA-ES settles on a different trade
        // between them from a different draw, so a change that moves one axis
        // has to be read against the spread rather than against one run.
        refineOptions.seed = (unsigned) args.value ("--seed", 1.0);

        const auto started = juce::Time::getMillisecondCounterHiRes();
        const auto refined = autosynth::Refine::run (patch, target.data(), (int) target.size(),
                                                     sampleRate, refineOptions);
        const auto elapsed = juce::Time::getMillisecondCounterHiRes() - started;

        patch = refined.patch;
        patchFile.replaceWithText (patch.toJson());

        std::printf ("fitted %s through Vital: loss %.4f -> %.4f over %d evaluations "
                     "in %.1f s -> %s\n",
                     fitPath.toRawUTF8(), refined.initialLoss, refined.finalLoss,
                     refined.evaluations, elapsed / 1000.0,
                     patchFile.getFullPathName().toRawUTF8());
    }

    const auto requestedHz = args.value ("--note", patch.rootHz);
    const auto midiNote = noteFor (requestedHz);
    const auto renderedHz = 440.0 * std::pow (2.0, (midiNote - 69) / 12.0);

    const auto presetPath = args.text ("--preset");
    if (presetPath.isNotEmpty())
        cwd.getChildFile (presetPath)
            .replaceWithText (autosynth::VitalExport::toJson (patch, patch.name));

    loadPreset (patch);

    // What the plugin thinks it is holding, which is the only way to tell a
    // preset that failed to load from one that loaded and sounds wrong. The
    // first run of this tool reported a mathematically exact sawtooth with an
    // instant attack -- Vital's init patch -- and the numbers alone could not
    // say whether the exporter or the handover was at fault.
    const auto dumpPath = args.text ("--dump-state");
    if (dumpPath.isNotEmpty())
    {
        juce::MemoryBlock state;
        plugin->getStateInformation (state);
        const auto inner = unwrapVst3State (state);
        cwd.getChildFile (dumpPath).replaceWithText (inner);
        std::printf ("plugin state: %d bytes wrapped, %d bytes of preset -> %s\n",
                     (int) state.getSize(), inner.length(), dumpPath.toRawUTF8());
    }

    const auto rendered = renderPatch (patch, requestedHz, duration, gate);

    // What one evaluation costs when this is the renderer inside refinement:
    // the plugin loads once per fit, but the preset load and the render happen
    // once per candidate. Reported separately because they are paid at
    // different rates and only the second is irreducible.
    const auto repeat = (int) args.value ("--repeat", 1.0);
    if (repeat > 1)
    {
        const auto clock = [] { return juce::Time::getMillisecondCounterHiRes(); };

        // Repeatability is measured alongside the cost, because an objective
        // that returns a different number for the same patch is a worse problem
        // than a slow one: CMA-ES would be reading noise as signal.
        auto loadMs = 0.0, renderMs = 0.0, worstDrift = 0.0;
        std::vector<float> previous;
        for (int i = 0; i < repeat; ++i)
        {
            const auto a = clock();
            loadPreset (patch);
            const auto b = clock();
            const auto again = renderPatch (patch, requestedHz, duration, gate);
            loadMs += b - a;
            renderMs += clock() - b;

            const auto worstAgainst = [&again] (const std::vector<float>& other)
            {
                auto worst = 0.0;
                const auto n = juce::jmin (other.size(), again.size());
                for (size_t s = 0; s < n; ++s)
                    worst = juce::jmax (worst, (double) std::abs (other[s] - again[s]));
                return worst;
            };

            const auto vsFirst = worstAgainst (rendered);
            const auto vsPrevious = previous.empty() ? 0.0 : worstAgainst (previous);
            std::printf ("  render %d: %.6f against the first, %.6f against the previous\n",
                         i + 1, vsFirst, vsPrevious);
            worstDrift = juce::jmax (worstDrift, vsPrevious);
            previous = again;
        }

        std::printf ("repeatability over %d renders of one patch: worst "
                     "consecutive difference %.6f\n", repeat, worstDrift);

        // The load is counted twice over -- renderPatch loads too -- so the
        // render figure is the pair and the load figure is one of them.
        std::printf ("timing over %d evaluations:\n"
                     "  preset load %7.1f ms each\n"
                     "  load+render %7.1f ms each (%.2f s of audio)\n"
                     "  one evaluation %7.1f ms -> %.1f s for 192\n",
                     repeat, loadMs / repeat, renderMs / repeat, duration,
                     renderMs / repeat, renderMs / repeat * 192.0 / 1000.0);
    }

    juce::AudioBuffer<float> buffer (1, (int) rendered.size());
    buffer.copyFrom (0, 0, rendered.data(), (int) rendered.size());

    plugin->releaseResources();

    // Peak-limit rather than normalise: the absolute level is part of what the
    // comparison is checking.
    auto peak = 0.0f;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
        peak = juce::jmax (peak, std::abs (buffer.getSample (0, i)));
    if (peak > 1.0f)
        buffer.applyGain (1.0f / peak);

    outFile.deleteFile();
    std::unique_ptr<juce::FileOutputStream> stream (outFile.createOutputStream());
    if (stream == nullptr)
    {
        std::fprintf (stderr, "error: cannot write %s\n",
                      outFile.getFullPathName().toRawUTF8());
        return 1;
    }

    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatWriter> writer (
        wav.createWriterFor (stream.get(), sampleRate, 1, 24, {}, 0));
    if (writer == nullptr)
    {
        std::fprintf (stderr, "error: cannot create WAV writer\n");
        return 1;
    }
    stream.release(); // writer owns it now

    writer->writeFromAudioSampleBuffer (buffer, 0, buffer.getNumSamples());
    writer.reset();

    std::printf ("rendered %s through %s at note %d (%.2f Hz, asked %.2f) -> %s (%.2fs, peak %.3f)\n",
                 patch.name.toRawUTF8(), plugin->getName().toRawUTF8(), midiNote, renderedHz,
                 requestedHz, outFile.getFullPathName().toRawUTF8(),
                 buffer.getNumSamples() / sampleRate, peak);
    return 0;
}
