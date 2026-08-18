// Renders a patch through Vital itself.
//
// Until now the exporter's output could only be checked by reading it: the
// preset loaded, the parameters were in range, and whether it *sounded* like
// the fit was a question for a listener. That gap is not small -- "vital export
// sounds a bit different" is the one report no measurement here could confirm
// or deny -- and it is structural, because refinement optimises against this
// project's engine and the preset is then played by a different synth.
//
// Hosting the installed VST3 closes it without a submodule and without a build
// dependency. Vital's plugin state chunk *is* the preset JSON -- its
// getStateInformation calls LoadSave::stateToJson, the same function that
// writes a .vital file -- so the exported text can be handed to the plugin
// directly, which tests the exporter end to end rather than a re-encoding of
// it.
//
// It renders the version that is installed, which is the version the presets
// will actually be opened in. That is a feature rather than a compromise: a
// submodule would render the public source drop, which is not necessarily the
// same engine.
//
// Usage mirrors autosynth_render, so the two outputs are comparable:
//   autosynth_vital patch.json out.wav [--note 220] [--dur 2.0] [--gate 1.5]
//                   [--sr 48000] [--plugin path/to/Vital.vst3] [--preset out.vital]
//
// The comparison itself is autosynth_diff's job:
//   autosynth_render patch.json ours.wav
//   autosynth_vital  patch.json theirs.wav
//   autosynth_diff   ours.wav   theirs.wav

#include "eval/Recovery.h"
#include "fit/PartialFit.h"
#include "fit/Refine.h"
#include "ir/Patch.h"
#include "ir/VitalExport.h"

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

// Hand-parsed for the same reason autosynth_render is: juce::ArgumentList
// treats an option's value as a separate positional argument.
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

// The usual install locations, system first. JUCE's own scan would find these
// too, but it wants a directory to walk and a callback to wait on; the plugin
// is either at one of two paths or the user passes --plugin.
juce::File findVital (const juce::String& override)
{
    if (override.isNotEmpty())
        return juce::File::getCurrentWorkingDirectory().getChildFile (override);

    const juce::File candidates[] = {
        juce::File ("C:/Program Files/Common Files/VST3/Vital.vst3"),
        juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
            .getChildFile ("Programs/Common/VST3/Vital.vst3"),
        juce::File::getSpecialLocation (juce::File::commonApplicationDataDirectory)
            .getChildFile ("VST3/Vital.vst3"),
    };

    for (const auto& file : candidates)
        if (file.exists())
            return file;

    return {};
}

// Vital's plugin state chunk is the preset JSON, but a *hosted* plugin's is
// not: JUCE's VST3 host wraps the component and controller states as base64
// inside an XML document, and its setStateInformation quietly does nothing at
// all when handed anything else -- getXmlFromBinary returns null and there is
// no error path. Passing the preset text directly therefore rendered Vital's
// init patch while reporting success, which is a failure only an actual
// measurement catches.
//
// Wrapping it puts the JSON where the chain expects it. Vital's own VST3
// wrapper reads the component stream back out and passes those bytes to its
// AudioProcessor::setStateInformation, which is the preset loader.
juce::MemoryBlock wrapAsVst3State (const juce::String& json)
{
    const juce::MemoryBlock component (json.toRawUTF8(), json.getNumBytesAsUTF8());

    juce::XmlElement state ("VST3PluginState");
    state.createNewChildElement ("IComponent")->addTextElement (component.toBase64Encoding());

    juce::MemoryBlock wrapped;
    juce::AudioProcessor::copyXmlToBinary (state, wrapped);
    return wrapped;
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
                      "--fit <target.wav> [--refine-evals n]\n");
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

    const auto pluginFile = findVital (args.text ("--plugin"));
    if (pluginFile == juce::File() || ! pluginFile.exists())
    {
        std::fprintf (stderr, "error: Vital.vst3 not found; pass --plugin <path>\n");
        return 1;
    }

    juce::VST3PluginFormat format;
    juce::OwnedArray<juce::PluginDescription> descriptions;
    format.findAllTypesForFile (descriptions, pluginFile.getFullPathName());
    if (descriptions.isEmpty())
    {
        std::fprintf (stderr, "error: no plugin types in %s\n",
                      pluginFile.getFullPathName().toRawUTF8());
        return 1;
    }

    // A VST3 bundle can hold several entries; the instrument is the one to
    // play. Vital ships an effect build too under the same format.
    const juce::PluginDescription* chosen = descriptions[0];
    for (const auto* description : descriptions)
        if (description->isInstrument)
        {
            chosen = description;
            break;
        }

    const int blockSize = 512;
    juce::AudioPluginFormatManager formats;
    formats.addFormat (new juce::VST3PluginFormat());

    juce::String createError;
    std::unique_ptr<juce::AudioPluginInstance> plugin (
        formats.createPluginInstance (*chosen, sampleRate, blockSize, createError));
    if (plugin == nullptr)
    {
        std::fprintf (stderr, "error: could not instantiate: %s\n", createError.toRawUTF8());
        return 1;
    }

    plugin->enableAllBuses();
    plugin->setNonRealtime (true);
    plugin->prepareToPlay (sampleRate, blockSize);

    // Loading a preset, as its own step so it can be timed and repeated.
    //
    // The settle gives the plugin's message thread a turn, on the theory that
    // loading might not be synchronous -- a wavetable could be built on an
    // async update, and a console host that never dispatches would then render
    // whatever was loaded before the state arrived.
    //
    // Measured, it is not needed: rendering the same patch at settles of 0, 10,
    // 50, 100 and 200 ms gives five bit-identical files, and that includes the
    // first load of each process, which is a cold load into a plugin still
    // holding its init patch. So the load is synchronous and the wait was
    // costing 200 ms per evaluation for nothing -- half the per-evaluation
    // budget, which is what makes it worth knowing.
    //
    // The flag stays because the failure it guards against would be silent, and
    // this is the knob to reach for if a future preset renders as the init
    // patch. `--dump-state` is the other one.
    const auto settleMs = (int) args.value ("--settle", 0.0);
    const auto loadPreset = [&] (const autosynth::Patch& p)
    {
        const auto wrapped = wrapAsVst3State (autosynth::VitalExport::toJson (p, p.name));
        plugin->setStateInformation (wrapped.getData(), (int) wrapped.getSize());
        if (settleMs > 0)
            juce::MessageManager::getInstance()->runDispatchLoopUntil (settleMs);
        // Loading a preset can reallocate the engine; prepare again so it is
        // configured for this rate whatever the state did.
        plugin->prepareToPlay (sampleRate, blockSize);

        // Flush what the last render left ringing.
        //
        // Without this, evaluation N begins inside evaluation N-1's reverb
        // tail: rendering one patch six times gave renders differing by 0.035
        // at the sample, which is exactly this patch's tail level and not a
        // coincidence. Every candidate would have been scored against a
        // different amount of the previous candidate's decay, which is a
        // dependence on evaluation *order* -- the kind of thing that makes an
        // optimiser's path irreproducible rather than merely noisy.
        plugin->reset();
    };

    // One rendered note, which is both the deliverable and, when fitting, one
    // evaluation of the objective.
    //
    // Vital is played rather than tuned to a frequency: it gets a MIDI note.
    // Rounding to the nearest and reporting the exact frequency of *that* note
    // is what makes the output comparable with autosynth_render -- otherwise
    // the diff reads up to half a semitone of pitch error that neither synth
    // committed.
    const auto noteFor = [] (double hz)
    {
        return juce::jlimit (0, 127, (int) std::lround (69.0 + 12.0 * std::log2 (hz / 440.0)));
    };

    const auto renderPatch = [&] (const autosynth::Patch& p, double noteHz,
                                  double dur, double gateSeconds)
    {
        loadPreset (p);

        const auto midiNote = noteFor (noteHz);

        // Reported latency is trimmed from the front rather than ignored,
        // because the diff measures attack time and a few hundred samples of it
        // is a measurable slower-attack verdict that nothing in the patch
        // caused.
        const auto latency = juce::jmax (0, plugin->getLatencySamples());
        const auto wanted = (int) std::ceil (dur * sampleRate);
        const auto total = wanted + latency;
        const auto gateSample = (int) std::lround (gateSeconds * sampleRate) + latency;
        const auto channels = juce::jmax (1, plugin->getTotalNumOutputChannels());

        std::vector<float> collected ((size_t) total, 0.0f);
        juce::AudioBuffer<float> block (channels, blockSize);
        bool noteStarted = false, noteStopped = false;

        for (int position = 0; position < total; position += blockSize)
        {
            const auto thisBlock = juce::jmin (blockSize, total - position);
            block.clear();

            juce::MidiBuffer midi;
            if (! noteStarted && position + thisBlock > latency)
            {
                midi.addEvent (juce::MidiMessage::noteOn (1, midiNote, 1.0f),
                               juce::jmax (0, latency - position));
                noteStarted = true;
            }
            if (! noteStopped && noteStarted && position + thisBlock > gateSample)
            {
                midi.addEvent (juce::MidiMessage::noteOff (1, midiNote),
                               juce::jlimit (0, thisBlock - 1, gateSample - position));
                noteStopped = true;
            }

            juce::AudioBuffer<float> view (block.getArrayOfWritePointers(), channels, 0, thisBlock);
            plugin->processBlock (view, midi);

            // Mono by averaging rather than taking the left channel: Vital's
            // unison and its effects spread energy across the pair, and one
            // channel of a stereo-spread patch is not the same signal.
            for (int i = 0; i < thisBlock; ++i)
            {
                float sum = 0.0f;
                for (int ch = 0; ch < channels; ++ch)
                    sum += view.getSample (ch, i);
                collected[(size_t) (position + i)] = sum / (float) channels;
            }
        }

        return std::vector<float> (collected.begin() + latency, collected.begin() + latency + wanted);
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

    // The recovery harness, with Vital as the synth being measured.
    //
    // Not the same question autosynth_eval asks. That one renders targets from
    // this engine and checks the fitter gets its parameters back, which is
    // blind to everything the other synth does differently. This moves the
    // whole experiment into Vital's world: random patches rendered *by Vital*
    // are the targets, and the control and every refinement candidate are
    // rendered there too. For a project whose deliverable is a preset for
    // someone else's synth, that is the question worth answering.
    //
    // Read against autosynth_eval's own control rather than against its fitted
    // scores. Different targets, so the absolute numbers are not comparable;
    // what compares is how far each run beats the control it ran with.
    if (evaluating)
    {
        autosynth::Recovery::Options evalOptions;
        evalOptions.trials = (int) args.value ("--trials", 12.0);
        evalOptions.seed = (unsigned) args.value ("--seed", 0.0);
        evalOptions.sampleRate = sampleRate;
        evalOptions.refineEvaluations = (int) args.value ("--refine-evals", 192.0);
        evalOptions.refine = args.value ("--no-refine", 0.0) == 0.0;
        evalOptions.renderer = [&] (const autosynth::Patch& p, double dur, double gateSeconds)
        {
            return renderPatch (p, p.rootHz, dur, gateSeconds);
        };

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
    // This is the point of the whole exercise: analysis decides the structure
    // as before, and refinement then lands the values against the synth that
    // will actually play them. Every difference between this engine and Vital
    // stops needing to be found and hand-corrected in the exporter, because the
    // optimiser is measuring the real output.
    if (fitting)
    {
        autosynth::PartialFit::Options fitOptions;
        fitOptions.gateSeconds = gate;
        patch = autosynth::PartialFit::fit (target.data(), (int) target.size(),
                                            sampleRate, fitOptions);

        autosynth::Refine::Options refineOptions;
        refineOptions.maxEvaluations = (int) args.value ("--refine-evals", 192.0);
        refineOptions.gateSeconds = gate;
        refineOptions.renderer = [&] (const autosynth::Patch& candidate,
                                      double dur, double gateSeconds)
        {
            return renderPatch (candidate, candidate.rootHz, dur, gateSeconds);
        };

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
        std::printf ("timing over %d evaluations at settle %d ms:\n"
                     "  preset load %7.1f ms each\n"
                     "  load+render %7.1f ms each (%.2f s of audio)\n"
                     "  one evaluation %7.1f ms -> %.1f s for 192\n",
                     repeat, settleMs, loadMs / repeat, renderMs / repeat, duration,
                     renderMs / repeat, renderMs / repeat * 192.0 / 1000.0);
    }

    juce::AudioBuffer<float> buffer (1, (int) rendered.size());
    buffer.copyFrom (0, 0, rendered.data(), (int) rendered.size());

    plugin->releaseResources();
    plugin.reset();

    // Peak-limit rather than normalise, matching autosynth_render: the absolute
    // level is part of what the comparison is checking.
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
                 patch.name.toRawUTF8(), chosen->name.toRawUTF8(), midiNote, renderedHz,
                 requestedHz, outFile.getFullPathName().toRawUTF8(),
                 buffer.getNumSamples() / sampleRate, peak);
    return 0;
}
