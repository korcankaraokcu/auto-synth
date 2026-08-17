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

#include "ir/Patch.h"
#include "ir/VitalExport.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include <cmath>
#include <cstdio>
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

    if (positional.size() < 2)
    {
        std::fprintf (stderr,
                      "usage: autosynth_vital <patch.json> <out.wav> "
                      "[--note hz] [--dur s] [--gate s] [--sr rate] "
                      "[--plugin Vital.vst3] [--preset out.vital]\n");
        return 2;
    }

    const auto cwd = juce::File::getCurrentWorkingDirectory();
    const auto patchFile = cwd.getChildFile (positional[0]);
    const auto outFile = cwd.getChildFile (positional[1]);

    juce::String error;
    const auto patch = autosynth::Patch::fromFile (patchFile, &error);
    if (error.isNotEmpty())
    {
        std::fprintf (stderr, "error: %s\n", error.toRawUTF8());
        return 1;
    }

    const auto sampleRate = args.value ("--sr", 48000.0);
    const auto duration = args.value ("--dur", 2.0);
    const auto gate = args.value ("--gate", 1.5);
    const auto requestedHz = args.value ("--note", patch.rootHz);

    // Vital is played, not tuned to a frequency: it gets a MIDI note. Rounding
    // to the nearest one and reporting the exact frequency of *that* note is
    // what makes this comparable with autosynth_render -- otherwise the diff
    // reads up to half a semitone of pitch error that neither synth committed.
    const auto midiNote = juce::jlimit (0, 127,
        (int) std::lround (69.0 + 12.0 * std::log2 (requestedHz / 440.0)));
    const auto renderedHz = 440.0 * std::pow (2.0, (midiNote - 69) / 12.0);

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

    // The exporter's own output, handed straight to the plugin. Writing it to
    // disk as well is optional and useful: the file that failed to load is the
    // one worth looking at.
    const auto presetJson = autosynth::VitalExport::toJson (patch, patch.name);
    const auto presetPath = args.text ("--preset");
    if (presetPath.isNotEmpty())
        cwd.getChildFile (presetPath).replaceWithText (presetJson);

    const auto wrapped = wrapAsVst3State (presetJson);
    plugin->setStateInformation (wrapped.getData(), (int) wrapped.getSize());

    // Give the plugin's message thread a turn before rendering. Loading a
    // preset is not necessarily synchronous -- a wavetable can be built on an
    // async update -- and a console host that never dispatches would render
    // whatever was loaded before the state arrived.
    juce::MessageManager::getInstance()->runDispatchLoopUntil (200);

    // Loading a preset can reallocate the engine; prepare again so it is
    // configured for this rate whatever the state did.
    plugin->prepareToPlay (sampleRate, blockSize);

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

    // Reported latency is trimmed from the front rather than ignored, because
    // the diff measures attack time and a few hundred samples of it is a
    // measurable slower-attack verdict that nothing in the patch caused.
    const auto latency = juce::jmax (0, plugin->getLatencySamples());
    const auto wanted = (int) std::ceil (duration * sampleRate);
    const auto total = wanted + latency;
    const auto gateSample = (int) std::lround (gate * sampleRate) + latency;

    const auto channels = juce::jmax (1, plugin->getTotalNumOutputChannels());
    juce::AudioBuffer<float> collected (1, total);
    collected.clear();

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

        // Mono by averaging rather than taking the left channel: Vital's unison
        // and its effects spread energy across the pair, and one channel of a
        // stereo-spread patch is not the same signal.
        for (int i = 0; i < thisBlock; ++i)
        {
            float sum = 0.0f;
            for (int ch = 0; ch < channels; ++ch)
                sum += view.getSample (ch, i);
            collected.setSample (0, position + i, sum / (float) channels);
        }
    }

    juce::AudioBuffer<float> buffer (1, wanted);
    buffer.copyFrom (0, 0, collected, 0, latency, wanted);

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
