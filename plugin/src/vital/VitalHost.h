#pragma once

#include "ir/Patch.h"
#include "ir/VitalExport.h"

#include <juce_audio_processors/juce_audio_processors.h>

#include <cmath>
#include <vector>

namespace autosynth
{

// Renders patches through the installed Vital.
//
// One implementation, shared by the tools and by the test suite, because there
// is now no other synth here: this project writes Vital presets, so the thing
// that says whether a patch is any good is Vital.
//
// It links no Vital code and ships none. The plug-in is whatever is installed,
// which is also the build the presets will be opened in -- a submodule would
// render the public source drop instead, which is not necessarily the same
// engine.
class VitalHost
{
public:
    // Wherever this platform keeps its VST3s, asked rather than assumed: JUCE
    // knows the standard locations and picks up any the user has configured.
    static juce::File find (const juce::String& pathOverride = {})
    {
        if (pathOverride.isNotEmpty())
            return juce::File::getCurrentWorkingDirectory().getChildFile (pathOverride);

        juce::VST3PluginFormat format;
        const auto search = format.getDefaultLocationsToSearch();

        for (int i = 0; i < search.getNumPaths(); ++i)
        {
            const auto candidate = search[i].getChildFile ("Vital.vst3");
            if (candidate.exists())
                return candidate;
        }

        return {};
    }

    bool open (double sampleRateToUse, juce::String& errorOut,
               const juce::String& pathOverride = {})
    {
        sampleRate = sampleRateToUse;

        const auto file = find (pathOverride);
        if (file == juce::File() || ! file.exists())
        {
            errorOut = "Vital.vst3 not found";
            return false;
        }

        juce::VST3PluginFormat format;
        juce::OwnedArray<juce::PluginDescription> descriptions;
        format.findAllTypesForFile (descriptions, file.getFullPathName());
        if (descriptions.isEmpty())
        {
            errorOut = "no plugin types in " + file.getFullPathName();
            return false;
        }

        // A VST3 bundle can hold several entries; the instrument is the one to
        // play. Vital ships an effect build under the same format.
        const juce::PluginDescription* chosen = descriptions[0];
        for (const auto* description : descriptions)
            if (description->isInstrument)
            {
                chosen = description;
                break;
            }

        formats.addFormat (new juce::VST3PluginFormat());
        instance = formats.createPluginInstance (*chosen, sampleRate, blockSize, errorOut);
        if (instance == nullptr)
            return false;

        instance->enableAllBuses();
        instance->setNonRealtime (true);
        instance->prepareToPlay (sampleRate, blockSize);
        return true;
    }

    bool isOpen() const noexcept { return instance != nullptr; }
    juce::AudioPluginInstance* plugin() const noexcept { return instance.get(); }

    // Vital's *plugin* state chunk is the preset JSON -- getStateInformation
    // calls the same LoadSave::stateToJson that writes a .vital file -- but a
    // *hosted* plug-in's is not: JUCE's VST3 host wraps the component and
    // controller states as base64 inside an XML document, and its
    // setStateInformation quietly does nothing when handed anything else.
    // getXmlFromBinary returns null and there is no error path, so passing the
    // preset text directly renders Vital's init patch while reporting success.
    void loadJson (const juce::String& json)
    {
        const juce::MemoryBlock component (json.toRawUTF8(), json.getNumBytesAsUTF8());

        juce::XmlElement state ("VST3PluginState");
        state.createNewChildElement ("IComponent")->addTextElement (component.toBase64Encoding());

        juce::MemoryBlock wrapped;
        juce::AudioProcessor::copyXmlToBinary (state, wrapped);

        instance->setStateInformation (wrapped.getData(), (int) wrapped.getSize());

        // Loading a preset can reallocate the engine, so prepare again. No
        // message-loop pump is needed: settles of 0 to 200 ms give bit-identical
        // renders including the first, cold load of a process, so the load is
        // synchronous.
        instance->prepareToPlay (sampleRate, blockSize);
        instance->reset();
    }

    void loadPreset (const Patch& patch)
    {
        loadJson (VitalExport::toJson (patch, patch.name));
    }

    // Renders whatever is currently loaded.
    //
    // Vital is played rather than tuned to a frequency: it gets a MIDI note.
    // Rounding to the nearest and reporting that note's own frequency is what
    // keeps a comparison honest -- otherwise a diff reads up to half a semitone
    // of pitch error that neither side committed.
    std::vector<float> renderLoaded (double noteHz, double duration, double gate)
    {
        const auto midiNote = noteFor (noteHz);

        // Reported latency is trimmed from the front rather than ignored: a few
        // hundred samples of it is a measurable slower-attack verdict that
        // nothing in the patch caused.
        const auto latency = juce::jmax (0, instance->getLatencySamples());
        const auto wanted = (int) std::ceil (duration * sampleRate);
        const auto total = wanted + latency;
        const auto gateSample = (int) std::lround (gate * sampleRate) + latency;
        const auto channels = juce::jmax (1, instance->getTotalNumOutputChannels());

        settle (channels);

        std::vector<float> collected ((size_t) total, 0.0f);
        juce::AudioBuffer<float> block (channels, blockSize);
        bool started = false, stopped = false;

        for (int position = 0; position < total; position += blockSize)
        {
            const auto thisBlock = juce::jmin (blockSize, total - position);
            block.clear();

            juce::MidiBuffer midi;
            if (! started && position + thisBlock > latency)
            {
                midi.addEvent (juce::MidiMessage::noteOn (1, midiNote, 1.0f),
                               juce::jmax (0, latency - position));
                started = true;
            }
            if (! stopped && started && position + thisBlock > gateSample)
            {
                midi.addEvent (juce::MidiMessage::noteOff (1, midiNote),
                               juce::jlimit (0, thisBlock - 1, gateSample - position));
                stopped = true;
            }

            juce::AudioBuffer<float> view (block.getArrayOfWritePointers(), channels, 0, thisBlock);
            instance->processBlock (view, midi);

            // Mono by averaging rather than taking the left channel: unison and
            // the effects spread energy across the pair, and one channel of a
            // stereo-spread patch is not the same signal.
            for (int i = 0; i < thisBlock; ++i)
            {
                float sum = 0.0f;
                for (int ch = 0; ch < channels; ++ch)
                    sum += view.getSample (ch, i);
                collected[(size_t) (position + i)] = sum / (float) channels;
            }
        }

        return { collected.begin() + latency, collected.begin() + latency + wanted };
    }

    std::vector<float> render (const Patch& patch, double noteHz,
                               double duration, double gate)
    {
        loadPreset (patch);

        // The first render of a process is still not quite like the others,
        // even with the settle: something a plug-in does lazily on its opening
        // blocks leaves a patch about a thousandth away from its own second
        // render. Discarding one render of *this* patch -- rather than of some
        // arbitrary warm-up note -- costs a second once and makes every
        // evaluation of a fit see the same conditions, including the first,
        // whose loss is what normalises every term's weight.
        if (! warmedUp)
        {
            warmedUp = true;
            renderLoaded (noteHz, duration, gate);
            loadPreset (patch);
        }

        return renderLoaded (noteHz, duration, gate);
    }

    // Everything the previous render left behind, cleared: the voices it was
    // still playing, the tails they fed, and the parameter ramps the new
    // preset started.
    //
    // Each of the three was found by measurement and each one alone is enough
    // to make two renders of the same patch differ.
    //
    // A note still held when the buffer ends -- every render whose gate is its
    // whole duration -- survives reset() and the next preset load and goes on
    // playing underneath what comes next: unstopped, a suite that renders 55 Hz
    // and then 220 measures 55 both times, the older voice being the lowest
    // thing in the mix.
    //
    // It is *released* rather than killed, and that distinction cost an
    // afternoon. All sound off silences it just as well, and leaves Vital's
    // voice allocator one place further on, because the voice never finished --
    // so the next note lands on a different voice with a different phase. Two
    // renders of one patch then alternate between two states, differing by 0.30
    // at the sample while their loudness agrees to 0.04 dB and their spectra to
    // 0.0015: invisible to every term in the objective and plainly visible in a
    // peak. Every gate short of the buffer was already exact, which is what
    // made it look like a property of the patch rather than of the host.
    // Released and allowed to finish, the voice goes back where it came from
    // and all four renders of a repeat agree bit for bit.
    //
    // Tails outlive the note that fed them, so the run continues until the
    // output goes quiet rather than for a fixed time.
    //
    // And Vital ramps to a newly loaded preset's values rather than jumping.
    // Rendering immediately after a load catches the tail of that ramp: with no
    // settle two renders of one patch differed by 0.051 at the sample -- six per
    // cent of full scale, on a *dry* patch -- and the first render was the odd
    // one out. Settling 85 ms leaves 0.0036, 171 ms leaves 0.00024, and 256 ms
    // leaves nothing at all. A quarter of a second is the floor here, with a
    // little margin, because a measurement taken during someone else's ramp is
    // not a measurement of the patch.
    void settle (int channels)
    {
        juce::MidiBuffer stop;
        for (int channel = 1; channel <= 16; ++channel)
            stop.addEvent (juce::MidiMessage::allNotesOff (channel), 0);

        const auto minBlocks = (int) std::ceil (0.3 * sampleRate / blockSize);
        const auto maxBlocks = (int) std::ceil (4.0 * sampleRate / blockSize);
        juce::AudioBuffer<float> flush (channels, blockSize);

        for (int i = 0; i < maxBlocks; ++i)
        {
            flush.clear();
            instance->processBlock (flush, i == 0 ? stop : emptyMidi);

            if (i >= minBlocks && flush.getMagnitude (0, blockSize) < 1.0e-7f)
                break;
        }

        instance->reset();
    }

    static int noteFor (double hz)
    {
        return juce::jlimit (0, 127, (int) std::lround (69.0 + 12.0 * std::log2 (hz / 440.0)));
    }

    static double frequencyOf (int midiNote)
    {
        return 440.0 * std::pow (2.0, (midiNote - 69) / 12.0);
    }

    double rate() const noexcept { return sampleRate; }

private:
    juce::AudioPluginFormatManager formats;
    std::unique_ptr<juce::AudioPluginInstance> instance;
    juce::MidiBuffer emptyMidi;
    double sampleRate = 48000.0;
    int blockSize = 512;
    bool warmedUp = false;
};

} // namespace autosynth
