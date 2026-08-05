// Headless renderer: patch JSON in, WAV out.
//
// This exists for the conformance test. The C++ engine and the Python
// reference engine are genuinely different implementations -- most visibly the
// filter, which is an STFT magnitude response there and a state-variable
// filter here -- so "did the port stay faithful?" is a question that needs an
// answer in numbers rather than by inspection.
//
// Usage mirrors `autosynth render`:
//   autosynth_render patch.json out.wav [--note 220] [--dur 2.0] [--gate 1.5] [--sr 48000]
//
// `--monitor a,b,c` additionally exposes the editor's solo/mute gains, which
// are monitoring state rather than patch state and so have no other route out
// of the plugin.

#include "dsp/Voice.h"
#include "ir/Patch.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <array>
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

// Parsed by hand rather than with juce::ArgumentList. That class treats an
// option's value as a separate positional argument, so "--sr 48000" left the
// sample rate unset and silently rendered a zero-length buffer -- the kind of
// failure that looks like a DSP bug for a while.
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

} // namespace

int main (int argc, char* argv[])
{
    const auto args = parseArgs (argc, argv);
    const auto& positional = args.positional;

    if (positional.size() < 2)
    {
        std::fprintf (stderr,
                      "usage: autosynth_render <patch.json> <out.wav> "
                      "[--note hz] [--dur s] [--gate s] [--sr rate] "
                      "[--monitor a,b,c]\n");
        return 2;
    }

    const juce::File patchFile (juce::File::getCurrentWorkingDirectory()
                                    .getChildFile (positional[0]));
    const juce::File outFile (juce::File::getCurrentWorkingDirectory()
                                  .getChildFile (positional[1]));

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
    // Default to the patch's own root pitch, matching `render(note_hz=None)`.
    const auto noteHz = args.value ("--note", patch.rootHz);

    autosynth::Engine engine;
    engine.prepare (sampleRate, 512);
    engine.setPatch (patch);

    // Solo/mute is a monitoring control the editor owns, so it is not part of
    // the patch and nothing outside the plugin could reach it. It went
    // unnoticed for that reason: the voice stored the mask and never applied
    // it. Exposing it here is what lets a test render prove otherwise.
    if (args.options.count ("--monitor") > 0)
    {
        std::array<float, autosynth::kNumOsc> mask { 1.0f, 1.0f, 1.0f };
        auto parts = juce::StringArray::fromTokens (
            args.options.at ("--monitor"), ",", "");
        for (int i = 0; i < juce::jmin (parts.size(), (int) mask.size()); ++i)
            mask[(size_t) i] = parts[i].getFloatValue();
        engine.setMonitorMask (mask);
    }

    juce::AudioBuffer<float> buffer;
    engine.renderOffline (buffer, noteHz, duration, gate);

    // Peak-limit rather than normalise, matching audio.write_wav: a fitted
    // patch's absolute level is information the conformance test compares.
    auto peak = 0.0f;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
        peak = juce::jmax (peak, std::abs (buffer.getSample (0, i)));
    if (peak > 1.0f)
        buffer.applyGain (1.0f / peak);

    outFile.deleteFile();
    std::unique_ptr<juce::FileOutputStream> stream (outFile.createOutputStream());
    if (stream == nullptr)
    {
        std::fprintf (stderr, "error: cannot write %s\n", outFile.getFullPathName().toRawUTF8());
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

    std::printf ("rendered %s at %.2f Hz -> %s (%.2fs)\n",
                 patch.name.toRawUTF8(), noteHz, outFile.getFullPathName().toRawUTF8(),
                 buffer.getNumSamples() / sampleRate);
    return 0;
}
