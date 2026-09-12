#pragma once

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <map>
#include <vector>

namespace autosynth::cli
{

// One command line, parsed once, shared by every verb.
//
// Hand-parsed because juce::ArgumentList treats an option's value as a separate
// positional argument, which is exactly the distinction a verb needs to make.
struct Args
{
    std::map<juce::String, juce::String> options;
    juce::StringArray positional;

    bool has (const char* flag) const
    {
        return options.count (flag) > 0;
    }

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
        return it == options.end() || it->second.isEmpty() ? fallback : it->second;
    }

    // Positional arguments are numbered from after the verb, so a command reads
    // its own arguments without knowing it was dispatched to.
    juce::String at (int index, const juce::String& fallback = {}) const
    {
        return index < positional.size() ? positional[index] : fallback;
    }

    juce::File file (int index, const juce::String& fallback = {}) const
    {
        return juce::File::getCurrentWorkingDirectory().getChildFile (at (index, fallback));
    }

    juce::File fileFor (const char* flag) const
    {
        return juce::File::getCurrentWorkingDirectory().getChildFile (text (flag));
    }

    // A copy with one more argument in front, for the case where what looked
    // like a verb turns out to be one of the arguments.
    Args withPositional (const juce::String& first) const
    {
        Args out = *this;
        out.positional.insert (0, first);
        return out;
    }
};

// `from` skips the program name and the verb.
inline Args parseArgs (int argc, char* argv[], int from)
{
    Args out;
    std::vector<juce::String> raw;
    for (int i = from; i < argc; ++i)
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

// Mono samples from a WAV, downmixed the same way analysis reads them, so that
// what a verb measures is what the fitter saw.
inline std::vector<float> readMono (const juce::File& file, double& sampleRateOut)
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

inline bool writeWav (const juce::File& file, const std::vector<float>& samples, double sampleRate)
{
    juce::AudioBuffer<float> buffer (1, (int) samples.size());
    buffer.copyFrom (0, 0, samples.data(), (int) samples.size());

    // Peak-limit rather than normalise: the absolute level is part of what a
    // comparison is checking.
    auto peak = 0.0f;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
        peak = juce::jmax (peak, std::abs (buffer.getSample (0, i)));
    if (peak > 1.0f)
        buffer.applyGain (1.0f / peak);

    file.deleteFile();
    std::unique_ptr<juce::FileOutputStream> stream (file.createOutputStream());
    if (stream == nullptr)
        return false;

    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatWriter> writer (
        wav.createWriterFor (stream.get(), sampleRate, 1, 24, {}, 0));
    if (writer == nullptr)
        return false;

    stream.release();   // the writer owns it now
    writer->writeFromAudioSampleBuffer (buffer, 0, buffer.getNumSamples());
    return true;
}

} // namespace autosynth::cli
