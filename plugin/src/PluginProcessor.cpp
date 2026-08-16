#include "PluginProcessor.h"
#include "PluginEditor.h"

#include "fit/PartialFit.h"
#include "fit/Refine.h"

#include "analysis/Stft.h"

#include <juce_audio_formats/juce_audio_formats.h>

namespace autosynth
{

AutoSynthProcessor::AutoSynthProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      state (*this, nullptr, "autosynth", params::createLayout())
{
    // Every parameter is listened to. The alternative -- polling the state each
    // block -- would either miss host automation between blocks or rebuild the
    // patch constantly for nothing.
    for (const auto* parameter : getParameters())
        if (const auto* withId = dynamic_cast<const juce::AudioProcessorParameterWithID*> (parameter))
            state.addParameterListener (withId->paramID, this);

    pushPatchToParameters (engine.getPatch());
}

void AutoSynthProcessor::prepareToPlay (double sampleRate, int maxBlockSize)
{
    currentSampleRate = sampleRate;
    // Table construction allocates and runs FFTs. prepareToPlay is not
    // realtime, which is exactly why it happens here and not on note-on.
    engine.prepare (sampleRate, maxBlockSize);
}

bool AutoSynthProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo();
}

void AutoSynthProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;

    // Never block the audio thread waiting on a patch load; if the editor holds
    // the lock this block, output silence rather than glitch or stall.
    const juce::ScopedTryLock lock (patchLock);
    if (! lock.isLocked())
    {
        buffer.clear();
        return;
    }

    // Merge the on-screen keyboard into the host's MIDI before rendering, so
    // clicked notes and played notes take exactly the same path.
    keyboardState.processNextMidiBuffer (midi, 0, buffer.getNumSamples(), true);

    engine.render (buffer, midi);

    // A/B playback is mixed in on top rather than replacing the engine, so a
    // held note keeps sounding while you audition the source.
    const auto what = playing.load();
    if (what != Playing::none)
    {
        const juce::ScopedTryLock playLock (playbackLock);
        if (playLock.isLocked())
        {
            const auto& source = (what == Playing::source) ? sourceAudio : rebuiltAudio;
            if (source.getNumSamples() > 0)
            {
                // Source is stored at its own rate; step by the ratio instead of
                // resampling on load.
                const auto ratio = (what == Playing::source)
                                 ? sourceRate / juce::jmax (getSampleRate(), 1.0)
                                 : 1.0;
                const auto* data = source.getReadPointer (0);
                const auto available = source.getNumSamples();
                auto position = playPosition.load();

                for (int n = 0; n < buffer.getNumSamples(); ++n)
                {
                    const auto index = static_cast<int> (position * ratio);
                    if (index >= available - 1)
                    {
                        playing = Playing::none;
                        break;
                    }
                    const auto frac = static_cast<float> (position * ratio - index);
                    const auto sample = data[index] + (data[index + 1] - data[index]) * frac;
                    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
                        buffer.addSample (ch, n, sample);
                    ++position;
                }
                playPosition = position;
            }
            else
            {
                playing = Playing::none;
            }
        }
    }
}

void AutoSynthProcessor::setSolo (int osc, bool shouldSolo)
{
    if (juce::isPositiveAndBelow (osc, kNumOsc))
    {
        solo[static_cast<size_t> (osc)] = shouldSolo;
        applyMonitorMask();
    }
}

void AutoSynthProcessor::setMute (int osc, bool shouldMute)
{
    if (juce::isPositiveAndBelow (osc, kNumOsc))
    {
        mute[static_cast<size_t> (osc)] = shouldMute;
        applyMonitorMask();
    }
}

bool AutoSynthProcessor::isSolo (int osc) const noexcept
{
    return juce::isPositiveAndBelow (osc, kNumOsc) && solo[static_cast<size_t> (osc)];
}

bool AutoSynthProcessor::isMute (int osc) const noexcept
{
    return juce::isPositiveAndBelow (osc, kNumOsc) && mute[static_cast<size_t> (osc)];
}

void AutoSynthProcessor::applyMonitorMask()
{
    // Any solo anywhere means everything unsoloed is silent -- the usual mixer
    // convention, and the one that makes "what does osc 2 contribute" a single
    // click rather than three.
    const auto anySolo = std::any_of (solo.begin(), solo.end(), [] (bool b) { return b; });

    std::array<float, kNumOsc> mask {};
    for (size_t i = 0; i < mask.size(); ++i)
    {
        const auto audible = anySolo ? solo[i] : ! mute[i];
        mask[i] = audible ? 1.0f : 0.0f;
    }

    const juce::ScopedLock lock (patchLock);
    engine.setMonitorMask (mask);
}

void AutoSynthProcessor::startPlayback (Playing what)
{
    if (what == Playing::source && ! hasSource())
        return;
    playPosition = 0;
    playing = what;
}

void AutoSynthProcessor::stopPlayback()
{
    playing = Playing::none;
}

void AutoSynthProcessor::renderComparison()
{
    // Rendered on a private engine so the live one's voices and delay state are
    // untouched -- auditioning a comparison must not interrupt playing notes.
    const auto patch = engine.getPatch();
    const auto rate = juce::jmax (getSampleRate(), 8000.0);
    const auto seconds = hasSource()
                       ? juce::jlimit (0.25, 8.0, sourceAudio.getNumSamples() / sourceRate)
                       : 2.0;

    Engine offline;
    offline.prepare (rate, 512);
    offline.setPatch (patch);

    juce::AudioBuffer<float> rendered;
    offline.renderOffline (rendered, patch.rootHz, seconds, seconds);

    {
        const juce::ScopedLock lock (playbackLock);
        rebuiltAudio = rendered;
    }

    // Average magnitude spectra, resampled onto a log-frequency grid. Averaging
    // over frames rather than taking one keeps a transient from standing in for
    // the whole sound.
    const auto averageDb = [rate] (const juce::AudioBuffer<float>& audio,
                                   const std::vector<float>& grid)
    {
        std::vector<float> out (grid.size(), -100.0f);
        if (audio.getNumSamples() < 2048)
            return out;

        const auto spectrogram = Stft::magnitudeSpectrogram (audio.getReadPointer (0),
                                                             audio.getNumSamples(),
                                                             2048, 1024, rate);
        if (spectrogram.numFrames <= 0)
            return out;

        std::vector<double> mean (static_cast<size_t> (spectrogram.numBins), 0.0);
        for (int f = 0; f < spectrogram.numFrames; ++f)
        {
            const auto* frame = spectrogram.frame (f);
            for (int b = 0; b < spectrogram.numBins; ++b)
                mean[static_cast<size_t> (b)] += frame[b];
        }
        for (auto& v : mean)
            v /= spectrogram.numFrames;

        const auto binHz = rate / 2048.0;
        for (size_t g = 0; g < grid.size(); ++g)
        {
            const auto bin = juce::jlimit (0, spectrogram.numBins - 1,
                                           static_cast<int> (std::lround (grid[g] / binHz)));
            out[g] = static_cast<float> (20.0 * std::log10 (mean[static_cast<size_t> (bin)] + 1.0e-6));
        }
        return out;
    };

    constexpr int kPoints = 192;
    std::vector<float> grid (kPoints);
    const auto lo = std::log (30.0), hi = std::log (juce::jmin (18000.0, rate * 0.45));
    for (int i = 0; i < kPoints; ++i)
        grid[static_cast<size_t> (i)] =
            static_cast<float> (std::exp (lo + (hi - lo) * i / (kPoints - 1)));

    Spectra next;
    next.frequencies = grid;
    next.fitDb = averageDb (rendered, grid);
    if (hasSource())
        next.sourceDb = averageDb (sourceAudio, grid);
    else
        next.sourceDb.assign (grid.size(), -100.0f);
    next.valid = true;

    {
        const juce::ScopedLock lock (spectraLock);
        spectra = std::move (next);
    }
}

void AutoSynthProcessor::refreshComparison()
{
    renderComparison();
    if (onPatchChanged != nullptr)
        onPatchChanged();
}

AutoSynthProcessor::Spectra AutoSynthProcessor::getSpectra() const
{
    const juce::ScopedLock lock (spectraLock);
    return spectra;
}

double AutoSynthProcessor::getTailLengthSeconds() const
{
    const auto& p = engine.getPatch();
    auto tail = p.ampEnv.release;
    for (const auto& osc : p.oscs)
        if (osc.enabled && osc.envEnabled)
            tail = juce::jmax (tail, osc.env.release);
    return tail;
}

void AutoSynthProcessor::applyPatch (const Patch& patch)
{
    // The FFTs happen here, outside the lock, so what the audio thread is kept
    // out of is a pointer swap rather than a millisecond of table building.
    // `setPatch` still rebuilds if it has to -- it just finds nothing to do.
    Engine::FrameTableSet staged;
    Engine::buildFrameTables (patch, getSampleRate(), staged);

    const juce::ScopedLock lock (patchLock);
    engine.adoptFrameTables (std::move (staged));
    engine.setPatch (patch);
}

namespace
{
// What a frame is worth before anyone draws on it.
std::array<float, Oscillator::kFrameHarmonics> generatedProfile (const Oscillator& osc)
{
    const auto amps = WaveTables::blendedHarmonics (osc.waveform, osc.waveformB, osc.waveMorph,
                                                    osc.pulseWidth, Oscillator::kFrameHarmonics);
    std::array<float, Oscillator::kFrameHarmonics> out {};
    for (size_t k = 0; k < out.size() && k < amps.size(); ++k)
        out[k] = amps[k];
    return out;
}
} // namespace

void AutoSynthProcessor::setFrameHarmonic (int oscIndex, int frameIndex, int harmonic, float amount)
{
    if (! juce::isPositiveAndBelow (oscIndex, kNumOsc)
        || ! juce::isPositiveAndBelow (frameIndex, Oscillator::kMaxFrames)
        || ! juce::isPositiveAndBelow (harmonic, Oscillator::kFrameHarmonics))
        return;

    auto patch = getPatchSnapshot();
    auto& osc = patch.oscs[static_cast<size_t> (oscIndex)];
    auto& frame = osc.frames[static_cast<size_t> (frameIndex)];

    // Seeded from the shape it was generated from, so the first drag moves one
    // bar rather than replacing the sound with a single sine.
    if (! frame.custom)
    {
        frame.custom = true;
        frame.harmonics = generatedProfile (osc);
    }

    frame.harmonics[static_cast<size_t> (harmonic)] = juce::jlimit (0.0f, 1.0f, amount);
    applyPatch (patch);
    if (onPatchChanged != nullptr)
        onPatchChanged();
}

Patch AutoSynthProcessor::getPatchSnapshot() const
{
    const juce::ScopedLock lock (patchLock);
    return engine.getPatch();
}

void AutoSynthProcessor::pushPatchToParameters (const Patch& patch)
{
    syncing = true;
    params::applyPatch (state, patch);
    syncing = false;
}

void AutoSynthProcessor::pullPatchFromParameters()
{
    applyPatch (params::toPatch (state, engine.getPatch()));
    if (onPatchChanged != nullptr)
        onPatchChanged();
}

void AutoSynthProcessor::parameterChanged (const juce::String&, float)
{
    if (syncing.load() || analysing.load())
        return;
    pullPatchFromParameters();
}

bool AutoSynthProcessor::loadPatchFromJson (const juce::String& json, juce::String& errorOut)
{
    juce::String parseError;
    const auto patch = Patch::fromJsonString (json, &parseError);
    if (parseError.isNotEmpty())
    {
        errorOut = parseError;
        return false;
    }

    applyPatch (patch);
    pushPatchToParameters (patch);
    loadedJson = json;
    patchName = patch.name;
    if (onPatchChanged != nullptr)
        onPatchChanged();
    return true;
}

bool AutoSynthProcessor::loadPatchFromFile (const juce::File& file, juce::String& errorOut)
{
    if (! file.existsAsFile())
    {
        errorOut = "no such file: " + file.getFullPathName();
        return false;
    }
    return loadPatchFromJson (file.loadFileAsString(), errorOut);
}

juce::String AutoSynthProcessor::getLoadedPatchName() const { return patchName; }

juce::String AutoSynthProcessor::getStatus() const
{
    const juce::ScopedLock lock (statusLock);
    return status;
}

void AutoSynthProcessor::analyseFileAsync (const juce::File& file, bool refine)
{
    // One at a time. A second drop while busy is ignored rather than queued:
    // the user almost certainly wants the newest sample, and cancelling a
    // half-finished CMA-ES run cleanly is not worth the machinery.
    if (analysing.exchange (true))
        return;

    if (pool == nullptr)
        pool = std::make_unique<juce::ThreadPool> (1);

    {
        const juce::ScopedLock lock (statusLock);
        status = "analysing " + file.getFileName() + "...";
    }
    if (onPatchChanged != nullptr)
        onPatchChanged();

    pool->addJob ([this, file, refine]
    {
        juce::String message;
        juce::AudioFormatManager formats;
        formats.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (file));

        if (reader == nullptr)
        {
            message = "cannot read " + file.getFileName();
        }
        else
        {
            const auto numSamples = static_cast<int> (reader->lengthInSamples);
            juce::AudioBuffer<float> buffer (static_cast<int> (reader->numChannels),
                                             juce::jmax (1, numSamples));
            reader->read (&buffer, 0, numSamples, 0, true, true);

            // Downmix to mono, matching audio.read_wav.
            if (buffer.getNumChannels() > 1)
            {
                for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
                    buffer.addFrom (0, 0, buffer, ch, 0, numSamples);
                buffer.applyGain (0, 0, numSamples, 1.0f / buffer.getNumChannels());
            }

            {
                // Kept for A/B and the spectrum overlay. Stored before fitting
                // so a failed fit still leaves something to compare against.
                const juce::ScopedLock lock (playbackLock);
                sourceAudio.setSize (1, numSamples);
                sourceAudio.copyFrom (0, 0, buffer, 0, 0, numSamples);
                sourceRate = reader->sampleRate;
                sourceLength = numSamples;
            }

            // Analysis runs at the file's own rate. The patch is rate-agnostic,
            // so nothing needs resampling before handing it to the engine.
            PartialFit::Options options;
            auto patch = PartialFit::fit (buffer.getReadPointer (0), numSamples,
                                          reader->sampleRate, options);
            patch.name = file.getFileNameWithoutExtension();

            if (refine)
            {
                Refine::Options refineOptions;
                const auto result = Refine::run (patch, buffer.getReadPointer (0), numSamples,
                                                 reader->sampleRate, refineOptions);
                patch = result.patch;
            }

            applyPatch (patch);
            // Parameters have to follow the fit, or the knobs would still show
            // the previous patch while the engine plays the new one.
            pushPatchToParameters (patch);
            // Persist the analysed patch as state too, or reopening a session
            // would lose it -- there is no source .json to reload from.
            loadedJson = patch.toJson();
            patchName = patch.name;
            message = juce::String (patch.activeOscCount()) + " osc, root "
                    + juce::String (patch.rootHz, 1) + " Hz";
        }

        {
            const juce::ScopedLock lock (statusLock);
            status = message;
        }
        analysing = false;
        renderComparison();
        if (onPatchChanged != nullptr)
            onPatchChanged();
    });
}

int AutoSynthProcessor::getActiveOscCount() const { return engine.getPatch().activeOscCount(); }

void AutoSynthProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // The patch JSON *is* the state. Storing the text rather than a flattened
    // parameter list means a session reopens with exactly the patch that was
    // loaded, and stays readable if the IR gains fields later.
    //
    // Serialised from the *engine's* current patch, not from whatever file was
    // loaded: once parameters exist the two diverge the moment a knob moves,
    // and persisting the loaded file would silently discard every edit.
    const auto text = engine.getPatch().toJson();
    destData.replaceAll (text.toRawUTF8(), text.getNumBytesAsUTF8());
}

void AutoSynthProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (data == nullptr || sizeInBytes <= 0)
        return;
    const juce::String text (juce::CharPointer_UTF8 (static_cast<const char*> (data)),
                             static_cast<size_t> (sizeInBytes));
    juce::String error;
    loadPatchFromJson (text, error);
}

juce::AudioProcessorEditor* AutoSynthProcessor::createEditor()
{
    return new AutoSynthEditor (*this);
}

} // namespace autosynth

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new autosynth::AutoSynthProcessor();
}
