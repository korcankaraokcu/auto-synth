#include "dsp/Voice.h"

#include <cmath>

namespace autosynth
{
namespace
{
// Symmetric detune spread, matching _unison_offsets in engine/synth.py.
double unisonOffsetCents (int voiceIndex, int numVoices, float spreadCents) noexcept
{
    if (numVoices <= 1)
        return 0.0;
    const auto t = static_cast<double> (voiceIndex) / static_cast<double> (numVoices - 1);
    return (t - 0.5) * static_cast<double> (spreadCents);
}
} // namespace

// --------------------------------------------------------------------------
// Voice
// --------------------------------------------------------------------------

void Voice::prepare (double sr, const WaveTables* sharedTables)
{
    sampleRate = sr;
    tables = sharedTables;
    ampEnv.prepare (sr);
    filterEnv.prepare (sr);
    for (auto& l : lfos)
        l.prepare (sr);
    filter.prepare (sr);
    sendFilter.prepare (sr);
    for (auto& osc : oscs)
    {
        osc.env.prepare (sr);
        osc.filterEnv.prepare (sr);
        osc.filter.prepare (sr);
    }
    reset();
}

void Voice::setPatch (const Patch& p)
{
    patch = p;
    ampEnv.setParameters (patch.ampEnv);
    filterEnv.setParameters (patch.filter.env);
    for (size_t i = 0; i < lfos.size(); ++i)
        lfos[i].setParameters (patch.lfos[i]);
    filter.setType (patch.filter.type);
    sendFilter.setType (patch.filter.type);

    for (size_t i = 0; i < oscs.size(); ++i)
    {
        const auto& src = patch.oscs[i];
        auto& dst = oscs[i];
        dst.numVoices = juce::jlimit (1, kMaxUnison, src.unisonVoices);
        dst.gain = src.enabled ? src.level : 0.0f;
        dst.useEnv = src.envEnabled;
        dst.env.setParameters (src.env);
        dst.useFilter = src.filterEnabled && src.filter.type != FilterType::off;
        dst.filterEnv.setParameters (src.filter.env);
        dst.filter.setType (src.filter.type);
        for (int v = 0; v < dst.numVoices; ++v)
            dst.detuneRatio[static_cast<size_t> (v)] =
                std::pow (2.0, unisonOffsetCents (v, dst.numVoices, src.unisonDetune) / 1200.0);
    }
}

void Voice::reset()
{
    active = false;
    filter.reset();
    sendFilter.reset();
    for (auto& osc : oscs)
    {
        osc.phase.fill (0.0);
        osc.filter.reset();
    }
}

float Voice::baseFrequency() const noexcept
{
    return static_cast<float> (440.0 * std::pow (2.0, (midiNote - 69) / 12.0));
}

void Voice::noteOn (int note, float vel)
{
    midiNote = note;
    velocity = juce::jlimit (0.0f, 1.0f, vel);
    active = true;

    ampEnv.noteOn();
    filterEnv.noteOn();
    for (auto& l : lfos)
        l.reset();
    filter.reset();
    sendFilter.reset();

    // Reseed the noise so a note is reproducible.
    //
    // A default-constructed juce::Random seeds itself from the clock, which
    // made every render of the same patch a different signal. That is not just
    // untidy: level calibration builds a least-squares column by rendering the
    // noise source, so the fitted oscillator levels moved from run to run, and
    // a marginal oscillator would appear or vanish depending on nothing at all.
    // It also denied CMA-ES common random numbers, leaving candidates to be
    // compared against different noise.
    //
    // Seeded from the note, so simultaneous notes still get different noise.
    noise.setSeed (0x9e3779b9LL + note);

    for (auto& osc : oscs)
    {
        osc.env.noteOn();
        osc.filterEnv.noteOn();
        osc.filter.reset();
        // All unison voices start coherent, matching the reference engine.
        //
        // Spreading them was tried -- it avoids the brief in-phase burst at the
        // attack of a wide unison, and sounds better in isolation -- but it is
        // a change to the *sound*, not to the implementation, and it broke
        // conformance by several dB. If spread phases are wanted they belong in
        // both engines, decided once and applied on both sides. A port is not
        // the place to improve the model.
        osc.phase.fill (0.0);
    }
}

void Voice::noteOff()
{
    ampEnv.noteOff();
    filterEnv.noteOff();
    for (auto& osc : oscs)
    {
        osc.env.noteOff();
        osc.filterEnv.noteOff();
    }
}

void Voice::render (float* out, float* send, int numSamples)
{
    if (! active || tables == nullptr)
        return;

    const auto noteHz = static_cast<double> (baseFrequency());

    for (int n = 0; n < numSamples; ++n)
    {
        // Every LFO is summed into its destination. Summing rather than
        // replacing is what makes two slots useful: both can target pitch for a
        // compound wobble, or split across pitch and amp for vibrato plus
        // tremolo, which one slot could never express.
        auto pitchSemis = 0.0f;
        auto ampMod = 0.0f;
        auto cutoffMod = 0.0f;
        for (size_t l = 0; l < lfos.size(); ++l)
        {
            const auto value = lfos[l].nextSample();
            switch (patch.lfos[l].dest)
            {
                case LfoDest::pitch:  pitchSemis += value * kLfoPitchSemitones; break;
                case LfoDest::amp:    ampMod += value; break;
                case LfoDest::cutoff: cutoffMod += value * kLfoCutoffOctaves; break;
                case LfoDest::none:   break;
            }
        }

        const auto ampValue = ampEnv.nextSample();
        const auto filtValue = filterEnv.nextSample();
        const auto pitchRatio = std::pow (2.0, pitchSemis / 12.0);

        float mix = 0.0f;
        // Parallel bus carrying each oscillator scaled by its reverb send,
        // tapped after that oscillator's own filter and envelope so the reverb
        // hears what it actually sounds like.
        float sendMix = 0.0f;
        for (size_t i = 0; i < oscs.size(); ++i)
        {
            auto& osc = oscs[i];
            if (osc.gain <= 1.0e-6f)
            {
                // Keep the per-oscillator modulators advancing in lockstep so a
                // level change mid-note does not jump the envelope.
                if (osc.useEnv)
                    osc.env.nextSample();
                if (osc.useFilter)
                    osc.filterEnv.nextSample();
                continue;
            }

            const auto& spec = patch.oscs[i];
            const auto tuned = noteHz
                             * std::pow (2.0, (spec.semitones + spec.cents / 100.0) / 12.0)
                             * pitchRatio;

            const auto* table = tables->tableFor (spec.waveform, spec.pulseWidth, tuned);
            // Both tables are band-limited for this note, so crossfading them
            // cannot introduce anything above Nyquist.
            const auto morph = juce::jlimit (0.0f, 1.0f, spec.waveMorph);
            const auto* tableB = morph > 1.0e-6f
                               ? tables->tableFor (spec.waveformB, spec.pulseWidth, tuned)
                               : nullptr;

            float acc = 0.0f;
            for (int v = 0; v < osc.numVoices; ++v)
            {
                const auto idx = static_cast<size_t> (v);
                const auto freq = tuned * osc.detuneRatio[idx];
                auto sample = WaveTables::lookup (table, osc.phase[idx]);
                if (tableB != nullptr)
                    sample += (WaveTables::lookup (tableB, osc.phase[idx]) - sample) * morph;
                acc += sample;
                osc.phase[idx] += freq / sampleRate;
                if (osc.phase[idx] >= 1.0)
                    osc.phase[idx] -= std::floor (osc.phase[idx]);
            }
            // Equal-power normalisation, so unison is not a second gain knob.
            acc /= std::sqrt (static_cast<float> (osc.numVoices));

            if (osc.useEnv)
                acc *= osc.env.nextSample();

            // Per-oscillator filter, before summing. Filtering the sum instead
            // would make every oscillator share one tone again, which is the
            // one thing this exists to avoid.
            if (osc.useFilter)
            {
                const auto oct = spec.filter.envAmount * osc.filterEnv.nextSample();
                const auto cut = juce::jlimit (20.0f,
                                               static_cast<float> (sampleRate) * 0.49f,
                                               spec.filter.cutoffHz * std::pow (2.0f, oct));
                osc.filter.update (cut, spec.filter.resonance);
                acc = osc.filter.processSample (acc);
            }

            // Solo/mute is monitoring, not a patch parameter, so it scales the
            // send as well -- a muted oscillator should not still be audible
            // through the reverb return.
            const auto contribution = acc * osc.gain * monitor[i];
            mix += contribution;
            sendMix += contribution * spec.reverbSend;
        }

        if (patch.noiseLevel > 1.0e-6f)
            mix += (noise.nextFloat() * 2.0f - 1.0f) * patch.noiseLevel;

        const auto ampGain = ampValue * juce::jlimit (0.0f, 2.0f, 1.0f + ampMod);
        mix *= ampGain;
        sendMix *= ampGain;

        const auto cutoffOct = patch.filter.envAmount * filtValue + cutoffMod;
        const auto cutoffHz = patch.filter.cutoffHz * std::pow (2.0f, cutoffOct);
        filter.update (cutoffHz, patch.filter.resonance);
        sendFilter.update (cutoffHz, patch.filter.resonance);
        mix = filter.processSample (mix);
        sendMix = sendFilter.processSample (sendMix);

        const auto gain = patch.masterLevel * velocity;
        out[n] += mix * gain;
        send[n] += sendMix * gain;
    }

    if (ampEnv.isFinished())
        active = false;
}

// --------------------------------------------------------------------------
// Engine
// --------------------------------------------------------------------------

void Engine::prepare (double sr, int)
{
    sampleRate = sr;
    tables.prepare (sr);
    delay.prepare (sr);
    delay.setParameters (patch.delay);
    reverb.prepare (sr);
    reverb.setParameters (patch.reverb);
    for (auto& v : voices)
    {
        v.prepare (sr, &tables);
        v.setPatch (patch);
        v.setMonitorMask (monitor);
    }
}

void Engine::setMonitorMask (const std::array<float, kNumOsc>& mask)
{
    monitor = mask;
    for (auto& v : voices)
        v.setMonitorMask (mask);
}

void Engine::setPatch (const Patch& p)
{
    patch = p;
    delay.setParameters (p.delay);
    reverb.setParameters (p.reverb);
    for (auto& v : voices)
        if (! v.isActive())
            v.setPatch (p);
}

void Engine::noteOn (int midiNote, float velocity)
{
    for (auto& v : voices)
    {
        if (! v.isActive())
        {
            v.setPatch (patch);
            v.setMonitorMask (monitor);
            v.noteOn (midiNote, velocity);
            return;
        }
    }
    // All busy: steal the first. Oldest-note stealing would be better; this is
    // adequate until polyphony is actually exercised.
    voices[0].setPatch (patch);
    voices[0].setMonitorMask (monitor);
    voices[0].noteOn (midiNote, velocity);
}

void Engine::noteOff (int midiNote)
{
    for (auto& v : voices)
        if (v.isActive() && v.getMidiNote() == midiNote)
            v.noteOff();
}

void Engine::allNotesOff()
{
    for (auto& v : voices)
        v.reset();
}

void Engine::renderBlock (float* out, int numSamples)
{
    if (static_cast<int> (sendBuffer.size()) < numSamples)
        sendBuffer.resize (static_cast<size_t> (numSamples));
    std::fill (sendBuffer.begin(), sendBuffer.begin() + numSamples, 0.0f);

    for (auto& v : voices)
        v.render (out, sendBuffer.data(), numSamples);

    // Reverb after the delay, so repeats are reverberated rather than the
    // reverse, and summed in as a return: the dry path already carries whatever
    // the sends did not take.
    for (int n = 0; n < numSamples; ++n)
        out[n] = delay.processSample (out[n]) + reverb.processSample (sendBuffer[static_cast<size_t> (n)]);
}

void Engine::render (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    buffer.clear();
    auto* out = buffer.getWritePointer (0);
    const auto total = buffer.getNumSamples();

    int position = 0;
    for (const auto meta : midi)
    {
        const auto eventTime = juce::jlimit (0, total, meta.samplePosition);
        if (eventTime > position)
        {
            renderBlock (out + position, eventTime - position);
            position = eventTime;
        }

        const auto message = meta.getMessage();
        if (message.isNoteOn())
            noteOn (message.getNoteNumber(), message.getFloatVelocity());
        else if (message.isNoteOff())
            noteOff (message.getNoteNumber());
        else if (message.isAllNotesOff() || message.isAllSoundOff())
            allNotesOff();
    }

    if (position < total)
        renderBlock (out + position, total - position);

    // Mono engine, duplicated across outputs. Stereo is a real gap -- the IR
    // has no pan or stereo width yet, so there is nothing to widen.
    for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
        buffer.copyFrom (ch, 0, buffer, 0, 0, total);
}

void Engine::renderOffline (juce::AudioBuffer<float>& out, double noteHz,
                            double durationSeconds, double gateSeconds)
{
    const auto totalSamples = static_cast<int> (std::llround (durationSeconds * sampleRate));
    const auto gateSamples = juce::jlimit (0, totalSamples,
                                           static_cast<int> (std::llround (gateSeconds * sampleRate)));
    out.setSize (1, totalSamples);
    out.clear();

    // Map the requested frequency onto the nearest MIDI note plus a detune, so
    // the offline path uses exactly the same voice code as live playback rather
    // than a parallel implementation that could drift from it.
    const auto exactNote = 69.0 + 12.0 * std::log2 (juce::jmax (noteHz, 1.0) / 440.0);
    const auto note = static_cast<int> (std::lround (exactNote));
    const auto residualCents = (exactNote - note) * 100.0;

    auto tuned = patch;
    for (auto& osc : tuned.oscs)
        osc.cents = static_cast<float> (juce::jlimit (-50.0, 50.0, osc.cents + residualCents));

    allNotesOff();
    delay.setParameters (tuned.delay);
    delay.reset();
    reverb.setParameters (tuned.reverb);
    reverb.reset();
    for (auto& v : voices)
        v.setPatch (tuned);

    auto& voice = voices[0];
    voice.noteOn (note, 1.0f);

    auto* data = out.getWritePointer (0);
    std::vector<float> send (static_cast<size_t> (juce::jmax (1, totalSamples)), 0.0f);

    if (gateSamples > 0)
        voice.render (data, send.data(), gateSamples);
    voice.noteOff();
    if (totalSamples > gateSamples)
        voice.render (data + gateSamples, send.data() + gateSamples,
                      totalSamples - gateSamples);

    // Applied over the finished buffer because this path calls the voice
    // directly rather than going through renderBlock.
    for (int n = 0; n < totalSamples; ++n)
        data[n] = delay.processSample (data[n])
                + reverb.processSample (send[static_cast<size_t> (n)]);
}

} // namespace autosynth
