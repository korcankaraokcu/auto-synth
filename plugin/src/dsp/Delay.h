#pragma once

#include "ir/Patch.h"
#include <vector>

namespace autosynth
{

// Feedback delay line, matching engine/delay.py.
//
// The Python side updates a whole block of `d` samples at once, because samples
// less than `d` apart never interact. This one runs sample by sample. Both
// evaluate the identical recurrence ``y[n] += f * y[n-d]``, so unlike the
// filter -- where the two engines genuinely differ -- the delay is expected to
// agree to floating-point noise, and the conformance test holds it there.
class Delay
{
public:
    void prepare (double sr)
    {
        sampleRate = sr;
        // One second of headroom covers the IR's maximum delay time. Sized once
        // in prepare so nothing allocates on the audio thread.
        buffer.assign (static_cast<size_t> (sr) + 4, 0.0f);
        reset();
    }

    void reset() noexcept
    {
        std::fill (buffer.begin(), buffer.end(), 0.0f);
        writeIndex = 0;
    }

    void setParameters (const DelayParams& p) noexcept { params = p; }

    float processSample (float x) noexcept
    {
        if (! params.enabled || params.mix <= 1.0e-6f || buffer.empty())
            return x;

        const auto size = static_cast<int> (buffer.size());
        const auto d = juce::jlimit (1, size - 1,
                                     static_cast<int> (std::lround (params.time * sampleRate)));

        auto readIndex = writeIndex - d;
        if (readIndex < 0)
            readIndex += size;

        const auto delayed = buffer[static_cast<size_t> (readIndex)];
        const auto wet = x + juce::jlimit (0.0f, 0.95f, params.feedback) * delayed;

        buffer[static_cast<size_t> (writeIndex)] = wet;
        if (++writeIndex >= size)
            writeIndex = 0;

        const auto m = juce::jlimit (0.0f, 1.0f, params.mix);
        return x * (1.0f - m) + wet * m;
    }

private:
    DelayParams params {};
    std::vector<float> buffer;
    double sampleRate = 44100.0;
    int writeIndex = 0;
};

} // namespace autosynth
