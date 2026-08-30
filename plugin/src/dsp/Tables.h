#pragma once

#include "ir/Patch.h"

#include <vector>

namespace autosynth
{

// The harmonic series each named waveform stands for, matching
// engine/tables.py -- so a "saw" here holds the same spectrum it held there.
//
// Only the series, and no way to play it. Vital plays; this project decides
// what Vital should play, and both of the places that decide -- the waveform
// fitter, which scores an observed profile against these shapes, and the
// exporter, which writes one into a Vital wavetable -- need the definition
// rather than a rendered table. Sharing the arithmetic is the only way to
// guarantee they agree on what "saw" means.
class WaveTables
{
public:
    // `pulseWidth` is ignored unless the waveform is pulse. `cosinePhase` says
    // whether the series is a cosine one, which only the pulse is.
    static void harmonicAmplitudes (Waveform waveform, int numHarmonics, float pulseWidth,
                                    std::vector<float>& amps, bool& cosinePhase);

    // The blend of two fixed shapes, peak-normalised. One definition, used by
    // the fitter that scores against it and the exporter that writes it out.
    static std::vector<float> blendedHarmonics (Waveform a, Waveform b, float morph,
                                                float pulseWidth, int numHarmonics);
};

} // namespace autosynth
