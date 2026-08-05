#pragma once

#include "ir/Patch.h"
#include <juce_audio_processors/juce_audio_processors.h>

namespace autosynth
{

// Host-automatable parameters, mirroring ir.PARAM_SPECS.
//
// The project's whole claim is that the output is an *editable* patch. Until
// these existed the plugin could load a fitted patch and play it, and that was
// all -- which is a resynthesiser, not the thing this is supposed to be.
//
// Every parameter in the IR is exposed, including the discrete ones. That is a
// deliberate departure from the fitter, which searches only continuous
// parameters: refinement must not be allowed to destroy a structural decision,
// but a *person* editing a patch absolutely should be able to change a waveform
// or switch an oscillator off.
//
// IDs are the IR paths with dots replaced by underscores, so they stay
// recognisable in a host's automation list and stay stable across versions.
namespace params
{

juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

// Patch -> parameter state. Used after analysis or a patch load, so the host
// and the UI see what the fitter produced.
void applyPatch (juce::AudioProcessorValueTreeState& state, const Patch& patch);

// Parameter state -> patch. Metadata that is not a parameter (`rootHz`, `name`)
// is carried over from `previous` rather than invented.
Patch toPatch (const juce::AudioProcessorValueTreeState& state, const Patch& previous);

// An IR path ("reverb.size") to its parameter id ("reverb_size"). The editor
// needs the same mapping the layout was built with, and a second copy of the
// rule would be a second thing to get wrong.
juce::String idFor (const juce::String& path);

} // namespace params
} // namespace autosynth
