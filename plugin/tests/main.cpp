// Test entry point.
//
// Catch2 supplies a main of its own, but two things must happen before any test
// runs. JUCE needs initialising before its singletons are touched -- the audio
// format manager the fixture loader uses among them, and the message manager
// the VST3 host scans on. And Vital is opened once here rather than on first
// use, so the cost is paid once and its absence is reported once.
//
// The host lives here rather than in a static so that it is torn down while
// JUCE is still standing: declared after the initialiser, it goes first. Held
// in a static instead, every test passes and the process then exits with an
// access violation -- a green suite and a red exit code.

#include "Helpers.h"

#include <catch2/catch_session.hpp>
#include <juce_events/juce_events.h>

#include <cstdio>

int main (int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI juceInit;
    autosynth::VitalHost host;
    autotest::vitalHost() = &host;

    juce::String error;
    if (! host.open (autotest::kSampleRate, error))
        std::fprintf (stderr,
                      "warning: %s -- every case that renders will be skipped\n",
                      error.toRawUTF8());

    const auto result = Catch::Session().run (argc, argv);
    autotest::vitalHost() = nullptr;
    return result;
}
