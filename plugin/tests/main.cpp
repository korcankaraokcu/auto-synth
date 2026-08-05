// Test entry point.
//
// Catch2 supplies a main of its own, but JUCE needs initialising before any
// of its singletons are touched -- the audio format manager the fixture loader
// uses among them -- so the session is driven by hand instead.

#include <catch2/catch_session.hpp>
#include <juce_events/juce_events.h>

int main (int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI juceInit;
    return Catch::Session().run (argc, argv);
}
