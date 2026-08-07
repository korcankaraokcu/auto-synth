// Ground-truth recovery harness: how good is the fitter?
//
// Renders random patches from this engine, fits them back, and reports how much
// of the patch was recovered -- against a control (an untouched default patch)
// so the absolute numbers mean something.
//
// Usage:
//   autosynth_eval [--trials 24] [--seed 0] [--no-refine] [--evals 192] [--json]
//
// Refinement costs roughly eight seconds per second of audio, so a 24-trial run
// with it enabled is a couple of minutes. `--no-refine` is the quick look.

#include "eval/Recovery.h"

#include <juce_core/juce_core.h>

#include <cstdio>
#include <map>

namespace
{

struct Args
{
    std::map<juce::String, juce::String> options;

    bool has (const char* flag) const { return options.count (flag) > 0; }

    double value (const char* flag, double fallback) const
    {
        const auto it = options.find (flag);
        if (it == options.end() || it->second.isEmpty())
            return fallback;
        return it->second.getDoubleValue();
    }
};

// Parsed by hand rather than with juce::ArgumentList, which treats an option's
// value as a separate positional argument -- that silently left the sample rate
// unset in the renderer and rendered a zero-length file.
Args parseArgs (int argc, char* argv[])
{
    Args out;
    juce::StringArray raw;
    for (int i = 1; i < argc; ++i)
        raw.add (juce::String (argv[i]));

    for (int i = 0; i < raw.size(); ++i)
    {
        auto key = raw[i];
        if (! key.startsWith ("--"))
            continue;

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

    if (args.has ("--help"))
    {
        std::printf ("usage: autosynth_eval [--trials N] [--seed N] [--no-refine] "
                     "[--evals N] [--json]\n");
        return 0;
    }

    autosynth::Recovery::Options options;
    options.trials = (int) args.value ("--trials", 24.0);
    options.seed = (unsigned) args.value ("--seed", 0.0);
    options.refine = ! args.has ("--no-refine");
    options.refineEvaluations = (int) args.value ("--evals", 192.0);

    if (! args.has ("--json"))
        std::printf ("running %d trials%s...\n", options.trials,
                     options.refine ? " with refinement" : "");

    const auto summary = autosynth::Recovery::run (options);

    if (args.has ("--json"))
        std::printf ("%s\n", autosynth::Recovery::toJson (summary).toRawUTF8());
    else
        std::printf ("\n%s", autosynth::Recovery::toText (summary).toRawUTF8());

    return summary.trials > 0 ? 0 : 1;
}
