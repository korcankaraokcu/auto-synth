// One tool, several verbs.
//
// These were four executables, and three of them are now one because the
// arguments had stopped meaning anything on their own. `autosynth_vital
// <patch.json> <out.wav>` was an input and an output, unless `--fit` was
// passed, in which case the first became an output too, unless `--eval` was
// passed, in which case neither was read at all -- six modes sharing two
// positional slots, each bolted on after the last. A verb settles that: it says
// what the arguments are before they are read.
//
// The test suite stays its own binary. Catch2 owns a command line of its own --
// tags, filters, --list-tests -- and burying that under a subcommand would mean
// fighting it or losing it. It also needs the golden fixtures from the
// repository, so it means nothing to someone who downloaded one file.

#include "Cli.h"

#include <juce_core/juce_core.h>

#include <cstdio>

// Each verb lives with the code it drives rather than here.
int runDiff (const autosynth::cli::Args& args);
int runProbe (const autosynth::cli::Args& args);
int runVital (const juce::String& verb, const autosynth::cli::Args& args);

namespace
{

int usage()
{
    std::printf (
        "autosynth -- turn a recording into an editable Vital preset\n"
        "\n"
        "  autosynth fit <recording.wav> [--preset out.vital] [--patch out.json]\n"
        "                                [--render out.wav] [--dur s] [--gate s]\n"
        "                                [--refine-evals n] [--seed n]\n"
        "      Fit a preset to a recording. This is the one you want. Names no\n"
        "      output and it writes <recording>.vital next to the recording.\n"
        "\n"
        "  autosynth diff <target.wav> <fit.wav>\n"
        "      How far apart two recordings are, on eleven named axes.\n"
        "\n"
        "  autosynth render <patch.json> <out.wav> [--note hz] [--dur s] [--gate s]\n"
        "      Play a patch through Vital and write the audio.\n"
        "\n"
        "  autosynth probe <recording.wav> [--patch out.json] [--hop n] [--fft n]\n"
        "      Every analysis stage as JSON. No renderer, so the patch it writes\n"
        "      is unfinished by design -- use `fit` for a preset.\n"
        "\n"
        "  autosynth score <patch.json> <recording.wav>\n"
        "      What the fitting objective makes of a patch, term by term.\n"
        "\n"
        "  autosynth eval [--trials n] [--seed n]\n"
        "      The ground-truth recovery harness: how good is the fitter?\n"
        "\n"
        "  autosynth selftest <patch.json>\n"
        "      Whether rendering repeats, and which parameters move the sound.\n"
        "\n"
        "Vital must be installed for anything that makes a sound. It is hosted\n"
        "from wherever the platform keeps its VST3s; --plugin overrides that.\n");
    return 2;
}

} // namespace

int main (int argc, char* argv[])
{
    if (argc < 2)
        return usage();

    const juce::String verb { juce::CharPointer_UTF8 (argv[1]) };
    if (verb == "--help" || verb == "-h" || verb == "help")
    {
        usage();
        return 0;
    }

    // Parsed from after the verb, so a command numbers its arguments from its
    // own first one and never has to know it was dispatched to.
    const auto args = autosynth::cli::parseArgs (argc, argv, 2);

    if (verb == "diff")  return runDiff (args);
    if (verb == "probe") return runProbe (args);

    if (verb == "fit" || verb == "render" || verb == "score"
        || verb == "eval" || verb == "selftest")
        return runVital (verb, args);

    std::fprintf (stderr, "autosynth: no such command: %s\n\n", verb.toRawUTF8());
    return usage();
}
