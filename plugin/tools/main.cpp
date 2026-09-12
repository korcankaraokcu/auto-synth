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
//
// The UI is in this binary too. On Windows that is a decision about the PE
// subsystem flag rather than about code: an executable is CONSOLE or it is
// WINDOWS, and there is no third option. This one is CONSOLE.
//
// WINDOWS is the worse choice. Such a binary gets no console from a terminal
// and can attach to its parent's, but the shell does not *wait* for it: it has
// no reason to think it will write anything. `autosynth fit x.wav` would return
// to the prompt immediately and then print over it, and `&&` would stop working.
// That breaks the CLI to add a UI.
//
// CONSOLE costs a console window on a double-click, which can be dismissed:
// `GetConsoleProcessList` reports one process only when Windows allocated the
// console for us, which is what a double-click does and what running from a
// terminal does not. When we own it the UI frees it; when we do not, it is the
// user's terminal and is left alone.

#include "Cli.h"

#include <juce_core/juce_core.h>

#include <cstdio>

#if JUCE_WINDOWS
 #include <windows.h>
#endif

// Set by CMake from the project version. The fallback is for a build that
// somehow bypassed it, and says so rather than claiming a number.
#ifndef AUTOSYNTH_VERSION
 #define AUTOSYNTH_VERSION "unknown"
#endif

// Each verb lives with the code it drives rather than here.
int runDiff (const autosynth::cli::Args& args);
int runProbe (const autosynth::cli::Args& args);
int runGui (const autosynth::cli::Args& args);
int runVital (const juce::String& verb, const autosynth::cli::Args& args);

namespace
{

// Explorer hands us the file path where a verb would go when a recording is
// dropped on the exe. That is not a typo, so it is not treated as one.
bool looksLikeARecording (const juce::String& argument)
{
    const juce::File file (argument);
    return file.existsAsFile() && file.hasFileExtension ("wav;aiff;aif;flac");
}

// Did Windows make this console for us, or are we running inside an existing one?
bool ownsItsConsole()
{
   #if JUCE_WINDOWS
    DWORD processes[2] = {};
    return GetConsoleProcessList (processes, 2) == 1;
   #else
    return false;
   #endif
}

// Nothing is written to it and nothing reads it: a window with a dead black
// rectangle behind it looks broken.
void releaseConsole()
{
   #if JUCE_WINDOWS
    if (ownsItsConsole())
        FreeConsole();
   #endif
}

int usage()
{
    std::printf (
        "autosynth " AUTOSYNTH_VERSION " -- turn a recording into an editable Vital preset\n"
        "\n"
        "  autosynth fit <recording.wav> [--preset out.vital] [--patch out.json]\n"
        "                                [--render out.wav] [--dur s] [--gate s]\n"
        "                                [--refine-evals n] [--seed n]\n"
        "      Fit a preset to a recording. This is the main one. With no output\n"
        "      named it writes <recording>.vital next to the recording.\n"
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
        "  autosynth gui\n"
        "      The same fit in a window: drop a recording, see it against the\n"
        "      original. Opens by itself when launched without a terminal.\n"
        "\n"
        "Vital must be installed for anything that makes a sound -- get it from\n"
        "vital.audio, the free version is enough. It is hosted from wherever the\n"
        "platform keeps its VST3s; --plugin overrides that. `diff` and `probe`\n"
        "make no sound and work without it.\n");
    return 2;
}

} // namespace

int main (int argc, char* argv[])
{
    // No arguments and no terminal means a double-click, and a double-click is
    // someone wanting to convert something, not a list of verbs with nowhere to
    // type them. No arguments *in* a terminal is a question about what this is,
    // which the usage text answers.
    if (argc < 2)
    {
        if (! ownsItsConsole())
            return usage();

        releaseConsole();
        return runGui ({});
    }

    const juce::String verb { juce::CharPointer_UTF8 (argv[1]) };
    if (verb == "--help" || verb == "-h" || verb == "help")
    {
        usage();
        return 0;
    }

    // Asked on its own rather than only sitting in the usage text, because the
    // question "which build is this" comes up against a binary somebody
    // downloaded months ago, and reading it out of the file properties is not
    // something a script can do.
    if (verb == "--version" || verb == "-v" || verb == "version")
    {
        std::printf ("autosynth %s\n", AUTOSYNTH_VERSION);
        return 0;
    }

    // Parsed from after the verb, so a command numbers its arguments from its
    // own first one and never has to know it was dispatched to.
    const auto args = autosynth::cli::parseArgs (argc, argv, 2);

    if (verb == "diff")  return runDiff (args);
    if (verb == "probe") return runProbe (args);

    if (verb == "gui")
    {
        releaseConsole();
        return runGui (args);
    }

    if (verb == "fit" || verb == "render" || verb == "score"
        || verb == "eval" || verb == "selftest")
        return runVital (verb, args);

    // A recording where a verb should be, which is what Explorer sends when one
    // is dropped on the exe. From a double-click that opens the UI on it; typed
    // into a terminal, saying so is more use than a list of verbs.
    if (looksLikeARecording (verb))
    {
        if (ownsItsConsole())
        {
            releaseConsole();
            return runGui (args.withPositional (verb));
        }

        std::fprintf (stderr, "autosynth: %s is a recording, not a command."
                              " Did you mean:\n\n    autosynth fit %s\n\n",
                      verb.toRawUTF8(), verb.toRawUTF8());
        return 2;
    }

    std::fprintf (stderr, "autosynth: no such command: %s\n\n", verb.toRawUTF8());
    return usage();
}
