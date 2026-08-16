cmake_minimum_required(VERSION 3.22)
# C is required as well as C++: JUCE compiles a few C sources and refuses to
# configure without it enabled.
project(AutoSynth VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE is fetched rather than vendored: nothing to check in, and the version is
# pinned here so a fresh clone builds the same thing. 8.0.15 is the settled 8.x
# line; JUCE has no LTS track, and 9.0.0 is too new to adopt for no gain.
include(FetchContent)
FetchContent_Declare(JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.15
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(JUCE)

# --------------------------------------------------------------------------
# Shared DSP, built once and linked into both the plugin and the headless
# renderer. The renderer is not an afterthought: it is what lets the Python
# conformance test drive this engine and compare it against the reference one.
# --------------------------------------------------------------------------
add_library(autosynth_dsp STATIC
    src/dsp/Tables.cpp
    src/dsp/Voice.cpp
    src/ir/Patch.cpp
    src/ir/VitalExport.cpp
    src/analysis/Stft.cpp
    src/analysis/Yin.cpp
    src/analysis/Partials.cpp
    src/analysis/Grouping.cpp
    src/analysis/Roles.cpp
    src/fit/Nmf.cpp
    src/fit/WaveformFit.cpp
    src/fit/WavetableFit.cpp
    src/fit/EnvelopeFit.cpp
    src/fit/FilterFit.cpp
    src/fit/Modulation.cpp
    src/fit/PartialFit.cpp
    src/fit/CmaEs.cpp
    src/fit/Refine.cpp
    src/fit/EffectsFit.cpp
    src/eval/Recovery.cpp)

target_include_directories(autosynth_dsp PUBLIC src)
target_link_libraries(autosynth_dsp
    PUBLIC juce::juce_audio_basics juce::juce_core juce::juce_dsp
    PRIVATE juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)
target_compile_definitions(autosynth_dsp PUBLIC JUCE_STANDALONE_APPLICATION=0)

# --------------------------------------------------------------------------
# The plugin
# --------------------------------------------------------------------------
juce_add_plugin(AutoSynth
    COMPANY_NAME "autosynth"
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    # Off deliberately. JUCE's auto-copy installs to C:/Program Files/Common
    # Files/VST3, which needs administrator rights -- the build otherwise
    # compiles and links successfully and then fails at the very last step with
    # a permission error, which reads like a build failure but is not one.
    # `install_plugin` below copies to the per-user folder instead.
    COPY_PLUGIN_AFTER_BUILD FALSE
    PLUGIN_MANUFACTURER_CODE Asyn
    PLUGIN_CODE Asy1
    FORMATS VST3 Standalone
    PRODUCT_NAME "auto-synth")

target_sources(AutoSynth PRIVATE
    src/PluginProcessor.cpp
    src/PluginEditor.cpp
    src/Parameters.cpp)

target_compile_definitions(AutoSynth PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0)

target_link_libraries(AutoSynth
    PRIVATE autosynth_dsp juce::juce_audio_utils
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)

# --------------------------------------------------------------------------
# Install to the per-user VST3 folder, which needs no elevation. Reaper and
# every other host scan it alongside the system one.
#     cmake --build build --config Release --target install_plugin
# --------------------------------------------------------------------------
if(WIN32)
    set(AUTOSYNTH_VST3_USER_DIR "$ENV{LOCALAPPDATA}/Programs/Common/VST3")
elseif(APPLE)
    set(AUTOSYNTH_VST3_USER_DIR "$ENV{HOME}/Library/Audio/Plug-Ins/VST3")
else()
    set(AUTOSYNTH_VST3_USER_DIR "$ENV{HOME}/.vst3")
endif()

add_custom_target(install_plugin
    COMMAND ${CMAKE_COMMAND} -E make_directory "${AUTOSYNTH_VST3_USER_DIR}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${AUTOSYNTH_VST3_USER_DIR}/auto-synth.vst3"
    # Spelled out rather than read from JUCE_PLUGIN_ARTEFACT_FILE: that
    # property holds nested generator expressions which do not get evaluated
    # when substituted here, and the copy is handed the literal "$<TARGET_..."
    # text as a path.
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_BINARY_DIR}/AutoSynth_artefacts/$<CONFIG>/VST3/auto-synth.vst3"
            "${AUTOSYNTH_VST3_USER_DIR}/auto-synth.vst3"
    COMMAND ${CMAKE_COMMAND} -E echo "installed to ${AUTOSYNTH_VST3_USER_DIR}/auto-synth.vst3"
    DEPENDS AutoSynth_VST3
    VERBATIM)

# --------------------------------------------------------------------------
# Headless renderer: patch JSON in, WAV out. Mirrors `autosynth render`.
# --------------------------------------------------------------------------
juce_add_console_app(autosynth_render PRODUCT_NAME "autosynth_render")
target_sources(autosynth_render PRIVATE tools/render_main.cpp)
target_link_libraries(autosynth_render
    PRIVATE autosynth_dsp juce::juce_audio_formats
    PUBLIC juce::juce_recommended_config_flags)

# --------------------------------------------------------------------------
# Analysis probe: WAV in, intermediates out as JSON. Analysis produces no audio
# to compare, so the port is validated by dumping each stage and diffing it
# against the Python implementation element-wise.
# --------------------------------------------------------------------------
juce_add_console_app(autosynth_probe PRODUCT_NAME "autosynth_probe")
target_sources(autosynth_probe PRIVATE tools/probe_main.cpp)
target_link_libraries(autosynth_probe
    PRIVATE autosynth_dsp juce::juce_audio_formats
    PUBLIC juce::juce_recommended_config_flags)

# --------------------------------------------------------------------------
# Fit diagnosis: why does the fit not sound like the target?
#
# A single distance number cannot say, and can mislead -- envelope distance
# rated a hissing violin better than a good clarinet. This reports the
# difference on axes a person can name and a parameter can fix.
# --------------------------------------------------------------------------
juce_add_console_app(autosynth_diff PRODUCT_NAME "autosynth_diff")
target_sources(autosynth_diff PRIVATE tools/diff_main.cpp)
target_link_libraries(autosynth_diff
    PRIVATE autosynth_dsp juce::juce_audio_formats
    PUBLIC juce::juce_recommended_config_flags)

# --------------------------------------------------------------------------
# Ground-truth recovery harness: how good is the fitter?
#
# The golden fixtures prove behaviour has not drifted. They say nothing about
# whether a fit is any *good*. This is the measurement that does, and every
# remaining item on the roadmap is a fitter change that needs it.
# --------------------------------------------------------------------------
juce_add_console_app(autosynth_eval PRODUCT_NAME "autosynth_eval")
target_sources(autosynth_eval PRIVATE tools/eval_main.cpp)
target_link_libraries(autosynth_eval
    PRIVATE autosynth_dsp juce::juce_audio_formats
    PUBLIC juce::juce_recommended_config_flags)

# --------------------------------------------------------------------------
# Tests
#
# Built on Catch2, fetched and pinned the same way JUCE is. Enabled by default
# but switchable off, because a plugin user building only the VST3 has no
# reason to pay for a test framework download.
#
#     cmake --build build --config Release --target autosynth_tests
#     ctest --test-dir build -C Release --output-on-failure
# --------------------------------------------------------------------------
option(AUTOSYNTH_BUILD_TESTS "Build the C++ test suite" ON)

if(AUTOSYNTH_BUILD_TESTS)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
    FetchContent_MakeAvailable(Catch2)

    enable_testing()

    juce_add_console_app(autosynth_tests PRODUCT_NAME "autosynth_tests")
    target_sources(autosynth_tests PRIVATE
        tests/main.cpp
        tests/test_ir.cpp
        tests/test_engine.cpp
        tests/test_analysis.cpp
        tests/test_fit.cpp
        tests/test_capabilities.cpp
        tests/test_recovery.cpp
        tests/test_unison.cpp
        tests/test_nmf.cpp
        tests/test_vital_export.cpp
        tests/test_wavetable.cpp
        tests/test_golden_engine.cpp
        tests/test_golden_analysis.cpp)

    # The fixtures are found by walking up from the executable, so the suite
    # runs from any working directory -- including ctest's, which is not the
    # source tree.
    target_compile_definitions(autosynth_tests PRIVATE
        AUTOSYNTH_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden")

    # Catch2::Catch2 rather than Catch2WithMain: tests/main.cpp drives the
    # session itself so JUCE can be initialised before any test runs.
    target_link_libraries(autosynth_tests
        PRIVATE autosynth_dsp juce::juce_audio_formats juce::juce_events Catch2::Catch2
        PUBLIC juce::juce_recommended_config_flags)

    # Register each Catch2 test case with CTest individually, so a failure
    # names the case rather than the whole binary.
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
    catch_discover_tests(autosynth_tests
        DISCOVERY_MODE PRE_TEST)
endif()
