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
# Analysis, fitting and the exporter, built once and linked into every tool.
#
# There is no synth in here. This project writes Vital presets, so the synth is
# Vital: every render goes through the installed plug-in (src/vital/VitalHost.h),
# which is why the fitter and the harness take a renderer rather than owning
# one. Anything that renders therefore links juce_audio_processors too.
# --------------------------------------------------------------------------
add_library(autosynth_dsp STATIC
    src/dsp/Tables.cpp
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

# The switches that turn JUCE into a plug-in host, shared with the test suite
# because it renders the same way.
set(AUTOSYNTH_HOST_DEFINITIONS
    JUCE_PLUGINHOST_VST3=1
    JUCE_PLUGINHOST_VST=0
    JUCE_PLUGINHOST_AU=0
    JUCE_PLUGINHOST_LV2=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0)

# --------------------------------------------------------------------------
# Render, fit and evaluate through Vital itself.
#
# This hosts the installed VST3 and hands it the exported JSON directly --
# Vital's plug-in state chunk is the preset format -- so what it renders is the
# preset a user would open, not a re-encoding of it. That makes it the renderer
# for everything that needs sound: `--fit` refines against it and `--eval` runs
# the ground-truth recovery harness through it.
#
# It links no Vital code and ships none: the plug-in is whatever the user has
# installed, which is also the version their presets will be opened in.
# --------------------------------------------------------------------------
juce_add_console_app(autosynth_vital PRODUCT_NAME "autosynth_vital")
target_sources(autosynth_vital PRIVATE tools/vital_main.cpp)
target_compile_definitions(autosynth_vital PRIVATE ${AUTOSYNTH_HOST_DEFINITIONS})
target_link_libraries(autosynth_vital
    PRIVATE autosynth_dsp
            juce::juce_audio_processors
            juce::juce_audio_formats
            juce::juce_gui_basics
            juce::juce_events
    PUBLIC juce::juce_recommended_config_flags)

# --------------------------------------------------------------------------
# Tests
#
# Built on Catch2, fetched and pinned the same way JUCE is. Enabled by default
# but switchable off, so a build that only wants the tools does not pay for a
# test framework download.
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
        tests/test_analysis.cpp
        tests/test_fit.cpp
        tests/test_capabilities.cpp
        tests/test_recovery.cpp
        tests/test_unison.cpp
        tests/test_nmf.cpp
        tests/test_vital_export.cpp
        tests/test_wavetable.cpp
        tests/test_golden_analysis.cpp)

    # The suite renders through Vital, so it hosts a plug-in like the tool does.
    # A machine without Vital installed cannot run the audible half of it: those
    # cases are skipped with a note rather than silently passing.
    target_compile_definitions(autosynth_tests PRIVATE ${AUTOSYNTH_HOST_DEFINITIONS})

    # The fixtures are found by walking up from the executable, so the suite
    # runs from any working directory -- including ctest's, which is not the
    # source tree.
    target_compile_definitions(autosynth_tests PRIVATE
        AUTOSYNTH_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden")

    # Catch2::Catch2 rather than Catch2WithMain: tests/main.cpp drives the
    # session itself so JUCE can be initialised before any test runs.
    target_link_libraries(autosynth_tests
        PRIVATE autosynth_dsp
                juce::juce_audio_processors
                juce::juce_audio_formats
                juce::juce_gui_basics
                juce::juce_events
                Catch2::Catch2
        PUBLIC juce::juce_recommended_config_flags)

    # Register each Catch2 test case with CTest individually, so a failure
    # names the case rather than the whole binary.
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
    catch_discover_tests(autosynth_tests
        DISCOVERY_MODE PRE_TEST)
endif()
