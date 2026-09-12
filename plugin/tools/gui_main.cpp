// The UI: drop a recording, get a preset.
//
// It calls `job::fit`, the same fit the `fit` verb runs, so there is only one
// order of operations to maintain. It adds two things the command line cannot:
// somewhere to drop a file, and a picture of how close the fit came.
//
// It is not a synth editor. No keyboard and no knobs -- the preset is the
// output, Vital is where it gets edited, and a second set of controls here
// would invite editing the copy that is thrown away. It shows a comparison
// instead: the two loudness contours over each other, and the same eleven axes
// `diff` prints, from the same measurement.
//
// Hosting: the plug-in is opened on the message thread, once, after the window
// is up. Rendering then happens on the worker. Opening it up front would leave
// a double-click showing nothing for a second or two, and opening it on the
// worker would mean instantiating a VST3 off the message thread, which is not
// something to rely on.

#include "Cli.h"
#include "FitJob.h"

#include "ir/VitalExport.h"

#include <juce_gui_basics/juce_gui_basics.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>

// Set by CMake from the project version, the same one the command line prints
// and Windows shows in the file properties.
#ifndef AUTOSYNTH_VERSION
 #define AUTOSYNTH_VERSION "unknown"
#endif

namespace
{

using autosynth::cli::Args;

constexpr int kMargin = 18;

const juce::Colour kBackground  { 0xff16181c };
const juce::Colour kPanel       { 0xff1e2127 };
const juce::Colour kEdge        { 0xff2c3039 };
const juce::Colour kText        { 0xffd8dce3 };
const juce::Colour kDim         { 0xff858c99 };
const juce::Colour kTargetLine  { 0xff6fb7ff };
const juce::Colour kFitLine     { 0xffffb454 };
const juce::Colour kOk          { 0xff63c98a };
const juce::Colour kOff         { 0xffe0704f };

juce::Font uiFont (float height, bool bold = false)
{
    return juce::Font (juce::FontOptions (height)
                           .withStyle (bold ? "Bold" : "Regular"));
}

// --- the picture -----------------------------------------------------------

// Both notes' loudness contours, over each other, in decibels below the
// recording's own peak.
//
// In decibels and against a *shared* reference on purpose. Every axis in the
// table is shape-relative -- each signal normalised by its own peak before
// being compared -- and that is exactly how a preset once lost two thirds of
// its level after the attack while the whole diagnostic read fine. Drawn this
// way, a fit that is quiet looks quiet.
class ContourPlot : public juce::Component
{
public:
    void setContours (const autosynth::Diagnose::Measurements& targetIn,
                      const autosynth::Diagnose::Measurements& fitIn)
    {
        target = targetIn;
        fit = fitIn;
        hasData = ! target.loudness.empty();
        repaint();
    }

    void clear()
    {
        hasData = false;
        repaint();
    }

    void paint (juce::Graphics& g) override
    {
        auto area = getLocalBounds().toFloat();
        g.setColour (kPanel);
        g.fillRoundedRectangle (area, 6.0f);
        g.setColour (kEdge);
        g.drawRoundedRectangle (area.reduced (0.5f), 6.0f, 1.0f);

        const auto plot = area.reduced (44.0f, 26.0f);

        if (! hasData)
        {
            g.setColour (kDim);
            g.setFont (uiFont (13.0f));
            g.drawText ("loudness over time, once there is something to compare",
                        area, juce::Justification::centred);
            return;
        }

        const auto seconds = juce::jmax (duration (target), duration (fit));
        if (seconds <= 0.0)
            return;

        // Both against the recording's peak, so the two curves share a zero.
        const auto reference = peakOf (target);
        const auto floorDb = floorFor (target, reference);
        const auto step = floorDb > -24.0 ? 3.0 : (floorDb > -48.0 ? 6.0 : 12.0);

        g.setFont (uiFont (10.0f));
        for (double db = 0.0; db >= floorDb; db -= step)
        {
            const auto y = plot.getY() + (float) (db / floorDb) * plot.getHeight();
            g.setColour (kEdge);
            g.drawHorizontalLine ((int) y, plot.getX(), plot.getRight());
            g.setColour (kDim);
            g.drawText (juce::String ((int) db) + " dB",
                        juce::Rectangle<float> (area.getX() + 4.0f, y - 7.0f, 38.0f, 14.0f),
                        juce::Justification::centredRight);
        }

        g.setColour (kDim);
        g.drawText (juce::String (seconds, 2) + " s",
                    juce::Rectangle<float> (plot.getRight() - 60.0f, plot.getBottom() + 4.0f,
                                            60.0f, 16.0f),
                    juce::Justification::centredRight);

        g.setColour (kTargetLine);
        g.strokePath (contour (target, reference, floorDb, seconds, plot),
                      juce::PathStrokeType (1.6f));
        g.setColour (kFitLine);
        g.strokePath (contour (fit, reference, floorDb, seconds, plot),
                      juce::PathStrokeType (1.6f));

        g.setFont (uiFont (11.0f));
        g.setColour (kTargetLine);
        g.drawText ("recording", plot.withHeight (16.0f).translated (2.0f, -20.0f),
                    juce::Justification::centredLeft);
        g.setColour (kFitLine);
        g.drawText ("fit", plot.withHeight (16.0f).translated (76.0f, -20.0f),
                    juce::Justification::centredLeft);
    }

private:
    static double duration (const autosynth::Diagnose::Measurements& m)
    {
        return m.loudness.size() * m.secondsPerFrame;
    }

    static float peakOf (const autosynth::Diagnose::Measurements& m)
    {
        auto peak = 0.0f;
        for (auto v : m.loudness)
            peak = juce::jmax (peak, v);
        return juce::jmax (peak, 1.0e-9f);
    }

    // How far down the scale goes, from what is actually in the recording.
    //
    // A fixed floor is wrong for both shapes this gets given. At -48 dB a steady
    // tone is a flat line across the top, and its tremolo -- one of the axes
    // below, and one of the weaker ones -- is a few pixels of wobble. Scaled to
    // the note's own range, what is being compared fills the box.
    //
    // From a percentile rather than the minimum, because every recording ends in
    // silence: the minimum is the noise floor it was recorded at, not a
    // property of the note.
    static double floorFor (const autosynth::Diagnose::Measurements& m, float reference)
    {
        std::vector<double> db;
        db.reserve (m.loudness.size());
        for (auto v : m.loudness)
            db.push_back (20.0 * std::log10 (juce::jmax (v, 1.0e-6f) / reference));

        if (db.empty())
            return -48.0;

        std::sort (db.begin(), db.end());
        const auto low = db[(size_t) (0.05 * (db.size() - 1))];
        return juce::jlimit (-60.0, -12.0, std::floor ((low - 6.0) / 3.0) * 3.0);
    }

    static juce::Path contour (const autosynth::Diagnose::Measurements& m, float reference,
                               double floorDb, double seconds, juce::Rectangle<float> plot)
    {
        // Tracked rather than asked, because juce::Path::isEmpty() counts only
        // drawing elements: a path holding one startNewSubPath still reports
        // empty, so asking it per point produces a path of moves and no lines,
        // which strokes to nothing at all.
        juce::Path path;
        auto started = false;

        for (size_t i = 0; i < m.loudness.size(); ++i)
        {
            const auto t = i * m.secondsPerFrame;
            const auto db = 20.0 * std::log10 (juce::jmax (m.loudness[i], 1.0e-6f) / reference);
            const auto x = plot.getX() + (float) (t / seconds) * plot.getWidth();
            const auto y = plot.getY()
                         + (float) (juce::jlimit (floorDb, 0.0, db) / floorDb)
                               * plot.getHeight();

            if (! std::exchange (started, true))
                path.startNewSubPath (x, y);
            else
                path.lineTo (x, y);
        }
        return path;
    }

    autosynth::Diagnose::Measurements target, fit;
    bool hasData = false;
};

// --- the verdicts ----------------------------------------------------------

// The eleven axes, exactly as `diff` prints them, from the same comparison.
class AxisTable : public juce::Component
{
public:
    void setAxes (std::vector<autosynth::Diagnose::Axis> newAxes)
    {
        axes = std::move (newAxes);
        repaint();
    }

    void clear()
    {
        axes.clear();
        repaint();
    }

    int preferredHeight() const
    {
        return kHeaderHeight + (int) juce::jmax<size_t> (axes.size(), 11) * kRowHeight + 8;
    }

    void paint (juce::Graphics& g) override
    {
        auto area = getLocalBounds().toFloat();
        g.setColour (kPanel);
        g.fillRoundedRectangle (area, 6.0f);
        g.setColour (kEdge);
        g.drawRoundedRectangle (area.reduced (0.5f), 6.0f, 1.0f);

        if (axes.empty())
        {
            g.setColour (kDim);
            g.setFont (uiFont (13.0f));
            g.drawText ("eleven named axes, once there is a fit to measure",
                        area, juce::Justification::centred);
            return;
        }

        auto rows = getLocalBounds().reduced (14, 8);
        const auto nameWidth = juce::jmax (120, rows.getWidth() * 22 / 100);
        const auto valueWidth = juce::jmax (90, rows.getWidth() * 20 / 100);

        auto header = rows.removeFromTop (kHeaderHeight);
        g.setColour (kDim);
        g.setFont (uiFont (11.0f));
        header.removeFromLeft (nameWidth);
        g.drawText ("recording", header.removeFromLeft (valueWidth),
                    juce::Justification::centredLeft);
        g.drawText ("fit", header.removeFromLeft (valueWidth),
                    juce::Justification::centredLeft);
        g.drawText ("verdict", header, juce::Justification::centredLeft);

        for (const auto& axis : axes)
        {
            auto row = rows.removeFromTop (kRowHeight);

            g.setColour (kText);
            g.setFont (uiFont (12.0f));
            g.drawText (axis.name, row.removeFromLeft (nameWidth),
                        juce::Justification::centredLeft);

            g.setColour (kDim);
            g.drawText (axis.target, row.removeFromLeft (valueWidth),
                        juce::Justification::centredLeft);
            g.setColour (kText);
            g.drawText (axis.fit, row.removeFromLeft (valueWidth),
                        juce::Justification::centredLeft);

            g.setColour (axis.ok ? kOk : kOff);
            g.setFont (uiFont (12.0f, ! axis.ok));
            g.drawText (axis.verdict, row, juce::Justification::centredLeft);
        }
    }

private:
    static constexpr int kHeaderHeight = 20;
    static constexpr int kRowHeight = 20;

    std::vector<autosynth::Diagnose::Axis> axes;
};

// --- the work --------------------------------------------------------------

// Everything the UI needs back from one fit, in one object, so it crosses to
// the message thread in a single hand-over rather than as separate fields the
// timer has to poll.
struct Result
{
    autosynth::Patch patch;
    juce::String presetJson;
    double seconds = 0.0;
    autosynth::job::Comparison comparison;
};

class FitThread : public juce::Thread
{
public:
    using Done = std::function<void (Result)>;

    FitThread (autosynth::VitalHost& hostToUse,
               std::vector<float> targetIn, double sampleRateIn,
               autosynth::job::Options optionsIn, Done doneIn)
        : juce::Thread ("autosynth fit"),
          host (hostToUse),
          target (std::move (targetIn)),
          sampleRate (sampleRateIn),
          options (optionsIn),
          done (std::move (doneIn))
    {
    }

    ~FitThread() override
    {
        cancel();
        stopThread (8000);
    }

    void cancel() { cancelled = true; }

    void setStage (double f, const juce::String& s)
    {
        fraction = f;
        const juce::ScopedLock lock (stageLock);
        stage = s;
    }

    // Read from the message thread by the timer, written here. Atomics rather
    // than a lock because nothing here is worth blocking a repaint for.
    std::atomic<double> fraction { 0.0 };
    juce::String stageName() const
    {
        const juce::ScopedLock lock (stageLock);
        return stage;
    }

    void run() override
    {
        const auto fitted = autosynth::job::fit (
            host, target, sampleRate, options,
            [this] (double f, const juce::String& s) { setStage (f, s); },
            &cancelled);

        if (fitted.cancelled || cancelled.load())
            return;

        setStage (0.95, "rendering");
        Result result;
        result.patch = fitted.patch;
        result.presetJson = autosynth::VitalExport::toJson (fitted.patch, fitted.patch.name);
        result.seconds = fitted.seconds;

        // The same duration the recording has, so the picture is two notes of
        // the same length rather than one cut short.
        result.comparison = autosynth::job::audition (
            host, fitted.patch, target, sampleRate,
            juce::jmax (1.0, target.size() / sampleRate), fitted.gateSeconds);

        if (cancelled.load())
            return;

        setStage (1.0, "done");
        juce::MessageManager::callAsync (
            [callback = done, r = std::move (result)]() mutable { callback (std::move (r)); });
    }

private:
    autosynth::VitalHost& host;
    std::vector<float> target;
    double sampleRate = 0.0;
    autosynth::job::Options options;
    Done done;

    std::atomic<bool> cancelled { false };
    mutable juce::CriticalSection stageLock;
    juce::String stage { "starting" };
};

// --- the window ------------------------------------------------------------

class MainComponent : public juce::Component,
                      public juce::FileDragAndDropTarget,
                      private juce::Timer
{
public:
    MainComponent (juce::String pluginOverrideIn, juce::File startWith)
        : pluginOverride (std::move (pluginOverrideIn)), waiting (std::move (startWith))
    {
        addAndMakeVisible (plot);
        addAndMakeVisible (table);

        choose.setButtonText ("Choose a recording...");
        choose.onClick = [this] { browse(); };
        addAndMakeVisible (choose);

        saveAs.setButtonText ("Save preset as...");
        saveAs.setEnabled (false);
        saveAs.onClick = [this] { browseForSave(); };
        addAndMakeVisible (saveAs);

        cancelButton.setButtonText ("Cancel");
        cancelButton.onClick = [this] { cancelFit(); };
        addChildComponent (cancelButton);

        progress = std::make_unique<juce::ProgressBar> (progressValue);
        addChildComponent (*progress);

        setOpaque (true);
        setSize (940, 680);

        // The window first, the plug-in scan a moment later, so the scan is not
        // the first thing a double-click shows.
        juce::Timer::callAfterDelay (80, [safe = juce::Component::SafePointer<MainComponent> (this)]
                                         {
                                             if (safe != nullptr)
                                                 safe->openHost();
                                         });
    }

    ~MainComponent() override
    {
        stopTimer();
        worker.reset();
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (kBackground);

        auto area = getLocalBounds().reduced (kMargin);

        auto header = area.removeFromTop (kHeaderHeight);
        auto titleRow = header.removeFromTop (28);

        // Right of the title, so "which build is this" is answerable without a
        // terminal -- the same string `autosynth --version` prints.
        g.setColour (kDim);
        g.setFont (uiFont (12.0f));
        g.drawText (AUTOSYNTH_VERSION, titleRow, juce::Justification::bottomRight);

        g.setColour (kText);
        g.setFont (uiFont (22.0f, true));
        g.drawText ("AutoSynth", titleRow, juce::Justification::topLeft);

        g.setColour (kDim);
        g.setFont (uiFont (12.0f));
        g.drawText ("Convert a recording into a Vital preset",
                    header, juce::Justification::topLeft);

        area.removeFromTop (8);

        // The drop zone, which is also where the status lives: one place to
        // look, whether it is asking for a file or telling you what happened.
        auto zone = area.removeFromTop (kZoneHeight).toFloat();
        g.setColour (dragging ? kEdge.brighter (0.25f) : kPanel);
        g.fillRoundedRectangle (zone, 6.0f);
        g.setColour (dragging ? kTargetLine : kEdge);
        // Dashed, so the box reads as somewhere to put something rather than as
        // a panel. Into a second path, because createDashedStroke reads its
        // source while writing its destination.
        const float dashes[] = { 6.0f, 4.0f };
        juce::Path outline, dashed;
        outline.addRoundedRectangle (zone.reduced (0.5f), 6.0f);
        juce::PathStrokeType (1.0f).createDashedStroke (dashed, outline, dashes, 2);
        g.strokePath (dashed, juce::PathStrokeType (dragging ? 1.8f : 1.0f));

        g.setColour (status.isError ? kOff : kText);
        g.setFont (uiFont (15.0f));
        g.drawText (status.headline, zone.withTrimmedBottom (zone.getHeight() * 0.42f),
                    juce::Justification::centredBottom);

        g.setColour (kDim);
        g.setFont (uiFont (12.0f));
        g.drawText (status.detail, zone.withTrimmedTop (zone.getHeight() * 0.58f),
                    juce::Justification::centredTop);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (kMargin);
        area.removeFromTop (kHeaderHeight + 8);

        auto zone = area.removeFromTop (kZoneHeight);
        auto bar = zone.removeFromBottom (26).reduced (60, 6);
        progress->setBounds (bar);

        area.removeFromTop (12);

        auto buttons = area.removeFromBottom (34);
        choose.setBounds (buttons.removeFromLeft (170));
        buttons.removeFromLeft (10);
        saveAs.setBounds (buttons.removeFromLeft (170));
        buttons.removeFromLeft (10);
        cancelButton.setBounds (buttons.removeFromLeft (110));

        area.removeFromBottom (12);

        // The table gets what it needs and the plot takes the rest: eleven axes
        // with two of them cut off is worse than a shorter picture.
        table.setBounds (area.removeFromBottom (
            juce::jlimit (0, table.preferredHeight(), area.getHeight() - kMinPlotHeight)));
        area.removeFromBottom (12);
        plot.setBounds (area);
    }

    // --- dropping ---

    bool isInterestedInFileDrag (const juce::StringArray& files) override
    {
        return ready() && files.size() == 1 && isAudio (files[0]);
    }

    void fileDragEnter (const juce::StringArray&, int, int) override
    {
        dragging = true;
        repaint();
    }

    void fileDragExit (const juce::StringArray&) override
    {
        dragging = false;
        repaint();
    }

    void filesDropped (const juce::StringArray& files, int, int) override
    {
        dragging = false;
        repaint();
        if (! files.isEmpty())
            startFit (juce::File (files[0]));
    }

private:
    struct Status
    {
        juce::String headline;
        juce::String detail;
        bool isError = false;
    };

    static constexpr int kHeaderHeight = 54;
    static constexpr int kZoneHeight = 116;
    static constexpr int kMinPlotHeight = 150;

    static bool isAudio (const juce::String& path)
    {
        return juce::File (path).hasFileExtension ("wav;aiff;aif;flac");
    }

    bool ready() const { return host.isOpen() && worker == nullptr; }

    void setStatus (juce::String headline, juce::String detail, bool isError = false)
    {
        status = { std::move (headline), std::move (detail), isError };
        repaint();
    }

    void openHost()
    {
        setStatus ("Looking for Vital...", "It is the synth: every candidate is rendered by it.");

        juce::String error;
        if (! host.open (48000.0, error, pluginOverride))
        {
            // Named with somewhere to go and get it. Whoever opened this has no
            // terminal to read an error in, and "not found" on its own leaves
            // someone who has never heard of Vital with nothing to do next.
            setStatus ("Vital is not installed",
                       "Install it from vital.audio -- the free version is "
                       "enough -- then start this again.",
                       true);
            choose.setEnabled (false);
            return;
        }

        setStatus ("Drop a recording here",
                   "or use the button below. One sustained note, mono or stereo .wav.");

        if (waiting != juce::File())
            startFit (std::exchange (waiting, juce::File()));
    }

    void browse()
    {
        chooser = std::make_unique<juce::FileChooser> ("Choose a recording",
                                                       juce::File(), "*.wav;*.aiff;*.flac");
        chooser->launchAsync (juce::FileBrowserComponent::openMode
                                  | juce::FileBrowserComponent::canSelectFiles,
                              [this] (const juce::FileChooser& fc)
                              {
                                  const auto file = fc.getResult();
                                  if (file != juce::File())
                                      startFit (file);
                              });
    }

    void startFit (const juce::File& recording)
    {
        if (! ready())
            return;

        if (! recording.existsAsFile())
        {
            setStatus ("That file is not there", recording.getFullPathName(), true);
            return;
        }

        double sampleRate = 0.0;
        auto target = autosynth::cli::readMono (recording, sampleRate);
        if (target.empty() || sampleRate <= 0.0)
        {
            setStatus ("That file could not be read",
                       "A .wav, .aiff or .flac of one sustained note.", true);
            return;
        }

        // The plug-in renders at the recording's own rate, so the fit compares
        // like with like rather than resampling one of the two.
        host.setSampleRate (sampleRate);

        plot.clear();
        table.clear();
        saveAs.setEnabled (false);
        lastResult.reset();
        fitted = recording;

        // Defaults throughout, including a note-off detected from the recording
        // rather than assumed: the UI has no flag to correct one with, so whatever
        // it assumes it is stuck with.
        const autosynth::job::Options options;

        setStatus ("Fitting " + recording.getFileName(),
                   "A couple of hundred renders through Vital. About a minute.");

        progressValue = 0.0;
        progress->setVisible (true);
        cancelButton.setVisible (true);
        choose.setEnabled (false);

        worker = std::make_unique<FitThread> (
            host, std::move (target), sampleRate, options,
            [safe = juce::Component::SafePointer<MainComponent> (this)]
            (Result result)
            {
                if (safe != nullptr)
                    safe->finished (std::move (result));
            });
        worker->startThread();
        startTimerHz (20);
    }

    void cancelFit()
    {
        if (worker == nullptr)
            return;

        setStatus ("Stopping...", "Finishing the render in flight.");
        worker->cancel();
        worker.reset();
        idle();
        setStatus ("Stopped", "Drop another recording when you are ready.");
    }

    void idle()
    {
        stopTimer();
        progress->setVisible (false);
        cancelButton.setVisible (false);
        choose.setEnabled (true);
    }

    void timerCallback() override
    {
        if (worker == nullptr)
            return;

        progressValue = worker->fraction.load();
        const auto stage = worker->stageName();
        if (stage != shownStage)
        {
            shownStage = stage;
            setStatus (status.headline, stageDetail (stage));
        }
    }

    static juce::String stageDetail (const juce::String& stage)
    {
        if (stage == "analysing") return "Reading the note: pitch, partials, envelope, modulation.";
        if (stage == "fitting")   return "Refining against Vital, one render per candidate.";
        if (stage == "rendering") return "Playing the answer back.";
        if (stage == "measuring") return "Comparing the two on eleven axes.";
        return {};
    }

    void finished (Result result)
    {
        idle();
        worker.reset();

        plot.setContours (result.comparison.targetMeasured, result.comparison.fitMeasured);
        table.setAxes (result.comparison.axes);

        // Written next to the recording without being asked, the same as the
        // `fit` verb does: a converter that converts and saves nothing is a
        // stopwatch. "Save as" is for putting it somewhere else.
        const auto preset = fitted.withFileExtension (".vital");
        const auto written = preset.replaceWithText (result.presetJson);

        auto inTolerance = 0;
        for (const auto& axis : result.comparison.axes)
            inTolerance += axis.ok ? 1 : 0;

        lastResult = std::move (result);
        saveAs.setEnabled (true);
        resized();

        setStatus (written ? "Saved " + preset.getFileName()
                           : "Fitted, but could not write " + preset.getFileName(),
                   juce::String (inTolerance) + " of " + juce::String (lastResult->comparison.axes.size())
                       + " axes inside tolerance, in "
                       + juce::String (lastResult->seconds, 1) + " s. Open it in Vital.",
                   ! written);
    }

    void browseForSave()
    {
        if (! lastResult.has_value())
            return;

        chooser = std::make_unique<juce::FileChooser> ("Save the preset",
                                                       fitted.withFileExtension (".vital"),
                                                       "*.vital");
        chooser->launchAsync (juce::FileBrowserComponent::saveMode
                                  | juce::FileBrowserComponent::warnAboutOverwriting,
                              [this] (const juce::FileChooser& fc)
                              {
                                  const auto file = fc.getResult();
                                  if (file == juce::File() || ! lastResult.has_value())
                                      return;

                                  const auto ok = file.replaceWithText (lastResult->presetJson);
                                  setStatus (ok ? "Saved " + file.getFileName()
                                                : "Could not write " + file.getFileName(),
                                             ok ? file.getParentDirectory().getFullPathName()
                                                : "Check the folder is writable.",
                                             ! ok);
                              });
    }

    juce::String pluginOverride;
    juce::File waiting;   // handed to us on the command line, fitted once Vital is up
    autosynth::VitalHost host;

    ContourPlot plot;
    AxisTable table;
    juce::TextButton choose, saveAs, cancelButton;
    std::unique_ptr<juce::ProgressBar> progress;
    double progressValue = 0.0;

    std::unique_ptr<juce::FileChooser> chooser;
    std::unique_ptr<FitThread> worker;
    juce::String shownStage;

    juce::File fitted;
    std::optional<Result> lastResult;

    Status status { "Starting...", {} };
    bool dragging = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainComponent)
};

class Window : public juce::DocumentWindow
{
public:
    Window (const juce::String& pluginOverride, const juce::File& startWith)
        : juce::DocumentWindow ("AutoSynth", kBackground, juce::DocumentWindow::allButtons)
    {
        setUsingNativeTitleBar (true);
        setContentOwned (new MainComponent (pluginOverride, startWith), true);
        setResizable (true, false);
        setResizeLimits (760, 560, 2400, 1800);
        centreWithSize (getWidth(), getHeight());
        setVisible (true);
    }

    void closeButtonPressed() override
    {
        juce::MessageManager::getInstance()->stopDispatchLoop();
    }
};

} // namespace

int runGui (const Args& args)
{
    juce::ScopedJuceInitialiser_GUI juceInit;

    juce::LookAndFeel_V4 look { juce::LookAndFeel_V4::getDarkColourScheme() };
    juce::LookAndFeel::setDefaultLookAndFeel (&look);

    // A recording named on the command line is fitted as soon as Vital is up,
    // which is what dropping one on the exe in Explorer amounts to.
    const auto startWith = args.positional.isEmpty() ? juce::File()
                                                     : args.file (0);

    auto window = std::make_unique<Window> (args.text ("--plugin"), startWith);
    juce::MessageManager::getInstance()->runDispatchLoop();

    window.reset();
    juce::LookAndFeel::setDefaultLookAndFeel (nullptr);
    return 0;
}
