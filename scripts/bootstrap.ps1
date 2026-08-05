<#
.SYNOPSIS
    Configure and build auto-synth on Windows.

.DESCRIPTION
    Checks the toolchain, configures CMake, builds the plugin and the test
    suite, and optionally installs the VST3 to the per-user folder.

    This exists because the interesting failures in this build are not build
    errors. JUCE's plugin copy step needs administrator rights and fails at the
    very last step of an otherwise successful build, which reads like a
    compilation failure. A missing C compiler makes JUCE refuse to configure
    with a message that does not mention C. Neither is obvious the first time.

.PARAMETER Config
    Build configuration. Release by default; the test suite is slow in Debug.

.PARAMETER Install
    Also copy the built VST3 into the per-user VST3 folder.

.PARAMETER SkipTests
    Configure and build the plugin only.

.EXAMPLE
    .\scripts\bootstrap.ps1
    .\scripts\bootstrap.ps1 -Install
    .\scripts\bootstrap.ps1 -Config Debug -SkipTests
#>

[CmdletBinding()]
param(
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug')]
    [string]$Config = 'Release',
    [switch]$Install,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$pluginDir = Join-Path $repoRoot 'plugin'
$buildDir = Join-Path $pluginDir 'build'

function Write-Step($message) {
    Write-Host ''
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Write-Problem($message) {
    Write-Host "    $message" -ForegroundColor Yellow
}

# --- toolchain ------------------------------------------------------------

Write-Step 'Checking the toolchain'

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if ($null -eq $cmake) {
    Write-Problem 'cmake was not found on PATH.'
    Write-Problem 'Install it from https://cmake.org/download/ or with: pip install cmake'
    throw 'cmake is required'
}
$cmakeVersion = (& cmake --version | Select-Object -First 1)
Write-Host "    $cmakeVersion"

# JUCE 8 needs a C compiler as well as a C++ one, and says so only obliquely.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (Test-Path $vswhere) {
    $vs = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property displayName
    if ([string]::IsNullOrWhiteSpace($vs)) {
        Write-Problem 'No Visual Studio C++ toolset found.'
        Write-Problem 'Install "Desktop development with C++" (the Build Tools alone are enough).'
        throw 'MSVC toolset is required'
    }
    Write-Host "    $vs"
} else {
    Write-Problem 'vswhere.exe not found; skipping the Visual Studio check.'
    Write-Problem 'If configuration fails, install the VS Build Tools with the C++ workload.'
}

# --- configure ------------------------------------------------------------

Write-Step "Configuring ($Config)"
Write-Host '    JUCE and Catch2 are fetched on the first run; expect a few minutes.'

$configureArgs = @('-S', $pluginDir, '-B', $buildDir)
if ($SkipTests) {
    $configureArgs += '-DAUTOSYNTH_BUILD_TESTS=OFF'
}

& cmake @configureArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

# --- build ----------------------------------------------------------------

$targets = @('AutoSynth_VST3', 'AutoSynth_Standalone', 'autosynth_render', 'autosynth_probe')
if (-not $SkipTests) {
    $targets += 'autosynth_tests'
}

Write-Step "Building: $($targets -join ', ')"
& cmake --build $buildDir --config $Config --target @targets
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

# --- test -----------------------------------------------------------------

if (-not $SkipTests) {
    Write-Step 'Running tests'
    & ctest --test-dir $buildDir -C $Config --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "tests failed ($LASTEXITCODE)" }
}

# --- install --------------------------------------------------------------

if ($Install) {
    Write-Step 'Installing the VST3'
    # Deliberately the per-user folder: JUCE's own copy step targets
    # C:\Program Files\Common Files\VST3, which needs elevation.
    & cmake --build $buildDir --config $Config --target install_plugin
    if ($LASTEXITCODE -ne 0) { throw "install failed ($LASTEXITCODE)" }
}

# --- summary --------------------------------------------------------------

$artefacts = Join-Path $buildDir "AutoSynth_artefacts\$Config"
Write-Step 'Done'
Write-Host "    Standalone : $(Join-Path $artefacts 'Standalone\auto-synth.exe')"
Write-Host "    VST3       : $(Join-Path $artefacts 'VST3\auto-synth.vst3')"
if (-not $Install) {
    Write-Host '    Run with -Install to copy the VST3 into your per-user VST3 folder.'
}
