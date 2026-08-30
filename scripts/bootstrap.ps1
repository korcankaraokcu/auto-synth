<#
.SYNOPSIS
    Configure and build auto-synth on Windows.

.DESCRIPTION
    Checks the toolchain, configures CMake, and builds the tools and the test
    suite.

    This exists because the interesting failure in this build is not a build
    error: a missing C compiler makes JUCE refuse to configure with a message
    that does not mention C, which is not obvious the first time.

.PARAMETER Config
    Build configuration. Release by default; the test suite is slow in Debug.

.PARAMETER SkipTests
    Configure and build the tools only.

.EXAMPLE
    .\scripts\bootstrap.ps1
    .\scripts\bootstrap.ps1 -Config Debug -SkipTests
#>

[CmdletBinding()]
param(
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug')]
    [string]$Config = 'Release',
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

$targets = @('autosynth_probe', 'autosynth_diff', 'autosynth_vital')
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

# --- summary --------------------------------------------------------------

Write-Step 'Done'
foreach ($tool in $targets) {
    if ($tool -eq 'autosynth_tests') { continue }
    Write-Host "    $tool : $(Join-Path $buildDir "${tool}_artefacts\$Config\$tool.exe")"
}
Write-Host ''
Write-Host '    Vital must be installed: it is the synth, and the test suite renders through it.'
