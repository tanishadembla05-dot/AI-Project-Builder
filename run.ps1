param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Requirements = Join-Path $ScriptDir "requirements.txt"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment Python not found at: $VenvPython" -ForegroundColor Red
    Write-Host "Create it first from project root: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

Set-Location $ScriptDir

if (-not $SkipInstall) {
    Write-Host "Installing/updating dependencies..." -ForegroundColor Cyan
    & $VenvPython -m pip install -r $Requirements
}

Write-Host "Starting Streamlit app..." -ForegroundColor Green
& $VenvPython -m streamlit run app.py
