# VoxShield - start everything.
#
#   .\start.ps1              backend on :8000, frontend on :3000
#   .\start.ps1 -BackendOnly
#   .\start.ps1 -Check       run the checks and exit, start nothing
#
# Each service opens in its own window so you can read its log and stop it
# with Ctrl+C independently. Close both windows to shut down.

[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [switch]$BackendOnly,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

function Say([string]$text, [string]$colour = 'Gray') {
    Write-Host $text -ForegroundColor $colour
}

function Fail([string]$text) {
    Write-Host "  [X] $text" -ForegroundColor Red
    $script:problems++
}

function Ok([string]$text) {
    Write-Host "  [+] $text" -ForegroundColor Green
}

function Warn([string]$text) {
    Write-Host "  [!] $text" -ForegroundColor Yellow
}

$script:problems = 0

Say ""
Say "VoxShield" Cyan
Say "=========" Cyan
Say ""

# --- interpreter ----------------------------------------------------------

$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($python) {
        Warn "No .venv found - using $python"
        Warn "A virtual environment is safer: python -m venv .venv"
    }
}

if (-not $python) {
    Fail "No Python found. Install Python 3.11+ and re-run."
    exit 1
}
Ok "python  $python"

# --- dependencies ---------------------------------------------------------

$missing = & $python -c @"
import importlib.util, sys
need = ['fastapi','uvicorn','torch','transformers','soundfile','librosa',
        'numpy','sklearn','pandas','speechbrain','requests','multipart']
print(','.join(m for m in need if importlib.util.find_spec(m) is None))
"@ 2>&1

if ($LASTEXITCODE -ne 0) {
    Fail "Could not run Python: $missing"
    exit 1
}

if ($missing) {
    Fail "Missing packages: $missing"
    Say ""
    Say "    Install them with:" Yellow
    Say "      $python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchaudio" Yellow
    Say "      $python -m pip install -r backend\requirements.txt" Yellow
    Say ""
} else {
    Ok "python packages present"
}

# --- GPU ------------------------------------------------------------------

$device = & $python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>$null
if ($device -and $device -ne 'CPU') {
    Ok "GPU     $device"
} else {
    Warn "No CUDA GPU - inference runs on CPU, roughly 10x slower but it works"
}

# --- model artifacts ------------------------------------------------------

$checkpoint = Join-Path $root 'models\voxshield_antispoof.pt'
if (Test-Path $checkpoint) {
    $mb = [math]::Round((Get-Item $checkpoint).Length / 1MB, 1)
    Ok "model   voxshield_antispoof.pt ($mb MB)"
} else {
    Fail "No checkpoint at models\voxshield_antispoof.pt"
    Say "      The API will start, but every call reports model_status 'untrained'." Yellow
}

$calibration = Join-Path $root 'results\calibration\antispoof_calibration.json'
if (Test-Path $calibration) {
    Ok "calibration present"
} else {
    Warn "No calibration - raw scores are NOT probabilities."
    Warn "Fix: cd backend; $python scripts\calibrate_detector.py"
}

# --- test corpus ----------------------------------------------------------

$corpus = Join-Path $root 'datasets\test-audio\spoof'
if (Test-Path $corpus) {
    $count = (Get-ChildItem $corpus -Filter *.wav -ErrorAction SilentlyContinue).Count
    Ok "test corpus: $count synthetic sample(s)"
} else {
    Warn "No test corpus. Generate one: cd backend; $python scripts\make_test_audio.py"
}

# --- frontend -------------------------------------------------------------

if (-not $BackendOnly) {
    if (Test-Path (Join-Path $root 'frontend\node_modules')) {
        Ok "frontend dependencies installed"
    } else {
        Warn "frontend\node_modules missing - running npm install now"
        Push-Location (Join-Path $root 'frontend')
        npm install
        Pop-Location
    }
}

# --- ports ----------------------------------------------------------------

function Test-PortFree([int]$port) {
    $null -eq (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-PortFree $BackendPort)) {
    Fail "Port $BackendPort is already in use."
    Say "      Find it:  Get-NetTCPConnection -LocalPort $BackendPort -State Listen" Yellow
    Say "      Or pick another:  .\start.ps1 -BackendPort 8001" Yellow
}

if (-not $BackendOnly -and -not (Test-PortFree $FrontendPort)) {
    Warn "Port $FrontendPort is in use - Next.js will pick the next free one."
}

Say ""

if ($script:problems -gt 0) {
    Say "$($script:problems) problem(s) above must be fixed first." Red
    exit 1
}

if ($Check) {
    Say "All checks passed. Nothing started (-Check)." Green
    exit 0
}

# --- launch ---------------------------------------------------------------

Say "Starting backend on http://localhost:$BackendPort ..." Cyan

$backendCommand = @"
Set-Location '$root\backend'
Write-Host 'VoxShield backend - http://localhost:$BackendPort' -ForegroundColor Cyan
Write-Host 'Docs: http://localhost:$BackendPort/docs   Status: http://localhost:$BackendPort/api/v1/status' -ForegroundColor DarkGray
Write-Host ''
& '$python' -m uvicorn app.main:app --host 0.0.0.0 --port $BackendPort
"@

Start-Process powershell -ArgumentList '-NoExit', '-Command', $backendCommand

# The first request loads a 380 MB checkpoint, so give it a moment before the
# frontend starts polling /status and reports the backend as down.
Say "Waiting for the backend to answer ..." DarkGray

$ready = $false
foreach ($attempt in 1..60) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-RestMethod "http://127.0.0.1:$BackendPort/api/v1/health" -TimeoutSec 2
        if ($response.status -eq 'healthy') { $ready = $true; break }
    } catch { }
}

if ($ready) {
    Ok "backend is up"
} else {
    Warn "backend has not answered yet - check its window for errors"
}

if (-not $BackendOnly) {
    Say "Starting frontend on http://localhost:$FrontendPort ..." Cyan

    $frontendCommand = @"
Set-Location '$root\frontend'
Write-Host 'VoxShield frontend - http://localhost:$FrontendPort' -ForegroundColor Cyan
Write-Host ''
`$env:NEXT_PUBLIC_API_URL = 'http://localhost:$BackendPort'
`$env:NEXT_PUBLIC_WS_URL = 'ws://localhost:$BackendPort'
npm run dev -- --port $FrontendPort
"@

    Start-Process powershell -ArgumentList '-NoExit', '-Command', $frontendCommand
}

Say ""
Say "Open:" Cyan
Say "  http://localhost:$FrontendPort              the testing console"
Say "  http://localhost:$BackendPort/docs          interactive API docs"
Say "  http://localhost:$BackendPort/api/v1/status what is loaded and calibrated"
Say ""
Say "Verify everything end to end:" Cyan
Say "  cd backend; & '$python' scripts\verify_system.py --url http://127.0.0.1:$BackendPort"
Say ""
Say "See TESTING.md for what to try and what each result should be." DarkGray
Say ""
