# ---------------------------------------------------------------------------
# Resumable download of ASVspoof 2019 LA (LA.zip, 7.12 GB).
#
# Runs detached, so it survives whatever launched it. Every attempt uses
# `curl -C -`, which resumes from the bytes already on disk, so a dropped
# connection costs seconds rather than restarting a 7 GB transfer. The loop
# keeps retrying until the file reaches its exact expected size.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\download_dataset.ps1
#
# Progress:  Get-Content datasets\raw\download.log -Tail 5
# Size now:  (Get-Item datasets\raw\LA.zip).Length
# ---------------------------------------------------------------------------

$ErrorActionPreference = 'Continue'

$Url = 'https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Dir = Join-Path $Root 'datasets\raw'
$Out = Join-Path $Dir 'LA.zip'
$Log = Join-Path $Dir 'download.log'

# Verified from the server's own headers on 2026-09-08:
#   Content-Type: application/zip   Content-Length: 7640952520
#   Accept-Ranges: bytes            (no authentication required)
$Expected = 7640952520

New-Item -ItemType Directory -Force -Path $Dir | Out-Null

function Write-Log($Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -Path $Log -Value $line -Encoding utf8
}

function Get-Size {
    if (Test-Path $Out) { return (Get-Item $Out).Length }
    return 0
}

Write-Log "START  target=$Expected bytes  ->  $Out"

$maxAttempts = 200

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {

    $have = Get-Size

    if ($have -ge $Expected) { break }

    $pct = [math]::Round($have / $Expected * 100, 1)
    Write-Log "attempt $attempt : have $have bytes ($pct%), resuming"

    # --no-progress-meter keeps the log readable; progress is tracked by
    # polling the file size instead. --retry handles transient failures
    # inside a single curl run; the outer loop handles the rest.
    & curl.exe `
        --location `
        --continue-at - `
        --retry 10 `
        --retry-delay 5 `
        --retry-all-errors `
        --connect-timeout 30 `
        --no-progress-meter `
        --output $Out `
        $Url 2>&1 | ForEach-Object { if ($_) { Write-Log "curl: $_" } }

    $now = Get-Size

    if ($now -ge $Expected) { break }

    # No forward progress at all means something structural is wrong (the
    # server refused a range request, the disk is full). Back off before
    # hammering it again.
    if ($now -le $have) {
        Write-Log "no progress this attempt (still $now bytes), backing off 30s"
        Start-Sleep -Seconds 30
    }
    else {
        Start-Sleep -Seconds 5
    }
}

$final = Get-Size

if ($final -eq $Expected) {
    Write-Log "COMPLETE  $final bytes"
    Write-Log "Next: extract, then build manifests -"
    Write-Log "  Expand-Archive datasets\raw\LA.zip -DestinationPath datasets\raw\ASVspoof2019"
    Write-Log "  python backend\scripts\build_manifest.py --asvspoof2019-la datasets\raw\ASVspoof2019\LA"
}
else {
    Write-Log "INCOMPLETE  $final of $Expected bytes - re-run this script to resume"
}
