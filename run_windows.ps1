# EMO Robot - Windows Launcher
$ErrorActionPreference = "Stop"

$ROOT_DIR = $PSScriptRoot
Set-Location $ROOT_DIR

$VENV_PY = Join-Path $ROOT_DIR ".venv\Scripts\python.exe"
if (-not (Test-Path $VENV_PY)) {
    Write-Host "Virtual environment not found. Running setup_windows.ps1 first..." -ForegroundColor Yellow
    & (Join-Path $ROOT_DIR "setup_windows.ps1")
}

Write-Host "Starting EMO Robot on Windows..." -ForegroundColor Cyan
Write-Host "Controls:" -ForegroundColor White
Write-Host "  - Press SPACE or click the window to wake EMO" -ForegroundColor White
Write-Host "  - Say 'EMO' to wake with voice" -ForegroundColor White
Write-Host "  - Press ESC or Q to quit" -ForegroundColor White
Write-Host "  - Remote controller: check http://localhost:8766/" -ForegroundColor White
Write-Host "--------------------------------------------------" -ForegroundColor DarkGray

& $VENV_PY emo.py
