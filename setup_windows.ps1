# EMO Robot - Windows Setup Script
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "       EMO Robot Windows Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Check Python
Write-Host "`n[1/7] Checking Python..." -ForegroundColor Yellow
try {
    $pyVersion = & python --version 2>&1
    Write-Host "Found: $pyVersion" -ForegroundColor Green
} catch {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10+ from python.org."
    exit 1
}

# 2. Virtual Environment
Write-Host "`n[2/7] Creating virtual environment (.venv)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & python -m venv .venv
}
Write-Host "Virtual environment ready at .venv" -ForegroundColor Green

$VENV_PY = Join-Path (Get-Location) ".venv\Scripts\python.exe"

# 3. Core dependencies
Write-Host "`n[3/7] Installing core and Windows dependencies..." -ForegroundColor Yellow
& $VENV_PY -m pip install --upgrade pip setuptools wheel
& $VENV_PY -m pip install -r requirements.txt
& $VENV_PY -m pip install -r requirements-faster-whisper.txt

# 4. Prepare local folders and .env
Write-Host "`n[4/7] Preparing folders and configuration (.env)..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "assets\songs" | Out-Null
New-Item -ItemType Directory -Force -Path "data\known_faces" | Out-Null

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example" -ForegroundColor Green
}

& $VENV_PY scripts\configure_env.py

# 5. Download Vosk wake-word model
Write-Host "`n[5/7] Checking Vosk wake-word model..." -ForegroundColor Yellow
& $VENV_PY scripts\download_vosk_model.py

# 6. Download Faster-Whisper tiny.en model
Write-Host "`n[6/7] Checking English Faster-Whisper tiny.en model..." -ForegroundColor Yellow
$whisperCache = Join-Path ([System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::UserProfile)) ".cache\emo\whisper-en"
& $VENV_PY scripts\download_whisper_en.py --model tiny.en --cache-dir $whisperCache

# 7. Check and download all models (Vosk, Whisper, llama.cpp binaries, and GGUF)
Write-Host "`n[7/7] Checking and downloading AI models..." -ForegroundColor Yellow
& $VENV_PY model_manager.py

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "  EMO Setup Complete on Windows!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "To start EMO on Windows:"
Write-Host "  .\run_windows.ps1" -ForegroundColor Yellow
Write-Host "Or activate the virtualenv manually:"
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "  python emo.py" -ForegroundColor Yellow
