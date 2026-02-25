# Create venv with uv and install packages
# Run from project root (e.g. PowerShell):
#   .\scripts\setup_uv.ps1
# Or manually (with uv on PATH):
#   uv venv
#   .\.venv\Scripts\Activate.ps1
#   uv pip install -r requirements.txt

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv via pip..."
    python -m pip install uv
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Could not find 'uv'. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }
}

Write-Host "Creating virtual environment with uv..."
uv venv

Write-Host "Installing packages..."
& .\.venv\Scripts\python.exe -m uv pip install -r requirements.txt

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Then run: streamlit run app.py"
