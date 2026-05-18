$ErrorActionPreference = "Stop"

Write-Host "== KANEA setup ==" -ForegroundColor Cyan

if (-not (Test-Path ".\\venv")) {
    Write-Host "Creation de l'environnement virtuel..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "Activation de l'environnement virtuel..." -ForegroundColor Yellow
. .\venv\Scripts\Activate.ps1

Write-Host "Mise a jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "Installation des dependances..." -ForegroundColor Yellow
python -m pip install -r requirements.txt

Write-Host "Setup termine." -ForegroundColor Green
Write-Host "Dashboard : python scripts/run_dashboard.py" -ForegroundColor Green
Write-Host "API       : python scripts/run_api.py" -ForegroundColor Green
