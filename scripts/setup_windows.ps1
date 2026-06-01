$ErrorActionPreference = "Stop"
$PYTHON = "C:\Python314\python.exe"

Write-Host "== KANEA setup (Python systeme) ==" -ForegroundColor Cyan
Write-Host "Python : $((& $PYTHON --version 2>&1))" -ForegroundColor Green

Write-Host "`nVerification des packages requis..." -ForegroundColor Yellow
& $PYTHON -m pip install --upgrade --quiet streamlit streamlit-folium pandas numpy scipy plotly folium scikit-learn xgboost joblib lightgbm shap reportlab qrcode pillow requests tqdm onnxruntime torch torchvision 2>&1

Write-Host "`nSetup termine." -ForegroundColor Green
Write-Host "Dashboard  : scripts\run_dashboard.bat" -ForegroundColor Green
Write-Host "API        : scripts\run_api.bat" -ForegroundColor Green
Write-Host "Entrainement malaria :" -ForegroundColor Green
Write-Host "  $PYTHON main.py --module malaria --skip-download --epochs 20" -ForegroundColor Cyan
