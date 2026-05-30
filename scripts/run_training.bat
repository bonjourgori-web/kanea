@echo off
set PYTHONIOENCODING=utf-8
cd /d c:\Users\HP\gori\KANEA
python scripts\train_malaria_resnet34.py > scripts\training_log.txt 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> scripts\training_log.txt
