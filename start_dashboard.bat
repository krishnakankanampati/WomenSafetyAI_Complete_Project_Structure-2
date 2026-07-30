@echo off
cd /d "%~dp0"

echo Starting backend...
start "Women Safety AI - Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo Starting frontend...
start "Women Safety AI - Frontend" cmd /k "cd frontend && set PATH=C:\Program Files\nodejs;%PATH% && npm run dev"

echo Waiting for servers to start...
timeout /t 12 /nobreak >nul

start http://localhost:5173
