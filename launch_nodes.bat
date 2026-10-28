@echo off
cd /d "%~dp0"
echo ======================================================================
echo  Clearing Existing Processes on Ports 8000-8003...
echo ======================================================================
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8001 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8002 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8003 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo ======================================================================
echo  Launching SHIELD-RAG 4-Node Microservices (4 Isolated Trust Domains)
echo ======================================================================
echo [1/4] Spawning KDC Node (Keystone IAM & QR Subspace Authority) on :8001...
start "SHIELD-RAG [1: KDC Authority] :8001" cmd /k "title SHIELD-RAG [1: KDC Authority] :8001 && color 0B && python -m services.kdc_node"

timeout /t 1 /nobreak >nul
echo [2/4] Spawning Revocation Proxy (Neutron Perimeter Noise Gating) on :8002...
start "SHIELD-RAG [2: Revocation Proxy] :8002" cmd /k "title SHIELD-RAG [2: Revocation Proxy] :8002 && color 0E && python -m services.proxy_node"

timeout /t 1 /nobreak >nul
echo [3/4] Spawning Cloud Storage Oracle (OpenStack Swift/Cinder) on :8003...
start "SHIELD-RAG [3: Cloud Storage Oracle] :8003" cmd /k "title SHIELD-RAG [3: Cloud Storage Oracle] :8003 && color 09 && python -m services.storage_node"

timeout /t 1 /nobreak >nul
echo [4/4] Spawning Query Gateway (Nova Ephemeral Attention & RAM Zeroization) on :8000...
start "SHIELD-RAG [4: Query Gateway] :8000" cmd /k "title SHIELD-RAG [4: Query Gateway] :8000 && color 0A && python -m services.gateway_node"

timeout /t 3 /nobreak >nul
echo.
echo ======================================================================
echo  All 4 Nodes Active! Opening Swagger UI in browser:
echo  * Query Gateway Docs:  http://localhost:8000/docs
echo ======================================================================
start http://localhost:8000/docs
