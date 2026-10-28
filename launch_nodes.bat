@echo off
cd /d "%~dp0"
echo ======================================================================
echo  Launching SHIELD-RAG 4-Node Microservices (4 Isolated Trust Domains)
echo ======================================================================
echo [1/4] Spawning KDC Node (Keystone IAM & QR Subspace Authority) on :8001...
start "SHIELD-RAG [KDC Authority] :8001" cmd /k "title SHIELD-RAG [KDC Authority] :8001 && color 0B && python -m services.kdc_node"

timeout /t 1 /nobreak >nul
echo [2/4] Spawning Revocation Proxy (Neutron Perimeter Noise Gating) on :8002...
start "SHIELD-RAG [Revocation Proxy] :8002" cmd /k "title SHIELD-RAG [Revocation Proxy] :8002 && color 0E && python -m services.proxy_node"

timeout /t 1 /nobreak >nul
echo [3/4] Spawning Cloud Storage Oracle (OpenStack Swift/Cinder) on :8003...
start "SHIELD-RAG [Cloud Storage Oracle] :8003" cmd /k "title SHIELD-RAG [Cloud Storage Oracle] :8003 && color 09 && python -m services.storage_node"

timeout /t 1 /nobreak >nul
echo [4/4] Spawning Query Gateway (Nova Ephemeral Attention & RAM Zeroization) on :8000...
start "SHIELD-RAG [Query Gateway] :8000" cmd /k "title SHIELD-RAG [Query Gateway] :8000 && color 0A && python -m services.gateway_node"

echo.
echo ======================================================================
echo  All 4 Nodes Active! Open in browser or run demo:
echo  * Query Gateway Docs:  http://localhost:8000/docs
echo  * KDC Authority:       http://localhost:8001/health
echo  * Revocation Proxy:    http://localhost:8002/health
echo  * Storage Oracle:      http://localhost:8003/health
echo  * Run Live Demo:       python demo.py
echo ======================================================================
