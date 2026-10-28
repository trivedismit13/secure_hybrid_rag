# SHIELD-RAG 4-Node PowerShell Launcher
Set-Location -Path $PSScriptRoot
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Launching SHIELD-RAG 4-Node Microservices (4 Isolated Trust Domains)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "[1/4] Spawning KDC Node on :8001..." -ForegroundColor Yellow
Start-Process cmd.exe -ArgumentList '/k "title SHIELD-RAG [KDC Authority] :8001 && color 0B && python -m services.kdc_node"'
Start-Sleep -Seconds 1

Write-Host "[2/4] Spawning Revocation Proxy on :8002..." -ForegroundColor Yellow
Start-Process cmd.exe -ArgumentList '/k "title SHIELD-RAG [Revocation Proxy] :8002 && color 0E && python -m services.proxy_node"'
Start-Sleep -Seconds 1

Write-Host "[3/4] Spawning Cloud Storage Oracle on :8003..." -ForegroundColor Yellow
Start-Process cmd.exe -ArgumentList '/k "title SHIELD-RAG [Cloud Storage Oracle] :8003 && color 09 && python -m services.storage_node"'
Start-Sleep -Seconds 1

Write-Host "[4/4] Spawning Query Gateway on :8000..." -ForegroundColor Yellow
Start-Process cmd.exe -ArgumentList '/k "title SHIELD-RAG [Query Gateway] :8000 && color 0A && python -m services.gateway_node"'

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host " All 4 Nodes Active!" -ForegroundColor Green
Write-Host " * Query Gateway Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host " * KDC Authority:       http://localhost:8001/health" -ForegroundColor White
Write-Host " * Revocation Proxy:    http://localhost:8002/health" -ForegroundColor White
Write-Host " * Storage Oracle:      http://localhost:8003/health" -ForegroundColor White
Write-Host " * Live Demo:           python demo.py" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Green
