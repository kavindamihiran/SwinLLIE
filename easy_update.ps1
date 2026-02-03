# =============================================================================
# 🚀 EASY DEPLOY/UPDATE SCRIPT
# Run this to push ANY changes to Hugging Face
# =============================================================================

$ErrorActionPreference = "Stop"
$spacePath = "..\swinlle"

Write-Host "🔄 syncing files to Space..." -ForegroundColor Cyan

# 1. Ensure Space directory exists
if (-not (Test-Path $spacePath)) {
    Write-Error "Space folder not found at $spacePath. Make sure you cloned it there!"
}

# 2. Copy all essential files (Overwrites existing ones)
Copy-Item "app.py" -Destination $spacePath -Force
Copy-Item "Dockerfile" -Destination $spacePath -Force
Copy-Item ".dockerignore" -Destination $spacePath -Force
Copy-Item "requirements.txt" -Destination $spacePath -Force
Copy-Item "README_HUGGINGFACE.md" -Destination "$spacePath\README.md" -Force

if (Test-Path "swinllie") { 
    if (Test-Path "$spacePath\swinllie") { Remove-Item "$spacePath\swinllie" -Recurse -Force }
    Copy-Item "swinllie" -Destination $spacePath -Recurse -Force
}

# 3. Commit and Push
Set-Location $spacePath
Write-Host "📦 Committing changes..." -ForegroundColor Yellow

git add .
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Update from script at $timestamp"

Write-Host "🚀 Pushing to Hugging Face..." -ForegroundColor Green
git push

Write-Host ""
Write-Host "✅ Done! Watch build at: https://huggingface.co/spaces/kvindatemp/swinlle" -ForegroundColor Cyan
Write-Host "   (Press Enter to exit)"
Read-Host
