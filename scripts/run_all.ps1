$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

Write-Host "Generating synthetic data..."
& $python "src\data_generation.py"

Write-Host "Training models..."
& $python "src\train_model.py"

Write-Host "Evaluating the saved model..."
& $python "src\evaluate_model.py"

Write-Host "Running tests..."
& $python -m pytest

Write-Host "Generating recruiter-facing visuals..."
& $python "scripts\generate_visuals.py"

Write-Host "All pipeline steps completed successfully."
