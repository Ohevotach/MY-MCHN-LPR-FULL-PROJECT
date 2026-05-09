param(
    [int]$SamplesPerLevel = 300,
    [int]$BatchSize = 384,
    [int]$CnnEpochs = 20,
    [int]$NumWorkers = 0,
    [string]$Pollution = "core",
    [switch]$Fast
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$commonArgs = @(
    "main_eval.py",
    "--pollution", $Pollution,
    "--samples-per-level", "$SamplesPerLevel",
    "--batch-size", "$BatchSize",
    "--cnn-epochs", "$CnnEpochs",
    "--num-workers", "$NumWorkers",
    "--split-mode", "group",
    "--mchn-topk", "10",
    "--mchn-maxsim-weight", "0.50",
    "--skip-e2e"
)

if ($Fast) {
    $commonArgs += @(
        "--skip-confusion",
        "--skip-balanced-eval",
        "--skip-ablation",
        "--skip-beta-ablation",
        "--skip-attention-errors",
        "--skip-capacity",
        "--skip-random-capacity"
    )
}

Write-Host "Running final MCHN experiments from $ProjectRoot"
python @commonArgs
