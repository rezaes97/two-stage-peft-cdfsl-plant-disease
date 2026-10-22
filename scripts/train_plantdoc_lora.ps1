param(
    # Dataset and clean pretrained checkpoint are intentionally required.
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [Parameter(Mandatory=$true)][string]$InitialCheckpoint,

    [string]$OutputDir = ".\outputs\plantdoc_offline_lora",
    [string]$PythonExe = "",
    [string]$VenvPath = "",

    # Saved older PlantDoc split: 23 source classes, 5 held-out target classes.
    [int]$NumClasses = 23,
    [int]$Epochs = 35,
    [int]$BatchSize = 32,
    [int]$Workers = 8,

    # Offline LoRA defaults. The proxy-learning optimizer/iteration settings apply
    # to the episodic stage below, not this supervised offline classifier training.
    [string]$Optimizer = "adamw",
    [double]$LearningRate = 0.001,
    [double]$WeightDecay = 0.05,
    [int]$WarmupEpochs = 3,
    [int]$LoraRank = 8,
    [int]$LoraAlpha = 1,
    [double]$LoraDropout = 0.05,
    [string[]]$LoraTargetModules = @("qkv"),
    [int]$Seed = 42,
    [switch]$NoSimpleAug
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$TrainScript = Join-Path $RepoRoot "offline\adapt.py"

# Resolve Python from the repository's venv by default; allow explicit overrides.
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    if ([string]::IsNullOrWhiteSpace($VenvPath)) {
        $VenvPath = Join-Path $RepoRoot "venv"
    }
    $VenvPython = Join-Path $VenvPath "Scripts\python.exe"
    if (!(Test-Path $VenvPython)) {
        throw "Virtual-environment Python not found: $VenvPython. Pass -VenvPath '<venv directory>' or -PythonExe '<full path to python.exe>'."
    }
    $PythonExe = $VenvPython
}
if (!(Test-Path $PythonExe)) { throw "Python executable not found: $PythonExe" }
$PythonExe = (Resolve-Path $PythonExe).Path
Write-Host "Python executable: $PythonExe"

# Fail early if this exact interpreter cannot import PyTorch.
& $PythonExe -c "import sys, torch; print('Python:', sys.version); print('Executable:', sys.executable); print('PyTorch:', torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "The selected interpreter cannot import torch. Install the project's dependencies into that venv and rerun."
}

$DataRoot = (Resolve-Path $DataRoot).Path
$InitialCheckpoint = (Resolve-Path $InitialCheckpoint).Path
if (!(Test-Path $TrainScript)) { throw "Offline training script not found: $TrainScript" }
if (!(Test-Path (Join-Path $DataRoot "train"))) { throw "Expected '$DataRoot\train'" }

# Use source-class validation only; held-out target classes must never select the checkpoint.
$ValName = if (Test-Path (Join-Path $DataRoot "validation")) {
    "validation"
} elseif (Test-Path (Join-Path $DataRoot "val")) {
    "val"
} else {
    throw "Expected a source-class validation folder: '$DataRoot\validation' or '$DataRoot\val'."
}
$TrainClasses = @(Get-ChildItem (Join-Path $DataRoot "train") -Directory | ForEach-Object Name | Sort-Object)
$ValClasses = @(Get-ChildItem (Join-Path $DataRoot $ValName) -Directory | ForEach-Object Name | Sort-Object)
if ($TrainClasses.Count -ne $NumClasses) {
    throw "Training split has $($TrainClasses.Count) class folders; NumClasses=$NumClasses. Check the saved source-class split."
}
if (Compare-Object -ReferenceObject $TrainClasses -DifferenceObject $ValClasses) {
    throw "train and validation class folder names differ. Keep the source classes identical across these folders."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Preserve the existing offline training pipeline; expose the relevant values as parameters.
$ArgsList = @(
    $DataRoot,
    "--dataset", "custom",
    "--train-split", "train",
    "--val-split", $ValName,
    "--model", "vit_small_patch16_224",
    "--num-classes", "$NumClasses",
    "--initial-checkpoint", $InitialCheckpoint,
    "--batch-size", "$BatchSize",
    "--epochs", "$Epochs",
    "--opt", $Optimizer,
    "--weight-decay", "$WeightDecay",
    "--warmup-lr", "1e-6",
    "--warmup-epochs", "$WarmupEpochs",
    "--lr", "$LearningRate",
    "--min-lr", "1e-8",
    "--drop-path", "0.1",
    "--img-size", "224",
    "--model-ema",
    "--amp",
    "--native-amp",
    "--workers", "$Workers",
    "--smoothing", "0",
    "--tuning-mode", "lora",
    "--lora-r", "$LoraRank",
    "--lora-alpha", "$LoraAlpha",
    "--lora-dropout", "$LoraDropout",
    "--target-modules"
)
$ArgsList += $LoraTargetModules
$ArgsList += @("--seed", "$Seed", "--output", $OutputDir)

# simple_aug is on by default to match the reported configuration.
if (!$NoSimpleAug) { $ArgsList += "--simple-aug" }

Write-Host "Offline LoRA training: $($TrainClasses.Count) source classes; validation split '$ValName'."
Write-Host "LoRA: rank=$LoraRank alpha=$LoraAlpha dropout=$LoraDropout targets=$($LoraTargetModules -join ','); simple_aug=$(!$NoSimpleAug)"
& $PythonExe -u $TrainScript @ArgsList
if ($LASTEXITCODE -ne 0) { throw "Offline LoRA training failed (exit code $LASTEXITCODE)." }
