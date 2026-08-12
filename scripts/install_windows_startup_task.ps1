param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [string]$TaskName = "CV-DP Camera Scheduler",
    [string]$PythonExecutable = "",
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $ProjectRoot "scripts\scheduled_runner.py"
$ManifestPath = (Resolve-Path $Manifest).Path
$StopFile = "$ManifestPath.scheduler-stop"
Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue

if (-not $PythonExecutable) {
    $PythonExecutable = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
$PythonPath = (Resolve-Path $PythonExecutable).Path

$ActionArgs = '"{0}" --manifest "{1}" --headless' -f $Runner, $ManifestPath
$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument $ActionArgs `
    -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries
$Principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Description "CV-DP scheduled camera supervisor | Manifest=$ManifestPath" `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Force | Out-Null

$RegisteredTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
if ($RegisteredTask.TaskName -ne $TaskName) {
    throw "Windows did not register startup task: $TaskName"
}
if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
}

Write-Host "Installed startup task: $TaskName"
Write-Host "Manifest: $ManifestPath"
if ($StartNow) {
    Write-Host "The scheduler task was started."
}
Write-Host "The scheduler will start at every boot, including boots inside an active window."
