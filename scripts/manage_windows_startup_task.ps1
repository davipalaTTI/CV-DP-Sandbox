param(
    [ValidateSet("Status", "Remove")]
    [string]$Operation = "Status",
    [string]$TaskName = "CV-DP Camera Scheduler"
)

$ErrorActionPreference = "Stop"
$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($Operation -eq "Status") {
    if ($null -eq $Task) {
        [ordered]@{
            installed = $false
            task_name = $TaskName
            state = "Not installed"
            manifest = ""
            last_run_time = ""
            last_result = ""
        } | ConvertTo-Json -Compress
        exit 0
    }

    $TaskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
    $TaskAction = @($Task.Actions)[0]
    $Arguments = [string]$TaskAction.Arguments
    $ManifestPath = ""
    if ($Arguments -match '--manifest\s+"([^"]+)"') {
        $ManifestPath = $Matches[1]
    }
    $LastRunTime = ""
    if ($null -ne $TaskInfo.LastRunTime -and $TaskInfo.LastRunTime.Year -gt 1900) {
        $LastRunTime = $TaskInfo.LastRunTime.ToString("yyyy-MM-dd HH:mm:ss")
    }

    [ordered]@{
        installed = $true
        task_name = $TaskName
        state = [string]$Task.State
        manifest = $ManifestPath
        last_run_time = $LastRunTime
        last_result = $TaskInfo.LastTaskResult
    } | ConvertTo-Json -Compress
    exit 0
}

if ($null -eq $Task) {
    Write-Host "Startup task is not installed: $TaskName"
    exit 0
}

$TaskAction = @($Task.Actions)[0]
$Arguments = [string]$TaskAction.Arguments
$ManifestPath = ""
if ($Arguments -match '--manifest\s+"([^"]+)"') {
    $ManifestPath = $Matches[1]
}
$StopFile = if ($ManifestPath) { "$ManifestPath.scheduler-stop" } else { "" }

if ($Task.State -eq "Running" -and $StopFile) {
    Set-Content -LiteralPath $StopFile -Value "stop" -Encoding ASCII
    $Deadline = (Get-Date).AddSeconds(45)
    do {
        Start-Sleep -Milliseconds 500
        $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    } while ($null -ne $Task -and $Task.State -eq "Running" -and (Get-Date) -lt $Deadline)
}

if ($null -ne $Task -and $Task.State -eq "Running") {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
if ($StopFile) {
    Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
}

Write-Host "Removed startup task: $TaskName"
Write-Host "Saved manifests, camera configs, and output data were not deleted."
