param(
    [ValidateSet("List", "Status", "Stop", "Remove")]
    [string]$Operation = "Status",
    [string]$TaskName = "CV-DP Camera Scheduler",
    [string]$TaskPath = "\"
)

$ErrorActionPreference = "Stop"

function Get-TaskManifest {
    param([object]$Task)

    $TaskAction = @($Task.Actions)[0]
    if ($null -eq $TaskAction) {
        return ""
    }
    $Arguments = [string]$TaskAction.Arguments
    if ($Arguments -match '--manifest(?:\s+|=)(?:"([^"]+)"|''([^'']+)''|(\S+))') {
        foreach ($Index in 1..3) {
            if ($Matches[$Index]) {
                return $Matches[$Index]
            }
        }
    }
    return ""
}

function Convert-TaskStatus {
    param([object]$Task)

    $TaskInfo = Get-ScheduledTaskInfo -TaskName $Task.TaskName -TaskPath $Task.TaskPath
    $TaskAction = @($Task.Actions)[0]
    $LastRunTime = ""
    if (
        $TaskInfo.LastTaskResult -ne 267011 -and
        $null -ne $TaskInfo.LastRunTime -and
        $TaskInfo.LastRunTime.Year -gt 1900
    ) {
        $LastRunTime = $TaskInfo.LastRunTime.ToString("yyyy-MM-dd HH:mm:ss")
    }
    $LastResult = if ($TaskInfo.LastTaskResult -eq 267011) {
        "Never run"
    }
    elseif ($TaskInfo.LastTaskResult -eq 0) {
        "Success"
    }
    else {
        [string]$TaskInfo.LastTaskResult
    }
    return [ordered]@{
        installed = $true
        task_name = [string]$Task.TaskName
        task_path = [string]$Task.TaskPath
        state = [string]$Task.State
        manifest = Get-TaskManifest -Task $Task
        last_run_time = $LastRunTime
        last_result = $LastResult
        executable = if ($null -ne $TaskAction) { [string]$TaskAction.Execute } else { "" }
        arguments = if ($null -ne $TaskAction) { [string]$TaskAction.Arguments } else { "" }
    }
}

function Get-CVDPTasks {
    return @(Get-ScheduledTask | Where-Object {
        $Task = $_
        $ActionText = (@($Task.Actions) | ForEach-Object {
            "{0} {1}" -f ([string]$_.Execute), ([string]$_.Arguments)
        }) -join " "
        $Task.TaskName -eq "CV-DP Camera Scheduler" -or
        ($ActionText -match '(?i)scheduled_runner\.py' -and $ActionText -match '(?i)--manifest(?:\s|=)')
    })
}

function Get-SelectedTask {
    return Get-ScheduledTask `
        -TaskName $TaskName `
        -TaskPath $TaskPath `
        -ErrorAction SilentlyContinue
}

function Stop-CVDPTask {
    param([object]$Task)

    if ($null -eq $Task -or $Task.State -ne "Running") {
        return
    }

    $ManifestPath = Get-TaskManifest -Task $Task
    $StopFile = if ($ManifestPath) { "$ManifestPath.scheduler-stop" } else { "" }
    if ($StopFile) {
        try {
            Set-Content -LiteralPath $StopFile -Value "stop" -Encoding ASCII
            $Deadline = (Get-Date).AddSeconds(45)
            do {
                Start-Sleep -Milliseconds 500
                $Task = Get-ScheduledTask `
                    -TaskName $Task.TaskName `
                    -TaskPath $Task.TaskPath `
                    -ErrorAction SilentlyContinue
            } while ($null -ne $Task -and $Task.State -eq "Running" -and (Get-Date) -lt $Deadline)
        }
        catch {
            Write-Warning "Graceful stop marker could not be written: $($_.Exception.Message)"
        }
    }

    if ($null -ne $Task -and $Task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $Task.TaskName -TaskPath $Task.TaskPath -ErrorAction Stop
    }
    if ($StopFile) {
        Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
    }
}

if ($Operation -eq "List") {
    $Rows = @(Get-CVDPTasks | ForEach-Object { Convert-TaskStatus -Task $_ })
    if ($Rows.Count -eq 0) {
        Write-Output "[]"
    }
    else {
        ConvertTo-Json -InputObject @($Rows) -Compress -Depth 4
    }
    exit 0
}

$Task = Get-SelectedTask

if ($Operation -eq "Status") {
    if ($null -eq $Task) {
        [ordered]@{
            installed = $false
            task_name = $TaskName
            task_path = $TaskPath
            state = "Not installed"
            manifest = ""
            last_run_time = ""
            last_result = ""
        } | ConvertTo-Json -Compress
        exit 0
    }

    Convert-TaskStatus -Task $Task | ConvertTo-Json -Compress -Depth 4
    exit 0
}

if ($null -eq $Task) {
    Write-Host "Startup task is not installed: $TaskPath$TaskName"
    exit 0
}

Stop-CVDPTask -Task $Task
if ($Operation -eq "Stop") {
    Write-Host "Stopped startup task: $TaskPath$TaskName"
    Write-Host "The task remains registered and will run again at the next configured trigger."
    exit 0
}

Unregister-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Confirm:$false
Write-Host "Removed startup task: $TaskPath$TaskName"
Write-Host "Saved manifests, camera configs, and output data were not deleted."
