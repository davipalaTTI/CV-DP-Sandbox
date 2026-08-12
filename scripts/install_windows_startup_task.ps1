param(
    [Parameter(Mandatory = $true)]
    [string]$Manifest,
    [string]$TaskName = "CV-DP Camera Scheduler",
    [string]$PythonExecutable = "",
    [string]$ViewerUserSid = "",
    [string]$OperationLog = "",
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $OperationLog) {
    $OperationLog = Join-Path $ProjectRoot "logs\startup_service_operation.log"
}
$LogDirectory = Split-Path -Parent $OperationLog
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
Set-Content -LiteralPath $OperationLog -Encoding UTF8 -Value @(
    "Operation=Install Windows startup task"
    "Started=$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    "TaskName=$TaskName"
    "Manifest=$Manifest"
    "ViewerUserSid=$ViewerUserSid"
)

function Grant-TaskReadAccess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$UserSid
    )

    $Identity = [System.Security.Principal.SecurityIdentifier]::new($UserSid)
    $TaskService = New-Object -ComObject "Schedule.Service"
    $TaskService.Connect()
    $TaskFolder = $TaskService.GetFolder("\")
    $ComTask = $TaskFolder.GetTask($Name)
    $ExistingSddl = $ComTask.GetSecurityDescriptor(0x7)
    $Descriptor = [System.Security.AccessControl.RawSecurityDescriptor]::new(
        $ExistingSddl
    )

    # FILE_GENERIC_READ | FILE_GENERIC_EXECUTE. Management operations still use UAC.
    $ReadExecuteMask = 0x1200A9
    $HasAccess = $false
    foreach ($ExistingAce in $Descriptor.DiscretionaryAcl) {
        if (
            $ExistingAce.AceQualifier -eq `
                [System.Security.AccessControl.AceQualifier]::AccessAllowed -and
            $null -ne $ExistingAce.SecurityIdentifier -and
            $ExistingAce.SecurityIdentifier.Value -eq $Identity.Value -and
            ($ExistingAce.AccessMask -band $ReadExecuteMask) -eq $ReadExecuteMask
        ) {
            $HasAccess = $true
            break
        }
    }
    if (-not $HasAccess) {
        $Ace = [System.Security.AccessControl.CommonAce]::new(
            [System.Security.AccessControl.AceFlags]::None,
            [System.Security.AccessControl.AceQualifier]::AccessAllowed,
            $ReadExecuteMask,
            $Identity,
            $false,
            $null
        )
        $Descriptor.DiscretionaryAcl.InsertAce(
            $Descriptor.DiscretionaryAcl.Count,
            $Ace
        )
        $Sections = `
            [System.Security.AccessControl.AccessControlSections]::Owner -bor `
            [System.Security.AccessControl.AccessControlSections]::Group -bor `
            [System.Security.AccessControl.AccessControlSections]::Access
        $ComTask.SetSecurityDescriptor($Descriptor.GetSddlForm($Sections), 0)
    }
}

try {
    $Runner = Join-Path $ProjectRoot "scripts\scheduled_runner.py"
    $ManifestPath = (Resolve-Path -LiteralPath $Manifest -ErrorAction Stop).Path
    $StopFile = "$ManifestPath.scheduler-stop"
    Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue

    if (-not $PythonExecutable) {
        $PythonExecutable = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    }
    $PythonPath = (Resolve-Path -LiteralPath $PythonExecutable -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
        throw "Scheduled runner does not exist: $Runner"
    }

    Add-Content -LiteralPath $OperationLog -Encoding UTF8 -Value @(
        "ProjectRoot=$ProjectRoot"
        "PythonExecutable=$PythonPath"
        "ResolvedManifest=$ManifestPath"
    )

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
        -TaskPath "\" `
        -Description "CV-DP scheduled camera supervisor | Manifest=$ManifestPath" `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Force | Out-Null

    $RegisteredTask = Get-ScheduledTask `
        -TaskName $TaskName `
        -TaskPath "\" `
        -ErrorAction Stop
    if ($RegisteredTask.TaskName -ne $TaskName) {
        throw "Windows did not register startup task: $TaskName"
    }
    if ($ViewerUserSid) {
        Grant-TaskReadAccess -Name $TaskName -UserSid $ViewerUserSid
        Add-Content -LiteralPath $OperationLog -Encoding UTF8 -Value `
            "ViewerAccess=ReadExecute"
    }
    if ($StartNow) {
        Start-ScheduledTask -TaskName $TaskName -TaskPath "\" -ErrorAction Stop
    }

    Add-Content -LiteralPath $OperationLog -Encoding UTF8 -Value @(
        "Status=SUCCESS"
        "Completed=$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    )

    Write-Host "Installed startup task: $TaskName"
    Write-Host "Manifest: $ManifestPath"
    if ($StartNow) {
        Write-Host "The scheduler task was started."
    }
    Write-Host "The scheduler will start at every boot, including boots inside an active window."
}
catch {
    $Failure = ($_ | Out-String).Trim()
    Add-Content -LiteralPath $OperationLog -Encoding UTF8 -Value @(
        "Status=FAILED"
        "Completed=$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
        "Error=$Failure"
    )
    Write-Error $Failure
    exit 1
}
