[CmdletBinding()]
param(
    [ValidateRange(1, 4)]
    [int] $Workers = 4,
    [ValidateSet('all', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5')]
    [string] $Stage = 'all',
    [switch] $SmokeOnly,
    [switch] $SkipSmoke
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'

if ($SmokeOnly -and $SkipSmoke) {
    throw 'SmokeOnly and SkipSmoke are mutually exclusive.'
}

Push-Location (Join-Path $PSScriptRoot '..')
try {
    $logs = Join-Path (Get-Location) 'outputs/adaptive_belief_feedback_logs'
    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $uvCommand = (Get-Command uv -ErrorAction Stop).Source
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'

    function Invoke-LoggedRun {
        param([string] $Job, [string[]] $Arguments)
        $logFile = Join-Path $logs "$($stamp)_$Job.log"
        & $uvCommand @Arguments 2>&1 | Tee-Object -FilePath $logFile
        $nativeStatus = $LASTEXITCODE
        if ($nativeStatus -ne 0) {
            throw "$Job exited with code $nativeStatus. Log: $logFile"
        }
    }

    Invoke-LoggedRun -Job 'tests' -Arguments @('run', '--extra', 'dev', '--extra', 'research', 'pytest', '-q')
    $entry = @('run', '--extra', 'research', 'python', 'scripts/adaptive_belief_feedback_study.py',
               '--workers', $Workers.ToString(), '--stage', $Stage)
    if (-not $SkipSmoke) {
        Invoke-LoggedRun -Job 'smoke' -Arguments ($entry + @('--smoke'))
    }
    if (-not $SmokeOnly) {
        Invoke-LoggedRun -Job 'study' -Arguments $entry
    }
}
finally {
    Pop-Location
}
