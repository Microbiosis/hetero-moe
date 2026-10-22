<#
.SYNOPSIS
    WiFi cycle: disconnect now, reconnect automatically after N seconds.

.DESCRIPTION
    HARD SAFETY RULE
    ----------------
    `netsh wlan disconnect` must NEVER be run on its own. Once the link is
    down, a session that dies mid-flight can never come back, and the machine
    is stranded with no network.

    This script makes the disconnect safe by construction:

      1. WATCHDOG FIRST.  Before touching the adapter it spawns a DETACHED
         helper (separate powershell process, hidden window) that sleeps
         (DelaySeconds + MarginSeconds) and then reconnects on its own.
         If this script is killed, crashes, or the whole process tree dies,
         the helper still brings the WiFi back.
      2. FRONTEND RECONNECT.  The foreground also waits and reconnects.
         Both paths are idempotent -- reconnecting twice is harmless.
      3. BOUNDED VERIFICATION.  The connection probe runs in a child process
         with a hard timeout, so a driver that blocks for 40s+ during the
         WLAN handshake can never eat the verify window.

    The disconnect therefore never happens as a standalone step: by the time
    the adapter goes down, a reconnect is already scheduled by an independent
    process.

    IMPLEMENTATION NOTES (learned the hard way on a MediaTek MT7925 box)
    -----------------------------------------------------------------
    * `Get-NetAdapter` without -Name returns several WLAN-ish entries
      (WLAN 2 / WLAN 5 are "Not Present" virtual adapters whose
      MediaConnectionState is 'Unknown'). Picking the first match lands on
      the wrong one. We therefore resolve the ACTIVE interface name from
      `netsh wlan show interfaces` ONCE, before disconnecting, and query
      that name explicitly.
    * `Get-NetAdapter` / `netsh wlan show interfaces` can BLOCK for 40s+
      while the adapter is re-associating. A blocking probe is not a
      negative answer, so the probe is run in a child job with a timeout and
      a timeout is treated as "still transitioning", never as "disconnected".

.PARAMETER DelaySeconds
    Seconds to stay disconnected. Default 30.

.PARAMETER MarginSeconds
    Extra headroom the watchdog waits on top of DelaySeconds, so the
    frontend's reconnect happens first and the watchdog is the fallback.
    Default 20.

.PARAMETER ProfileName
    Optional WiFi profile to reconnect to. When empty, `netsh wlan connect`
    is used, which re-associates the most recently used profile.

.PARAMETER RetryTimes
    How many times to retry the reconnect. Default 3.

.PARAMETER VerifyTimeoutSec
    How long to wait for the adapter to report "connected" after each
    reconnect attempt. Default 60.

.EXAMPLE
    pwsh -NoProfile -File .\wifi-cycle.ps1

.EXAMPLE
    pwsh -NoProfile -File .\wifi-cycle.ps1 -DelaySeconds 30 -ProfileName "MyWiFi"
#>
[CmdletBinding()]
param(
    [int]$DelaySeconds = 30,
    [int]$MarginSeconds = 20,
    [string]$ProfileName = "",
    [int]$RetryTimes = 3,
    [int]$VerifyTimeoutSec = 60,
    [int]$ProbeTimeoutSec = 6
)

$ErrorActionPreference = 'Continue'
Set-StrictMode -Version Latest

$script:WifiName = $null      # active interface name, resolved before we cut the link

# ------------------------------------------------------------------ helpers
function Write-Step {
    param([string]$Message, [string]$Level = "INFO")
    $ts = (Get-Date).ToString("HH:mm:ss")
    Write-Host "[$ts][$Level] $Message"
}

function Get-WlanInterfaces {
    $raw = netsh wlan show interfaces 2>$null | Out-String
    if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
    return $raw
}

function Get-CurrentSsid {
    $raw = Get-WlanInterfaces
    if (-not $raw) { return $null }
    $m = [regex]::Match($raw, '(?im)^\s*SSID\s*:\s*(.+?)\s*$')
    if ($m.Success) { return $m.Groups[1].Value }
    return $null
}

function Get-ProfileLabel {
    $raw = Get-WlanInterfaces
    if (-not $raw) { return $null }
    $m = [regex]::Match($raw, '(?im)^\s*(?:Profile|配置文件)\s*:\s*(.+?)\s*$')
    if ($m.Success) { return $m.Groups[1].Value }
    return $null
}

function Get-ActiveWifiName {
    $raw = Get-WlanInterfaces
    if (-not $raw) { return $null }
    $m = [regex]::Match($raw, '(?im)^\s*Name\s*:\s*(.+?)\s*$')
    if ($m.Success) { return $m.Groups[1].Value }
    return $null
}

function Get-WifiBssid {
    $raw = Get-WlanInterfaces
    if (-not $raw) { return $null }
    # Label is "AP BSSID" on modern Windows, plain "BSSID" on older ones.
    # Separator is spaces (en) or colons (zh) depending on locale.
    $m = [regex]::Match($raw, '(?im)^\s*(?:AP\s+)?BSSID\s*:\s*([0-9a-fA-F]{2}(?:[\s:]+[0-9a-fA-F]{2}){5})')
    if (-not $m.Success) { return $null }
    return ($m.Groups[1].Value -replace '[\s:]', '')
}

function Get-MediaConnectionStateFast {
    # Runs the CIM query in a child job with a hard timeout. The driver can
    # block for 40s+ while re-associating; a timeout means "transitioning",
    # NOT "disconnected".
    param([string]$Name, [int]$TimeoutSec = 6)
    if (-not $Name) { return $null }
    $job = Start-Job -ScriptBlock {
        param($n)
        $a = Get-NetAdapter -Name $n -ErrorAction SilentlyContinue
        if (-not $a) { return "Absent" }
        # IMPORTANT: cast to string INSIDE the child. MediaConnectionState is
        # an enum and remoting serializes it as its integer value (1 = Connected),
        # which would never match a string comparison in the parent.
        return [string]$a.MediaConnectionState
    } -ArgumentList $Name
    try {
        if (Wait-Job $job -Timeout $TimeoutSec) {
            $v = Receive-Job $job -ErrorAction SilentlyContinue
            if ($null -ne $v) {
                $s = ("$v").Trim()
                switch ($s) {
                    '1' { return 'Connected' }
                    '2' { return 'Disconnected' }
                    '0' { return 'Unknown' }
                    default { return $s }
                }
            }
        }
        return $null
    } finally {
        Remove-Job $job -Force -ErrorAction SilentlyContinue
    }
}

function Test-WifiAssociated {
    # $true  = associated
    # $false = definitively not associated
    # $null  = still transitioning / unknown (NOT a negative answer)
    $st = Get-MediaConnectionStateFast -Name $script:WifiName -TimeoutSec $ProbeTimeoutSec
    if ($st -eq 'Connected') { return $true }
    if ($st -eq 'Disconnected') { return $false }
    return $null
}

function Invoke-Reconnect {
    if ($ProfileName) {
        Write-Step "netsh wlan connect name='$ProfileName'"
        netsh wlan connect name="$ProfileName" 2>&1 | Out-Null
    } else {
        Write-Step "netsh wlan connect (last used profile)"
        netsh wlan connect 2>&1 | Out-Null
    }
}

function Wait-WifiAssociated {
    param([int]$TimeoutSec)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ($true) {
        $ok = Test-WifiAssociated
        if ($ok -eq $true) { return $true }
        if ((Get-Date) -ge $deadline) { return $false }
        Start-Sleep -Seconds 2
    }
}

# ------------------------------------------------------------------ watchdog
function Start-Watchdog {
    param([int]$DelaySec, [string]$Profile)

    $wd = Join-Path $env:TEMP ("wifi_reconnect_watchdog_{0}.ps1" -f ([guid]::NewGuid().ToString('N')))
    $connectLine = if ($Profile) {
        'netsh wlan connect name="{0}" 2>&1 | Out-Null' -f ($Profile -replace '"', '`"')
    } else {
        'netsh wlan connect 2>&1 | Out-Null'
    }

    @(
        '# Detached WiFi reconnect watchdog (auto-generated, self-deleting).'
        '$ErrorActionPreference = "Continue"'
        'Start-Sleep -Seconds {0}' -f $DelaySec
        $connectLine
        'Start-Sleep -Seconds 8'
        $connectLine
        'Start-Sleep -Seconds 2'
        'Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force -ErrorAction SilentlyContinue'
    ) | Set-Content -LiteralPath $wd -Encoding UTF8

    # -WindowStyle Hidden + independent process: survives the parent exiting.
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", $wd) `
        -WindowStyle Hidden | Out-Null

    Write-Step "watchdog armed -> $wd (fires in $DelaySec s, detached)"
    return $wd
}

# ------------------------------------------------------------------ main
Write-Step "=== WiFi cycle ===" "INFO"
Write-Step "delay=${DelaySeconds}s margin=${MarginSeconds}s retries=$RetryTimes"

$script:WifiName = Get-ActiveWifiName
$ssid = Get-CurrentSsid
if ($ssid) { Write-Step "current SSID: $ssid" } else { Write-Step "current SSID: <unknown>" "WARN" }
$prof = Get-ProfileLabel
if ($prof) { Write-Step "current profile: $prof" }
if ($script:WifiName) { Write-Step "active interface: $($script:WifiName)" }
if ($ProfileName) { Write-Step "target profile (explicit): $ProfileName" }

$state0 = Get-MediaConnectionStateFast -Name $script:WifiName -TimeoutSec $ProbeTimeoutSec
Write-Step "media state now: $state0"
if ($state0 -ne 'Connected') {
    Write-Step "WiFi is NOT associated right now - nothing to cycle. Aborting safely." "WARN"
    exit 2
}

# 1) arm the watchdog BEFORE the disconnect
$watchdog = Start-Watchdog -DelaySec ($DelaySeconds + $MarginSeconds) -Profile $ProfileName

# 2) only now is it safe to drop the link
Write-Step "disconnecting WiFi ..." "WARN"
netsh wlan disconnect 2>&1 | Out-Null
Start-Sleep -Seconds 2
Write-Step "disconnected. A reconnect is already scheduled (watchdog + frontend)." "WARN"

# 3) frontend reconnect (the watchdog is the fallback)
Write-Step "waiting ${DelaySeconds}s ..."
Start-Sleep -Seconds $DelaySeconds

$ok = $false
for ($i = 1; $i -le $RetryTimes; $i++) {
    Write-Step "reconnect attempt $i/$RetryTimes"
    Invoke-Reconnect
    if (Wait-WifiAssociated -TimeoutSec $VerifyTimeoutSec) {
        $ok = $true
        break
    }
    Write-Step "not associated yet, retrying in 5s" "WARN"
    Start-Sleep -Seconds 5
}

$finalState = Get-MediaConnectionStateFast -Name $script:WifiName -TimeoutSec $ProbeTimeoutSec
$bssid = Get-WifiBssid
$newSsid = Get-CurrentSsid

if ($ok) {
    Write-Step "reconnected: SSID='$newSsid' state=$finalState BSSID=$bssid" "OK"
    Write-Step "cycle complete"
    exit 0
}

Write-Step "verification did not confirm association (state=$finalState BSSID=$bssid)." "ERROR"
if ($finalState -eq 'Connected' -or $bssid) {
    Write-Step "note: the adapter IS actually up -- only the probe failed." "WARN"
}
Write-Step 'manual recovery: netsh wlan show profiles / netsh wlan connect name="<profile>"' "ERROR"
exit 1
