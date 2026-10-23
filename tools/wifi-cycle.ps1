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

      0. RESOLVE THE RECONNECT TARGET FIRST.  Before touching the adapter it
         determines WHICH WiFi to reconnect to: an explicit -ProfileName, else
         the profile name reported by `netsh wlan show interfaces`, else the
         current SSID.  If none of those resolve, the script **exits without
         disconnecting** (exit 3) -- a cycle whose reconnect target is unknown
         is a cycle that can strand the machine offline.
         (This was a real defect found on 2026-10-23: the script probed the
         current profile/SSID but then never passed it to the reconnect, so the
         only fallback was `netsh wlan connect` = "most recently used
         profile", which may not be the network you were on.)
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
    the adapter goes down, the reconnect target is known AND a reconnect is
    already scheduled by an independent process.

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
    * The two sources can DISAGREE right after a re-association (observed
      2026-10-22 22:11): `Get-NetAdapter` said MediaConnectionState=Disconnected
      while `netsh wlan show interfaces` said "connected" with 98% signal and
      the traffic actually worked. The old code trusted only the CIM state and
      therefore reported a FAILED cycle for a successful one.
      => `netsh` (the WLAN service's own view) is authoritative for association;
         ANY source saying Connected wins, and only when BOTH say Disconnected
         do we report a negative. The FINAL authoritative check decides the
         exit code, so a false negative on a per-attempt probe cannot turn a
         successful cycle into a reported failure.

.PARAMETER DelaySeconds
    Seconds to stay disconnected. Default 30.

.PARAMETER MarginSeconds
    Extra headroom the watchdog waits on top of DelaySeconds, so the
    frontend's reconnect happens first and the watchdog is the fallback.
    Default 20.

.PARAMETER ProfileName
    Optional WiFi profile to reconnect to. When empty, the script auto-detects
    the target: first the profile name reported by `netsh wlan show interfaces`,
    then the current SSID. If neither can be resolved the script refuses to
    disconnect (exit 3) rather than risk an unrecoverable cycle.

.PARAMETER RetryTimes
    How many times to retry the reconnect. Default 3.

.PARAMETER VerifyTimeoutSec
    How long to wait for the adapter to report "connected" after each
    reconnect attempt. Default 60.

.EXAMPLE
    pwsh -NoProfile -File .\wifi-cycle.ps1

.EXAMPLE
    pwsh -NoProfile -File .\wifi-cycle.ps1 -DelaySeconds 30 -ProfileName "MyWiFi"

.EXAMPLE
    pwsh -NoProfile -File .\wifi-cycle.ps1 -ResolveOnly
    只解析并打印「断开后要连回的 WiFi 名称」，不碰网卡（不冒断网风险）。
#>
[CmdletBinding()]
param(
    [int]$DelaySeconds = 30,
    [int]$MarginSeconds = 20,
    [string]$ProfileName = "",
    [int]$RetryTimes = 3,
    [int]$VerifyTimeoutSec = 60,
    [int]$ProbeTimeoutSec = 6,
    [switch]$ResolveOnly
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

function Resolve-ReconnectTarget {
    # 决定「断开之后连回哪个 WiFi」。
    #
    # ⚠️ 这里曾经是个真缺陷（2026-10-23 用户指出）：
    #    脚本**探测**到了当前 profile/SSID，却**没有把它传给重连**——
    #    `Invoke-Reconnect` 与看门狗只认 `-ProfileName` 参数，而它默认是空，
    #    于是回退到 `netsh wlan connect`（"最近使用的配置"）——
    #    那个猜测可能指到别的网络，结果就是**断了连不回来**。
    #
    # 现在的规则：显式参数 > 探测到的 profile 名 > 探测到的 SSID。
    # 三者都没有 ⇒ 返回 $null，调用方**必须拒绝断开**。
    if ($ProfileName) { return $ProfileName }
    $p = Get-ProfileLabel
    if ($p) { return $p }
    $s = Get-CurrentSsid
    if ($s) { return $s }
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

function Get-NetshStateFast {
    # `netsh wlan show interfaces` is the WLAN service's OWN view of association
    # and is the authoritative source for "is this interface associated right now".
    # It can block while re-associating, so it also runs under a hard timeout.
    # Returns the raw State string (e.g. 'connected' / '已连接'), or $null.
    param([int]$TimeoutSec = 8)
    $job = Start-Job -ScriptBlock {
        $raw = netsh wlan show interfaces 2>$null | Out-String
        if ([string]::IsNullOrWhiteSpace($raw)) { return $null }
        $m = [regex]::Match($raw, '(?im)^\s*State\s*:\s*(.+?)\s*$')
        if (-not $m.Success) {
            $m = [regex]::Match($raw, '(?im)^\s*状态\s*:\s*(.+?)\s*$')
        }
        if (-not $m.Success) { return $null }
        return $m.Groups[1].Value.Trim()
    }
    try {
        if (Wait-Job $job -Timeout $TimeoutSec) {
            $v = Receive-Job $job -ErrorAction SilentlyContinue
            if ($null -ne $v) { return ("$v").Trim() }
        }
        return $null
    } finally {
        Remove-Job $job -Force -ErrorAction SilentlyContinue
    }
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
    #
    # TWO independent sources, because they can disagree right after a
    # re-association (observed 2026-10-22 22:11: Get-NetAdapter reported
    # MediaConnectionState=Disconnected while `netsh wlan show interfaces`
    # reported "connected" with 98% signal, and the traffic worked).
    #   - netsh  = the WLAN service's own view -> authoritative for association
    #   - CIM    = the adapter object, can lag / report a stale state
    # => ANY source saying Connected wins; only when BOTH say Disconnected
    #    do we treat it as a negative answer.
    $net = Get-MediaConnectionStateFast -Name $script:WifiName -TimeoutSec $ProbeTimeoutSec
    $sh = Get-NetshStateFast -TimeoutSec $ProbeTimeoutSec
    $netConnected = ($net -eq 'Connected')
    $shConnected = ($sh -match '^(connected|已连接|已連接)$')
    $netDisc = ($net -eq 'Disconnected')
    $shDisc = ($sh -match '^(disconnected|已断开|未连接|已斷開)$')
    if ($netConnected -or $shConnected) { return $true }
    if ($netDisc -and $shDisc) { return $false }
    return $null
}

function Invoke-Reconnect {
    # 连回**已解析好的目标名**（$script:ReconnectName），不再靠"最近使用的配置"猜。
    $name = $script:ReconnectName
    if ($name) {
        Write-Step "netsh wlan connect name='$name'"
        netsh wlan connect name="$name" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            # profile 名与 SSID 不同、或 profile 不在库里时，带 ssid= 更稳
            if ($script:Ssid -and $script:Ssid -ne $name) {
                Write-Step "  带 ssid 重试：name='$name' ssid='$($script:Ssid)'"
                netsh wlan connect name="$name" ssid="$($script:Ssid)" 2>&1 | Out-Null
            }
            if ($LASTEXITCODE -ne 0) {
                Write-Step "  指定名连接失败，回退到'最近使用的配置'" "WARN"
                netsh wlan connect 2>&1 | Out-Null
            }
        }
    } else {
        Write-Step "netsh wlan connect (last used profile) —— **未解析到目标名**" "WARN"
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
    # 看门狗必须带着**已解析的目标名**；只有名字才能保证真的连得回去。
    $connectLine = if ($Profile) {
        'netsh wlan connect name="{0}" 2>&1 | Out-Null' -f ($Profile -replace '"', '`"')
    } else {
        'netsh wlan connect 2>&1 | Out-Null'
    }
    $fallbackLine = if ($Profile) {
        'if ($LASTEXITCODE -ne 0) { netsh wlan connect 2>&1 | Out-Null }'
    } else {
        ''
    }

    @(
        '# Detached WiFi reconnect watchdog (auto-generated, self-deleting).'
        '$ErrorActionPreference = "Continue"'
        'Start-Sleep -Seconds {0}' -f $DelaySec
        $connectLine
        $fallbackLine
        'Start-Sleep -Seconds 8'
        $connectLine
        $fallbackLine
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
$script:Ssid = $ssid
if ($ssid) { Write-Step "current SSID: $ssid" } else { Write-Step "current SSID: <unknown>" "WARN" }
$prof = Get-ProfileLabel
if ($prof) { Write-Step "current profile: $prof" }
if ($script:WifiName) { Write-Step "active interface: $($script:WifiName)" }
if ($ProfileName) { Write-Step "target profile (explicit): $ProfileName" }

# --- 断开之前必须先解决「连回哪个 WiFi」 ---
# ⚠️ 2026-10-23 用户指出的真缺陷：探测到的 profile/SSID 以前**没有**传给重连，
#    断开后只能靠 `netsh wlan connect`（"最近使用的配置"）去猜 ——
#    猜错就再也连不回来。现在：**解析不出目标名 ⇒ 根本不断开**。
$script:ReconnectName = Resolve-ReconnectTarget
if (-not $script:ReconnectName) {
    Write-Step "解析不到要重连的 WiFi 名称（-ProfileName 没给，profile/SSID 也探测不到）。" "ERROR"
    Write-Step '**不切断连接**：断了就可能连不回来。请用 -ProfileName "<你的WiFi名>" 再跑。' "ERROR"
    exit 3
}
Write-Step "reconnect target: '$($script:ReconnectName)'" "OK"

if ($ResolveOnly) {
    # 只解析目标、**完全不碰网卡**。用来验证「能不能连回去」这件事，
    # 而不必真的断开一次网络。
    Write-Step "resolve-only 模式：已解析出重连目标，未做任何断开/重连操作。"
    exit 0
}

$state0 = Get-MediaConnectionStateFast -Name $script:WifiName -TimeoutSec $ProbeTimeoutSec
Write-Step "media state now: $state0"
if ($state0 -ne 'Connected') {
    Write-Step "WiFi is NOT associated right now - nothing to cycle. Aborting safely." "WARN"
    exit 2
}

# 1) arm the watchdog BEFORE the disconnect（看门狗带着**已解析的目标名**）
$watchdog = Start-Watchdog -DelaySec ($DelaySeconds + $MarginSeconds) -Profile $script:ReconnectName

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
$finalNetsh = Get-NetshStateFast -TimeoutSec $ProbeTimeoutSec
$bssid = Get-WifiBssid
$newSsid = Get-CurrentSsid

# The FINAL authoritative check decides the outcome. A per-attempt probe that
# returned a false negative (see Test-WifiAssociated) must not turn a
# successful cycle into a reported failure.
$actuallyUp = (($finalState -eq 'Connected') -or
    ($finalNetsh -match '^(connected|已连接|已連接)$') -or
    ([bool]$bssid))

if ($ok -or $actuallyUp) {
    Write-Step "reconnected: SSID='$newSsid' state=$finalState netsh='$finalNetsh' BSSID=$bssid" "OK"
    if (-not $ok) {
        Write-Step "note: the per-attempt probe gave a false negative; the final authoritative check says we ARE associated." "WARN"
    }
    Write-Step "cycle complete"
    exit 0
}

Write-Step "verification did not confirm association (state=$finalState netsh='$finalNetsh' BSSID=$bssid)." "ERROR"
Write-Step 'manual recovery: netsh wlan show profiles / netsh wlan connect name="<profile>"' "ERROR"
exit 1
