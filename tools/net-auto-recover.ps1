<#
.SYNOPSIS
    net-auto-recover.ps1 — 网络坏掉时自动做一次 WiFi 重连（给 MCode Hook 用）。

.DESCRIPTION
    这台机器的网络通道会间歇性掐断（git push 空回复、HF 桶下载卡死、curl 52/56、
    DNS 超时）。手工处理很贵：每次都要判断「是不是网络死了」⇒「要不要断网换出口 IP」，
    而断网会改用户设备状态。

    ⇒ 本脚本把这个决策收进一条可审计的规则：
      ① 先探针：真的发 HTTPS 请求（**不是**只做 TCP 握手）—— 因为本机最常见的故障是
         「隧道还通、但出口节点坏了」（curl 52/56），TCP 探针抓不到那种坏；
      ② 再冷却：上一次循环在冷却窗口内 ⇒ 不重复断（防抖动）；
      ③ 最后才动手：把 tools/wifi-cycle.ps1 后台脱管启动（钩子不等它跑完），
         由 wifi-cycle.ps1 自己负责「先解析重连目标、解析不出就不断、断开前先武装看门狗」。

    ⇒ 本脚本自己不断网：它只负责探测 + 冷却 + 脱管启动；
       真正的断开/重连全在 tools/wifi-cycle.ps1 里，那里有硬安全规则。

    ⇒ 输出纪律：正常路径**一个字都不往 stdout/stderr 打**（钩子的 stderr 会进 Agent 上下文，
       每次调用都刷屏 = 污染上下文）⇒ 日志只写状态目录的文件。

.PARAMETER DryRun
    只探测与决策，不启动任何循环（用于验证钩子本身）。

.PARAMETER Force
    不管探测结果如何都走一次循环（人工测试用；钩子不会传这个参数）。

.PARAMETER CooldownSeconds
    两次循环之间的最小间隔（秒）。默认 300 ⇒ 防「一断一断地抖」。

.PARAMETER StateDir
    状态/日志目录。默认 ${MINIMAX_HOOKS_DATA}（MCode 给 Hook 的持久目录），
    没有就退回 $env:TEMP\net-auto-recover。

.EXAMPLE
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\net-auto-recover.ps1 -DryRun
    只探测网络并打印决策，不动网卡。

.EXAMPLE
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\net-auto-recover.ps1 -SelfTest
    自测（纯逻辑：冷却窗口 / 探测端点表 / 脚本定位），不碰网卡。
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Force,
    [switch]$SelfTest,
    [switch]$SimulateDown,
    [int]$CooldownSeconds = 300,
    [int]$DelaySeconds = 30,
    [string]$StateDir = ""
)

$ErrorActionPreference = 'Continue'
Set-StrictMode -Version Latest

# ⚠️ 本脚本要同时跑在 **PowerShell 7** 与 **Windows PowerShell 5.1** 上：
#    钩子的 Windows 入口是 `tools/hook-run.cmd`（**优先 pwsh 7、回落 5.1**，
#    理由与 cmd 的编码/引号坑都写在那份 .cmd 的注释里）⇒
#    ⇒ 下面几处写法是刻意保守的：
#    · $PSCommandPath 在脚本作用域取一次（函数内的 $MyInvocation.MyCommand.Path 是空的）；
#    · 计数器放脚本作用域（嵌套函数里的 $script: 才是同一份）；
#    · 任何可能为空的结果都先 @() 包一层再取 .Count（5.1 上 $null.Count 会炸）；
#    · epoch 用**纯 UTC 算术**（`[DateTime]::UtcNow - (Get-Date)` 在 UTC+8 上是负数）。
$script:ScriptPath = $PSCommandPath
$script:fail = 0

# 探测端点：**全部**失败才判「网络坏了」⇒ 单个站点抽风不会误触发。
# 顺序即优先级：先试 HF（本项目主要依赖），再试 GitHub。
$ProbeEndpoints = @(
    @{ Url = 'https://huggingface.co/api/whoami-v2'; TimeoutSec = 3 },
    @{ Url = 'https://api.github.com/zen'; TimeoutSec = 2 }
)

$script:LogFile = $null

function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $line = "[$ts][$LEVEL] $Message"
    if ($script:LogFile) {
        try { Add-Content -LiteralPath $script:LogFile -Value $line -Encoding UTF8 } catch { }
    } else {
        Write-Host $line
    }
}

function Write-Notice {
    # 只有「真的做了动作」才往 stderr 打（PostToolUse 的 stderr 会进 Agent 上下文）
    param([string]$Message)
    try { [Console]::Error.WriteLine($Message) } catch { Write-Host $Message }
}

function Resolve-StateDir {
    if ($StateDir) { return $StateDir }
    if ($env:MINIMAX_HOOKS_DATA) { return $env:MINIMAX_HOOKS_DATA }
    return (Join-Path $env:TEMP 'net-auto-recover')
}

function Get-StateFile {
    param([string]$Dir)
    return (Join-Path $Dir 'net-recover-state.json')
}

function Get-UtcEpoch {
    # ⚠️ 别写成 `[DateTime]::UtcNow - (Get-Date)`：那是拿 UTC 减**本地**时间，
    #    在 UTC+8 上算出的是「-8 小时」这种负数 ⇒ epoch 完全不对。
    #    这里用**纯 UTC 算术**（UTC epoch 显式构造，不依赖本地时区）。
    $epoch = New-Object DateTime 1970, 1, 1, 0, 0, 0, ([DateTimeKind]::Utc)
    return [double]([DateTime]::UtcNow - $epoch).TotalSeconds
}

function Read-LastCycle {
    param([string]$Path)
    try {
        if (-not (Test-Path -LiteralPath $Path)) { return 0.0 }
        $j = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($null -eq $j.last_cycle_utc) { return 0.0 }
        return [double]$j.last_cycle_utc
    } catch {
        return 0.0
    }
}

function Write-LastCycle {
    param([string]$Path, [double]$Utc)
    try {
        $dir = Split-Path -Parent $Path
        if (-not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
        @{ last_cycle_utc = $Utc; last_cycle_local = (Get-Date).ToString('s') } |
            ConvertTo-Json | Set-Content -LiteralPath $Path -Encoding UTF8
    } catch {
        Write-Log "写状态文件失败（不致命）: $($_.Exception.Message)" 'WARN'
    }
}

function Test-Endpoint {
    <#
      真发一次 HTTPS 请求。
      判定「通」的标准不是 200，而是**拿到了 HTTP 响应**（401/403 也说明链路是通的）；
      只有「连不上/超时/证书/TLS 层失败」才算不通。
    #>
    param([string]$Url, [int]$TimeoutSec = 3)
    try {
        $resp = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSec -UseBasicParsing -MaximumRedirection 2
        return ($null -ne $resp)
    } catch {
        $r = $null
        try { $r = $_.Exception.Response } catch { $r = $null }
        if ($null -ne $r) {
            # 有 HTTP 响应（例如 401/403）⇒ 链路通，只是没给凭证
            Write-Log "端点 $Url 返回 HTTP 错误但链路是通的: $($_.Exception.Message)" 'OK'
            return $true
        }
        return $false
    }
}

function Test-NetworkDown {
    # 依次探端点：**任一**通 ⇒ 网络可用；**全**不通 ⇒ 网络死了。
    $alive = @()
    foreach ($e in $ProbeEndpoints) {
        if (Test-Endpoint -Url $e.Url -TimeoutSec $e.TimeoutSec) { $alive += $e.Url }
    }
    if ($alive.Count -gt 0) {
        Write-Log "探测：连通 $($alive -join ',') ⇒ 网络可用" 'OK'
        return $false
    }
    Write-Log "探测：全部端点不通 ⇒ 判为网络不可用" 'WARN'
    return $true
}

function Resolve-WifiCycleScript {
    # 定位 tools/wifi-cycle.ps1：先看本脚本旁边（tools/），再看仓库根。
    $here = Split-Path -Parent $script:ScriptPath
    if (-not $here) { return $null }
    $cand = Join-Path $here 'wifi-cycle.ps1'
    if (Test-Path -LiteralPath $cand) { return (Resolve-Path -LiteralPath $cand).Path }
    $root = Split-Path -Parent $here
    if (-not $root) { return $null }
    $cand2 = Join-Path $root 'tools\wifi-cycle.ps1'
    if (Test-Path -LiteralPath $cand2) { return (Resolve-Path -LiteralPath $cand2).Path }
    return $null
}

function Start-CycleDetached {
    # 脱管启动：钩子的超时上限 10s，而一次循环要 30s+ ⇒ 同步跑必被杀掉，
    # 而被杀在「已断开、还没重连」的窗口里 = 机器断网 ⇒ 绝不能同步跑。
    param([string]$Script, [string]$Dir, [int]$Delay)
    $stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
    $out = Join-Path $Dir "wifi-cycle-$stamp.log"
    $err = Join-Path $Dir "wifi-cycle-$stamp.err"
    try {
        Start-Process -FilePath 'powershell.exe' `
            -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Script,
                            '-DelaySeconds', "$Delay") `
            -WindowStyle Hidden `
            -RedirectStandardOutput $out -RedirectStandardError $err | Out-Null
        return $out
    } catch {
        Write-Log "启动 wifi-cycle.ps1 失败: $($_.Exception.Message)" 'ERROR'
        return $null
    }
}

function Invoke-SelfTest {
    $script:fail = 0
    function Chk {
        param([string]$Name, [bool]$Cond)
        if ($Cond) { Write-Host "OK  $Name" } else { Write-Host "FAIL $Name"; $script:fail = $script:fail + 1 }
    }
    $badScheme = @($ProbeEndpoints | Where-Object { $_.Url -notlike 'https://*' })
    Chk '探测端点表非空且都是 https' (($ProbeEndpoints.Count -ge 2) -and ($badScheme.Count -eq 0))
    Chk '冷却窗口为正' ($CooldownSeconds -gt 0)
    Chk '断网秒数为正' ($DelaySeconds -gt 0)

    # 冷却逻辑（纯函数级验证）
    $tmp = Join-Path $env:TEMP ("net-recover-selftest-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    $sf = Get-StateFile -Dir $tmp
    $now = Get-UtcEpoch
    Write-LastCycle -Path $sf -Utc $now
    $last = Read-LastCycle -Path $sf
    Chk '状态文件能写能读回（last_cycle_utc > 0）' ($last -gt 0)
    Chk '刚写过 ⇒ 处于冷却窗口内' (($now - $last) -lt $CooldownSeconds)
    Write-LastCycle -Path $sf -Utc ($now - $CooldownSeconds - 60)
    $last2 = Read-LastCycle -Path $sf
    Chk '冷却窗口外的旧记录 ⇒ 不在冷却内' ((($now - $last2)) -ge $CooldownSeconds)
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue

    # 解析 wifi-cycle.ps1 的位置
    $w = Resolve-WifiCycleScript
    Chk '能定位 tools/wifi-cycle.ps1' ($null -ne $w)

    if ($script:fail -eq 0) { Write-Host "OK  net-auto-recover 自测全绿" } else { Write-Host "FAIL $script:fail 项失败" }
    return $script:fail
}

# ------------------------------------------------------------------ main
if ($SelfTest) {
    exit ([int]((Invoke-SelfTest) -gt 0))
}

$dir = Resolve-StateDir
try { if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null } } catch { }
$stateFile = Get-StateFile -Dir $dir
$script:LogFile = Join-Path $dir ("net-recover-" + (Get-Date).ToString('yyyyMMdd') + '.log')

Write-Log "=== 网络自动恢复探测 ==="

if (-not $Force) {
    $last = Read-LastCycle -Path $stateFile
    $nowUtc = Get-UtcEpoch
    if ($last -gt 0 -and ($nowUtc - $last) -lt $CooldownSeconds) {
        $left = [int]($CooldownSeconds - ($nowUtc - $last))
        Write-Log "冷却中（上次循环后 ${left}s 内）⇒ 本次不动作" 'INFO'
        exit 0
    }
}

$down = $false
if ($Force) {
    Write-Log "强制模式：跳过网络判定，直接走一次循环" 'WARN'
    $down = $true
} elseif ($SimulateDown) {
    # 只用于**测试判定分支**（不真的断网）：当作「探测全挂」。
    Write-Log "模拟模式：当作探测全挂（不真的断网）" 'WARN'
    $down = $true
} else {
    $down = Test-NetworkDown
}

if (-not $down) {
    Write-Log "网络正常 ⇒ 不动作" 'OK'
    exit 0
}

$wifi = Resolve-WifiCycleScript
if ($null -eq $wifi) {
    Write-Log "找不到 tools/wifi-cycle.ps1 ⇒ 不动作" 'ERROR'
    exit 0
}

if ($DryRun) {
    Write-Log "[DryRun] 判定：网络不可用 ⇒ 会后台启动 '$wifi'（断 ${DelaySeconds}s）" 'WARN'
    Write-Log "[DryRun] 未做任何断开/重连操作"
    Write-Host "[DryRun] 判定：网络不可用 ⇒ 会后台启动 wifi-cycle.ps1（断 ${DelaySeconds}s）；未做任何断开/重连。"
    exit 0
}

$logOut = Start-CycleDetached -Script $wifi -Dir $dir -Delay $DelaySeconds
if ($null -eq $logOut) {
    Write-Log "启动失败 ⇒ 不动作" 'ERROR'
    exit 0
}
Write-LastCycle -Path $stateFile -Utc (Get-UtcEpoch)
Write-Log "已在后台启动 WiFi 重连（断 ${DelaySeconds}s ⇒ 约 40s 后恢复）；日志: $logOut" 'WARN'
Write-Notice "⚠️ 检测到网络不可用（全部探测端点都不通）⇒ 已自动启动 WiFi 重连（断 ${DelaySeconds}s，约 40 秒后恢复）。请等待约 45 秒后重试刚才失败的命令。"
exit 0
