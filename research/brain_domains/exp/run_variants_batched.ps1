# run_variants_batched.ps1 — 阶段 D 第二轮：**分批**跑（每批 4 个），等上一批完成再发下一批。

# 为什么分批：这台机器上 **20 个并发会把 Job 饿死**（实测：20 个并发时
# 每个 Job 40 分钟还没跑完第一个被试，running_secs 只涨 100 s）；
# 降到 4 并发后每个 Job ~11 分钟跑完 10 个被试（与 running_secs≈650 s 吻合）。
# ⇒ 并发上限取 4。

param(
    [int]$BatchSize = 4,
    [string]$S1aCkpt = "/domains/runs/illusionbrain-align/exp80/k1024/tokenizer_sub-01.pt"
)

# ⚠️ 不能用 'Stop'：`hf jobs run` 会把一部分正常输出写 stderr
#    （如「Job configuration:」）⇒ 在 Stop 模式下会被当成错误而中断整个脚本。
$ErrorActionPreference = 'Continue'

$vol = @(
    "-v", "hf://buckets/illusion-brain/training-data:/training:ro",
    "-v", "hf://buckets/illusion-brain/brain-domains:/domains",
    "--flavor", "cpu-upgrade", "--timeout", "90m", "-d"
)

# (被试, 趟) 清单：stable 剩 05-10，dev 全部 01-10
$jobs = @()
foreach ($s in 5..10) { $jobs += @(@{ sid = '{0:d2}' -f $s; trip = "stable" }) }
foreach ($s in 1..10) { $jobs += @(@{ sid = '{0:d2}' -f $s; trip = "dev" }) }

function Get-JobState([string]$id) {
    $j = hf jobs inspect "redmingwei/$id" --format json 2>$null | ConvertFrom-Json
    if ($j -is [array]) { $j = $j[0] }
    if ($j.status -is [string]) { return $j.status }
    return $j.status.stage
}

function Find-Existing([string]$name) {
    # 同名 Job 已存在且未进入终态 ⇒ 跳过（避免重复发）
    $j = hf jobs list -a --format json 2>$null | ConvertFrom-Json
    foreach ($x in $j) {
        if ($x.name -eq $name) {
            $st = if ($x.status -is [string]) { $x.status } else { $x.status.stage }
            if ($st -notin @("COMPLETED", "ERROR", "CANCELED")) { return $x.id }
        }
    }
    return $null
}

$pending = @()
foreach ($j in $jobs) {
    $trip = $j.trip
    $name = "t77f-s$($j.sid)-$trip"
    $exist = Find-Existing $name
    if ($exist) {
        Write-Host ("[{0}] {1} 已在跑（{2}）⇒ 跳过" -f (Get-Date -Format 'HH:mm:ss'), $name, $exist)
        $pending += @{ name = $name; id = $exist }
    } else {
    if ($trip -eq "stable") {
        $cmd = @("python", "-u", "/domains/scripts/exp77_variant_eval.py",
            "--bd-version", "stable", "--donors", "signal-jepa,s1a-recon",
            "--s1a-ckpt", $S1aCkpt, "--only-subjects", $j.sid,
            "--out-dir", "/domains/runs/illusionbrain-align/variants3/stable")
    } else {
        $cmd = @("python", "-u", "/domains/scripts/exp77_variant_eval.py",
            "--bd-version", "dev", "--donors", "mirepnet,braintokenizer",
            "--only-subjects", $j.sid,
            "--out-dir", "/domains/runs/illusionbrain-align/variants3/dev")
    }
    $out = hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" @cmd @vol --name $name 2>&1
    $id = ($out | Select-String -Pattern 'id:\s*(\S+)' | Select-Object -First 1).Matches.Groups[1].Value
    if (-not $id) { $id = ($out | Select-String -Pattern 'redmingwei/(\S+)' | Select-Object -First 1).Matches.Groups[1].Value }
    Write-Host ("[{0}] {1} -> {2}" -f (Get-Date -Format 'HH:mm:ss'), $name, $id)
    $pending += @{ name = $name; id = $id }
    }
    if ($pending.Count -ge $BatchSize) {
        Write-Host "--- 等待本批 $($pending.Count) 个完成 ---"
        foreach ($p in $pending) {
            while ($true) {
                $st = Get-JobState $p.id
                if ($st -in @("COMPLETED", "ERROR", "CANCELED")) { break }
                Start-Sleep -Seconds 30
            }
            Write-Host ("  {0} => {1}" -f $p.name, (Get-JobState $p.id))
        }
        $pending = @()
    }
}
if ($pending.Count -gt 0) {
    Write-Host "--- 等待最后一批 ---"
    foreach ($p in $pending) {
        while ($true) {
            $st = Get-JobState $p.id
            if ($st -in @("COMPLETED", "ERROR", "CANCELED")) { break }
            Start-Sleep -Seconds 30
        }
        Write-Host ("  {0} => {1}" -f $p.name, (Get-JobState $p.id))
    }
}
Write-Host "全部批次完成"
