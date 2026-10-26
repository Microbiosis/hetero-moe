# upload_scripts.ps1 — 把 exp/ 下的脚本传到桶里，**带校验重试**。
#
# ## 为什么需要它
#
# 实测（2026-10-23 连续两轮）：`hf buckets cp` 的 **写通道**会随机断
# （`ConnectionError: ... xet-write-token`），同一份文件可能连挂 3 次；
# 而**上传失败没有任何本地提示** ⇒ 一度让 2 个评测 Job 跑在**旧版脚本**上
# （10 秒全 ERROR：`FileNotFoundError: donors/none.pt`）。
#
# ⇒ 规矩：**上传完必须核对桶里的大小 == 本地大小**，不一致就重传。
#
# 用法：
#   pwsh -File research/brain_domains/exp/upload_scripts.ps1
#   pwsh -File research/brain_domains/exp/upload_scripts.ps1 -Files exp87_shared_vocab.py

param(
    [string[]]$Files = @(),
    [int]$Retries = 6
)

$ErrorActionPreference = 'Continue'
$expDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$bucket = "hf://buckets/illusion-brain/brain-domains/scripts"

if ($Files.Count -eq 0) {
    $Files = @("exp86_assembled_train.py", "exp81_tokenizer_eval.py",
               "exp87_shared_vocab.py", "exp87b_image_codes.py",
               "exp88_codebook_geometry.py")
}

function Get-RemoteSize {
    param([string]$Name)
    $out = hf buckets ls $bucket 2>$null | Select-String -Pattern ([regex]::Escape($Name) + '$')
    if ($out -and $out.Line -match '^\s*(\d+)') { return [int64]$matches[1] }
    return $null
}

function Invoke-Hf {
    param([string[]]$Args, [int]$TimeoutSec = 120)
    """跑 `hf` 并**带超时**（实测：`hf buckets cp` 可能**挂起**而不是快速失败，
    不设超时的话重试逻辑会卡死在一个调用上）。"""
    $p = Start-Process -FilePath "hf" -ArgumentList $Args -NoNewWindow -PassThru `
        -RedirectStandardOutput "$env:TEMP\hf_out.txt" -RedirectStandardError "$env:TEMP\hf_err.txt"
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        try { $p.Kill() } catch { }
        return "TIMEOUT"
    }
    $o = (Get-Content "$env:TEMP\hf_out.txt" -Raw -ErrorAction SilentlyContinue)
    $e = (Get-Content "$env:TEMP\hf_err.txt" -Raw -ErrorAction SilentlyContinue)
    return ("$o`n$e")
}

$fail = @()
foreach ($f in $Files) {
    $src = Join-Path $expDir $f
    if (-not (Test-Path $src)) { Write-Host "✗ 本地没有 $f"; $fail += $f; continue }
    $local = (Get-Item $src).Length
    $ok = $false
    for ($i = 1; $i -le $Retries; $i++) {
        $out = Invoke-Hf -Args @("buckets", "cp", $src, "$bucket/$f")
        Write-Host ("[{0}]   尝试 {1}（{2}）：{3}" -f (Get-Date -Format 'HH:mm:ss'), $f, $i,
            (($out -split "`n" | Where-Object { $_ -match 'Uploaded|Error|error|TIMEOUT' } |
              Select-Object -First 1) -replace '\s+', ' '))
        $remote = Get-RemoteSize -Name $f
        if ($remote -eq $local) {
            Write-Host ("[{0}] ✓ {1}（{2} 字节，一致）" -f (Get-Date -Format 'HH:mm:ss'), $f, $local)
            $ok = $true
            break
        }
        Start-Sleep -Seconds 8
    }
    if (-not $ok) { $fail += $f }
}

if ($fail.Count -gt 0) {
    Write-Host "✗ 没能确认上传：$($fail -join ', ')"
    exit 1
}
Write-Host "全部脚本已在桶里且大小一致"
