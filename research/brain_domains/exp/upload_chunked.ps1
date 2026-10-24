# upload_chunked.ps1 — **分块上传的续传工具**（upload_via_job.ps1 的补充）。

# ## 为什么要有它
#
# 2026-10-24 晚：`upload_via_job.ps1` 在 Job 提交被限流时会**丢掉个别分块**
# （实测 exp93f：4 块里第 0 块连续 6 次没发出去，其余 3 块都落了），
# 而拼接用的是 `glob('.../file.*')` ⇒ **少一块就拼不出正确文件**，
# 且旧脚本**不校验落地字节数** ⇒ 可能把半个文件当成成功。

# ⇒ 本脚本：**先看哪些块已经在桶里，只补缺的那些**，再拼接、**并校验字节数**。

param(
    [Parameter(Mandatory = $true)][string]$File,
    [int]$Chunk = 2000,
    [int]$Retries = 8
)

$ErrorActionPreference = 'Continue'
$expDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $expDir $File
if (-not (Test-Path $src)) { Write-Host "✗ 本地没有 $File"; exit 1 }

$vol = @("-v", "hf://buckets/illusion-brain/brain-domains:/domains",
    "--flavor", "cpu-upgrade", "--timeout", "20m", "-d")
$chunkDir = "/domains/.chunks"

# ---- 算 base64 ----
$bytes = [IO.File]::ReadAllBytes($src)
$ms = New-Object System.IO.MemoryStream
$gz = New-Object System.IO.Compression.GZipStream($ms, [System.IO.Compression.CompressionMode]::Compress)
$gz.Write($bytes, 0, $bytes.Length); $gz.Close()
$b64 = [Convert]::ToBase64String($ms.ToArray())
$n = [math]::Ceiling($b64.Length / $Chunk)
Write-Host ("[{0}] {1}：{2} 字节 → base64 {3} 字符 ⇒ {4} 块 × {5}" -f
    (Get-Date -Format 'HH:mm:ss'), $File, $bytes.Length, $b64.Length, $n, $Chunk)

function Send-Job {
    param([string]$Name, [string]$Code)
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" python -u -c $Code `
        @vol --name $Name 2>&1 |
        Select-String -Pattern 'id:|Error|error' | Select-Object -First 1
}
function Test-Submitted { param($Out) return ($Out -and ($Out.ToString() -match 'id:\s*[0-9a-f]{8,}')) }

# ---- 桶里现有哪些块 ----
$have = @{}
$ls = hf buckets ls "hf://buckets/illusion-brain/brain-domains/.chunks" 2>&1 | Out-String
foreach ($m in [regex]::Matches($ls, "(?m)^\s*(\d+)\s+\S+\s+\S+\.chunks/$([regex]::Escape($File))\.(\d+)\s*$")) {
    $have[[int]$m.Groups[2].Value] = [int]$m.Groups[1].Value
}
if ($have.Count) { Write-Host ("  桶里已有块：" + (($have.Keys | Sort-Object | ForEach-Object { ".$_($($have[$_]))" }) -join ' ')) }

# ---- 补缺块 ----
for ($c = 0; $c -lt $n; $c++) {
    $len = [math]::Min($Chunk, $b64.Length - $c * $Chunk)
    if ($have.ContainsKey($c) -and $have[$c] -eq $len) { continue }
    $part = $b64.Substring($c * $Chunk, $len)
    $cp = "import pathlib;p=pathlib.Path('$chunkDir/$File.$c');" +
        "p.parent.mkdir(parents=True,exist_ok=True);" +
        "p.write_text('$part');print('CHUNK',p,p.stat().st_size)"
    $ok = $false
    for ($k = 1; $k -le $Retries; $k++) {
        $o = Send-Job -Name "chunk-$($File -replace '\.py$','')-$c" -Code $cp
        if (Test-Submitted $o) { $ok = $true; break }
        Write-Host ("    第 {0} 块第 {1} 次没发出去" -f $c, $k)
        Start-Sleep -Seconds 10
    }
    if (-not $ok) { Write-Host "✗ 第 $c 块最终没发出去 ⇒ 放弃（不拼半截文件）"; exit 1 }
    Write-Host ("    ✓ 第 {0} 块已发（{1} 字符）" -f $c, $len)
    Start-Sleep -Seconds 3
}

# ---- 拼接 ----
$join = "import base64,zlib,pathlib,glob;" +
    "parts=[pathlib.Path(p).read_text() for p in sorted(glob.glob('$chunkDir/$File.*'),key=lambda q:int(q.rsplit('.',1)[1]))];" +
    "b=''.join(parts);" +
    "p=pathlib.Path('/domains/scripts/$File');" +
    "p.write_bytes(zlib.decompress(base64.b64decode(b),47));" +
    "print('WROTE',p,p.stat().st_size)"
$oj = Send-Job -Name "join-$($File -replace '\.py$','')" -Code $join
if (-not (Test-Submitted $oj)) { Write-Host "✗ 拼接 Job 没发出去"; exit 1 }
Write-Host "✓ 拼接 Job 已发"

# ---- 校验落地字节数 ----
for ($w = 1; $w -le 15; $w++) {
    Start-Sleep -Seconds 20
    # ⚠️ **必须先剥掉 ANSI 颜色码**：`hf` 的输出带转义序列，
    #    否则正则 `^\s*(\d+)` 匹配不到 ⇒ 会把「已落地」误报成「迟迟没落地」。
    $out = hf buckets ls "hf://buckets/illusion-brain/brain-domains/scripts" 2>&1 | Out-String
    $out = $out -replace "`e\[[0-9;]*[A-Za-z]", ""
    $m = [regex]::Match($out, "(?m)^\s*(\d+)\s+\S+\s+\S+\s+scripts/$([regex]::Escape($File))\s*$")
    if ($m.Success) {
        $sz = [int]$m.Groups[1].Value
        Write-Host ("  落地 {0} 字节（期望 {1}）" -f $sz, $bytes.Length)
        if ($sz -eq $bytes.Length) { Write-Host "✓ 字节数一致"; exit 0 }
        else { Write-Host "✗ 字节数不符 ⇒ 文件被截断/损坏"; exit 1 }
    }
}
Write-Host "✗ 迟迟没落地"
exit 1
