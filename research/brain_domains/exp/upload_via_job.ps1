# upload_via_job.ps1 — **用 Job 把脚本写进桶**（绕开本机坏掉的写通道）。
#
# ## 为什么需要它
#
# 实测（2026-10-24 凌晨）：本机的**写通道整个断了**——
# - `hf buckets cp` → `ConnectionError: ... /api/buckets/.../xet-write-token`
#   （连 `HF_HUB_DISABLE_XET=1` 也照样走那个端点）
# - `hf cp` 到 dataset repo → `httpx2.RemoteProtocolError: Server disconnected`
# - `hf jobs run -v LOCAL_DIR:/mnt`（本地目录挂载）→ 提交阶段就挂住
#   （同步也是客户端做的 ⇒ 同样撞坏通道）
#
# 而**读通道一直正常**，且 **Job 容器里的写路径是好的**
# （checkpoint 就是 Job 写进桶的）。
#
# ⇒ 结论：**让 Job 替我把脚本写进桶**。
# 做法：把脚本 base64 编码，作为 `python -c` 的一个 argv 传进去（argv 不走 shell ⇒ 无引号问题），
# Job 里 `pathlib.Path('/domains/scripts/xxx.py').write_bytes(base64.b64decode(...))`。
#
# 用法：
#   pwsh -File research/brain_domains/exp/upload_via_job.ps1
#   pwsh -File research/brain_domains/exp/upload_via_job.ps1 -Files exp87_shared_vocab.py

param(
    [string[]]$Files = @(),
    [int]$Retries = 3
)

$ErrorActionPreference = 'Continue'
$expDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($Files.Count -eq 0) {
    $Files = @("exp80_eeg_tokenizer_train.py", "exp81_tokenizer_eval.py",
               "exp86_assembled_train.py", "exp87_shared_vocab.py",
               "exp87b_image_codes.py", "exp88_codebook_geometry.py",
               "eval_brain_alignment.py")
}

$vol = @(
    "-v", "hf://buckets/illusion-brain/brain-domains:/domains",
    "--flavor", "cpu-upgrade", "--timeout", "20m", "-d"
)

function Send-Job {
    param([string]$Name, [string]$Code)
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" python -u -c $Code `
        @vol --name $Name 2>&1 |
        Select-String -Pattern 'id:|Error|error' | Select-Object -First 1
}

# ⚠️ **只认「真的拿到 id」**：提交失败时 `hf jobs run` 会打 RemoteProtocolError，
#    那行里有 "Error" 却没有 "id:" ⇒ 用 'id:' 判定，避免把失败当成功。
function Test-Submitted {
    param($Out)
    return ($Out -and ($Out.ToString() -match 'id:\s*[0-9a-f]{8,}'))
}

# ⚠️ **落地校验**：Job 写进去的字节数必须等于本地字节数，否则就是半个文件。
function Test-Landed {
    param([string]$File, [int]$Expect)
    $out = hf buckets ls "hf://buckets/illusion-brain/brain-domains/scripts" 2>&1 | Out-String
    # ⚠️ **必须先剥 ANSI 颜色码**（`hf` 输出带转义序列 ⇒ 否则正则匹配不到）
    $out = $out -replace "`e\[[0-9;]*[A-Za-z]", ""
    $m = [regex]::Match($out, "(?m)^\s*(\d+)\s+\S+\s+\S+\s+scripts/$([regex]::Escape($File))\s*$")
    if (-not $m.Success) { return $false }
    $sz = [int]$m.Groups[1].Value
    Write-Host ("[{0}]   落地校验：{1} 字节（期望 {2}）" -f (Get-Date -Format 'HH:mm:ss'), $sz, $Expect)
    return ($sz -eq $Expect)
}

$chunkDir = "/domains/.chunks"
$failed = 0

foreach ($f in $Files) {
    $src = Join-Path $expDir $f
    if (-not (Test-Path $src)) { Write-Host "✗ 本地没有 $f"; continue }
    # ⚠️ **先 gzip 再 base64**：Job 创建接口对大 payload 不稳
    #    （实测 23 KB 的脚本连挂 2 次 RemoteProtocolError；18 KB 的第 2 次就过了）
    $bytes = [IO.File]::ReadAllBytes($src)
    $ms = New-Object System.IO.MemoryStream
    $gz = New-Object System.IO.Compression.GZipStream($ms, [System.IO.Compression.CompressionMode]::Compress)
    $gz.Write($bytes, 0, $bytes.Length); $gz.Close()
    $b64 = [Convert]::ToBase64String($ms.ToArray())
    Write-Host ("[{0}] {1}：{2} 字节 → base64 {3} 字符" -f (Get-Date -Format 'HH:mm:ss'),
        $f, $bytes.Length, $b64.Length)
    # argv 直接传（不走 shell ⇒ base64 里没有引号/空格，天然安全）
    $py = "import base64,zlib,pathlib;p=pathlib.Path('/domains/scripts/$f');" +
          "p.write_bytes(zlib.decompress(base64.b64decode('$b64'),47));" +
          "print('WROTE',p,p.stat().st_size)"
    $ok = $false
    for ($i = 1; $i -le 4; $i++) {
        $out = Send-Job -Name "upload-$($f -replace '\.py$','')" -Code $py
        Write-Host ("[{0}]   整传第 {1} 次：{2}" -f (Get-Date -Format 'HH:mm:ss'), $i,
            ($out -replace '\s+', ' '))
        if (Test-Submitted $out) { $ok = $true; break }
        Start-Sleep -Seconds 15
    }
    # --- 分块兜底：接口对大 payload 持续失败时用 ---
    if (-not $ok) {
        # ⚠️ **先把同名旧分块清掉**：拼接用的是 `glob('$chunkDir/$f.*')`
        #    ⇒ 上一次失败留下的残块会被一起拼进去 ⇒ 产出**损坏**的文件。
        $clean = "import pathlib,glob;" +
                 "[pathlib.Path(p).unlink() for p in glob.glob('$chunkDir/$f.*')];" +
                 "print('CLEANED')"
        $oc = Send-Job -Name "clean-$($f -replace '\.py$','')" -Code $clean
        if (Test-Submitted $oc) { Start-Sleep -Seconds 25 }
        $chunk = 2000
        $n = [math]::Ceiling($b64.Length / $chunk)
        Write-Host ("[{0}]   改分块：{1} 块 × {2} 字符" -f (Get-Date -Format 'HH:mm:ss'), $n, $chunk)
        $sent = 0
        for ($c = 0; $c -lt $n; $c++) {
            $len = [math]::Min($chunk, $b64.Length - $c * $chunk)
            $part = $b64.Substring($c * $chunk, $len)
            $cp = "import pathlib;p=pathlib.Path('$chunkDir/$f.$c');" +
                  "p.parent.mkdir(parents=True,exist_ok=True);" +
                  "p.write_text('$part');print('CHUNK',p,p.stat().st_size)"
            $s = $false
            for ($k = 1; $k -le 6; $k++) {
                $o2 = Send-Job -Name "chunk-$($f -replace '\.py$','')-$c" -Code $cp
                if (Test-Submitted $o2) { $s = $true; break }
                Start-Sleep -Seconds 12
            }
            if ($s) { $sent++ } else { Write-Host "    ✗ 第 $c 块没发出去" }
            Start-Sleep -Seconds 4
        }
        if ($sent -eq $n) {
            $join = "import base64,zlib,pathlib,glob;" +
                    "parts=[pathlib.Path(p).read_text() for p in sorted(glob.glob('$chunkDir/$f.*'))];" +
                    "b=''.join(parts);" +
                    "p=pathlib.Path('/domains/scripts/$f');" +
                    "p.write_bytes(zlib.decompress(base64.b64decode(b),47));" +
                    "print('WROTE',p,p.stat().st_size)"
            $oj = Send-Job -Name "join-$($f -replace '\.py$','')" -Code $join
            if (Test-Submitted $oj) { $ok = $true; Write-Host "    ✓ 拼接 Job 已发" }
        }
    }
    if ($ok) {
        # ⚠️ **等 Job 跑完再校验**（刚提交时文件还没写进去）
        # ⚠️ 2026-10-25 修：原来这里**只提交就算成功**，落地校验失败也照样往下走
        #    ⇒ 末尾那句「上传 Job 已发出（字节数已校验）」是**假消息**（曾误导我以为传好了）。
        #    ⇒ 现在：**校验没过就明确报失败**，并给出重试提示。
        $landed = $false
        for ($w = 1; $w -le 30; $w++) {
            Start-Sleep -Seconds 20
            if (Test-Landed -File $f -Expect $bytes.Length) { $landed = $true; break }
        }
        if (-not $landed) {
            Write-Host ("  ✗✗ {0}：**提交成功但落地校验未过**（文件可能还在排队/Job 失败）" -f $f)
            Write-Host "     ⇒ 不要当成已传好；先 `hf buckets ls` 确认，再重跑本脚本。"
            $failed += 1
        }
    }
    if (-not $ok) { Write-Host "  ✗ $f 没发出去"; $failed += 1 }
}
if ($failed -gt 0) {
    Write-Host ("✗ 有 {0} 个文件没落地（提交≠落地）" -f $failed)
    exit 1
}
Write-Host "✓ 上传完成：所有文件均已落地且字节数校验通过"
