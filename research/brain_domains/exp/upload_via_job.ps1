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

$chunkDir = "/domains/.chunks"

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
    for ($i = 1; $i -le 2; $i++) {
        $out = Send-Job -Name "upload-$($f -replace '\.py$','')" -Code $py
        Write-Host ("[{0}]   整传第 {1} 次：{2}" -f (Get-Date -Format 'HH:mm:ss'), $i,
            ($out -replace '\s+', ' '))
        if ($out -and $out -match 'id:') { $ok = $true; break }
    }
    # --- 分块兜底：接口对大 payload 持续失败时用 ---
    if (-not $ok) {
        $chunk = 3000
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
            for ($k = 1; $k -le 4; $k++) {
                $o2 = Send-Job -Name "chunk-$($f -replace '\.py$','')-$c" -Code $cp
                if ($o2 -and $o2 -match 'id:') { $s = $true; break }
                Start-Sleep -Seconds 8
            }
            if ($s) { $sent++ } else { Write-Host "    ✗ 第 $c 块没发出去" }
        }
        if ($sent -eq $n) {
            $join = "import base64,zlib,pathlib,glob;" +
                    "parts=[pathlib.Path(p).read_text() for p in sorted(glob.glob('$chunkDir/$f.*'))];" +
                    "b=''.join(parts);" +
                    "p=pathlib.Path('/domains/scripts/$f');" +
                    "p.write_bytes(zlib.decompress(base64.b64decode(b),47));" +
                    "print('WROTE',p,p.stat().st_size)"
            $oj = Send-Job -Name "join-$($f -replace '\.py$','')" -Code $join
            if ($oj -and $oj -match 'id:') { $ok = $true; Write-Host "    ✓ 拼接 Job 已发" }
        }
    }
    if (-not $ok) { Write-Host "  ✗ $f 没发出去" }
}
Write-Host "上传 Job 已发出（等它们跑完后校验桶里的大小）"
