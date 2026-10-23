# run_assembled_v3.ps1 — S1b-v3「共享词典」阶段的编排。
#
#   第 0 步：exp87b 建图像侧离散词表（k-means）→ 落桶
#   第 1 步：等图像码 → 发 3 个训练臂
#   第 2 步：等 checkpoint → 发 3 个 exp81 评测
#
# ## 三个臂（单因子分解）
#
# | 臂 | 码本机制 | CE 的码本路径 | 回答什么 |
# |---|---|---|---|
# | `grad-pos`   | 梯度 | 只有正类可导 | **候选配方**：码本吃语义梯度 |
# | `grad-full`  | 梯度 | 全部可导 | 负类压力行不行 |
# | `grad-none`  | 梯度 | 全 detach | 对照：码本**不吃**语义梯度 |
#
# （基线是 `v2-plain`：EMA 码本 + 无 CE，见 FINDINGS §二十三·三/·四。）
#
# ⚠️ 并发上限 4–5：第 1 步只发 3 个训练（够快），第 2 步发 3 个评测。
# ⚠️ `hf buckets ls` 必须带 -R（不带只列下一层的目录，列不到文件）。

$ErrorActionPreference = 'Continue'

$vol = @(
    "-v", "hf://buckets/illusion-brain/training-data:/training:ro",
    "-v", "hf://buckets/illusion-brain/brain-domains:/domains",
    "--flavor", "cpu-upgrade", "--timeout", "120m", "-d"
)
$root = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"
$codes = "/domains/runs/illusionbrain-align/shared-vocab/image_codes_k1024.pt"

$ARMS = @(
    @{ name = "grad-pos";  mode = "positive" },
    @{ name = "grad-full"; mode = "full" },
    @{ name = "grad-none"; mode = "none" }
)

function Wait-File {
    param([string]$Rel, [int]$MaxSec = 3600)
    $dl = (Get-Date).AddSeconds($MaxSec)
    while ((Get-Date) -lt $dl) {
        $out = hf buckets ls -R "$root/$Rel" 2>$null
        if ($out -match 'image_codes_k1024\.pt') { return $true }
        Start-Sleep -Seconds 30
    }
    return $false
}

function Run-Job {
    param([string]$Name, [string]$Script, [string[]]$JobArgs)
    $all = @("python", "-u", "/domains/scripts/$Script") + $JobArgs + $vol + @("--name", $Name)
    hf jobs run "hf.co/spaces/redmingwei/IllusionBrain" @all 2>&1 |
        Select-String -Pattern 'id:' | Select-Object -First 1
    Write-Host ("[{0}] 已发 {1}" -f (Get-Date -Format 'HH:mm:ss'), $Name)
}

# ---------- 第 0 步：图像侧词表 ----------
Write-Host "=== 第 0 步：exp87b 建图像侧离散词表 ==="
Run-Job "t87b-image-codes" "exp87b_image_codes.py" @(
    "--k", "1024", "--iters", "30",
    "--out-dir", "/domains/runs/illusionbrain-align/shared-vocab")

# ---------- 第 1 步：等图像码 → 发训练 ----------
Write-Host "=== 第 1 步：等图像侧码（最多 40 分钟）==="
if (-not (Wait-File "shared-vocab" 2400)) {
    Write-Host "✗ 图像侧码没出来，先停在这里"
    exit 1
}
foreach ($a in $ARMS) {
    Run-Job "t87-sharedvocab-$($a.name)" "exp87_shared_vocab.py" @(
        "--vq", "grad", "--sem-codebook-mode", $a.mode,
        "--image-codes", $codes,
        "--steps", "6000", "--batch", "64", "--log-every", "250",
        "--max-seconds", "2400",
        "--out-dir", "/domains/runs/illusionbrain-align/exp87/$($a.name)")
}

# ---------- 第 2 步：等 checkpoint → 发评测 ----------
Write-Host "=== 第 2 步：等 checkpoint（最多 60 分钟）==="
foreach ($a in $ARMS) {
    $dir = "exp87/$($a.name)"
    $dl = (Get-Date).AddSeconds(3600)
    $ok = $false
    while ((Get-Date) -lt $dl) {
        $out = hf buckets ls -R "$root/$dir/" 2>$null
        if ($out -match 'tokenizer_assembled\.pt') { $ok = $true; break }
        Start-Sleep -Seconds 45
    }
    if (-not $ok) { Write-Host "⚠ 缺 checkpoint：$dir"; continue }
    Run-Job "t81-eval-v3-$($a.name)" "exp81_tokenizer_eval.py" @(
        "--ckpt", "/domains/runs/illusionbrain-align/$dir/tokenizer_assembled.pt",
        "--no-metric-b", "--subjects", "1,2,3,4,5,6,7,8,9,10",
        "--out-dir", "/domains/runs/illusionbrain-align/exp81/v3-$($a.name)")
}
Write-Host "v3 阶段全部发出"
