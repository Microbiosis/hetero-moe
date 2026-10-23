# run_assembled_2x2.ps1 — 补齐 2×2 因子设计的两个缺失格。
#
# ## 为什么需要这个
#
# 第一轮（run_assembled_stage.ps1）的设计里，**「嫁接注意力」没有同配方的对照**：
#
# | | 无语义监督 | 有语义监督 |
# |---|---|---|
# | 嫁接注意力 | v2-none-nosem ✅ | v2-none ✅ |
# | 不嫁接注意力 | ❌ 缺 | ❌ 缺 |
#
# ⇒ 「v2-none 比 S1a 强」这件事**无法归因**：
#    S1a 与组装臂的配方不同（掩码重建 vs 普通重建、3000 vs 6000 步、
#    码本初始化不同）⇒ 差值里混着「语义监督」与「配方差异」两份贡献。
#
# ⇒ 补上左下/右下两格，得到**单因子**对照：
#    - 「注意力嫁接的净效应」 = v2-none − v2-plain（同配方，只差注意力）
#    - 「语义监督的净效应」   = v2-plain − v2-plain-nosem（同配方，只差 λ_nce）
#
# $ErrorActionPreference = 'Continue'

$vol = @(
    "-v", "hf://buckets/illusion-brain/training-data:/training:ro",
    "-v", "hf://buckets/illusion-brain/brain-domains:/domains",
    "--flavor", "cpu-upgrade", "--timeout", "120m", "-d"
)
$root = "hf://buckets/illusion-brain/brain-domains/runs/illusionbrain-align"

function Wait-Ckpt {
    param([string]$Dir, [int]$MaxSec = 3600)
    $dl = (Get-Date).AddSeconds($MaxSec)
    while ((Get-Date) -lt $dl) {
        $out = hf buckets ls -R "$root/$Dir/" 2>$null
        if ($out -match 'tokenizer_assembled\.pt') { return $true }
        Start-Sleep -Seconds 45
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

# --- 训练：两个缺失格 ---
Run-Job "t86v2-assembled-plain" "exp86_assembled_train.py" @(
    "--attn-donor", "none", "--donor-ffn", "none",
    "--steps", "6000", "--batch", "64", "--log-every", "250",
    "--max-seconds", "2400", "--codebook-init", "data",
    "--out-dir", "/domains/runs/illusionbrain-align/exp86v2/ffn-plain")

Run-Job "t86v2-assembled-plain-nosem" "exp86_assembled_train.py" @(
    "--attn-donor", "none", "--donor-ffn", "none",
    "--steps", "6000", "--batch", "64", "--log-every", "250",
    "--max-seconds", "2400", "--codebook-init", "data", "--lambda-nce", "0",
    "--out-dir", "/domains/runs/illusionbrain-align/exp86v2/ffn-plain-nosem")

# --- 等 checkpoint → 评测 ---
$arms = @('plain', 'plain-nosem')
foreach ($v in $arms) {
    if (-not (Wait-Ckpt "exp86v2/ffn-$v" 3600)) {
        Write-Host "⚠ 缺 checkpoint：ffn-$v"
        continue
    }
    Run-Job "t81-eval-v2-$v" "exp81_tokenizer_eval.py" @(
        "--ckpt", "/domains/runs/illusionbrain-align/exp86v2/ffn-$v/tokenizer_assembled.pt",
        "--no-metric-b", "--subjects", "1,2,3,4,5,6,7,8,9,10",
        "--out-dir", "/domains/runs/illusionbrain-align/exp81/v2-ffn-$v")
}
Write-Host "2×2 补齐完成"
