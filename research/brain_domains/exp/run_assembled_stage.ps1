# run_assembled_stage.ps1 — 组件级组装原型的三波编排。
#
#   波 A：4 个 v2 组装训练（none/mirepnet/biot/eegdino-small）+ 1 个「无语义监督」对照臂
#   波 B：5 个 exp81 评测（v2 的 4 个臂 + 对照臂）—— 判 S1b
#   波 C：4 个 exp81 评测（v1 的 4 个臂）—— 次要：另一套配方下的组件排序
#
# ⚠️ **并发实测上限 4–5**（20 并发会排队/饿死）⇒ 按波次发，不要一次全发。
# ⚠️ `hf buckets ls` **必须带 -R**：
#    不带时只列下一层的「目录」，列不到更深处的文件
#    ⇒ 旧版脚本的等待循环永远等不到 checkpoint（已修）。

$ErrorActionPreference = 'Continue'

$vol = @(
    "-v", "hf://buckets/illusion-brain/training-data:/training:ro",
    "-v", "hf://buckets/illusion-brain/brain-domains:/domains",
    "--flavor", "cpu-upgrade", "--timeout", "120m", "-d"
)
$variants = @('none', 'mirepnet', 'biot', 'eegdino-small')
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

# ================= 波 A：v2 组装训练 =================
Write-Host "=== 波 A：等 v1 checkpoint（最多 30 分钟）==="
$ready = @{}
foreach ($v in $variants) {
    if (Wait-Ckpt "exp86/ffn-$v" 1800) { $ready[$v] = $true }
}
Write-Host ("[{0}] v1 checkpoint 就绪 {1}/{2}" -f (Get-Date -Format 'HH:mm:ss'),
    $ready.Count, $variants.Count)

foreach ($v in $variants) {
    Run-Job "t86v2-assembled-$v" "exp86_assembled_train.py" @(
        "--donor-ffn", $v, "--steps", "6000", "--batch", "64",
        "--log-every", "250", "--max-seconds", "2400",
        "--codebook-init", "data",
        "--out-dir", "/domains/runs/illusionbrain-align/exp86v2/ffn-$v")
}
# 对照臂：**只去掉语义监督**（lambda-nce 0），其余配方与 v2-none 逐字相同
# ⇒ 与 v2-none 的差 = 语义监督这一个因子
Run-Job "t86v2-assembled-none-nosem" "exp86_assembled_train.py" @(
    "--donor-ffn", "none", "--steps", "6000", "--batch", "64",
    "--log-every", "250", "--max-seconds", "2400",
    "--codebook-init", "data", "--lambda-nce", "0",
    "--out-dir", "/domains/runs/illusionbrain-align/exp86v2/ffn-none-nosem")

# ================= 波 B：v2 评测（判 S1b）=================
Write-Host "=== 波 B：等 v2 checkpoint（最多 60 分钟）==="
$v2 = @('none', 'mirepnet', 'biot', 'eegdino-small', 'none-nosem')
$ready2 = @{}
foreach ($v in $v2) {
    if (Wait-Ckpt "exp86v2/ffn-$v" 3600) { $ready2[$v] = $true }
}
Write-Host ("[{0}] v2 checkpoint 就绪 {1}/{2}" -f (Get-Date -Format 'HH:mm:ss'),
    $ready2.Count, $v2.Count)

foreach ($v in $v2) {
    if (-not $ready2.ContainsKey($v)) { continue }
    Run-Job "t81-eval-v2-$v" "exp81_tokenizer_eval.py" @(
        "--ckpt", "/domains/runs/illusionbrain-align/exp86v2/ffn-$v/tokenizer_assembled.pt",
        "--no-metric-b", "--subjects", "1,2,3,4,5,6,7,8,9,10",
        "--out-dir", "/domains/runs/illusionbrain-align/exp81/v2-ffn-$v")
}

# ================= 波 C：v1 评测（次要）=================
Write-Host "=== 波 C：v1 评测（另一套配方下的组件排序）==="
foreach ($v in $variants) {
    if (-not $ready.ContainsKey($v)) { continue }
    Run-Job "t81-eval-v1-$v" "exp81_tokenizer_eval.py" @(
        "--ckpt", "/domains/runs/illusionbrain-align/exp86/ffn-$v/tokenizer_assembled.pt",
        "--no-metric-b", "--subjects", "1,2,3,4,5,6,7,8,9,10",
        "--out-dir", "/domains/runs/illusionbrain-align/exp81/v1-ffn-$v")
}
Write-Host "全部已发出"
