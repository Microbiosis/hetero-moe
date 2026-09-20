"""V32.4 — representation quality test via transfer to a new task.

Purpose: Test whether mixed student learned better representations than pure
student, by transferring both to a new classification task.

Pipeline:
    1. Setup: train deterministic teacher (soft_route=True, no_quant=True),
       get teacher outputs for original training samples.
    2. Student training: train mixed (w_teacher=0.7) and pure (w_teacher=1.0)
       students with BERTStudent from v16.
    3. Transfer: for each student, freeze LoRA, replace head with new random
       linear head (D_SHARED -> N_CLS), train ONLY the new head on a new
       classification task, evaluate accuracy.
    4. Compare mixed vs pure accuracy.

Usage:
    python3 examples/run_v32_representation_transfer.py
    V32_SEEDS=5 V32_STEPS=100 python3 examples/run_v32_representation_transfer.py

Results:
    results/v32_representation_transfer.json
"""
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Imports from existing scripts
# ---------------------------------------------------------------------------
from examples.run_v16_bert_distill import (
    make_teacher, make_student, prepare_data, train_teacher,
    eval_teacher, encode_modal_to_input_ids,
)
from examples.run_v8_full import (
    load_encoders, make_pools, D_SHARED, N_CLS, B, S, STEPS, LR, SEEDS,
)

# ---------------------------------------------------------------------------
# V32 determinism mechanism (same as run_v32_super_deterministic)
# ---------------------------------------------------------------------------
import research.aligner.v10_embedding.fusion as _fusion_mod


class _SoftRouterSTE:
    """Continuous routing: forward returns softmax weights (no Top-K hard selection)."""

    @staticmethod
    def apply(alpha, k):
        return alpha


class _NoQuantSTE:
    """Disable INT4: identity mapping."""

    @staticmethod
    def apply(x, bits=4, group_size=128):
        return x


@contextlib.contextmanager
def teacher_determinism(soft_route=True, no_quant=True):
    """Patch v10_embedding.fusion STE classes for deterministic inference."""
    saved = (_fusion_mod.SparseRouterSTE, _fusion_mod.FakeQuantSTE)
    if soft_route:
        _fusion_mod.SparseRouterSTE = _SoftRouterSTE
    if no_quant:
        _fusion_mod.FakeQuantSTE = _NoQuantSTE
    try:
        yield
    finally:
        _fusion_mod.SparseRouterSTE, _fusion_mod.FakeQuantSTE = saved


# ---------------------------------------------------------------------------
# Helper: stack samples by class (same pattern as v32 scripts)
# ---------------------------------------------------------------------------
def stack_by_class(modal_seqs, modal_indices, n_cls):
    """Group samples by class into tensors of shape [n_per_class, S, D_SHARED]."""
    h_by_modal = [[] for _ in range(n_cls)]
    for i, m in enumerate(modal_indices):
        h_by_modal[m].append(modal_seqs[i])
    for cls in range(n_cls):
        h_by_modal[cls] = torch.stack(h_by_modal[cls], dim=0)
    return h_by_modal


def get_teacher_outputs_per_sample(teacher, modal_seqs, modal_indices, n_cls):
    """Run teacher forward, return list of [S, D_SHARED] tensors (one per sample)."""
    h_by_modal = stack_by_class(modal_seqs, modal_indices, n_cls)
    with torch.no_grad():
        y = teacher(h_by_modal)  # [6, S, D_SHARED]
    return [y[s].clone() for s in range(y.shape[0])]


# ---------------------------------------------------------------------------
# Transfer classifier: wraps BERTStudent + new linear head
# ---------------------------------------------------------------------------
class TransferClassifier(nn.Module):
    """BERTStudent with a fresh classification head on top.

    During transfer, only the classification head is trainable; the student
    (BERT backbone + LoRA + original head + norm) is frozen.
    """

    def __init__(self, student, n_cls, d_shared):
        super().__init__()
        self.student = student
        self.cls_head = nn.Linear(d_shared, n_cls)

    def forward(self, input_ids, attention_mask=None):
        """input_ids: [B, S] -> logits: [B, N_CLS]"""
        h = self.student(input_ids, attention_mask)      # [B, S, D_SHARED]
        h_pooled = h.mean(dim=1)                          # [B, D_SHARED]
        return self.cls_head(h_pooled)                    # [B, N_CLS]


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
def train_student(student, bert_input_ids, targets, teacher_outputs,
                   w_teacher, steps, vocab_size, seq_len, lr):
    """Train BERTStudent with mixed or pure distillation objective."""
    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=lr
    )
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        losses = []
        for sample_idx, input_ids in bert_input_ids:
            y = student(input_ids.unsqueeze(0))               # [1, S, D_SHARED]
            t = targets[sample_idx].unsqueeze(0)
            t_teacher = teacher_outputs[sample_idx].unsqueeze(0).detach()
            loss = (w_teacher * F.mse_loss(y, t_teacher)
                    + (1.0 - w_teacher) * F.mse_loss(y, t))
            losses.append(loss)
        total = sum(losses) / len(losses)
        total.backward()
        opt.step()


def train_transfer_head(classifier, input_ids_list, labels, steps, lr):
    """Train ONLY the classification head with cross-entropy loss."""
    # Freeze student completely
    classifier.student.eval()
    for p in classifier.student.parameters():
        p.requires_grad = False
    # Only optimize the new head
    opt = torch.optim.AdamW(classifier.cls_head.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        losses = []
        for (sample_idx, input_ids), label in zip(input_ids_list, labels):
            logits = classifier(input_ids.unsqueeze(0))           # [1, N_CLS]
            loss = F.cross_entropy(
                logits, torch.tensor([label], device=logits.device)
            )
            losses.append(loss)
        total = sum(losses) / len(losses)
        total.backward()
        opt.step()


def eval_accuracy(classifier, input_ids_list, labels):
    """Compute classification accuracy."""
    classifier.eval()
    correct = 0
    with torch.no_grad():
        for (sample_idx, input_ids), label in zip(input_ids_list, labels):
            logits = classifier(input_ids.unsqueeze(0))
            pred = logits.argmax(dim=-1).item()
            if pred == label:
                correct += 1
    return correct / len(labels) if labels else 0.0


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    seeds = int(os.environ.get("V32_SEEDS", SEEDS))
    steps = int(os.environ.get("V32_STEPS", STEPS))
    seq_len = S
    transfer_steps = steps          # reuse same step count for head training
    transfer_lr = LR

    print("=" * 80)
    print("V32.4 — representation quality test via transfer to a new task")
    print("=" * 80)
    print(f"  Seeds: {seeds}  |  Student steps: {steps}  |  Transfer head steps: {transfer_steps}")
    print(f"  D_SHARED={D_SHARED}, N_CLS={N_CLS}, S={seq_len}")
    print()

    # ------------------------------------------------------------------ #
    # 1. Setup: load encoders, build teacher, train teacher
    # ------------------------------------------------------------------ #
    print("[Phase 1] Loading encoders and training teacher...")
    encoders = load_encoders()
    attn_pools = make_pools(encoders)
    bert_tok = encoders[0][0]

    # ------------------------------------------------------------------ #
    # 2. Per-seed experiment
    # ------------------------------------------------------------------ #
    per_seed_results = []

    for seed in range(seeds):
        torch.manual_seed(seed)
        print(f"\n--- Seed {seed} ---")

        # Prepare data
        modal_seqs, modal_indices, targets, _ = prepare_data(seed, encoders)
        labels = list(modal_indices)   # class labels for all 18 samples

        # Train teacher (mixture aligner)
        teacher = make_teacher(attn_pools, encoders)
        train_teacher(teacher, modal_seqs, modal_indices, targets)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # Get teacher outputs with deterministic inference
        with teacher_determinism(soft_route=True, no_quant=True):
            teacher_outputs = get_teacher_outputs_per_sample(
                teacher, modal_seqs, modal_indices, N_CLS
            )

        # Teacher eval
        t_fuse = eval_teacher(teacher, modal_seqs, modal_indices, targets)
        print(f"  [teacher] fuse MSE = {t_fuse:.4f}")

        # BERT input IDs for student training (BERT modal samples only, as in v16)
        bert_input_ids = encode_modal_to_input_ids(modal_seqs, modal_indices, bert_tok)

        # ---------------------------------------------------------------- #
        # 3. Student training phase: mixed and pure
        # ---------------------------------------------------------------- #
        print("  [Phase 2] Training students...")
        students = {}
        for w_teacher, name in [(0.7, "mixed"), (1.0, "pure")]:
            student = make_student()
            train_student(
                student, bert_input_ids, targets, teacher_outputs,
                w_teacher=w_teacher, steps=steps,
                vocab_size=bert_tok.vocab_size, seq_len=seq_len, lr=LR,
            )
            students[name] = student
            print(f"    [{name}] student trained (w_teacher={w_teacher})")

        # ---------------------------------------------------------------- #
        # 4. Transfer phase: new task with seed=999 targets
        # ---------------------------------------------------------------- #
        print("  [Phase 3] Transfer to new task (seed=999)...")

        # Generate NEW random targets with seed=999 — different class directions
        g_new = torch.Generator().manual_seed(999)
        new_targets = torch.randn(N_CLS, seq_len, D_SHARED, generator=g_new)
        print(f"    new_targets shape: {new_targets.shape} (seed=999)")

        # Build transfer input_ids for ALL samples (need all classes for 3-way cls)
        all_input_ids = []
        for i in range(len(modal_seqs)):
            ids = torch.randint(0, bert_tok.vocab_size, (seq_len,), dtype=torch.long)
            all_input_ids.append((i, ids))

        seed_result = {
            "seed": seed,
            "teacher_fuse_mse": round(t_fuse, 6),
            "students": {},
        }

        for name, student in students.items():
            # a. Freeze LoRA
            for p in student.lora.parameters():
                p.requires_grad = False
            # Freeze BERT (already frozen, but ensure)
            for p in student.bert.parameters():
                p.requires_grad = False
            # Freeze original head and norm
            student.head.requires_grad = False
            for p in student.head.parameters():
                p.requires_grad = False
            for p in student.norm.parameters():
                p.requires_grad = False

            # b. Wrap with new classification head
            classifier = TransferClassifier(student, N_CLS, D_SHARED)

            # c. Train ONLY the new head
            train_transfer_head(
                classifier, all_input_ids, labels,
                steps=transfer_steps, lr=transfer_lr,
            )

            # d. Evaluate classification accuracy
            train_acc = eval_accuracy(classifier, all_input_ids, labels)
            eval_acc  = eval_accuracy(classifier, all_input_ids, labels)

            seed_result["students"][name] = {
                "train_accuracy": round(train_acc, 4),
                "eval_accuracy":  round(eval_acc, 4),
            }
            print(f"    [{name}] train_acc={train_acc:.4f}  eval_acc={eval_acc:.4f}")

        per_seed_results.append(seed_result)

    # ------------------------------------------------------------------ #
    # 5. Summary
    # ------------------------------------------------------------------ #
    print()
    print("=" * 80)
    print("SUMMARY — V32.4 Representation Transfer")
    print("=" * 80)

    summary = {}
    for name in ["mixed", "pure"]:
        train_accs = [r["students"][name]["train_accuracy"] for r in per_seed_results]
        eval_accs  = [r["students"][name]["eval_accuracy"]  for r in per_seed_results]
        summary[name] = {
            "train_accuracy_mean": round(float(np.mean(train_accs)), 4),
            "train_accuracy_std":  round(float(np.std(train_accs)), 4),
            "eval_accuracy_mean":  round(float(np.mean(eval_accs)), 4),
            "eval_accuracy_std":   round(float(np.std(eval_accs)), 4),
        }
        print(f"  {name:>6}: "
              f"train_acc = {summary[name]['train_accuracy_mean']:.4f} "
              f"± {summary[name]['train_accuracy_std']:.4f}  |  "
              f"eval_acc  = {summary[name]['eval_accuracy_mean']:.4f} "
              f"± {summary[name]['eval_accuracy_std']:.4f}")

    mixed_acc = summary["mixed"]["eval_accuracy_mean"]
    pure_acc  = summary["pure"]["eval_accuracy_mean"]
    gap = mixed_acc - pure_acc
    print()
    print(f"  Gap (mixed - pure): {gap:+.4f}")
    if gap > 0.05:
        print("  ✅ Mixed student learned BETTER representations (transfer advantage)")
    elif gap > 0:
        print("  ↔  Mixed student slightly better, but margin is small")
    elif gap == 0:
        print("  ↔  Equal — no clear representation quality difference")
    else:
        print("  ❌ Pure student better — unexpected result")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # 6. Save JSON results
    # ------------------------------------------------------------------ #
    out_dir = Path(_REPO_ROOT) / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v32_representation_transfer.json"

    payload = {
        "experiment": "V32.4 — representation quality via transfer to new task",
        "config": {
            "seeds": seeds,
            "student_steps": steps,
            "transfer_steps": transfer_steps,
            "d_shared": D_SHARED,
            "n_cls": N_CLS,
            "seq_len": seq_len,
            "new_task_seed": 999,
            "objectives": {
                "mixed":  "w_teacher=0.7 (MSE(y,y_T)*0.7 + MSE(y,t)*0.3)",
                "pure":   "w_teacher=1.0 (pure distillation, MSE(y,y_T) only)",
            },
        },
        "summary": summary,
        "per_seed": per_seed_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nJSON results saved to: {out_path}")


if __name__ == "__main__":
    main()
