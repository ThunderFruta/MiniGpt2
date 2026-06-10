"""
reset_and_train_loras.py

Fully automatic LoRA pipeline runner for MiniGpt2.

Pipeline order (no prompts, no flags):
1. Backup all existing LoRA adapters
2. Train Stage 1 (TrainCode)
3. Merge Stage 1 → Stage 2 base
4. Train Stage 2
5. Merge Stage 2 → Stage 3 base
6. Train Stage 3
7. Merge Stage 3 → Stage 4 base
8. Train Stage 4

All adapters are MOVED (never deleted) into:
  MiniGpt2/adapters_backups/<timestamp>/
"""

import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# -------------------------------------------------
# Paths
# -------------------------------------------------
# `ROOT` should be the repository root (folder containing this script).
# Previously it used `.parent.parent` which pointed one level above the repo.
ROOT = Path(__file__).resolve().parent
MINIGPT = ROOT
LORA_DIR = MINIGPT / "LoraAdapters"
BACKUP_ROOT = MINIGPT / "AdapterBackups"

ADAPTER_INDICATOR = "adapter_config.json"

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def run(script: Path, label: str | None = None):
    if label:
        print(f"\n▶ {label}")
    print(f"[RUN] {script.name}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT)
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script.name} failed")


def dry_run_checks():
    """Check repository filepaths and report status without executing anything.

    Prints adapters present, merged models, dataset files, and required scripts.
    """
    print("\n=== Dry-run: repository path checks ===")
    print(f"Repo root: {ROOT}")
    print(f"MiniGpt (scripts dir): {MINIGPT}")
    print(f"LORA_DIR: {LORA_DIR}")

    # Adapters
    if LORA_DIR.exists() and LORA_DIR.is_dir():
        adapters = [p.name for p in LORA_DIR.iterdir() if p.is_dir()]
        print(f"Adapters found ({len(adapters)}): {adapters}")
    else:
        print("Adapters directory missing or empty.")

    # Merged models
    for s in range(1, 5):
        merged = LORA_DIR / f"merged_stage{s}"
        print(f"merged_stage{s}: {'OK' if merged.exists() else 'MISSING'} -> {merged}")

    # Dataset files
    for s in range(1, 5):
        ds = ROOT / "Voices" / f"Stage{s}.jsonl"
        print(f"Voices/Stage{s}.jsonl: {'OK' if ds.exists() else 'MISSING'} -> {ds}")

    # Scripts
    scripts = [
        MINIGPT / "TrainStages" / "TrainCode.py",
        MINIGPT / "TrainStages" / "TrainStage2.py",
        MINIGPT / "TrainStages" / "TrainStage3.py",
        MINIGPT / "TrainStages" / "TrainStage4.py",
        MINIGPT / "MergeStages" / "MergeStagesB-1.py",
        MINIGPT / "MergeStages" / "MergeStages1-2.py",
        MINIGPT / "MergeStages" / "MergeStages2-3.py",
        MINIGPT / "MergeStages" / "MergeStages3-4.py",
    ]
    for sp in scripts:
        print(f"Script {sp.name}: {'OK' if sp.exists() else 'MISSING'} -> {sp}")

    print("\nDry-run checks complete. No actions taken.")

def backup_all_adapters():
    if not LORA_DIR.exists():
        print("[INFO] LoraAdapters directory does not exist — nothing to back up")
        return

    items = list(LORA_DIR.iterdir())

    if not items:
        print("[INFO] LoraAdapters is empty — nothing to back up")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_root = BACKUP_ROOT / ts
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[BACKUP] Moving EVERYTHING from LoraAdapters → {dst_root}")

    for item in items:
        print(f"  - {item.name}")
        shutil.move(str(item), str(dst_root / item.name))


# -------------------------------------------------
# Pipeline
# -------------------------------------------------
def main():
    print("\n=== MiniGpt2 LoRA AUTO PIPELINE START ===")

    # CLI / mode handling
    parser = argparse.ArgumentParser(description="Reset & Train LoRA pipeline controller")
    parser.add_argument("--dry-run", action="store_true", help="Check filepaths and report instead of running")
    parser.add_argument("--action", choices=["missing", "new", "no"], help="Non-interactive action")
    parsed = parser.parse_args()

    if parsed.dry_run:
        dry_run_checks()
        return

    # Ask user whether to train new adapters (interactive fallback)
    if parsed.action:
        resp = parsed.action
    else:
        resp = input(
            "Action? Type 'missing' to train only missing adapters, "
            "'new' to retrain everything, or 'no' to abort: "
        ).strip().lower()

    if resp == "no":
        print("No action taken. Exiting.")
        return

    elif resp == "new":
        print("[ACTION] Full reset requested — backing up all adapters")
        backup_all_adapters()
        print("[ACTION] Retraining ALL stages from scratch")

    elif resp == "missing":
        print("[ACTION] Training only missing adapters (existing ones will be kept)")

    else:
        print(f"[ERROR] Unknown option '{resp}'. Use: missing | new | no")
        return


    # Helper checks
    def adapter_exists(stage: int) -> bool:
        # Candidate adapter folder names
        candidates = [f"stage{stage}-lora", f"stage{stage}_lora", f"stage{stage}", f"stage{stage}lora"]
        for c in candidates:
            if (LORA_DIR / c).exists():
                return True
        return False

    def merged_exists(stage: int) -> bool:
        return (LORA_DIR / f"merged_stage{stage}").exists()

    # 1. Stage 1
    if adapter_exists(1):
        print("[SKIP] Stage 1 adapter already present — skipping TrainCode.py")
    else:
        run(MINIGPT / "TrainStages" / "TrainCode.py", "Training Stage 1 (base personality / style)")
        print("✅ Stage 1 complete")

    # 2. Merge Stage 1 → Stage 2 base
    if merged_exists(1):
        print("[SKIP] merged_stage1 already exists — skipping MergeStagesB-1.py")
    else:
        print("\n🔗 Fusing base model + Stage 1 LoRA → merged_stage1")
        print(f"[SCRIPT] Executing strict merge script: {'MergeStagesB-1.py'}")
        run(MINIGPT / "MergeStages" / "MergeStagesB-1.py")
        print("✅ Merge Stage 1 → Stage 2 base complete")

    # 3. Stage 2
    if adapter_exists(2):
        print("[SKIP] Stage 2 adapter already present — skipping TrainStage2.py")
    else:
        run(MINIGPT / "TrainStages" / "TrainStage2.py", "Training Stage 2 (chaos / voice / behavior)")
        print("✅ Stage 2 complete")

    # 4. Merge Stage 2 → Stage 3 base
    if merged_exists(2):
        print("[SKIP] merged_stage2 already exists — skipping MergeStages1-2.py")
    else:
        print("\n🔗 Fusing merged_stage1 + Stage 2 LoRA → merged_stage2")
        print(f"[SCRIPT] Executing strict merge script: {'MergeStages1-2.py'}")
        run(MINIGPT / "MergeStages" / "MergeStages1-2.py")
        print("✅ Merge Stage 2 → Stage 3 base complete")

    # 5. Stage 3
    if adapter_exists(3):
        print("[SKIP] Stage 3 adapter already present — skipping TrainStage3.py")
    else:
        run(MINIGPT / "TrainStages" / "TrainStage3.py", "Training Stage 3 (hard canon / facts)")
        print("✅ Stage 3 complete")

    # 6. Merge Stage 3 → Stage 4 base
    if merged_exists(3):
        print("[SKIP] merged_stage3 already exists — skipping MergeStages2-3.py")
    else:
        print("\n🔗 Fusing merged_stage2 + Stage 3 LoRA → merged_stage3")
        print(f"[SCRIPT] Executing strict merge script: {'MergeStages2-3.py'}")
        run(MINIGPT / "MergeStages" / "MergeStages2-3.py")
        print("✅ Merge Stage 3 → Stage 4 base complete")

    # 7. Stage 4
    if adapter_exists(4):
        print("[SKIP] Stage 4 adapter already present — skipping TrainStage4.py")
    else:
        run(MINIGPT / "TrainStages" / "TrainStage4.py", "Training Stage 4 (response length & control)")
        print("✅ Stage 4 complete")

    if merged_exists(4):
        print("[SKIP] merged_stage4 already exists — skipping MergeStages3-4.py")
    else:
        print("\n🔗 Fusing merged_stage3 + Stage 4 LoRA → merged_stage4")
        print(f"[SCRIPT] Executing strict merge script: {'MergeStages3-4.py'}")
        run(MINIGPT / "MergeStages" / "MergeStages3-4.py")
        print("✅ Merge Stage 4 → Final complete")

    print("\n🎉 PIPELINE COMPLETE")
    print("📦 Final fused base model: merged_stage4")

# -------------------------------------------------
if __name__ == "__main__":
    main()
