"""Prepare and run a 250-update nnU-Net MPS pilot on fixed fold 0."""
from pathlib import Path
import os
ROOT = Path.home() / "Documents/medical/nnunet_workspace"
for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    os.environ[key] = str(ROOT/key)
os.environ["nnUNet_compile"] = "false"
os.environ["nnUNet_n_proc_DA"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["MPLCONFIGDIR"] = str(ROOT/"matplotlib_cache")
import argparse
import json
import subprocess
import sys
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerMacPilot(nnUNetTrainer):
    """Uses the standard trainer; run length is shortened by the launcher."""
    pass

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true",help="Validate settings without preprocessing or training")
    parser.add_argument("--resume",action="store_true",help="Continue this same pilot from its latest checkpoint")
    args=parser.parse_args()
    folder=ROOT/"nnUNet_preprocessed/Dataset501_COVID19Lesions"
    splits=json.loads((folder/"splits_final.json").read_text())
    manifest=json.loads((Path(__file__).resolve().parents[1]/"splits.json").read_text())
    assert len(splits)==1
    assert splits[0]["train"]==[c["case_id"] for c in manifest["train"]]
    assert splits[0]["val"]==[c["case_id"] for c in manifest["validation"]]
    assert not set(splits[0]["train"]+splits[0]["val"]) & {c["case_id"] for c in manifest["test"]}
    plans=json.loads((folder/"nnUNetPlans.json").read_text())
    # Retain the planned architecture, patch, spacing and preprocessing.
    # Batch 1 is the configuration already benchmarked on this Mac.
    plans["plans_name"]="nnUNetPlans_MacBatch1"
    plans["configurations"]["3d_fullres"]["batch_size"]=1
    path=folder/"nnUNetPlans_MacBatch1.json"
    if path.exists() and json.loads(path.read_text())!=plans:
        raise RuntimeError("Existing Mac plan differs; refusing to overwrite.")
    if not torch.backends.mps.is_available():raise RuntimeError("MPS unavailable")
    print("Fixed split: 139 training / 30 validation / 30 tests excluded",flush=True)
    print("Pilot: 5 epochs x 50 updates = 250 updates; batch 1; MPS",flush=True)
    print("Standard nnU-Net SGD/loss/sampling/augmentation; shortened learning-rate schedule.")
    print("This is a training pilot, not a fully trained nnU-Net baseline.")
    if args.dry_run:
        print("Dry run passed; no preprocessing or training started.");return
    path.write_text(json.dumps(plans,indent=2)+"\n")
    # Reuses existing processed cases; limited to the selected configuration.
    subprocess.run([str(Path(sys.executable).parent/"nnUNetv2_preprocess"),"-d","501","-plans_name","nnUNetPlans_MacBatch1","-c","3d_fullres","-np","2"],check=True)
    assert json.loads((folder/"splits_final.json").read_text())==splits
    plans["continue_training"]=args.resume
    trainer=nnUNetTrainerMacPilot(plans=plans,configuration="3d_fullres",fold=0,dataset_json=json.loads((folder/"dataset.json").read_text()),device=torch.device("mps"))
    trainer.num_epochs=5
    trainer.num_iterations_per_epoch=50
    trainer.num_val_iterations_per_epoch=10
    trainer.save_every=1
    output=Path(trainer.output_folder)
    if args.resume:
        checkpoints=[output/"checkpoint_latest.pth",output/"checkpoint_final.pth"]
        checkpoint=next((c for c in checkpoints if c.exists()),None)
        if checkpoint is None:raise FileNotFoundError("No pilot checkpoint to resume")
        trainer.load_checkpoint(str(checkpoint))
    elif output.exists() and any(output.glob("checkpoint*.pth")):
        raise RuntimeError("Pilot checkpoint exists; use --resume rather than overwrite it.")
    # Same allocator cap as the successful synthetic batch-1 benchmark.
    torch.mps.set_per_process_memory_fraction(0.65)
    print("Training output:",output,flush=True)
    trainer.run_training()
    best=output/"checkpoint_best.pth"
    if best.exists():trainer.load_checkpoint(str(best))
    print("Whole-volume validation on the 30 validation cases...",flush=True)
    trainer.perform_actual_validation(save_probabilities=False)
    print("Pilot complete:",output)
if __name__=="__main__":main()
