from pathlib import Path
import os,json,importlib.metadata
ROOT=Path.home()/"Documents/medical/nnunet_workspace"
for name in ("nnUNet_raw","nnUNet_preprocessed","nnUNet_results"):
    os.environ[name]=str(ROOT/name)
os.environ["MPLCONFIGDIR"]=str(ROOT/"matplotlib_cache")
os.environ["nnUNet_compile"]="false"

def main():
    import torch
    from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    print("nnU-Net:",importlib.metadata.version("nnunetv2"),flush=True)
    print("PyTorch:",torch.__version__,"MPS:",torch.backends.mps.is_available(),flush=True)
    if not torch.backends.mps.is_available():raise RuntimeError("MPS unavailable in this execution context")
    model=torch.nn.Conv3d(1,2,3,padding=1).to("mps")
    x=torch.randn(1,1,8,8,8,device="mps")
    model(x).square().mean().backward()
    torch.mps.synchronize()
    print("MPS Conv3d forward/backward: PASS (not a full-model benchmark)",flush=True)
    verify_dataset_integrity(str(ROOT/"nnUNet_raw/Dataset501_COVID19Lesions"),num_processes=2)
    splits=json.loads((ROOT/"nnUNet_preprocessed/Dataset501_COVID19Lesions/splits_final.json").read_text())
    assert len(splits)==1 and len(splits[0]["train"])==139 and len(splits[0]["val"])==30
    print("Fixed fold 0: 139 train / 30 validation. PASS",flush=True)
if __name__=="__main__":main()
