import sys
import torch
import monai
import nibabel as nib

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("MONAI:", monai.__version__)
print("NiBabel:", nib.__version__)
print("Apple GPU available:", torch.backends.mps.is_available())