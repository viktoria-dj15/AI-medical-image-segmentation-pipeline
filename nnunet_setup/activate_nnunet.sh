# Source this file in a terminal; it does not launch preprocessing/training.
source "$HOME/3D segmentation + slicer/.venv-nnunet/bin/activate"
export nnUNet_raw="$HOME/Documents/medical/nnunet_workspace/nnUNet_raw"
export nnUNet_preprocessed="$HOME/Documents/medical/nnunet_workspace/nnUNet_preprocessed"
export nnUNet_results="$HOME/Documents/medical/nnunet_workspace/nnUNet_results"
export nnUNet_n_proc_DA=2
export MPLCONFIGDIR="$HOME/Documents/medical/nnunet_workspace/matplotlib_cache"
# Benchmark MPS first; avoid compilation overhead during initial compatibility checks.
export nnUNet_compile=false
