# Isolated nnU-Net baseline setup

Environment: `../.venv-nnunet` (Python 3.12, nnU-Net 2.8.1, PyTorch 2.14.1).
The existing MONAI environment and experiment script are unchanged.

Activate in a terminal:

```sh
source "$HOME/3D segmentation + slicer/nnunet_setup/activate_nnunet.sh"
```

## Data and split

Dataset: `~/Documents/medical/nnunet_workspace/nnUNet_raw/Dataset501_COVID19Lesions`.
Images: `imagesTr/{case_id}_0000.nii.gz`; labels: `labelsTr/{case_id}.nii.gz`.
These are byte-identical copies of original HU CTs and binary labels, not MONAI-resampled outputs.
`dataset.json` explicitly uses NibabelIO to preserve the effective affine convention previously audited.

Standard nnU-Net development layout includes 169 cases in imagesTr/labelsTr.
The custom `nnUNet_preprocessed/Dataset501_COVID19Lesions/splits_final.json`
assigns exactly the original 139 training and 30 validation cases to **fold 0**.
Use fold `0`, never `all`, and retain this split file. There is no five-fold experiment here.
Default nnU-Net fingerprinting/planning will use these 169 development cases;
validation cases are excluded from gradient updates but not default dataset-level planning.
If strict train-only fingerprinting is required, prepare a train-only planning dataset
and transfer its plans before any planning command is run.

The 30 test case IDs are stored only in `manifests/held_out_test.json` outside raw/preprocessed.
No test CT or label was copied or used for setup checks. Test data must not be used for
planning, normalization, training, hyperparameter selection or postprocessing choices.
Original splits use inferred case-ID groups, not independently verified patient identities.

## Reproducibility

`prepare_dataset.py` recreates/validates the copies using the project's original splits.json.
`manifests/conversion_manifest.csv` records filenames, original split and SHA-256 hashes.
`manifests/preparation_report.json` records integrity checks.
`requirements-lock.txt` records installed versions.
`verify_setup.py` checks installed imports, a tiny MPS 3D forward/backward operation,
and nnU-Net's dataset integrity checker. This is not a full-network memory/performance test.

Fingerprint extraction and default experiment planning are complete. The full-resolution model passed a synthetic MPS batch-1 benchmark (4.87 seconds/update after warm-up; highest sampled driver allocation 7.09 GiB). Dataset preprocessing and nnU-Net lesion training remain pending; run the pilot below.
Keep multiprocessing modest on the 16 GB Mac. Do not assume default full nnU-Net training fits.

References:
- https://github.com/MIC-DKFZ/nnUNet
- https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
- https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md


## Start a training pilot

Run `train_pilot.py` with `.venv-nnunet/bin/python`, or add `--dry-run` first.
The launcher preprocesses only `3d_fullres` with two workers, preserves fold 0,
and uses a separately named Mac plan with batch size 1 (other planned settings unchanged).
It runs five epochs of 50 updates and 10 patch-validation batches each, then runs
whole-volume validation. The schedule and batch differ from standard nnU-Net;
this pilot must not be reported as a fully trained reference model.
Training uses nnU-Net's loss, SGD, augmentation and foreground sampling, not the MONAI settings.
The successful synthetic benchmark does not guarantee augmentation and full-volume
validation fit the memory budget. Checkpoints are saved each epoch; use `--resume`
for interrupted training. Increasing the pilot's schedule later changes its learning-rate
trajectory and requires a separately designed experiment.
Custom trainer inference should be run through a launcher that imports this trainer class;
stock CLI class discovery does not install this project-local class.
