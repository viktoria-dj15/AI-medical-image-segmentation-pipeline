# COVID-19 lesion segmentation: experiment report

## Objective and data

Binary volumetric segmentation of COVID-19 lung lesions, using original HU CTs and
matching binary annotations from COVID-19-20. There are 199 labeled pairs.
The official 50-scan validation subset has no public masks and is not our labeled validation set.
The local fixed split contains 139 training, 30 validation, and 30 untouched test cases.
Case-group separation is based on identifiers, not independently verified patient identities.
Original images are preserved. Geometry auditing gave 198 PASS, one REVIEW, zero FAIL.
The review case had differing NIfTI transform codes but matching effective affines.

## MONAI baseline

Small 3D U-Net: channels 8/16/32/64, 150118 parameters, instance normalization,
Dice+BCE objective, Adam at 0.001. RAS orientation, 1.5x1.5x2 mm spacing, HU window
-1000 to 400 scaled to 0-1; training patches 96x96x64.

Initial five-case runs failed on training and validation volumes. Controlled tests
established that learning was possible: one fixed patch reached Dice 0.9694;
ten fixed positive patches reached mean 0.9619; changing positive crops reached
0.8100 on a fixed training-case diagnostic collection. These are memorization or
training-case diagnostics, not independent validation scores.

| Experiment | Mean validation Dice | Notes |
|---|---:|---|
| 139 cases, positive-only, 2000 updates | 0.1721 | Constant blending, 25% overlap |
| 139 cases, 75/25 crop centres, 2000 updates | 0.2139 | Same inference |
| Same mixed model, Gaussian/50% inference | 0.2506 | All 30 validation cases |
| Mixed sampling, 5000-update run, best step 4000 | 0.3932 | Gaussian/50%, threshold 0.5 |
| Same run, final step 5000 | 0.3324 | Lower precision, higher recall |

Best-step mean precision was 0.4769 and recall 0.4294. Threshold 0.4 produced
mean validation Dice 0.3953, a small gain with more false positives; threshold 0.5
reproduced the saved baseline exactly. No threshold has independent test validation.
Repeated model selection on the same validation set can give optimistic estimates.
False positives, missed lesions and strong case-to-case variability remain.

## Engineering corrections

CT/mask geometry and binary labels are checked. Separate output directories prevent
mixing checkpoints/inference settings. Local persistent caches preserve spatial metadata;
only locally generated preprocessing caches use unrestricted metadata deserialization.
Model checkpoints remain separately handled. Inspection flags are heuristics, not quality certification.
Long-run MONAI checkpoints now save optimizer and random-generator states.

## nnU-Net setup and hardware feasibility

Separate environment: Python 3.12, nnU-Net 2.8.1, PyTorch 2.14.1.
Dataset501_COVID19Lesions uses byte-verified NIfTI copies and NibabelIO.
The standard development layout contains 169 pairs and a custom fold 0 assigning
139 train / 30 validation. Test cases are not copied. Standard fingerprinting/planning
uses the 169 development cases, including validation; it is not train-only planning.

Default full-resolution plan: patch 28x256x256, spacing 5x0.743x0.743 mm in internal
axis order, planned batch 2, 44.6 million parameters. On the local 16 GB Apple Silicon
Mac, a three-step synthetic MPS benchmark at batch 1 passed with deep supervision,
backpropagation and SGD. Measured updates averaged 4.87 seconds after warm-up.
Highest sampled driver allocation was 7.09 GiB under a 7.70 GiB allocator cap.
These are sampled allocations, not true peaks; loading/augmentation are excluded.
Batch 2 has not been verified. Newer residual-encoder presets were not benchmarked.

## Next experiment

The provided launcher uses the benchmarked batch 1 and a separate plan name,
preprocesses the full-resolution configuration and starts a 250-update pilot with
whole-volume validation afterward. This differs from standard nnU-Net training;
its quality results must be labeled as a short pilot. Full training and independent
test evaluation remain pending. Do not use fold `all` with the fixed evaluation split.

## Sources

- [COVID-19-20 data](https://covid-segmentation.grand-challenge.org/Data/)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [Dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)
- [Custom splits](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md)

This is an educational research project; performance does not establish clinical utility.
