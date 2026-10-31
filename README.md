# 3D COVID-19 Lesion Segmentation

A research and learning project building a reproducible chest CT segmentation workflow with **Python, PyTorch, MONAI, PyCharm, and 3D Slicer**.

**Current status:** dataset inspection and geometry auditing are complete. A reproducible split script is prepared. Preprocessing, model training, and quantitative model evaluation are not yet implemented. The displayed segmentations are supplied ground-truth annotations, not model predictions.

## Problem

The goal is to train a 3D U-Net from scratch to assign each CT voxel to background or a COVID-19 lesion. The resulting binary volume should preserve the input scan's physical geometry so that predictions can be inspected in 3D Slicer and used to calculate lesion volumes.

This task segments lesions, not whole lungs or lung lobes. It is an educational research workflow, not a validated diagnostic system.

## Dataset

We use the [COVID-19-20 Lung CT Lesion Segmentation Challenge](https://covid-segmentation.grand-challenge.org/Data/) dataset, obtained through its [official download page](https://covid-segmentation.grand-challenge.org/Download/).

The downloaded archive contains:

| Subset | CT volumes | Available lesion masks |
| --- | ---: | ---: |
| Training | 199 | 199 |
| Official validation | 50 | 0 |

Volumes and masks are compressed NIfTI files (`.nii.gz`). Training pairs share a case identifier and end in `_ct.nii.gz` and `_seg.nii.gz`. Mask labels are `0` for background and `1` for lesion.

The accompanying spreadsheet lists 199 training and 50 validation filenames, all matching the downloaded files. It provides no independent patient-ID column. No duplicate filenames, repeated base case IDs, or shared base IDs between the official subsets were found. Treating base case IDs as patient groups remains an explicit assumption, not an independently verified identity mapping. The bundled README mentions 200 training patients; the actual archive and spreadsheet contain 199 training cases, which this project uses.

**Download the data separately.** CT volumes, masks, archives, and Slicer scenes are excluded from Git. Keep the original data unchanged.

## Environment and setup

Development uses PyCharm with Python 3.12. Slicer uses its own embedded Python environment. Run the Slicer script in **Slicer → View → Python Console**, not with the PyCharm interpreter.

Observed local environment: Python 3.12.16, PyTorch 2.14.1, MONAI 1.6.1, NiBabel 5.4.2, and Slicer 5.12.4. Core package imports and a CPU 3D convolution were tested. PyTorch reported Apple MPS available when run from PyCharm on the 16 GB Apple M5 Mac; complete 3D training on MPS has not yet been benchmarked.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Select `.venv/bin/python` as the project interpreter in PyCharm. The requirements file records the observed core package versions; a fresh installation on other platforms has not yet been tested.

By default the scripts expect:

```text
~/Documents/medical/covid19-20/COVID-19-20_v2/Train/
```

For another location, set `COVID19_TRAIN_DIR` to the absolute path of the `Train` folder in the PyCharm run configuration or shell. Slicer must receive the same environment variable separately if its data is elsewhere.

## Run the current workflow

```bash
python main.py                  # Check package imports and MPS availability
python inspect_first_case.py    # Inspect case 0003 and display an overlay
python audit_dataset.py         # Generate dataset_audit.csv
python create_splits.py         # Generate splits.json once
```

For the reviewed case, enter this in Slicer's Python Console, replacing the script path with your checkout location:

```python
exec(open("/absolute/path/to/project/slicer_inspect_0256_1.py").read())
```

This loads the CT and labelmap, converts the labelmap to a segmentation, configures a four-view layout, and creates a 3D lesion surface. Previous segmentations are hidden to avoid confusing overlays from different cases. Scroll through all three anatomical planes and compare the overlay with the CT.

## Completed work and observations

### 1. First-case inspection

Case `0003` was loaded using NiBabel. Both CT and mask have dimensions **512 × 512 × 291** and voxel spacing **0.74 × 0.74 × 1.0 mm**. Their effective affine transforms agree. The mask contains only 0 and 1.

For the Python plot, both volumes were reoriented to canonical axes without interpolation. The displayed axial slice was selected by maximum lesion area. The original files were also loaded into Slicer for inspection in axial, coronal, and sagittal views and as a 3D surface.

### 2. Dataset audit

The executed audit reported **198 PASS, 1 REVIEW, and 0 FAIL** across 199 training CTs. It checks paired filenames, dimensions, spacing, effective affine alignment, binary labels, and empty masks. It also checks for orphan masks.

Case `0256_1` has different qform/sform codes: CT `(1, 0)`, mask `(2, 1)`. Its effective affine passes the alignment check. It was loaded in Slicer and the displayed overlays showed no obvious large displacement. This supports retaining the case with the warning documented; it does not constitute expert verification of every voxel or slice. No header rewrite or corrective resampling was applied.

The audit is a geometry and label-format check. It does not establish annotation accuracy, detect every corrupted CT voxel, or prove the absence of patient duplication.

### 3. Split preparation

`create_splits.py` uses sorted base case IDs and random seed **42** to assign **139 training, 30 validation, and 30 test cases**. It checks group disjointness and complete coverage, and refuses to overwrite an existing split file. The script currently expects this exact 199-case dataset with unique base IDs; it stops if that assumption changes.

At the time this report was written, the script existed but no `splits.json` was present in the project folder. Therefore these assignments are prepared, not yet recorded as a completed run. The official 50-case validation subset is kept separate because its ground truth is unavailable locally.

`splits.json` contains machine-specific absolute paths and is excluded from Git. Preserve it locally across experiments. A future portable manifest should record relative paths and fixed case assignments for sharing across machines.

## Planned work

1. Run and preserve the split manifest. Use training data for fitting, validation for model selection, and the held-out test subset only for final evaluation.
2. Implement paired MONAI transforms: consistent orientation, chosen target spacing, CT intensity clipping/normalization, foreground cropping, and lesion/background patch sampling. Use nearest-neighbor interpolation for masks and retain inverse spatial transforms. Choose preprocessing settings using training data only.
3. Check transformed CT–mask overlays and run a small forward/backward training benchmark. Measure memory and runtime before choosing local MPS or a cloud CUDA GPU.
4. Train a compact 3D U-Net from random initialization with a documented loss, optimizer, augmentation policy, random seeds, and saved checkpoints. Dice plus cross-entropy is the planned starting loss.
5. Implement sliding-window inference and restore predictions to the original CT geometry before export.
6. Evaluate per case using Dice, IoU, precision, recall, surface metrics, and lesion-volume error. Define behavior for empty predictions and masks. Report distributions and failure cases, not only a single average score.
7. Import predicted masks into Slicer, compare with ground truth, and export lesion-volume tables using the binary labelmap representation. Keep surface visualization settings separate from quantitative measurement.

No training losses, segmentation scores, or claimed AI performance are available yet. Future results will be added only after the corresponding experiments run.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Environment check |
| `inspect_first_case.py` | Single-case geometry checks and overlay |
| `audit_dataset.py` | Dataset-wide checks and CSV report |
| `create_splits.py` | Deterministic local split manifest |
| `slicer_inspect_0256_1.py` | Review the flagged case inside Slicer |
| `requirements.txt` | Observed core Python dependencies |

## Data attribution and references

CT data are published under **CC BY 4.0**; challenge annotations are described on the official data page as **CC0**. Follow the source's attribution requirements when using or redistributing the data. These data terms do not automatically license this repository's code.

- Roth et al. (2022). *Rapid artificial intelligence solutions in a pandemic—The COVID-19-20 Lung CT Lesion Segmentation Challenge.* Medical Image Analysis, 82, 102605. [Challenge website](https://covid-segmentation.grand-challenge.org/).
- An et al. (2020). *CT Images in COVID-19.* The Cancer Imaging Archive. [Dataset DOI](https://doi.org/10.7937/tcia.2020.gqry-nc81).
- Clark et al. (2013). *The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository.* [DOI](https://doi.org/10.1007/s10278-013-9622-7).
- [MONAI challenge baseline](https://github.com/Project-MONAI/tutorials/tree/main/3d_segmentation/challenge_baseline): reference for planned training and inference design; this repository does not yet implement that baseline.
- [3D Slicer segmentation documentation](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmentations.html).

The annotation of the dataset was made possible through the joint work of Children's National Hospital, NVIDIA, and the National Institutes of Health for the COVID-19-20 Lung CT Lesion Segmentation Challenge.
