# 3D COVID-19 Lesion Segmentation

A research and learning project building a reproducible chest CT segmentation workflow with **Python, PyTorch, MONAI, PyCharm, and 3D Slicer**.

**Current status:** the MONAI baseline has been trained and evaluated on a fixed 139/30/30 split. Best mean validation Dice is 0.3932 (30 cases, step 4000, threshold 0.5). An isolated nnU-Net environment, converted dataset, plans and a successful batch-1 MPS benchmark are ready. A short nnU-Net training launcher is included; nnU-Net lesion-training results are pending.

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

Observed local environment: Python 3.12.16, PyTorch 2.14.1, MONAI 1.6.1, NiBabel 5.4.2, and Slicer 5.12.4. Core package imports and a CPU 3D convolution were tested. PyTorch reported Apple MPS available when run from PyCharm on the 16 GB Apple M5 Mac; MONAI training and a synthetic nnU-Net batch-1 training benchmark have since completed successfully on MPS.

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

The split script has now been run successfully: 139 training, 30 validation, and 30 test cases, with no base-ID overlap. The resulting `splits.json` is stored locally. The official 50-case validation subset is kept separate because its ground truth is unavailable locally.

`splits.json` contains machine-specific absolute paths and is excluded from Git. Preserve it locally across experiments. A future portable manifest should record relative paths and fixed case assignments for sharing across machines.

## Completed modeling and remaining work

Paired preprocessing, MPS forward/backward checks, controlled memorization tests,
full-split MONAI training, whole-volume validation, inference comparisons and
threshold analysis have completed. Best mean validation Dice is 0.3932 at threshold
0.5; the validation-selected 0.4 threshold gives 0.3953. These are not test results.
See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for the experiment sequence.

The nnU-Net environment, data conversion, fingerprint and default plan are ready.
Batch 1 passed a synthetic MPS benchmark. Run the supplied training pilot next.
Remaining work includes full nnU-Net training, restoring predictions to original
CT geometry for final export, additional surface/volume metrics, and final test
assessment after freezing model-selection decisions. Current quantitative outputs
are primarily evaluated on resampled CT grids.

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

## One-case preprocessing check

Run `python preprocess_one_case.py` in PyCharm. It reads only the first training entry from the saved split manifest. The initial settings are RAS orientation, 1.5 × 1.5 × 2.0 mm spacing, linear CT interpolation, nearest-neighbor label interpolation, and clipping CT intensities from −1000 to 400 HU before scaling to [0, 1]. An image-derived foreground crop is followed by sampling two 96 × 96 × 64 patches. Lesion/background center-selection probabilities are equal; a pair of samples is not guaranteed to contain one of each. Spatial transform history remains attached to MONAI MetaTensors for later inversion work.

The executed case was `0267`: original shape 512 × 512 × 68 with 0.915 × 0.915 × 5.0 mm spacing, processed shape 1 × 313 × 313 × 168. Geometry, finite values, binary labels, and retained foreground checks passed. A transformed overlay was saved in `outputs/preprocessing_overlay.png` and visually inspected. These settings are development choices, not tuned parameters or evidence of segmentation performance. Resampling does not recover detail absent from the original 5 mm slices.


## Training results and nnU-Net pilot

See [the experiment report](EXPERIMENT_REPORT.md) for the controlled experiments,
validation results, hardware benchmark and limitations.

To start the nnU-Net pilot on the prepared local dataset:

```bash
"$HOME/3D segmentation + slicer/.venv-nnunet/bin/python" \
"$HOME/3D segmentation + slicer/nnunet_setup/train_pilot.py"
```

It preprocesses the 3D configuration, trains 250 updates at batch 1, then validates.
Use `--dry-run` to check setup, or `--resume` for an interrupted pilot.
This shortened schedule is a feasibility pilot, not standard full nnU-Net training.
See [nnU-Net setup](nnunet_setup/README.md). Data, caches, environments and weights
are local only and excluded from this repository.
