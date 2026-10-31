from pathlib import Path
import os
import csv

import nibabel as nib
import numpy as np

DATA_DIR = Path(os.environ.get(
    "COVID19_TRAIN_DIR",
    str(Path.home() / "Documents/medical/covid19-20/COVID-19-20_v2/Train"),
)).expanduser()
REPORT = Path(__file__).resolve().parent / "dataset_audit.csv"

images = sorted(DATA_DIR.glob("*_ct.nii.gz"))
if not images:
    raise FileNotFoundError(f"No CT images found in {DATA_DIR}")

rows = []

for number, ct_path in enumerate(images, start=1):
    mask_path = ct_path.with_name(
        ct_path.name.replace("_ct.nii.gz", "_seg.nii.gz")
    )
    row = {
        "case": ct_path.name.replace("_ct.nii.gz", ""),
        "shape": "",
        "spacing": "",
        "labels": "",
        "status": "PASS",
        "notes": "",
    }
    issues = []
    notes = []

    try:
        if not mask_path.exists():
            raise FileNotFoundError("Matching mask is missing")

        ct = nib.load(ct_path)
        mask = nib.load(mask_path)

        row["shape"] = str(ct.shape)
        row["spacing"] = str(
            tuple(float(v) for v in ct.header.get_zooms()[:3])
        )

        if len(ct.shape) != 3 or len(mask.shape) != 3:
            issues.append("Not a 3D image")
        if ct.shape != mask.shape:
            issues.append("Dimensions differ")
        if not np.allclose(
            ct.header.get_zooms()[:3],
            mask.header.get_zooms()[:3],
            rtol=0, atol=1e-5,
        ):
            issues.append("Voxel spacing differs")
        if not np.allclose(
            ct.affine, mask.affine, rtol=0, atol=1e-3
        ):
            issues.append("Physical alignment differs")

        # Different header codes need review, but do not necessarily
        # mean the effective physical-coordinate alignment differs.
        ct_codes = (
            int(ct.header["qform_code"]),
            int(ct.header["sform_code"]),
        )
        mask_codes = (
            int(mask.header["qform_code"]),
            int(mask.header["sform_code"]),
        )
        if ct_codes != mask_codes:
            notes.append(
                f"Transform codes differ: CT={ct_codes}, mask={mask_codes}"
            )

        # Process one mask at a time to limit memory use.
        labels = np.unique(np.asanyarray(mask.dataobj))
        row["labels"] = str(labels.tolist())

        if not np.isin(labels, [0, 1]).all():
            issues.append("Mask has values other than 0 and 1")
        if not np.any(labels == 1):
            notes.append("Empty lesion mask: review")

    except Exception as error:
        issues.append(f"{type(error).__name__}: {error}")

    row["status"] = "FAIL" if issues else ("REVIEW" if notes else "PASS")
    row["notes"] = "; ".join(issues + notes)
    rows.append(row)

    print(
        f"[{number}/{len(images)}] "
        f"{row['case']}: {row['status']}",
        flush=True,
    )

# Find masks that have no corresponding CT.
for mask_path in sorted(DATA_DIR.glob("*_seg.nii.gz")):
    ct_path = mask_path.with_name(
        mask_path.name.replace("_seg.nii.gz", "_ct.nii.gz")
    )
    if not ct_path.exists():
        rows.append({
            "case": mask_path.name,
            "shape": "",
            "spacing": "",
            "labels": "",
            "status": "FAIL",
            "notes": "Mask has no corresponding CT",
        })

with REPORT.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nAUDIT SUMMARY")
print("CT images:", len(images))
for status in ("PASS", "REVIEW", "FAIL"):
    print(f"{status}: {sum(r['status'] == status for r in rows)}")

print("\nCases needing attention:")
for row in rows:
    if row["status"] != "PASS":
        print(row["case"], "—", row["notes"])

print("\nReport saved to:", REPORT)