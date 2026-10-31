from pathlib import Path
import os

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(os.environ.get(
    "COVID19_TRAIN_DIR",
    str(Path.home() / "Documents/medical/covid19-20/COVID-19-20_v2/Train"),
)).expanduser()

ct_path = DATA_DIR / "volume-covid19-A-0003_ct.nii.gz"
mask_path = DATA_DIR / "volume-covid19-A-0003_seg.nii.gz"

for path in (ct_path, mask_path):
    if not path.is_file():
        raise FileNotFoundError(path)

# Load NIfTI images and their spatial metadata.
ct_img = nib.load(ct_path)
mask_img = nib.load(mask_path)

ct_spacing = ct_img.header.get_zooms()[:3]
mask_spacing = mask_img.header.get_zooms()[:3]

print("CT dimensions:", ct_img.shape)
print("Mask dimensions:", mask_img.shape)
print("CT voxel spacing:", ct_spacing)
print("Mask voxel spacing:", mask_spacing)
print("CT spatial units:", ct_img.header.get_xyzt_units()[0])
print("CT orientation:", nib.aff2axcodes(ct_img.affine))
print("Mask orientation:", nib.aff2axcodes(mask_img.affine))

# Affines map voxel indices to physical coordinates.
same_shape = ct_img.shape == mask_img.shape
same_spacing = np.allclose(
    ct_spacing, mask_spacing, rtol=0, atol=1e-5
)
same_affine = np.allclose(
    ct_img.affine, mask_img.affine, rtol=0, atol=1e-3
)

print("\nDimensions match:", same_shape)
print("Voxel spacing matches:", same_spacing)
print("Physical-coordinate alignment matches:", same_affine)

if len(ct_img.shape) != 3:
    raise ValueError("Expected a 3D CT volume.")

if not (same_shape and same_spacing and same_affine):
    print("CT affine:\n", ct_img.affine)
    print("Mask affine:\n", mask_img.affine)
    raise ValueError(
        "Geometry mismatch. Inspect before overlaying or resampling."
    )

mask = mask_img.get_fdata(dtype=np.float32)
labels = np.unique(mask)
print("Mask label values:", labels)

if not np.isin(labels, [0, 1]).all():
    raise ValueError("Expected a binary mask containing only 0 and 1.")

if not np.any(mask):
    raise ValueError("This mask contains no lesion voxels.")

# Reorient both images consistently for plotting.
# This changes axis ordering/flips without interpolating the data.
ct_ras = nib.as_closest_canonical(ct_img)
mask_ras = nib.as_closest_canonical(mask_img)

ct = ct_ras.get_fdata(dtype=np.float32)
lesion = mask_ras.get_fdata(dtype=np.float32) > 0

# Select an axial slice containing the largest lesion area.
slice_index = int(np.argmax(lesion.sum(axis=(0, 1))))
ct_slice = ct[:, :, slice_index].T
mask_slice = lesion[:, :, slice_index].T

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
spacing = ct_ras.header.get_zooms()[:3]
aspect = spacing[1] / spacing[0]

for ax in axes:
    ax.imshow(
        ct_slice,
        cmap="gray",
        vmin=-1000,
        vmax=400,
        origin="lower",
        aspect=aspect,
    )
    ax.axis("off")

axes[0].set_title("Chest CT")
axes[1].imshow(
    np.ma.masked_where(~mask_slice, mask_slice),
    cmap="autumn",
    vmin=0,
    vmax=1,
    alpha=0.45,
    origin="lower",
    aspect=aspect,
)
axes[1].set_title("Ground-truth lesion overlay")
fig.suptitle(f"Case 0003 — axial slice {slice_index}")
plt.tight_layout()
plt.show()

print("\nGeometry checks passed.")
print("Visual inspection is still needed to confirm anatomical alignment.")