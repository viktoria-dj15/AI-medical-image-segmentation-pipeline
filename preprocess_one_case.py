"""Inspect deterministic MONAI preprocessing and sampled 3D training patches."""
from pathlib import Path
import argparse
import json
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImaged, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd,
    RandCropByPosNegLabeld,
)
from monai.utils import set_determinism


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'splits.json').read_text())
    case = manifest['train'][0]
    folder = Path(os.environ.get('COVID19_TRAIN_DIR', manifest['source'])).expanduser()
    pair = {
        'image': str(folder / f"{case['case_id']}_ct.nii.gz"),
        'label': str(folder / f"{case['case_id']}_seg.nii.gz"),
    }
    for path in pair.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    original = nib.load(pair['image'])
    print('Training case:', case['case_id'], flush=True)
    print('Original shape:', original.shape)
    print('Original spacing (mm):', original.header.get_zooms()[:3])

    # Initial development settings, to be assessed using training data only.
    # MONAI records spatial transform history on its MetaTensors.
    keys = ['image', 'label']
    transforms = Compose([
        LoadImaged(keys=keys, ensure_channel_first=True, dtype=np.float32),
        Orientationd(keys=keys, axcodes='RAS'),
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0),
                 mode=('bilinear', 'nearest')),
        ScaleIntensityRanged(keys='image', a_min=-1000, a_max=400,
                             b_min=0, b_max=1, clip=True),
        # Image-derived crop: never use the ground-truth mask to choose it.
        CropForegroundd(keys=keys, source_key='image', margin=8,
                        allow_smaller=True),
    ])
    data = transforms(pair)
    image, label = data['image'], data['label']
    if image.shape != label.shape:
        raise ValueError('Transformed CT and mask dimensions differ')
    if not torch.allclose(image.affine, label.affine, atol=1e-3, rtol=0):
        raise ValueError('Transformed physical alignment differs')
    if not torch.isfinite(image).all() or not torch.isfinite(label).all():
        raise ValueError('Nonfinite transformed values')
    if not torch.all((label == 0) | (label == 1)):
        raise ValueError('Mask is no longer binary')
    if not torch.any(label == 1):
        raise ValueError('Lesion disappeared during preprocessing')

    print('Processed shape [channel, X, Y, Z]:', tuple(image.shape))
    print('Processed spacing (mm):', nib.affines.voxel_sizes(image.affine.numpy()))
    print('CT range:', float(image.min()), float(image.max()))
    print('Mask labels:', torch.unique(label).tolist())
    print('Dimensions and physical alignment: PASS')

    set_determinism(seed=42)
    sampler = Compose([
        SpatialPadd(keys=keys, spatial_size=(96, 96, 64)),
        RandCropByPosNegLabeld(keys=keys, label_key='label',
                              spatial_size=(96, 96, 64), pos=1, neg=1,
                              num_samples=2, image_key='image', image_threshold=0),
    ])
    patches = sampler(data)
    print('Patch shapes:', [tuple(p['image'].shape) for p in patches])
    print('Patch lesion voxels:', [int(p['label'].sum()) for p in patches])
    # Sampling is probabilistic: pos=neg gives equal center-selection probability,
    # rather than guaranteeing one positive and one negative patch per call.

    im = image[0].numpy()
    seg = label[0].numpy() > 0
    z = int(np.argmax(seg.sum(axis=(0, 1))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.imshow(im[:, :, z].T, cmap='gray', vmin=0, vmax=1, origin='lower')
        ax.axis('off')
    axes[0].set_title('Preprocessed CT')
    axes[1].imshow(np.ma.masked_where(~seg[:, :, z].T, seg[:, :, z].T),
                   cmap='autumn', vmin=0, vmax=1, alpha=0.4, origin='lower')
    axes[1].set_title('Preprocessed ground-truth overlay')
    fig.suptitle(f"{case['case_id']} — axial slice {z}")
    fig.tight_layout()
    output = root / 'outputs'
    output.mkdir(exist_ok=True)
    fig.savefig(output / 'preprocessing_overlay.png', dpi=150)
    print('Overlay saved:', output / 'preprocessing_overlay.png')
    if not args.no_show:
        plt.show()
    plt.close(fig)
    print('PREPROCESSING CHECK PASSED')


if __name__ == '__main__':
    main()
