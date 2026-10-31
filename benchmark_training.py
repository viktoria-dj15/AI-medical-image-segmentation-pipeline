from pathlib import Path
import json
import os
import time

import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandCropByPosNegLabeld,
)
from monai.utils import set_determinism


def main():
    set_determinism(seed=42)

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple GPU is unavailable in this interpreter. "
            "Run this script using your configured PyCharm environment."
        )

    device = torch.device("mps")
    print("Device:", device, flush=True)

    # Load only a case assigned to training.
    project = Path(__file__).resolve().parent
    manifest = json.loads((project / "splits.json").read_text())
    case = manifest["train"][0]

    folder = Path(
        os.environ.get("COVID19_TRAIN_DIR", manifest["source"])
    ).expanduser()

    pair = {
        "image": str(folder / f"{case['case_id']}_ct.nii.gz"),
        "label": str(folder / f"{case['case_id']}_seg.nii.gz"),
    }

    for path in pair.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    print("Training case:", case["case_id"], flush=True)
    print("Preparing one patch...", flush=True)

    keys = ["image", "label"]

    transforms = Compose([
        LoadImaged(
            keys=keys,
            ensure_channel_first=True,
            dtype=np.float32,
        ),
        Orientationd(
            keys=keys,
            axcodes="RAS",
            labels=None,
        ),
        Spacingd(
            keys=keys,
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys="image",
            a_min=-1000,
            a_max=400,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        SpatialPadd(
            keys=keys,
            spatial_size=(96, 96, 64),
        ),
        # Force a lesion-centered patch for this hardware check.
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=(96, 96, 64),
            pos=1,
            neg=0,
            num_samples=1,
        ),
    ])

    patch = transforms(pair)[0]

    # Convert MetaTensors to ordinary tensors and add batch dimension.
    image = (
        patch["image"].as_tensor()
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    label = (
        patch["label"].as_tensor()
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )

    del patch

    print("Input shape [batch, channel, X, Y, Z]:", tuple(image.shape))
    print("Lesion voxels in patch:", int(label.sum().item()))

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=1,
        norm="INSTANCE",
    ).to(device)

    loss_function = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {parameters:,}")
    print("\nStarting training benchmark...", flush=True)

    warmup_steps = 2
    measured_steps = 5
    times = []
    losses = []

    for step in range(warmup_steps + measured_steps):
        # Synchronize to measure completed GPU work accurately.
        torch.mps.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        prediction = model(image)

        if prediction.shape != label.shape:
            raise RuntimeError("Prediction and label dimensions differ.")

        loss = loss_function(prediction, label)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Loss became NaN or infinite.")

        loss.backward()

        for parameter in model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all().item():
                    raise RuntimeError("A gradient became NaN or infinite.")

        optimizer.step()

        torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        value = float(loss.item())
        losses.append(value)

        phase = "warm-up" if step < warmup_steps else "measured"
        if step >= warmup_steps:
            times.append(elapsed)

        print(
            f"Step {step + 1}: loss={value:.4f}, "
            f"time={elapsed:.3f} seconds ({phase})",
            flush=True,
        )

    print("\nBENCHMARK SUMMARY")
    print("Device:", device)
    print("Mean measured step time:", round(float(np.mean(times)), 3), "seconds")
    print("First loss:", round(losses[0], 4))
    print("Last loss:", round(losses[-1], 4))
    print(
        "Current tensor memory:",
        round(torch.mps.current_allocated_memory() / 1024**3, 3),
        "GiB",
    )
    print(
        "Current GPU driver allocation:",
        round(torch.mps.driver_allocated_memory() / 1024**3, 3),
        "GiB",
    )
    print("Memory values are current allocations, not peak usage.")
    print("APPLE GPU TRAINING CHECK PASSED")
    print("This benchmark does not measure validation performance.")


if __name__ == "__main__":
    main()