from pathlib import Path
from datetime import datetime
import csv
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import sliding_window_inference
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
    RandFlipd,
)
from monai.utils import set_determinism


SEED = 42
EPOCHS = 100
PATCH_SIZE = (96, 96, 64)
PATCHES_PER_CASE = 4
LEARNING_RATE = 1e-3

def train_full_split_pilot():
    import csv
    import json
    import os
    import time
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import torch
    from monai.data import PersistentDataset
    from monai.data.utils import pickle_hashing
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceCELoss
    from monai.networks.nets import UNet
    from monai.transforms import (
        Compose, LoadImaged, Orientationd,
        Spacingd, ScaleIntensityRanged,
        SpatialPadd, RandCropByPosNegLabeld,
    )
    from monai.utils import set_determinism

    seed = 42
    total_steps = 5000
    validation_interval = 500
    learning_rate = 1e-3
    patch_size = (96, 96, 64)

    set_determinism(seed=seed)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU unavailable.")

    device = torch.device("mps")
    project = Path(__file__).resolve().parent
    manifest = json.loads((project / "splits.json").read_text())

    train_cases = manifest["train"]
    val_cases = manifest["validation"]
    test_cases = manifest["test"]

    if (len(train_cases), len(val_cases), len(test_cases)) != (139, 30, 30):
        raise ValueError("Expected the existing 139/30/30 split.")

    groups = [
        {case["group_id"] for case in cases}
        for cases in (train_cases, val_cases, test_cases)
    ]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Split groups overlap.")

    data_dir = Path(
        os.environ.get("COVID19_TRAIN_DIR", manifest["source"])
    ).expanduser()
    output = (
        project / "outputs"
        / f"full_split_mixed75_5000_gaussian50_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output.mkdir(parents=True, exist_ok=False)

    # Shared disk cache. Random cropping is NOT cached.
    cache_dir = project / "outputs" / "preprocessing_cache_full_split_v2_meta"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def records(cases):
        result = []
        for case in cases:
            image = data_dir / f"{case['case_id']}_ct.nii.gz"
            label = data_dir / f"{case['case_id']}_seg.nii.gz"
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(case["case_id"])
            result.append({
                "image": str(image),
                "label": str(label),
                "case_id": case["case_id"],
                # Include modification times in the cache key.
                "source_mtimes": [
                    image.stat().st_mtime_ns,
                    label.stat().st_mtime_ns,
                ],
            })
        return result

    preprocessing = Compose([
        LoadImaged(
            keys=["image", "label"],
            ensure_channel_first=True,
            dtype=np.float32,
        ),
        Orientationd(
            keys=["image", "label"], axcodes="RAS", labels=None
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys="image",
            a_min=-1000, a_max=400,
            b_min=0, b_max=1, clip=True,
        ),
    ])

    train_data = PersistentDataset(
        data=records(train_cases),
        transform=preprocessing,
        cache_dir=cache_dir,
        hash_transform=pickle_hashing,
        track_meta=True,
        # This cache contains only preprocessing outputs generated locally.
        weights_only=False,
    )
    val_data = PersistentDataset(
        data=records(val_cases),
        transform=preprocessing,
        cache_dir=cache_dir,
        hash_transform=pickle_hashing,
        track_meta=True,
        # This cache contains only preprocessing outputs generated locally.
        weights_only=False,
    )

    sampling = Compose([
        SpatialPadd(
            keys=["image", "label"], spatial_size=patch_size
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=3,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
    ])
    sampling.set_random_state(seed=seed)

    config = {
        "seed": seed,
        "steps": total_steps,
        "learning_rate": learning_rate,
        "patch_size": list(patch_size),
        "spacing_mm": [1.5, 1.5, 2.0],
        "intensity_window_hu": [-1000, 400],
        "validation_threshold": 0.5,
        "inference_overlap": 0.5,
        "inference_blend_mode": "gaussian",
        "initialization": "fresh weights, seed 42",
        "training_cases": [c["case_id"] for c in train_cases],
        "validation_cases": [c["case_id"] for c in val_cases],
        "sampling": "75% foreground / 25% background crop-centre probability",
        "foreground_center_probability": 0.75,
        "background_center_probability": 0.25,
        "background_center_image_threshold": 0,
        "augmentation": False,
        "validation_interval": validation_interval,
        "cache_dir": str(cache_dir),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))

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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    def tensor(value):
        return value.as_tensor() if hasattr(value, "as_tensor") else value

    def check_geometry(data):
        if data["image"].shape != data["label"].shape:
            raise RuntimeError(f"Shape mismatch: {data['case_id']}")
        if not torch.allclose(
            data["image"].affine, data["label"].affine,
            atol=1e-3, rtol=0,
        ):
            raise RuntimeError(f"Geometry mismatch: {data['case_id']}")

    def portable(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(portable(v) for v in value)
        return value

    def numpy_rng_state(state):
        # Plain Python values keep checkpoints compatible with weights_only=True.
        return {
            "algorithm": state[0], "keys": state[1].tolist(),
            "position": int(state[2]), "has_gauss": int(state[3]),
            "cached_gaussian": float(state[4]),
        }

    def sampler_states():
        states = {}
        def visit(obj, path):
            if hasattr(obj, "R"):
                states[path] = numpy_rng_state(obj.R.get_state())
            for index, child in enumerate(getattr(obj, "transforms", ())):
                visit(child, f"{path}.transforms.{index}")
            if hasattr(obj, "cropper"):
                visit(obj.cropper, path + ".cropper")
        visit(sampling, "sampling")
        return states

    def save_checkpoint(name, step, score):
        payload = {
            "step": step,
            "validation_dice": score,
            "config": config,
            "model_state_dict": portable(model.state_dict()),
            "optimizer_state_dict": portable(optimizer.state_dict()),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": numpy_rng_state(np.random.get_state()),
            "torch_rng_state": torch.get_rng_state(),
            "mps_rng_state": torch.mps.get_rng_state().cpu(),
            "case_order_rng_state": rng.bit_generator.state,
            "case_order": list(order),
            "next_case_position": step % len(train_data),
            "sampler_rng_states": sampler_states(),
            "best_validation_dice": best_score,
            "best_step": best_step,
            "history": list(history),
        }
        temporary = output / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output / name)

    validation_file = output / "validation_cases.csv"
    validation_fields = [
        "step", "case_id", "dice", "precision", "recall",
        "predicted_voxels", "gt_voxels", "flags",
    ]
    with validation_file.open("w", newline="") as file:
        csv.writer(file).writerow(validation_fields)

    def validate(step):
        model.eval()
        scores = []
        recalls = []
        started = time.perf_counter()

        with torch.inference_mode():
            for index in range(len(val_data)):
                data = val_data[index]
                check_geometry(data)
                image = tensor(data["image"]).float().unsqueeze(0)
                target = tensor(data["label"]).unsqueeze(0) > 0.5

                print(
                    f"Validation step {step}: "
                    f"{index + 1}/{len(val_data)} {data['case_id']}",
                    flush=True,
                )
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=patch_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=config["inference_overlap"],
                    mode=config["inference_blend_mode"],
                    sw_device=device,
                    device=torch.device("cpu"),
                )
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite validation output.")

                prediction = torch.sigmoid(logits) > 0.5
                intersection = int((prediction & target).sum())
                predicted = int(prediction.sum())
                actual = int(target.sum())
                dice = (
                    2 * intersection / (predicted + actual)
                    if predicted + actual else 1.0
                )
                precision = intersection / predicted if predicted else 0.0
                recall = intersection / actual if actual else None

                flags = []
                if actual:
                    if recall < 0.05:
                        flags.append("RECALL_BELOW_5_PERCENT")
                    if predicted < 0.1 * actual:
                        flags.append("SEVERE_UNDERPREDICTION")
                    if predicted > 10 * actual:
                        flags.append("VOLUME_OVER_10_TIMES_GT")

                row = {
                    "step": step,
                    "case_id": data["case_id"],
                    "dice": dice,
                    "precision": precision,
                    "recall": recall,
                    "predicted_voxels": predicted,
                    "gt_voxels": actual,
                    "flags": ";".join(flags) or "No heuristic flags triggered",
                }
                with validation_file.open("a", newline="") as file:
                    csv.DictWriter(
                        file, fieldnames=validation_fields
                    ).writerow(row)

                scores.append(dice)
                if recall is not None:
                    recalls.append(recall)
                print(
                    f"  Dice={dice:.4f}, precision={precision:.4f}, "
                    f"recall={recall}",
                    flush=True,
                )
                del data, image, target, logits, prediction

        return {
            "step": step,
            "mean_dice": float(np.mean(scores)),
            "median_dice": float(np.median(scores)),
            "mean_recall": float(np.mean(recalls)) if recalls else None,
            "validation_minutes": (time.perf_counter() - started) / 60,
        }

    print("Device:", device)
    print("Training cases:", len(train_data))
    print("Validation cases:", len(val_data))
    print("Output:", output)
    print("Cache:", cache_dir)
    print("Fresh 5000-update run; lr=1e-3; sampling=75/25.")
    print("Validation: Gaussian blending, 50% overlap, every 500 updates.")
    print("Preparing untrained validation baseline...", flush=True)

    started = time.perf_counter()
    history = []

    def record_validation(step):
        result = validate(step)
        history.append(result)
        with (output / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        return result["mean_dice"]

    rng = np.random.default_rng(seed)
    order = []
    best_score = record_validation(0)
    best_step = 0
    save_checkpoint("best_model.pt", 0, best_score)

    recent_losses = []
    train_log = output / "training_steps.csv"
    with train_log.open("w", newline="") as file:
        csv.writer(file).writerow([
            "step", "case_id", "loss", "lesion_voxels"
        ])

    for step in range(1, total_steps + 1):
        position = (step - 1) % len(train_data)
        if position == 0:
            order = rng.permutation(len(train_data)).tolist()

        data = train_data[order[position]]
        check_geometry(data)
        case_id = data["case_id"]
        patch = sampling(data)[0]
        del data

        image = tensor(patch["image"]).unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        label = tensor(patch["label"]).unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        lesion_voxels = int((label > 0.5).sum().item())

        # Background-centred crops may legitimately contain no lesions.
        if not torch.all((label == 0) | (label == 1)).item():
            raise RuntimeError(f"Nonbinary mask: {case_id}")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(image), label)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        recent_losses.append(loss_value)
        with train_log.open("a", newline="") as file:
            csv.writer(file).writerow([
                step, case_id, loss_value, lesion_voxels
            ])
        del patch, image, label, loss

        if step % 50 == 0:
            print(
                f"Step {step}/{total_steps}: "
                f"recent mean loss={np.mean(recent_losses):.4f}",
                flush=True,
            )
            recent_losses.clear()

        if step % validation_interval == 0:
            score = record_validation(step)
            if score > best_score:
                best_score = score
                best_step = step
                save_checkpoint("best_model.pt", step, score)
            save_checkpoint("last_model.pt", step, score)

    summary = {
        "training_cases": len(train_data),
        "validation_cases": len(val_data),
        "updates": total_steps,
        "best_validation_dice": best_score,
        "best_step": best_step,
        "final_validation_dice": history[-1]["mean_dice"],
        "total_minutes": (time.perf_counter() - started) / 60,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nFULL-SPLIT PILOT SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Saved to:", output)

def overfit_one_patch():
    from monai.losses import DiceLoss
    import torch.nn.functional as F

    set_determinism(seed=42)

    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU is unavailable.")

    device = torch.device("mps")
    project = Path(__file__).resolve().parent

    # Reuse the exact patch from the previous test.
    patch_path = (
        project / "outputs"
        / "overfit_patch_20261031_141152_110378"
        / "fixed_patch.pt"
    )
    saved = torch.load(
        patch_path,
        map_location="cpu",
        weights_only=True,
    )

    image = saved["image"].to(device=device, dtype=torch.float32)
    label = saved["label"].to(device=device, dtype=torch.float32)
    target = label > 0.5
    background = ~target

    if image.shape != label.shape:
        raise ValueError("CT and mask shapes differ.")
    if not torch.isfinite(image).all().item():
        raise ValueError("CT contains invalid values.")
    if not torch.all((label == 0) | (label == 1)).item():
        raise ValueError("Mask must contain only 0 and 1.")
    if not target.any().item() or not background.any().item():
        raise ValueError("Patch needs both lesion and background voxels.")

    steps = 2000
    learning_rate = 1e-3

    output = (
        project / "outputs"
        / f"overfit_diagnostics_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output.mkdir(parents=True, exist_ok=False)

    print("Device:", device)
    print("Saved patch:", patch_path)
    print("Case:", saved["case_id"])
    print("Shape:", tuple(image.shape))
    print("Lesion voxels:", int(target.sum().item()))
    print(
        "Lesion fraction:",
        f"{100 * target.float().mean().item():.3f}%",
        flush=True,
    )

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=1,
        norm="INSTANCE",
    ).to(device)

    # Keep the original optimization objective.
    loss_function = DiceCELoss(sigmoid=True)
    dice_component = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    def measure(step):
        model.eval()

        with torch.inference_mode():
            logits = model(image)

            total = loss_function(logits, label)
            dice_loss = dice_component(logits, label)
            bce_loss = F.binary_cross_entropy_with_logits(logits, label)

            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Nonfinite model output.")

            # With our settings, total should equal Dice loss + BCE.
            if not torch.allclose(
                total, dice_loss + bce_loss, atol=1e-5, rtol=1e-5
            ):
                raise RuntimeError("Loss components do not match total loss.")

            probability = torch.sigmoid(logits)
            prediction = probability > 0.5

            tp = int((prediction & target).sum().item())
            n_pred = int(prediction.sum().item())
            n_target = int(target.sum().item())

            lesion_probability = probability[target]
            background_probability = probability[background]

            # Soft Dice uses probabilities instead of a binary threshold.
            soft_dice = (
                (2 * (probability * label).sum() + 1e-5)
                / (probability.sum() + label.sum() + 1e-5)
            )

            row = {
                "step": step,
                "total_loss": float(total.item()),
                "dice_loss": float(dice_loss.item()),
                "bce_loss": float(bce_loss.item()),
                "soft_dice": float(soft_dice.item()),
                "hard_dice": 2 * tp / (n_pred + n_target),
                "precision": tp / n_pred if n_pred else 0.0,
                "recall": tp / n_target,
                "predicted_voxels": n_pred,
                "lesion_mean_probability": float(
                    lesion_probability.mean().item()
                ),
                "background_mean_probability": float(
                    background_probability.mean().item()
                ),
                "lesion_min_probability": float(
                    lesion_probability.min().item()
                ),
                "lesion_max_probability": float(
                    lesion_probability.max().item()
                ),
            }

        return row

    def save_checkpoint(filename, row):
        torch.save({
            "step": row["step"],
            "metrics": row,
            "case_id": saved["case_id"],
            "fixed_patch_path": str(patch_path),
            "seed": 42,
            "learning_rate": learning_rate,
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
        }, output / filename)

    def print_metrics(row):
        print(
            f"Step {row['step']:3d}: "
            f"total={row['total_loss']:.4f}, "
            f"Dice loss={row['dice_loss']:.4f}, "
            f"BCE={row['bce_loss']:.4f}\n"
            f"  hard Dice={row['hard_dice']:.4f}, "
            f"soft Dice={row['soft_dice']:.4f}, "
            f"predicted voxels={row['predicted_voxels']}\n"
            f"  mean probability: "
            f"lesion={row['lesion_mean_probability']:.4f}, "
            f"background={row['background_mean_probability']:.4f}",
            flush=True,
        )

    initial = measure(0)
    history = [initial]
    best = initial.copy()
    save_checkpoint("initial_model.pt", initial)
    save_checkpoint("best_model.pt", best)
    print_metrics(initial)

    start = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(image)
        loss = loss_function(logits, label)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Invalid training loss at step {step}.")

        loss.backward()

        for parameter in model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all().item():
                    raise RuntimeError(
                        f"Invalid gradient at step {step}."
                    )

        optimizer.step()

        if step % 20 == 0 or step == steps:
            row = measure(step)
            history.append(row)
            print_metrics(row)

            if row["hard_dice"] > best["hard_dice"]:
                best = row.copy()
                save_checkpoint("best_model.pt", best)

            with (output / "diagnostics.csv").open(
                "w", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=history[0].keys()
                )
                writer.writeheader()
                writer.writerows(history)

    final = history[-1]
    save_checkpoint("final_model.pt", final)

    summary = {
        "initial": initial,
        "best_by_hard_dice": best,
        "final": final,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    measured_steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for key, name in [
        ("total_loss", "Total"),
        ("dice_loss", "Dice loss"),
        ("bce_loss", "BCE"),
    ]:
        axes[0].plot(
            measured_steps,
            [row[key] for row in history],
            label=name,
        )
    axes[0].set(xlabel="Step", ylabel="Loss")
    axes[0].legend()

    for key, name in [
        ("hard_dice", "Hard Dice at 0.5"),
        ("soft_dice", "Soft Dice"),
    ]:
        axes[1].plot(
            measured_steps,
            [row[key] for row in history],
            label=name,
        )
    axes[1].set(xlabel="Step", ylabel="Dice", ylim=(0, 1))
    axes[1].legend()

    for key, name in [
        ("lesion_mean_probability", "Inside lesions"),
        ("background_mean_probability", "Background"),
    ]:
        axes[2].plot(
            measured_steps,
            [row[key] for row in history],
            label=name,
        )
    axes[2].set(
        xlabel="Step", ylabel="Mean predicted probability", ylim=(0, 1)
    )
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output / "diagnostic_curves.png", dpi=150)

    print("\nDIAGNOSTIC SUMMARY")
    print("Best measured hard Dice:", round(best["hard_dice"], 4))
    print("Best measured step:", best["step"])
    print("Final hard Dice:", round(final["hard_dice"], 4))
    print("Final soft Dice:", round(final["soft_dice"], 4))
    print("Final Dice loss:", round(final["dice_loss"], 4))
    print("Final BCE:", round(final["bce_loss"], 4))
    print(
        "Final lesion mean probability:",
        round(final["lesion_mean_probability"], 4),
    )
    print(
        "Final background mean probability:",
        round(final["background_mean_probability"], 4),
    )
    print("Final predicted voxels:", final["predicted_voxels"])
    print("Saved to:", output)

    plt.show()

def inspect_saved_experiment(run_dir, subset="training", checkpoint_name="best_model.pt", case_ids=None, overlap=0.25, blend_mode="constant", output_tag=None, show_plots=True):
    if subset not in ("training", "validation"):
        raise ValueError("subset must be 'training' or 'validation'")
    if not 0 <= overlap < 1 or blend_mode not in ("constant", "gaussian"):
        raise ValueError("Invalid inference settings")
    import nibabel as nib

    run_dir = Path(run_dir)
    checkpoint = torch.load(
        run_dir / checkpoint_name,
        map_location="cpu",
        weights_only=True,
    )
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # The random-positive experiment records its original
        # preprocessing settings and case lists through source_run.
        experiment_config = json.loads(
            (run_dir / "config.json").read_text()
        )
        source_run = Path(experiment_config["source_run"])
        config = json.loads(
            (source_run / "config.json").read_text()
        )

        # Confirm that the patch dimensions have not changed.
        if tuple(experiment_config["patch_size"]) != tuple(
                config["patch_size"]
        ):
            raise ValueError("Experiment patch sizes differ.")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    project = Path(__file__).resolve().parent
    manifest = json.loads((project / "splits.json").read_text())
    folder = Path(
        os.environ.get("COVID19_TRAIN_DIR", manifest["source"])
    ).expanduser()

    # Each checkpoint gets its own cache to avoid mixing model outputs.
    if "epoch" in checkpoint:
        checkpoint_unit = "epoch"
        checkpoint_number = checkpoint["epoch"]
    else:
        checkpoint_unit = "step"
        checkpoint_number = checkpoint["step"]

    inspection_dir = run_dir / (
        f"{subset}_inspection_{checkpoint_unit}_{checkpoint_number}"
    )
    if output_tag is not None:
        inspection_dir = inspection_dir.with_name(
            inspection_dir.name + "_" + output_tag
        )
    inspection_dir.mkdir(exist_ok=True)

    preprocessing = Compose([
        LoadImaged(
            keys=["image", "label"],
            ensure_channel_first=True,
            dtype=np.float32,
        ),
        Orientationd(
            keys=["image", "label"],
            axcodes="RAS",
            labels=None,
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=tuple(config["spacing_mm"]),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys="image",
            a_min=config["intensity_window_hu"][0],
            a_max=config["intensity_window_hu"][1],
            b_min=0,
            b_max=1,
            clip=True,
        ),
    ])

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=1,
        norm="INSTANCE",
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Inspecting checkpoint {checkpoint_unit}:", checkpoint_number)
    print("Device:", device)
    results = []

    available_cases = config[f"{subset}_cases"]
    if case_ids is None:
        selected_cases = available_cases
    else:
        unknown = set(case_ids) - set(available_cases)
        if unknown:
            raise ValueError(f"Cases not in the selected subset: {unknown}")
        selected_cases = case_ids

    for case_id in selected_cases:
        print(f"\nInspecting {case_id}...", flush=True)

        data = preprocessing({
            "image": str(folder / f"{case_id}_ct.nii.gz"),
            "label": str(folder / f"{case_id}_seg.nii.gz"),
        })

        if data["image"].shape != data["label"].shape:
            raise ValueError("CT and ground-truth dimensions differ.")
        if not torch.allclose(
            data["image"].affine,
            data["label"].affine,
            atol=1e-3,
            rtol=0,
        ):
            raise ValueError("CT and ground-truth alignment differs.")

        image = data["image"].as_tensor().unsqueeze(0)
        ct = image[0, 0].numpy()
        gt = data["label"].as_tensor()[0].numpy() > 0
        affine = data["image"].affine.cpu().numpy()

        prediction_path = inspection_dir / f"{case_id}_prediction.nii.gz"

        if prediction_path.exists():
            saved = nib.load(prediction_path)
            if saved.shape != ct.shape or not np.allclose(
                saved.affine, affine, atol=1e-3, rtol=0
            ):
                raise ValueError("Saved prediction geometry differs from CT.")
            pred = np.asanyarray(saved.dataobj) > 0
            print("Reusing saved prediction.")
        else:
            with torch.inference_mode():
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=tuple(config["patch_size"]),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                    mode=blend_mode,
                    sw_device=device,
                    device=torch.device("cpu"),
                )

            if not torch.isfinite(logits).all():
                raise RuntimeError("Prediction contains invalid values.")

            pred = (
                torch.sigmoid(logits)[0, 0].numpy()
                > config["validation_threshold"]
            )
            nib.save(
                nib.Nifti1Image(pred.astype(np.uint8), affine),
                prediction_path,
            )
            del logits
            print("Generated and saved prediction.")

        # Export matching resampled CT and ground truth for Slicer.
        nib.save(
            nib.Nifti1Image(ct.astype(np.float32), affine),
            inspection_dir / f"{case_id}_normalized_ct.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(gt.astype(np.uint8), affine),
            inspection_dir / f"{case_id}_ground_truth.nii.gz",
        )

        tp = int(np.count_nonzero(pred & gt))
        n_pred = int(np.count_nonzero(pred))
        n_gt = int(np.count_nonzero(gt))
        denominator = n_pred + n_gt

        dice = 1.0 if denominator == 0 else 2 * tp / denominator
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gt if n_gt else 0.0
        fraction = n_pred / pred.size

        flags = []
        if n_pred == 0:
            flags.append("NO LESIONS PREDICTED")
        if fraction > 0.90:
            flags.append("MORE THAN 90% OF VOLUME LABELED")
        if n_gt and n_pred > 10 * n_gt:
            flags.append("PREDICTED VOLUME >10 TIMES GROUND TRUTH")

        if n_gt and recall < 0.05:
            flags.append("RECALL BELOW 5%")
        if n_gt and n_pred < 0.1 * n_gt:
            flags.append("SEVERE UNDERPREDICTION")
        if n_pred and precision < 0.1:
            flags.append("PRECISION BELOW 10%")

        print(f"Dice: {dice:.4f}")
        print(f"Precision: {precision:.4f}; recall: {recall:.4f}")
        print(f"Ground-truth voxels: {n_gt}; predicted voxels: {n_pred}")
        print(f"Predicted foreground: {100 * fraction:.2f}%")
        print("Geometry: matches the resampled CT")
        print("Failure flags:", "; ".join(flags) if flags else "No heuristic flags triggered")

        results.append({
            "case_id": case_id,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "ground_truth_voxels": n_gt,
            "predicted_voxels": n_pred,
            "foreground_fraction": fraction,
            "flags": flags,
        })

        # Show where ground truth is largest and where prediction is largest.
        slices = [
            int(np.argmax(gt.sum(axis=(0, 1)))),
            int(np.argmax(pred.sum(axis=(0, 1)))) if n_pred else
            int(np.argmax(gt.sum(axis=(0, 1)))),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(13, 9))

        for row, z in enumerate(slices):
            for ax in axes[row]:
                ax.imshow(
                    ct[:, :, z].T,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    origin="lower",
                )
                ax.axis("off")

            axes[row, 0].set_title(f"CT — axial slice {z}")

            for column, mask, title, color in [
                (1, gt, "Ground truth", "autumn"),
                (2, pred, "AI prediction", "winter"),
            ]:
                section = mask[:, :, z].T
                axes[row, column].imshow(
                    np.ma.masked_where(~section, section),
                    cmap=color,
                    vmin=0,
                    vmax=1,
                    alpha=0.45,
                    origin="lower",
                )
                axes[row, column].set_title(title)

        fig.suptitle(
            f"{case_id} | Dice={dice:.4f}\n"
            "Top: largest ground-truth area; bottom: largest prediction area"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(inspection_dir / f"{case_id}_comparison.png", dpi=150)
        if show_plots:
            plt.show()
        plt.close(fig)

    (inspection_dir / "metrics.json").write_text(
        json.dumps(results, indent=2)
    )
    print("\nInspection files saved to:", inspection_dir)

    return inspection_dir, results

def main():
    set_determinism(seed=SEED)
    order_rng = random.Random(SEED)

    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU is unavailable. Check your interpreter.")

    device = torch.device("mps")
    project = Path(__file__).resolve().parent
    split_path = project / "splits.json"
    manifest = json.loads(split_path.read_text())

    training_cases = manifest["train"][:5]
    validation_cases = manifest["validation"][:2]

    if len(training_cases) != 5 or len(validation_cases) != 2:
        raise ValueError("Expected at least 5 training and 2 validation cases.")

    train_groups = {c["group_id"] for c in training_cases}
    val_groups = {c["group_id"] for c in validation_cases}
    if train_groups & val_groups:
        raise ValueError("Training and validation groups overlap.")

    data_dir = Path(
        os.environ.get("COVID19_TRAIN_DIR", manifest["source"])
    ).expanduser()

    output = (
        project
        / "outputs"
        / f"small_experiment_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output.mkdir(parents=True, exist_ok=False)

    print("Device:", device)
    print("Training cases:", [c["case_id"] for c in training_cases])
    print("Validation cases:", [c["case_id"] for c in validation_cases])
    print("Output folder:", output, flush=True)

    config = {
        "seed": SEED,
        "epochs": EPOCHS,
        "patch_size": PATCH_SIZE,
        "patches_per_case": PATCHES_PER_CASE,
        "learning_rate": LEARNING_RATE,
        "training_cases": [c["case_id"] for c in training_cases],
        "validation_cases": [c["case_id"] for c in validation_cases],
        "spacing_mm": [1.5, 1.5, 2.0],
        "intensity_window_hu": [-1000, 400],
        "validation_threshold": 0.5,
        "metric_space": "resampled volume, not original CT resolution",
        "torch_version": str(torch.__version__),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))

    keys = ["image", "label"]

    # The same deterministic preprocessing is used for both subsets.
    preprocessing = Compose([
        LoadImaged(
            keys=keys,
            ensure_channel_first=True,
            dtype=np.float32,
        ),
        Orientationd(keys=keys, axcodes="RAS", labels=None),
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
    ])

    # Random cropping and augmentation apply only during training.
    training_sampling = Compose([
        SpatialPadd(keys=keys, spatial_size=PATCH_SIZE),
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=1,
            neg=1,
            num_samples=PATCHES_PER_CASE,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
    ])

    def load_case(case):
        pair = {
            "image": str(data_dir / f"{case['case_id']}_ct.nii.gz"),
            "label": str(data_dir / f"{case['case_id']}_seg.nii.gz"),
        }
        for path in pair.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)

        data = preprocessing(pair)

        if data["image"].shape != data["label"].shape:
            raise ValueError(f"Shape mismatch: {case['case_id']}")
        if not torch.allclose(
            data["image"].affine,
            data["label"].affine,
            atol=1e-3,
            rtol=0,
        ):
            raise ValueError(f"Alignment mismatch: {case['case_id']}")
        if not torch.isfinite(data["image"]).all():
            raise ValueError(f"Invalid CT values: {case['case_id']}")
        if not torch.all((data["label"] == 0) | (data["label"] == 1)):
            raise ValueError(f"Nonbinary mask: {case['case_id']}")

        return data

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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    # Cache only the two validation cases on CPU.
    print("\nPreparing validation volumes...", flush=True)
    validation_data = [
        (case["case_id"], load_case(case))
        for case in validation_cases
    ]

    def validate(epoch):
        model.eval()
        scores = []

        with torch.inference_mode():
            for case_id, data in validation_data:
                print(f"  Validating {case_id}...", flush=True)

                # Keep complete volumes and stitched outputs on CPU.
                # Individual windows are evaluated on the Apple GPU.
                image = data["image"].as_tensor().unsqueeze(0)
                target = (
                    data["label"].as_tensor().unsqueeze(0) > 0
                )

                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                    mode="constant",
                    sw_device=device,
                    device=torch.device("cpu"),
                )

                if not torch.isfinite(logits).all():
                    raise RuntimeError("Validation produced invalid values.")

                prediction = torch.sigmoid(logits) > 0.5

                intersection = int((prediction & target).sum().item())
                predicted_voxels = int(prediction.sum().item())
                target_voxels = int(target.sum().item())
                denominator = predicted_voxels + target_voxels

                # Both empty: Dice=1. Only one empty: Dice=0.
                dice = (
                    1.0 if denominator == 0
                    else 2.0 * intersection / denominator
                )
                scores.append(dice)

                with (output / "validation_cases.csv").open(
                    "a", newline=""
                ) as file:
                    csv.writer(file).writerow([
                        epoch, case_id, dice,
                        predicted_voxels, target_voxels,
                    ])

                print(f"  {case_id}: Dice={dice:.4f}", flush=True)
                del logits, prediction

        return float(np.mean(scores))

    with (output / "validation_cases.csv").open("w", newline="") as file:
        csv.writer(file).writerow([
            "epoch", "case_id", "dice",
            "predicted_voxels", "target_voxels",
        ])

    def save_checkpoint(filename, epoch, score):
        # Save CPU weights for portability.
        torch.save({
            "epoch": epoch,
            "validation_dice": score,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "config": config,
        }, output / filename)

    print("\nEvaluating the untrained model...", flush=True)
    baseline_dice = validate(epoch=0)
    best_dice = baseline_dice
    best_epoch = 0
    save_checkpoint("best_model.pt", 0, best_dice)
    print(f"Untrained mean validation Dice: {baseline_dice:.4f}")

    history = []
    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        shuffled_cases = training_cases.copy()
        order_rng.shuffle(shuffled_cases)

        print(f"\nEpoch {epoch}/{EPOCHS}", flush=True)

        for case in shuffled_cases:
            # Load one full training case at a time to limit RAM usage.
            data = load_case(case)
            patches = training_sampling(data)
            del data

            for patch in patches:
                image = (
                    patch["image"].as_tensor().unsqueeze(0)
                    .to(device=device, dtype=torch.float32)
                )
                label = (
                    patch["label"].as_tensor().unsqueeze(0)
                    .to(device=device, dtype=torch.float32)
                )

                optimizer.zero_grad(set_to_none=True)
                logits = model(image)
                loss = loss_function(logits, label)

                if not torch.isfinite(loss).item():
                    raise RuntimeError("Training loss became invalid.")

                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            print(
                f"  Trained on {case['case_id']}",
                flush=True,
            )
            del patches, image, label, logits, loss

        mean_loss = float(np.mean(losses))
        mean_dice = validate(epoch)

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_epoch = epoch
            save_checkpoint("best_model.pt", epoch, mean_dice)

        history.append({
            "epoch": epoch,
            "training_loss": mean_loss,
            "validation_dice": mean_dice,
        })

        with (output / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=history[0].keys()
            )
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch}: loss={mean_loss:.4f}, "
            f"validation Dice={mean_dice:.4f}",
            flush=True,
        )

    save_checkpoint("last_model.pt", EPOCHS, history[-1]["validation_dice"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]

    axes[0].plot(
        epochs,
        [row["training_loss"] for row in history],
        marker="o",
    )
    axes[0].set(
        xlabel="Epoch", ylabel="Training loss",
        title="Random training patches",
    )

    axes[1].plot(
        [0] + epochs,
        [baseline_dice] + [row["validation_dice"] for row in history],
        marker="o",
    )
    axes[1].set(
        xlabel="Epoch", ylabel="Mean Dice",
        title="Two complete validation volumes",
        ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)

    print("\nEXPERIMENT SUMMARY")
    print("Training cases:", len(training_cases))
    print("Validation cases:", len(validation_cases))
    print("Training steps:", EPOCHS * 5 * PATCHES_PER_CASE)
    print("Untrained validation Dice:", round(baseline_dice, 4))
    print("Best validation Dice:", round(best_dice, 4))
    print("Best epoch:", best_epoch)
    print("Total training/validation minutes:",
          round((time.perf_counter() - start) / 60, 2))
    print("Results saved to:", output)
    print("SMALL EXPERIMENT COMPLETED")
    print("\nInspecting the best validation checkpoint on training cases...")
    inspect_saved_experiment(output, subset="training")

    print("\nInspecting the best validation checkpoint on validation cases...")
    inspect_saved_experiment(output, subset="validation")

def diagnose_patch_vs_volume():
    import csv
    import json
    from pathlib import Path

    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet
    from monai.transforms import (
        Compose,
        LoadImaged,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
        SpatialPadd,
    )

    project = Path(__file__).resolve().parent
    run_dir = (
        project / "outputs"
        / "small_experiment_20261031_150123_685153"
    )

    checkpoint = torch.load(
        run_dir / "last_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]
    manifest = json.loads((project / "splits.json").read_text())

    import os

    data_dir = Path(
        os.environ.get("COVID19_TRAIN_DIR", manifest["source"])
    ).expanduser()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    patch_size = tuple(config["patch_size"])
    threshold = float(config["validation_threshold"])
    rng = np.random.default_rng(42)

    output = run_dir / "patch_vs_volume_epoch_100"
    output.mkdir(exist_ok=True)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=1,
        norm="INSTANCE",
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    keys = ["image", "label"]
    preprocessing = Compose([
        LoadImaged(
            keys=keys,
            ensure_channel_first=True,
            dtype=np.float32,
        ),
        Orientationd(keys=keys, axcodes="RAS", labels=None),
        Spacingd(
            keys=keys,
            pixdim=tuple(config["spacing_mm"]),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys="image",
            a_min=config["intensity_window_hu"][0],
            a_max=config["intensity_window_hu"][1],
            b_min=0,
            b_max=1,
            clip=True,
        ),
        SpatialPadd(keys=keys, spatial_size=patch_size),
    ])

    def metrics(probability, target):
        probability = probability.detach().cpu().float()
        target = target.detach().cpu().bool()

        if not torch.isfinite(probability).all():
            raise RuntimeError("Non-finite probabilities.")

        prediction = probability > threshold
        intersection = int((prediction & target).sum())
        predicted = int(prediction.sum())
        actual = int(target.sum())

        return {
            "dice": (
                2 * intersection / (predicted + actual)
                if predicted + actual else 1.0
            ),
            "soft_dice": float(
                (2 * (probability * target).sum() + 1e-5)
                / (probability.sum() + target.sum() + 1e-5)
            ),
            "precision": intersection / predicted if predicted else 0.0,
            "recall": intersection / actual if actual else 0.0,
            "predicted_voxels": predicted,
            "gt_voxels": actual,
            "lesion_mean_probability": (
                float(probability[target].mean()) if actual else None
            ),
        }

    rows = []
    csv_path = output / "patch_comparison.csv"

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Device:", device)
    print("No training will run.", flush=True)

    with torch.inference_mode():
        for case_id in config["training_cases"]:
            print(f"\nLoading {case_id}...", flush=True)

            data = preprocessing({
                "image": str(data_dir / f"{case_id}_ct.nii.gz"),
                "label": str(data_dir / f"{case_id}_seg.nii.gz"),
            })

            if data["image"].shape != data["label"].shape:
                raise RuntimeError("CT/mask shape mismatch.")
            if not torch.allclose(
                data["image"].affine,
                data["label"].affine,
                atol=1e-3,
                rtol=0,
            ):
                raise RuntimeError("CT/mask affine mismatch.")

            # Complete volumes stay on CPU.
            image = data["image"].as_tensor().float().unsqueeze(0)
            target = data["label"].as_tensor().unsqueeze(0) > 0.5
            del data

            foreground = torch.nonzero(target[0, 0], as_tuple=False)
            if len(foreground) == 0:
                raise RuntimeError(f"{case_id} has no lesion voxels.")

            print("Running full-volume inference...", flush=True)
            volume_logits = sliding_window_inference(
                inputs=image,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
                mode="constant",
                sw_device=device,
                device=torch.device("cpu"),
            )
            volume_probability = torch.sigmoid(volume_logits)
            del volume_logits

            full_metrics = metrics(volume_probability, target)
            print(
                f"Full-volume Dice: {full_metrics['dice']:.4f}; "
                f"recall: {full_metrics['recall']:.4f}"
            )

            # Reproducible foreground-centred crops.
            # Crops may overlap; these are diagnostic samples.
            chosen = rng.choice(
                len(foreground),
                size=min(3, len(foreground)),
                replace=False,
            )

            for patch_number, index in enumerate(chosen, start=1):
                center = foreground[int(index)].tolist()
                starts = [
                    max(0, min(
                        int(c) - width // 2,
                        image.shape[axis + 2] - width,
                    ))
                    for axis, (c, width) in enumerate(
                        zip(center, patch_size)
                    )
                ]
                slices = tuple(
                    slice(start, start + width)
                    for start, width in zip(starts, patch_size)
                )
                selection = (slice(None), slice(None), *slices)

                patch = image[selection].contiguous()
                patch_target = target[selection]

                direct_logits = model(patch.to(device)).cpu()

                # With exactly one window, this should reproduce
                # direct model inference on the identical input.
                single_logits = sliding_window_inference(
                    inputs=patch,
                    roi_size=patch_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.25,
                    mode="constant",
                    sw_device=device,
                    device=torch.device("cpu"),
                )

                max_difference = float(
                    (direct_logits - single_logits).abs().max()
                )
                agrees = torch.allclose(
                    direct_logits,
                    single_logits,
                    atol=1e-4,
                    rtol=1e-4,
                )

                direct_probability = torch.sigmoid(direct_logits)
                contextual_probability = volume_probability[selection]

                direct = metrics(direct_probability, patch_target)
                contextual = metrics(
                    contextual_probability, patch_target
                )

                row = {
                    "case_id": case_id,
                    "patch": patch_number,
                    "start_x": starts[0],
                    "start_y": starts[1],
                    "start_z": starts[2],
                    "lesion_fraction": float(
                        patch_target.float().mean()
                    ),
                    "single_window_agrees": bool(agrees),
                    "single_window_max_logit_difference": max_difference,
                }
                row.update({
                    f"direct_{key}": value
                    for key, value in direct.items()
                })
                row.update({
                    f"volume_region_{key}": value
                    for key, value in contextual.items()
                })
                rows.append(row)

                print(
                    f"  Patch {patch_number}: "
                    f"direct Dice={direct['dice']:.4f}, "
                    f"volume-region Dice={contextual['dice']:.4f}\n"
                    f"    Direct recall={direct['recall']:.4f}, "
                    f"lesion probability="
                    f"{direct['lesion_mean_probability']:.4f}, "
                    f"GT voxels={direct['gt_voxels']}\n"
                    f"    Single-window agreement: {agrees}; "
                    f"max logit difference={max_difference:.2e}",
                    flush=True,
                )

                # Save incremental results in case execution stops.
                with csv_path.open("w", newline="") as file:
                    writer = csv.DictWriter(
                        file, fieldnames=list(rows[0])
                    )
                    writer.writeheader()
                    writer.writerows(rows)

                if not agrees:
                    raise RuntimeError(
                        "Direct and single-window inference disagree. "
                        "Results saved; investigate before continuing."
                    )

                del direct_logits, single_logits, direct_probability

            del image, target, foreground, volume_probability

    print("\nPATCH-VERSUS-VOLUME SUMMARY")
    print("Patches inspected:", len(rows))
    print(
        "Mean direct patch Dice:",
        round(float(np.mean([r["direct_dice"] for r in rows])), 4),
    )
    print(
        "Mean corresponding volume-region Dice:",
        round(float(np.mean([
            r["volume_region_dice"] for r in rows
        ])), 4),
    )
    print("Results saved to:", csv_path)
    print(
        "These are lesion-centred diagnostic patches, "
        "not an unbiased validation score."
    )

def overfit_fixed_patch_collection():
    import csv
    import json
    import os
    import time
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import torch
    from monai.losses import DiceCELoss
    from monai.networks.nets import UNet
    from monai.transforms import (
        Compose,
        LoadImaged,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
        SpatialPadd,
    )
    from monai.utils import set_determinism
    from monai.transforms import RandCropByPosNegLabeld
    from monai.data import MetaTensor

    seed = 42
    steps = 2000
    evaluate_every = 200
    learning_rate = 1e-3
    set_determinism(seed=seed)

    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU unavailable. Check the interpreter.")

    device = torch.device("mps")
    project = Path(__file__).resolve().parent
    previous_run = (
        project / "outputs"
        / "small_experiment_20261031_150123_685153"
    )
    config = json.loads((previous_run / "config.json").read_text())
    manifest = json.loads((project / "splits.json").read_text())
    data_dir = Path(
        os.environ.get("COVID19_TRAIN_DIR", manifest["source"])
    ).expanduser()
    patch_size = tuple(config["patch_size"])

    # Reuse the exact crop locations from the completed diagnostic.
    comparison_file = (
        previous_run / "patch_vs_volume_epoch_100"
        / "patch_comparison.csv"
    )
    with comparison_file.open(newline="") as file:
        comparison_rows = list(csv.DictReader(file))

    output = (
        project / "outputs"
        / f"random_positive_patches_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output.mkdir(parents=True, exist_ok=False)

    preprocessing = Compose([
        LoadImaged(
            keys=["image", "label"],
            ensure_channel_first=True,
            dtype=np.float32,
        ),
        Orientationd(
            keys=["image", "label"], axcodes="RAS", labels=None
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=tuple(config["spacing_mm"]),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys="image",
            a_min=config["intensity_window_hu"][0],
            a_max=config["intensity_window_hu"][1],
            b_min=0,
            b_max=1,
            clip=True,
        ),
        SpatialPadd(
            keys=["image", "label"], spatial_size=patch_size
        ),
    ])
    # Cache preprocessed training volumes on CPU.
    # Evaluation patches remain fixed and are not explicitly fed to training.
    training_volumes = {}

    positive_sampler = RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=patch_size,
        pos=1,
        neg=0,
        num_samples=1,
    )
    positive_sampler.set_random_state(seed=seed)

    patches = []
    for case_id in config["training_cases"]:
        selected = sorted(
            [r for r in comparison_rows if r["case_id"] == case_id],
            key=lambda r: int(r["patch"]),
        )[:2]
        if len(selected) != 2:
            raise RuntimeError(f"Expected two crop locations for {case_id}.")

        print(f"Caching patches from {case_id}...", flush=True)
        data = preprocessing({
            "image": str(data_dir / f"{case_id}_ct.nii.gz"),
            "label": str(data_dir / f"{case_id}_seg.nii.gz"),
        })

        if data["image"].shape != data["label"].shape:
            raise RuntimeError("CT/mask dimensions differ.")
        if not torch.allclose(
            data["image"].affine, data["label"].affine,
            atol=1e-3, rtol=0,
        ):
            raise RuntimeError("CT/mask geometry differs.")

        for row in selected:
            starts = [
                int(row["start_x"]),
                int(row["start_y"]),
                int(row["start_z"]),
            ]
            slices = tuple(
                slice(start, start + width)
                for start, width in zip(starts, patch_size)
            )
            selection = (slice(None), *slices)

            # clone() prevents retaining the entire source volume.
            image = (
                data["image"].as_tensor()[selection]
                .unsqueeze(0).float().clone().contiguous()
            )
            label = (
                data["label"].as_tensor()[selection]
                .unsqueeze(0).float().clone().contiguous()
            )

            expected_shape = (1, 1, *patch_size)
            if tuple(image.shape) != expected_shape:
                raise RuntimeError("Unexpected patch dimensions.")
            if image.shape != label.shape:
                raise RuntimeError("Patch dimensions differ.")
            if not torch.isfinite(image).all():
                raise RuntimeError("Non-finite CT values.")
            if not torch.all((label == 0) | (label == 1)):
                raise RuntimeError("Mask is not binary.")
            if not label.any():
                raise RuntimeError("Expected a lesion-containing patch.")

            patches.append({
                "id": f"{case_id}_patch_{row['patch']}",
                "image": image,
                "label": label,
                "starts": starts,
            })
            training_volumes[case_id] = {
                "image": data["image"],
                "label": data["label"],
            }
        del data

    if len(patches) != 10:
        raise RuntimeError(f"Expected 10 patches, got {len(patches)}.")

    # Preserve the exact inputs for reproducibility.
    torch.save(patches, output / "fixed_patches.pt")
    (output / "config.json").write_text(json.dumps({
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "evaluate_every": evaluate_every,
        "patch_size": patch_size,
        "source_run": str(previous_run),
        "training_cases": config["training_cases"],
        "loss": "DiceCELoss(sigmoid=True)",
        "augmentation": False,
        "training_sampling": "new foreground-centred crop each update",
        "evaluation": "10 fixed patches from the same training cases",
        "foreground_center_probability": 1.0,
        "threshold": 0.5,
    }, indent=2))

    # Ten small patches fit comfortably compared with full CT volumes.
    # Keep them on the GPU to avoid repeated preprocessing/transfers.
    gpu_patches = [
        (
            p["image"].to(device),
            p["label"].to(device),
        )
        for p in patches
    ]

    # Fresh initialization: do not load the previous model.
    set_determinism(seed=seed)
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    history = []
    evaluations = []
    best_mean_dice = -1.0
    best_step = 0

    def save_checkpoint(filename, step, mean_dice):
        torch.save({
            "step": step,
            "mean_patch_dice": mean_dice,
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "patch_file": str(output / "fixed_patches.pt"),
        }, output / filename)

    def evaluate(step):
        nonlocal best_mean_dice, best_step
        model.eval()
        current = []

        with torch.inference_mode():
            for patch, (image, label) in zip(patches, gpu_patches):
                logits = model(image)
                probability = torch.sigmoid(logits)
                if not torch.isfinite(probability).all():
                    raise RuntimeError("Non-finite predictions.")

                target = label > 0.5
                prediction = probability > 0.5
                intersection = int((prediction & target).sum().item())
                actual = int(target.sum().item())
                predicted = int(prediction.sum().item())

                row = {
                    "step": step,
                    "patch_id": patch["id"],
                    "loss": float(loss_function(logits, label).item()),
                    "dice": 2 * intersection / (actual + predicted),
                    "soft_dice": float((
                        (2 * (probability * label).sum() + 1e-5)
                        / (probability.sum() + label.sum() + 1e-5)
                    ).item()),
                    "precision": (
                        intersection / predicted if predicted else 0.0
                    ),
                    "recall": intersection / actual,
                    "gt_voxels": actual,
                    "predicted_voxels": predicted,
                    "lesion_probability": float(
                        probability[target].mean().item()
                    ),
                    "background_probability": float(
                        probability[~target].mean().item()
                    ),
                }
                current.append(row)
                evaluations.append(row)

        mean_dice = float(np.mean([r["dice"] for r in current]))
        minimum_dice = min(r["dice"] for r in current)
        history.append({
            "step": step,
            "mean_loss": float(np.mean([r["loss"] for r in current])),
            "mean_dice": mean_dice,
            "minimum_dice": minimum_dice,
            "mean_recall": float(np.mean([r["recall"] for r in current])),
        })

        print(
            f"\nStep {step}/{steps}: mean Dice={mean_dice:.4f}; "
            f"minimum Dice={minimum_dice:.4f}",
            flush=True,
        )
        for row in current:
            print(
                f"  {row['patch_id']}: Dice={row['dice']:.4f}, "
                f"recall={row['recall']:.4f}, "
                f"predicted/GT={row['predicted_voxels']}/"
                f"{row['gt_voxels']}",
                flush=True,
            )

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            best_step = step
            save_checkpoint("best_model.pt", step, mean_dice)

        for filename, records in (
            ("history.csv", history),
            ("patch_metrics.csv", evaluations),
        ):
            with (output / filename).open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)

        return mean_dice

    def plain_tensor(value):
        if isinstance(value, MetaTensor):
            return value.as_tensor()
        return value

    case_ids = list(config["training_cases"])
    sampling_log = []
    order_rng = np.random.default_rng(seed)
    order = []

    cached_bytes = sum(
        plain_tensor(value).numel()
        * plain_tensor(value).element_size()
        for pair in training_volumes.values()
        for value in pair.values()
    )

    print("\nDevice:", device)
    print("Training cases:", len(case_ids))
    print("Fixed evaluation patches:", len(patches))
    print("Training: new lesion-centred crop each update")
    print("Augmentation: disabled")
    print("Learning rate:", learning_rate)
    print("Total updates:", steps)
    print(
        "Cached volume tensor memory:",
        f"{cached_bytes / 1024 ** 3:.2f} GiB",
    )
    print("Output:", output, flush=True)

    start_time = time.perf_counter()
    initial_dice = evaluate(0)
    final_dice = initial_dice

    for step in range(1, steps + 1):
        # Visit every case once per cycle, in shuffled order.
        position = (step - 1) % len(case_ids)
        if position == 0:
            order = order_rng.permutation(len(case_ids)).tolist()

        case_id = case_ids[order[position]]

        # Only crop sampling is random. Cached CT/mask preprocessing
        # is identical to the fixed-patch experiment.
        sampled = positive_sampler(training_volumes[case_id])[0]

        image = (
            plain_tensor(sampled["image"])
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        label = (
            plain_tensor(sampled["label"])
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

        expected_shape = (1, 1, *patch_size)
        if tuple(image.shape) != expected_shape:
            raise RuntimeError(f"Unexpected input shape: {image.shape}")
        if image.shape != label.shape:
            raise RuntimeError("Image and label dimensions differ.")

        lesion_voxels = int((label > 0.5).sum().item())
        if lesion_voxels == 0:
            raise RuntimeError(
                f"Positive sampling returned an empty mask: {case_id}"
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = loss_function(logits, label)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss.")

        loss.backward()
        optimizer.step()

        sampling_log.append({
            "step": step,
            "case_id": case_id,
            "loss": float(loss.item()),
            "lesion_voxels": lesion_voxels,
            "lesion_fraction": lesion_voxels / label.numel(),
        })

        del sampled, image, label, logits, loss

        if step % evaluate_every == 0:
            recent = sampling_log[-evaluate_every:]
            print(
                "\nRecent training loss:",
                round(float(np.mean([r["loss"] for r in recent])), 4),
                "| Mean sampled lesion fraction:",
                f"{np.mean([r['lesion_fraction'] for r in recent]):.4%}",
                flush=True,
            )

            # evaluate() uses the same fixed 10 patches as before.
            final_dice = evaluate(step)

            with (output / "sampling_log.csv").open(
                    "w", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=sampling_log[0].keys()
                )
                writer.writeheader()
                writer.writerows(sampling_log)

    summary = {
        "initial_mean_dice": initial_dice,
        "best_mean_dice": best_mean_dice,
        "best_step": best_step,
        "final_mean_dice": final_dice,
        "final_minimum_dice": history[-1]["minimum_dice"],
        "elapsed_minutes": (time.perf_counter() - start_time) / 60,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nRANDOM-POSITIVE-PATCH SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Saved to:", output)
    print("Memorization diagnostic only—not validation performance.")


def compare_inference_settings():
    """Three-case diagnostic only; no training or test-set access."""
    import nibabel as nib

    run_dir = (
        Path(__file__).resolve().parent / "outputs"
        / "full_split_mixed75_20261031_170107_903324"
    )
    case_ids = [
        "volume-covid19-A-0165",
        "volume-covid19-A-0011",
        "volume-covid19-A-0354",
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = run_dir / f"inference_comparison_{stamp}"
    output.mkdir(exist_ok=False)
    variants = [
        ("constant25", 0.25, "constant"),
        ("gaussian50", 0.50, "gaussian"),
    ]
    (output / "settings.json").write_text(json.dumps({
        "checkpoint": "best_model.pt",
        "cases": case_ids,
        "variants": variants,
        "note": "Two inference parameters change together; three-case diagnostic, not full validation.",
    }, indent=2))
    directories = {}
    metrics = {}
    for name, overlap, mode in variants:
        print(f"\nINFERENCE VARIANT: {name}", flush=True)
        folder, rows = inspect_saved_experiment(
            run_dir, subset="validation", checkpoint_name="best_model.pt",
            case_ids=case_ids, overlap=overlap, blend_mode=mode,
            output_tag=f"{name}_{stamp}", show_plots=False,
        )
        directories[name] = folder
        metrics[name] = {r["case_id"]: r for r in rows}

    comparisons = []
    for case_id in case_ids:
        row = {"case_id": case_id}
        for name, _, _ in variants:
            for key in ("dice", "precision", "recall", "predicted_voxels"):
                row[f"{name}_{key}"] = metrics[name][case_id][key]
        row["dice_change"] = row["gaussian50_dice"] - row["constant25_dice"]
        comparisons.append(row)

        base = directories["constant25"]
        ct_img = nib.load(base / f"{case_id}_normalized_ct.nii.gz")
        ct = np.asanyarray(ct_img.dataobj)
        gt = np.asanyarray(nib.load(base / f"{case_id}_ground_truth.nii.gz").dataobj) > 0
        masks = []
        for name, _, _ in variants:
            img = nib.load(directories[name] / f"{case_id}_prediction.nii.gz")
            if img.shape != ct.shape or not np.allclose(img.affine, ct_img.affine, atol=1e-3, rtol=0):
                raise RuntimeError("Comparison geometry differs")
            masks.append(np.asanyarray(img.dataobj) > 0)

        # Identical slices across all columns: GT maximum and each prediction maximum.
        slices = list(dict.fromkeys(
            int(np.argmax(m.sum(axis=(0, 1)))) for m in [gt, *masks]
        ))
        fig, axes = plt.subplots(len(slices), 4, figsize=(16, 4 * len(slices)), squeeze=False)
        for r, z in enumerate(slices):
            for c, (title, mask) in enumerate([
                ("CT", None), ("Ground truth", gt),
                ("25% constant", masks[0]), ("50% Gaussian", masks[1]),
            ]):
                ax = axes[r, c]
                ax.imshow(ct[:, :, z].T, cmap="gray", vmin=0, vmax=1, origin="lower")
                if mask is not None:
                    section = mask[:, :, z].T
                    ax.imshow(np.ma.masked_where(~section, section), cmap="autumn" if c == 1 else "winter", vmin=0, vmax=1, alpha=0.45, origin="lower")
                ax.set_title(f"{title} | slice {z}")
                ax.axis("off")
        fig.suptitle(
            f"{case_id} | constant Dice={row['constant25_dice']:.4f} | Gaussian Dice={row['gaussian50_dice']:.4f}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output / f"{case_id}_side_by_side.png", dpi=150)
        plt.close(fig)

    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)
    print("\nINFERENCE COMPARISON SUMMARY")
    for row in comparisons:
        print(f"{row['case_id']}: constant Dice={row['constant25_dice']:.4f}; Gaussian Dice={row['gaussian50_dice']:.4f}; change={row['dice_change']:+.4f}")
    print("Saved comparison CSV and side-by-side images to:", output)
    print("No training performed. These three selected cases do not establish full-validation performance.")


def compare_full_validation_inference():
    """Evaluate Gaussian inference; compare with the same checkpoint's saved baseline."""
    run_dir = (
        Path(__file__).resolve().parent / "outputs"
        / "full_split_mixed75_20261031_170107_903324"
    )
    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    step = int(checkpoint["step"])
    config = checkpoint["config"]
    case_ids = config["validation_cases"]
    if len(case_ids) != 30 or len(set(case_ids)) != 30:
        raise ValueError("Expected 30 unique validation cases.")
    if float(config["validation_threshold"]) != 0.5:
        raise ValueError("Expected the unchanged 0.5 threshold.")

    with (run_dir / "validation_cases.csv").open(newline="") as file:
        baseline_rows = [
            row for row in csv.DictReader(file)
            if int(row["step"]) == step
        ]
    baseline = {row["case_id"]: row for row in baseline_rows}
    if len(baseline_rows) != 30 or set(baseline) != set(case_ids):
        raise ValueError("Missing or duplicate same-checkpoint baseline results.")
    del checkpoint

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = run_dir / f"full_validation_inference_comparison_{stamp}"
    output.mkdir(exist_ok=False)
    print(f"Comparing checkpoint step {step} on all 30 validation cases.")
    print("Reusing saved constant/25% metrics; generating Gaussian/50% predictions.")
    print("No training or test-set evaluation will run.", flush=True)
    started = time.perf_counter()
    prediction_dir, gaussian_rows = inspect_saved_experiment(
        run_dir,
        subset="validation",
        checkpoint_name="best_model.pt",
        case_ids=case_ids,
        overlap=0.50,
        blend_mode="gaussian",
        output_tag=f"gaussian50_all30_{stamp}",
        show_plots=False,
    )
    gaussian = {row["case_id"]: row for row in gaussian_rows}
    comparisons = []
    for case_id in case_ids:
        old = baseline[case_id]
        new = gaussian[case_id]
        if int(old["gt_voxels"]) != int(new["ground_truth_voxels"]):
            raise ValueError(f"Ground-truth voxel count changed: {case_id}")
        row = {"case_id": case_id, "gt_voxels": int(old["gt_voxels"])}
        for metric in ("dice", "precision", "recall", "predicted_voxels"):
            row[f"constant25_{metric}"] = float(old[metric])
            row[f"gaussian50_{metric}"] = float(new[metric])
            row[f"change_{metric}"] = float(new[metric]) - float(old[metric])
        comparisons.append(row)

    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)

    summary = {
        "checkpoint_step": step,
        "validation_cases": len(case_ids),
        "threshold": 0.5,
        "baseline_source": str(run_dir / "validation_cases.csv"),
        "gaussian_predictions": str(prediction_dir),
        "gaussian_evaluation_minutes": (time.perf_counter() - started) / 60,
    }
    for variant in ("constant25", "gaussian50"):
        scores = [r[f"{variant}_dice"] for r in comparisons]
        summary[variant] = {
            "mean_dice": float(np.mean(scores)),
            "median_dice": float(np.median(scores)),
            "mean_precision": float(np.mean([r[f"{variant}_precision"] for r in comparisons])),
            "mean_recall": float(np.mean([r[f"{variant}_recall"] for r in comparisons])),
            "cases_recall_below_5_percent": sum(r[f"{variant}_recall"] < 0.05 for r in comparisons),
            "cases_precision_below_10_percent": sum(r[f"{variant}_precision"] < 0.1 for r in comparisons),
            "cases_volume_over_10_times_gt": sum(r[f"{variant}_predicted_voxels"] > 10 * r["gt_voxels"] for r in comparisons if r["gt_voxels"] > 0),
        }
    summary["dice_improved_cases"] = sum(r["change_dice"] > 1e-8 for r in comparisons)
    summary["dice_worsened_cases"] = sum(r["change_dice"] < -1e-8 for r in comparisons)
    summary["dice_unchanged_cases"] = 30 - summary["dice_improved_cases"] - summary["dice_worsened_cases"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nFULL-VALIDATION INFERENCE COMPARISON")
    print(json.dumps(summary, indent=2))
    print("Comparison report saved to:", output)
    print("Validation comparison only; the test split remains untouched.")


def inspect_best_long_run():
    """Inspect the verified best checkpoint; never starts training."""
    run_dir = (
        Path(__file__).resolve().parent / "outputs"
        / "full_split_mixed75_5000_gaussian50_20261031_174802_323144"
    )
    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    if checkpoint["step"] != 4000:
        raise ValueError("Expected the verified step-4000 best checkpoint.")
    config = checkpoint["config"]
    print("Best checkpoint step:", checkpoint["step"])
    print("Saved mean validation Dice:", checkpoint["validation_dice"])
    with (run_dir / "validation_cases.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    print("\nSaved whole-validation results (30 cases):")
    for step in (4000, 5000):
        selected = [row for row in rows if int(row["step"]) == step]
        if len(selected) != 30:
            raise ValueError(f"Expected 30 results at step {step}.")
        print(f"Step {step}: " + ", ".join(
            f"mean {key}={np.mean([float(row[key]) for row in selected]):.4f}"
            for key in ("dice", "precision", "recall")
        ))
    del checkpoint
    print("\nGenerating overlays for the same three previously inspected cases.")
    print("No training or test-set evaluation will run.", flush=True)
    folder, results = inspect_saved_experiment(
        run_dir,
        subset="validation",
        checkpoint_name="best_model.pt",
        case_ids=[
            "volume-covid19-A-0165",
            "volume-covid19-A-0011",
            "volume-covid19-A-0354",
        ],
        overlap=config["inference_overlap"],
        blend_mode=config["inference_blend_mode"],
        output_tag="best4000_gaussian50_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        show_plots=False,
    )
    print("\nBEST-CHECKPOINT INSPECTION COMPLETE")
    print("Comparison images and masks:", folder)
    print("Three-case overlays are illustrative; the saved validation mean uses all 30 cases.")


def evaluate_validation_thresholds():
    """Validation-only sweep; save probabilities for subsequent analyses."""
    import nibabel as nib
    run_dir = Path(__file__).resolve().parent / "outputs" / "full_split_mixed75_5000_gaussian50_20261031_174802_323144"
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    if checkpoint["step"] != 4000:
        raise ValueError("Expected checkpoint step 4000.")
    config = checkpoint["config"]
    cases = config["validation_cases"]
    manifest = json.loads((Path(__file__).resolve().parent / "splits.json").read_text())
    if len(cases) != 30 or set(cases) != {c["case_id"] for c in manifest["validation"]}:
        raise ValueError("Expected the original 30 validation cases.")
    folder = Path(os.environ.get("COVID19_TRAIN_DIR", manifest["source"])).expanduser()
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    output = run_dir / f"threshold_sweep_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(exist_ok=False)
    (output / "probabilities").mkdir()
    (output / "settings.json").write_text(json.dumps({
        "checkpoint_step": 4000, "config": config,
        "thresholds": thresholds, "comparison": "probability > threshold",
        "selection_rule": "maximum mean per-case Dice; ties prefer threshold closest to 0.5",
    }, indent=2))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(8,16,32,64), strides=(2,2,2), num_res_units=1, norm="INSTANCE").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    saved_score = checkpoint["validation_dice"]
    del checkpoint
    preprocessing = Compose([
        LoadImaged(keys=["image","label"], ensure_channel_first=True, dtype=np.float32),
        Orientationd(keys=["image","label"], axcodes="RAS", labels=None),
        Spacingd(keys=["image","label"], pixdim=tuple(config["spacing_mm"]), mode=("bilinear","nearest")),
        ScaleIntensityRanged(keys="image", a_min=config["intensity_window_hu"][0], a_max=config["intensity_window_hu"][1], b_min=0, b_max=1, clip=True),
    ])
    rows = []
    def write_csv(path, records):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print("Device:", device)
    print("30 validation scans, six thresholds, one inference pass per scan.")
    print("Output:", output, flush=True)
    started = time.perf_counter()
    with torch.inference_mode():
        for index, case_id in enumerate(cases, start=1):
            print(f"\n{index}/30: {case_id}", flush=True)
            data = preprocessing({"image":str(folder/f"{case_id}_ct.nii.gz"), "label":str(folder/f"{case_id}_seg.nii.gz")})
            if data["image"].shape != data["label"].shape or not torch.allclose(data["image"].affine, data["label"].affine, atol=1e-3, rtol=0):
                raise ValueError("CT/mask geometry differs.")
            image = data["image"].as_tensor().unsqueeze(0)
            label = data["label"].as_tensor()[0].numpy()
            if not np.all((label == 0) | (label == 1)):
                raise ValueError("Nonbinary ground truth.")
            gt = label > 0.5
            affine = data["image"].affine.cpu().numpy()
            logits = sliding_window_inference(inputs=image, roi_size=tuple(config["patch_size"]), sw_batch_size=1, predictor=model, overlap=config["inference_overlap"], mode=config["inference_blend_mode"], sw_device=device, device=torch.device("cpu"))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite predictions.")
            probability = torch.sigmoid(logits)[0,0].numpy()
            del logits
            nib.save(nib.Nifti1Image(probability.astype(np.float32), affine), output/"probabilities"/f"{case_id}_probability.nii.gz")
            nib.save(nib.Nifti1Image(gt.astype(np.uint8), affine), output/"probabilities"/f"{case_id}_ground_truth.nii.gz")
            actual = int(np.count_nonzero(gt))
            voxel_ml = abs(float(np.linalg.det(affine[:3,:3]))) / 1000
            for threshold in thresholds:
                pred = probability > threshold
                predicted = int(np.count_nonzero(pred))
                tp = int(np.count_nonzero(pred & gt))
                row = {
                    "case_id":case_id, "threshold":threshold,
                    "dice":2*tp/(predicted+actual) if predicted+actual else 1.,
                    "precision":tp/predicted if predicted else 0.,
                    "recall":tp/actual if actual else None,
                    "gt_voxels":actual, "predicted_voxels":predicted,
                    "false_positive_voxels":predicted-tp, "false_negative_voxels":actual-tp,
                    "false_positive_ml":(predicted-tp)*voxel_ml,
                    "predicted_to_gt_ratio":predicted/actual if actual else None,
                }
                rows.append(row)
                print(f"  t={threshold:.1f}: Dice={row['dice']:.4f}, precision={row['precision']:.4f}, recall={row['recall']}", flush=True)
            write_csv(output/"case_threshold_metrics.csv", rows)
            del data,image,label,gt,probability,pred
    summaries = []
    for threshold in thresholds:
        subset = [r for r in rows if r["threshold"] == threshold]
        recalls = [r["recall"] for r in subset if r["recall"] is not None]
        summaries.append({
            "threshold":threshold,
            "mean_dice":float(np.mean([r["dice"] for r in subset])),
            "median_dice":float(np.median([r["dice"] for r in subset])),
            "mean_precision":float(np.mean([r["precision"] for r in subset])),
            "mean_recall":float(np.mean(recalls)) if recalls else None,
            "mean_false_positive_ml":float(np.mean([r["false_positive_ml"] for r in subset])),
            "cases_recall_below_5_percent":sum(r < 0.05 for r in recalls),
            "cases_volume_over_10_times_gt":sum(r["predicted_to_gt_ratio"] > 10 for r in subset if r["predicted_to_gt_ratio"] is not None),
        })
    write_csv(output/"threshold_summary.csv", summaries)
    best = max(summaries, key=lambda r:(r["mean_dice"],-abs(r["threshold"]-0.5)))
    baseline = next(r for r in summaries if r["threshold"] == 0.5)
    difference = baseline["mean_dice"] - saved_score
    result = {
        "best_threshold_by_mean_validation_dice":best,
        "baseline_threshold_05":baseline,
        "baseline_dice_difference_from_saved":difference,
        "baseline_reproduced_within_1e_minus_4":abs(difference) < 1e-4,
        "elapsed_minutes":(time.perf_counter()-started)/60,
        "note":"Validation-selected threshold, not independent test performance. No model settings modified.",
    }
    (output/"summary.json").write_text(json.dumps(result,indent=2))
    print("\nTHRESHOLD COMPARISON SUMMARY")
    for row in summaries:
        print(json.dumps(row))
    print(json.dumps(result,indent=2))
    if abs(difference) >= 1e-4:
        print("REVIEW: threshold-0.5 score differs from the saved checkpoint score.")
    print("Saved to:",output)
    print("No training or test-set evaluation performed.")

if __name__ == "__main__":
    evaluate_validation_thresholds()
