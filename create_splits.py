from pathlib import Path
import os
from collections import defaultdict
import json
import random
import re

DATA_DIR = Path(os.environ.get(
    "COVID19_TRAIN_DIR",
    str(Path.home() / "Documents/medical/covid19-20/COVID-19-20_v2/Train"),
)).expanduser()
OUTPUT = Path(__file__).resolve().parent / "splits.json"
SEED = 42

# Protect an existing split from accidental replacement.
if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Keep the same split for all experiments."
    )

groups = defaultdict(list)

for image in sorted(DATA_DIR.glob("*_ct.nii.gz")):
    case_id = image.name.removesuffix("_ct.nii.gz")
    mask = DATA_DIR / f"{case_id}_seg.nii.gz"

    if not mask.is_file():
        raise FileNotFoundError(f"Missing mask: {mask}")

    # Precaution: scans sharing a base ID must stay together.
    # This is inferred from filenames, not verified patient metadata.
    group_id = re.sub(r"_\d+$", "", case_id)

    groups[group_id].append({
        "case_id": case_id,
        "group_id": group_id,
        "image": str(image),
        "label": str(mask),
    })

if len(groups) != 199 or sum(map(len, groups.values())) != 199:
    raise ValueError(
        "Expected 199 cases with distinct base IDs. "
        "Recheck the dataset before creating splits."
    )

group_ids = sorted(groups)
random.Random(SEED).shuffle(group_ids)

assignments = {
    "train": group_ids[:139],
    "validation": group_ids[139:169],
    "test": group_ids[169:],
}

splits = {
    name: [
        case
        for group_id in assigned_groups
        for case in groups[group_id]
    ]
    for name, assigned_groups in assignments.items()
}

# Verify that groups never overlap between splits.
train_ids = set(assignments["train"])
val_ids = set(assignments["validation"])
test_ids = set(assignments["test"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)
assert train_ids | val_ids | test_ids == set(groups)

result = {
    "seed": SEED,
    "grouping_note": (
        "Base case IDs inferred from filenames; "
        "independent patient identifiers were not supplied."
    ),
    "source": str(DATA_DIR),
    **splits,
}

with OUTPUT.open("x") as file:
    json.dump(result, file, indent=2)

print("SPLIT SUMMARY")
for name, cases in splits.items():
    print(f"{name}: {len(cases)} cases")

print("Group overlap: NONE")
print("Saved to:", OUTPUT)