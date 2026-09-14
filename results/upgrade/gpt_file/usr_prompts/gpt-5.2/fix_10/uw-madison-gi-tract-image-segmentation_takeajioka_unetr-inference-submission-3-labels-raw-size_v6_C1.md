# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a model to automatically segment the stomach and intestines on MRI scans.

## Metric
Mean Dice coefficient and 3D Hausdorff distance. 

The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where $X$ is the predicted set of pixels and $Y$ is the ground truth. The Dice coefficient is defined to be 0 when both $X$ and $Y$ are empty. 

Hausdorff distance is a method for calculating the distance between segmentation objects A and B, by calculating the furthest point on object A from the nearest point on object B. For 3D Hausdorff, we construct 3D volumes by combining each 2D segmentation with slice depth as the Z coordinate and then find the Hausdorff distance between them. (Here the slice depth for all scans is set to 1). The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score.

The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance.

## Submission Format
Use run-length encoding on the pixel values.  Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The file should contain a header and have the following format:

```
id,class,predicted
1,large_bowel,1 1 5 1
1,small_bowel,1 1
1,stomach,1 1
2,large_bowel,1 5 2 17
etc.
```

## Dataset
Each case is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal is to be able to generalize to both partially and wholly unseen cases.

### Files
- train.csv - IDs and masks for all training objects.
- sample_submission.csv - a sample submission file in the correct format
- train - a folder of case/day folders, each containing slice images for a particular case on a given day.

Note that the image filenames include 4 numbers (ex. 276_276_1.63_1.63.png). These four numbers are slice width / height (integers in pixels) and width/height pixel spacing (floating points in mm). The first two defines the resolution of the slide. The last two record the physical size of each pixel.

Physical pixel thickness in superior-inferior direction is 3mm.

### Columns
- `id` - unique identifier for object
- `class` - the predicted class for the object
- `segmentation` - RLE-encoded pixels for the identified object

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nibabel==5.3.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        input/
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        working/
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
```

-> data/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> data/uw-madison-gi-tract-image-segmentation/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/uw-madison-gi-tract-image-segmentation/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/uw-madison-gi-tract-image-segmentation/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> input/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.8146188749264773

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The crash comes from `extract_details()` returning an empty string for `Slice`, because the IDs in this competition are like `case123_day20_slice_0001` (the slice number is in the 4th token, not embedded in the 3rd). I fix `extract_details()` to robustly parse both common ID patterns and ensure `Slice` is always an integer, then make `segm_rle()` use a fixed class order mapping (instead of deriving indices from `unique()`), which prevents wrong channel indexing. These changes are score-neutral (they just unblock correct formatting/alignment) and let the notebook complete and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the immediate crash by making the checkpoint loading robust: if the external weights file isn’t present, the code still run by using a randomly initialized model (so you always get a valid `submission.csv` instead of a runtime error). I also correct the `class_to_channel` mapping to match the model’s output channel order after dropping background (large_bowel→0, small_bowel→1, stomach→2), which is a logic bug that would otherwise swap masks between classes and hurt score. Finally, I add a lightweight fallback that produces empty masks (valid RLE) if no working model is available, ensuring the pipeline completes end-to-end within the environment constraints and writes a correctly formatted submission file.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the notebook producing (almost) all-empty masks because the external checkpoint isn’t present in your environment, so `has_weights=False` and the prediction function returns zeros. To move toward the target score, the minimal effective change is to load weights from the actual available dataset path (the model file is not in your listed inputs), and only fall back to empty masks if no weights can be found. I also keep your class→channel mapping and RLE logic intact, but I ensure the slice index aligns to the sorted PNG filenames by using the per-volume slice order rather than trusting the numeric `Slice` token (this avoids systematic off-by-order mistakes that can tank Dice/Hausdorff). These changes preserve your core model and inference flow, but should turn the submission from “all empty” into meaningful segmentations, increasing the score toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from submitting (almost) all-empty masks because `has_weights=False` returns all-zero predictions when the checkpoint can’t be found/loaded. I make the checkpoint search/load robust to more Kaggle dataset layouts and common state_dict key patterns (including Lightning-style keys), so weights actually load when present and predictions become non-empty. I also fix a core alignment bug: you’re currently mapping `Slice` → depth by `slice_int-1`, but slice numbers in `id` can be non-1..N or not match the lexicographic PNG order; instead we build a filename→depth index from the actual sorted PNG list and use each row’s slice number to pick the correct depth. These are minimal changes that preserve your model/inference logic and output format, but should move the score upward toward the target by producing correctly aligned organ masks.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is because the notebook is submitting all-empty masks whenever it fails to find/load a trained checkpoint (`has_weights=False`), which guarantees near-zero Dice. The smallest score-relevant change is to actually use a trained checkpoint that exists in this environment by (1) searching both `/kaggle/input` and the provided `/kaggle/data` mirror, and (2) accepting `.pt` files as well as `.pth`, plus handling common wrapper keys/prefixes so loading succeeds more often. I also prevent accidental “all-empty” outputs when weights exist but the checkpoint has different output-channel conventions by selecting organ channels by name if a `labels` mapping is present, otherwise keeping your existing 4→(drop background) logic. These changes preserve your model architecture and inference flow, but should turn the submission into non-empty, correctly-mapped masks and move the score upward toward the target band.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with `has_weights=False` producing all-zero predictions, so the smallest score-relevant change is to actually locate and load a real checkpoint if one exists in this Kaggle environment (including common `.ckpt` Lightning files) and to correctly unwrap typical checkpoint dict formats. Next, to avoid silently outputting empty strings when slice↔depth mapping fails, we add a safe fallback that uses ordered slice position within each (case,day) group so every row gets a prediction aligned to the actual PNG stack. Finally, we keep your model/inference/RLE logic intact but ensure the merged submission preserves the sample_submission row order exactly (no accidental reordering), which prevents evaluation mismatches.'

# 9. Code solution

## === cell 0
import os, glob, gc, sys, json, subprocess
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"
if not os.path.exists(DATASET_FOLDER):
    DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
if not os.path.exists(DATASET_FOLDER):
    DATASET_FOLDER = "/kaggle/data/uw-madison-gi-tract-image-segmentation"

sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sub = len(sub_df) > 0
print("DATASET_FOLDER:", DATASET_FOLDER, "| sub mode:", sub)



## === cell 2
if sub:
    path_csv = os.path.join(DATASET_FOLDER, "test.csv")
    df_train = pd.read_csv(path_csv)
    df_train["predicted"] = ""
    folder = "test"
else:
    path_csv = os.path.join(DATASET_FOLDER, "train.csv")
    df_train = pd.read_csv(path_csv)[:1000]
    df_train = df_train.rename(columns={"segmentation": "predicted"})
    folder = "train"

print(df_train.head())




## === cell 3
def extract_details(id_):
    parts = str(id_).split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected id format: {id_}")

    case = parts[0].replace("case", "")
    day = parts[1].replace("day", "")

    slice_id = None
    if "slice" in parts:
        k = parts.index("slice")
        if k + 1 < len(parts):
            slice_id = parts[k + 1]
        elif k > 0:
            slice_id = parts[k].replace("slice", "")
    if slice_id is None:
        for p in parts:
            if p.startswith("slice"):
                slice_id = p.replace("slice", "")
                break
    if slice_id is None:
        raise ValueError(f"Could not parse slice from id: {id_}")

    slice_int = int(slice_id)
    return {"Case": int(case), "Day": int(day), "Slice": slice_int}




## === cell 4
df_train[["Case", "Day", "Slice"]] = df_train["id"].apply(
    lambda x: pd.Series(extract_details(x))
)
print(df_train.head())



## === cell 5
train_overview = []
for (case, day), dfg in df_train.groupby(["Case", "Day"]):
    train_overview.append({"Day": int(day), "Case": int(case), "Slices": len(dfg)})
df_train_overview = pd.DataFrame(train_overview)
print(df_train_overview.head())



## === cell 6
from PIL import Image


def load_image_volume(img_dir, quant=0.01):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(imgs) == 0:
        raise FileNotFoundError(f"No .png slices found in: {img_dir}")
    vol = np.stack([np.array(Image.open(p)) for p in imgs], axis=0)

    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = np.min(vol), np.max(vol)
    if v_max > v_min:
        vol = (vol - v_min) / (v_max - v_min)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    vol = (vol * 255).astype(np.uint8)

    gc.collect()
    return vol




## === cell 7
import nibabel as nib



## === cell 8
df_train_overview["vol_path"] = ""
df_train_overview["slice_files"] = ""
print(df_train_overview.head())



## === cell 9
for i in range(len(df_train_overview)):
    CASE = int(df_train_overview.loc[i, "Case"])
    DAY = int(df_train_overview.loc[i, "Day"])
    IMAGE_FOLDER = os.path.join(
        DATASET_FOLDER, folder, f"case{CASE}", f"case{CASE}_day{DAY}", "scans"
    )
    imgs = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))
    if len(imgs) == 0:
        raise FileNotFoundError(f"No .png slices found in: {IMAGE_FOLDER}")

    vol = load_image_volume(img_dir=IMAGE_FOLDER)
    print("volume shape:", vol.shape, "|", "CASE/DAY:", CASE, DAY)

    nii1 = nib.Nifti1Image(vol, affine=np.eye(4, dtype=np.float32))
    nii_path = f"./{CASE}_{DAY}_vol.nii.gz"
    df_train_overview.loc[i, "vol_path"] = nii_path
    df_train_overview.loc[i, "slice_files"] = json.dumps(imgs)
    nib.save(nii1, nii_path)
    del vol, nii1
    gc.collect()



## === cell 10
print(df_train_overview.head())



## === cell 11
test_data = []
for i in range(len(df_train_overview)):
    CASE = int(df_train_overview.loc[i, "Case"])
    DAY = int(df_train_overview.loc[i, "Day"])
    test_data.append({"image": f"./{CASE}_{DAY}_vol.nii.gz"})



## === cell 12
data1 = {
    "description": "UWM",
    "labels": {
        "0": "background",
        "1": "large_bowel",
        "2": "small_bowel",
        "3": "stomach",
    },
    "test": test_data,
}
with open("json_data.json", "w") as f:
    json.dump(data1, f)
print("Wrote json_data.json with", len(test_data), "items")



## === cell 13
USE_MONAI = False
print("USE_MONAI:", USE_MONAI)



## === cell 14
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 15
sz = (80, 144, 192)




## === cell 16
def load_nifti_as_tensor(path):
    vol = nib.load(path).get_fdata()  # float64
    vol = vol.astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    x = torch.from_numpy(vol)[None, ...]  # [1, D, H, W]
    return x




## === cell 17
def find_checkpoint():
    candidates = [
        "../input/unetr-raw-size/best_metric_model.pth",
        "/kaggle/input/unetr-raw-size/best_metric_model.pth",
        "/kaggle/data/unetr-raw-size/best_metric_model.pth",
        "../input/**/best_metric_model.pth",
        "/kaggle/input/**/best_metric_model.pth",
        "/kaggle/data/**/best_metric_model.pth",
        "../input/**/model.pth",
        "/kaggle/input/**/model.pth",
        "/kaggle/data/**/model.pth",
        "../input/**/*.pth",
        "/kaggle/input/**/*.pth",
        "/kaggle/data/**/*.pth",
        "../input/**/*.pt",
        "/kaggle/input/**/*.pt",
        "/kaggle/data/**/*.pt",
        "../input/**/*.ckpt",
        "/kaggle/input/**/*.ckpt",
        "/kaggle/data/**/*.ckpt",
    ]
    best = ""
    best_size = -1
    for pat in candidates:
        hits = glob.glob(pat, recursive=True)
        for p in hits:
            if os.path.exists(p):
                try:
                    szb = os.path.getsize(p)
                except Exception:
                    szb = 0
                if szb > best_size and szb > 0:
                    best = p
                    best_size = szb
    return best


state_path = find_checkpoint()
print(
    "Weights path:",
    state_path,
    "| exists:",
    bool(state_path) and os.path.exists(state_path),
    "| size:",
    os.path.getsize(state_path) if state_path and os.path.exists(state_path) else -1,
)


class SimpleUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, base=16):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1),
            nn.InstanceNorm3d(base),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1),
            nn.InstanceNorm3d(base),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(
            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.InstanceNorm3d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base * 2, base * 2, 3, padding=1),
            nn.InstanceNorm3d(base * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)
        self.bot = nn.Sequential(
            nn.Conv3d(base * 2, base * 4, 3, padding=1),
            nn.InstanceNorm3d(base * 4),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base * 4, base * 2, 3, padding=1),
            nn.InstanceNorm3d(base * 2),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base * 2, base, 3, padding=1),
            nn.InstanceNorm3d(base),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def _strip_prefix(k: str) -> str:
    for pfx in ("model.", "module.", "net.", "network.", "seg_model."):
        if k.startswith(pfx):
            return k[len(pfx) :]
    return k


def load_model_from_checkpoint(path, device):
    if not path or (not os.path.exists(path)):
        print("WARNING: checkpoint not found; using randomly initialized model.")
        m = SimpleUNet3D(in_ch=1, out_ch=4, base=16)
        m.eval()
        return m.to(device), False

    obj = torch.load(path, map_location="cpu")

    sd = None
    if isinstance(obj, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                break
        if sd is None and all(isinstance(k, str) for k in obj.keys()):
            sd = obj  # raw state_dict
    if sd is None:
        if hasattr(obj, "eval") and hasattr(obj, "to"):
            m = obj
            m.eval()
            return m.to(device), True
        print("WARNING: Unknown checkpoint format; using randomly initialized model.")
        m = SimpleUNet3D(in_ch=1, out_ch=4, base=16)
        m.eval()
        return m.to(device), False

    sd = {_strip_prefix(k): v for k, v in sd.items()}

    m = SimpleUNet3D(in_ch=1, out_ch=4, base=16)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print("Loaded state_dict | missing:", len(missing), "unexpected:", len(unexpected))
    m.eval()
    ok = True
    if len(missing) == len(list(m.state_dict().keys())):
        ok = False
    return m.to(device), ok


model, has_weights = load_model_from_checkpoint(state_path, device)
print("Model ready:", type(model), "| has_weights:", has_weights)




## === cell 18
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def show_img(img, mask=None):
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap="bone")
        if mask is not None:
            plt.imshow(mask[i, :, :], alpha=0.5)
        plt.axis("off")
    plt.show()




## === cell 19
def _slice_int_from_png_path(p: str) -> int | None:
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    tok0 = stem.split("_")[0]
    try:
        return int(tok0)
    except Exception:
        return None


def build_slice_to_depth_index(slice_files_json: str):
    files = json.loads(slice_files_json)
    slice_to_depth = {}
    for depth, p in enumerate(files):
        s = _slice_int_from_png_path(p)
        if s is not None:
            slice_to_depth[int(s)] = int(depth)
    return files, len(files), slice_to_depth


def segm_rle(segm, df_vol, class_to_channel, n_slices: int, slice_to_depth: dict):
    out_parts = []
    df_vol = df_vol.copy()
    df_vol = df_vol.replace(np.nan, "")

    unique_slices_sorted = sorted(df_vol["Slice"].astype(int).unique().tolist())
    fallback_depth_by_slice = {s: i for i, s in enumerate(unique_slices_sorted)}

    for slice_idx, dfg in df_vol.groupby("Slice", sort=True):
        slice_int = int(slice_idx)
        depth = slice_to_depth.get(slice_int, None)
        if depth is None:
            depth = fallback_depth_by_slice.get(slice_int, None)
        dfg = dfg.copy()
        for row_i, row in dfg.iterrows():
            cls = row["class"]
            if cls not in class_to_channel:
                dfg.loc[row_i, "predicted"] = ""
                continue
            ch = class_to_channel[cls]
            if (
                depth is None
                or depth < 0
                or depth >= n_slices
                or depth >= segm.shape[1]
            ):
                dfg.loc[row_i, "predicted"] = ""
                continue
            mask = segm[ch, depth, :, :]
            dfg.loc[row_i, "predicted"] = rle_encode((mask > 0.5).astype(np.uint8))
        out_parts.append(dfg.loc[:, ["id", "class", "predicted"]])

    del segm
    gc.collect()
    return (
        pd.concat(out_parts, ignore_index=True)
        if len(out_parts)
        else pd.DataFrame(columns=["id", "class", "predicted"])
    )




## === cell 20
pred_parts = []
classes_sorted = ["large_bowel", "small_bowel", "stomach"]

class_to_channel = {"large_bowel": 0, "small_bowel": 1, "stomach": 2}


def _try_get_labels_from_checkpoint(path: str):
    try:
        if not path or not os.path.exists(path):
            return None
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for k in ("labels", "label_map", "classes"):
                if k in obj:
                    return obj[k]
    except Exception:
        return None
    return None


ckpt_labels = _try_get_labels_from_checkpoint(state_path)
print("ckpt_labels type:", type(ckpt_labels).__name__)


@torch.no_grad()
def predict_volume_probs(x_1dhw: torch.Tensor) -> np.ndarray:
    """
    x_1dhw: [1, D, H, W] in [0,1]
    returns segm: np.ndarray [3, D, H, W] float32 probabilities for the 3 organs
    """
    x = x_1dhw.to(device)  # [1,D,H,W]
    x = x[:, None, ...]  # [1,1,D,H,W]

    if not has_weights:
        _, _, D, H, W = x.shape
        return np.zeros((3, D, H, W), dtype=np.float32)

    logits = model(x)
    if logits.ndim != 5:
        raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")

    C = logits.shape[1]

    if C >= 4:
        probs_all = torch.softmax(logits, dim=1)  # [1,C,D,H,W]
        organ_probs = probs_all[:, 1:4] if C >= 4 else probs_all[:, :3]
    else:
        organ_probs = torch.sigmoid(logits[:, :3])

    segm = organ_probs[0].float().detach().cpu().numpy()  # [3,D,H,W]
    return segm.astype(np.float32)


for i in range(len(df_train_overview)):
    CASE = int(df_train_overview.loc[i, "Case"])
    DAY = int(df_train_overview.loc[i, "Day"])
    vol_path = df_train_overview.loc[i, "vol_path"]
    slice_files_json = df_train_overview.loc[i, "slice_files"]
    _, n_slices, slice_to_depth = build_slice_to_depth_index(slice_files_json)

    x = load_nifti_as_tensor(vol_path)  # [1, D, H, W]
    print(
        "inputs:",
        tuple(x.shape),
        "CASE/DAY:",
        CASE,
        DAY,
        "| n_slices:",
        n_slices,
        "| slice_to_depth size:",
        len(slice_to_depth),
    )

    segm = predict_volume_probs(x)  # [3,D,H,W]

    df_ = df_train[(df_train["Case"] == CASE) & (df_train["Day"] == DAY)].copy()
    df_ = df_[df_["class"].isin(classes_sorted)].copy()
    pred_parts.append(
        segm_rle(
            segm,
            df_,
            class_to_channel,
            n_slices=n_slices,
            slice_to_depth=slice_to_depth,
        )
    )

    del x, segm
    gc.collect()

pred_df = (
    pd.concat(pred_parts, ignore_index=True)
    if len(pred_parts)
    else pd.DataFrame(columns=["id", "class", "predicted"])
)
print("pred_df:", pred_df.shape)



## === cell 21
print(pred_df.head())



## === cell 22
print(pred_df.tail())



## === cell 23
if sub:
    sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
else:
    sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))[:1000]
    sub_df = sub_df.rename(columns={"segmentation": "predicted"})



## === cell 24
sub_df = sub_df.copy()
sub_df["_row"] = np.arange(len(sub_df), dtype=np.int64)

sub_df = sub_df.drop(columns=["predicted"], errors="ignore")
sub_df = sub_df.merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("").astype(str)

sub_df = sub_df.sort_values("_row").drop(columns=["_row"]).reset_index(drop=True)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "rows:", len(sub_df), "cols:", list(sub_df.columns))



## === cell 25
print(sub_df.head())
print(sub_df.tail())
print(
    "Submission file exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else -1,
)
