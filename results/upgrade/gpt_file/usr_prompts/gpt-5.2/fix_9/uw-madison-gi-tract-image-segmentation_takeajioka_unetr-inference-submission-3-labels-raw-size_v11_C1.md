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

0.8372741440309309

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the MONAI metadata KeyError by explicitly enabling metadata tracking (and using the correct meta key) so inference can map each loaded NIfTI back to its case/day. I also make the case/day mapping deterministic by storing an explicit `image` path in `df_train_overview` and building `test_data` from that, avoiding fragile basename logic. Then I ensure `pred_df` is always defined (even if something goes wrong) and that the final `submission.csv` is written with the exact required columns/row order from `sample_submission.csv`. These changes are runtime/IO alignment fixes and should be score-neutral except that they allow the model predictions to actually be produced instead of failing.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a submission that is effectively empty/invalid for the test set because the pipeline is reading `sample_submission.csv` instead of `test.csv`, so it builds case/day/slice groups that don’t exist and can’t align predictions to the real test IDs. I make the minimal fix of loading `test.csv` when `sub==True`, and I also extract the slice index correctly for test IDs (they have 3 parts, not 4), so grouping by `Slice` matches the real scan slices. Finally, I keep your model/inference/RLE logic intact but ensure `pred_df` is merged onto `sample_submission.csv` in the exact row order, producing a valid non-empty `submission.csv` and moving the score upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that contains mostly empty/incorrectly-mapped masks due to slice index parsing and volume slice ordering not matching the `id` slice numbers. I make two minimal, score-relevant fixes: (1) parse `slice` robustly for both train/test IDs and keep it as an integer for correct grouping/sorting, and (2) ensure the PNG stack order matches the numeric slice order (not lexicographic) when building the NIfTI volumes. These keep your MONAI UNETR inference + sliding window + RLE logic unchanged, but should turn “mostly wrong/empty” predictions into correctly-aligned ones, moving score upward toward your target. Submission writing/format is kept identical, just safer alignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with generating *systematically misaligned* masks: the model outputs are in the UNETR “img_size” space (80×144×192), but your RLE encoding assumes the original PNG resolution per slice, so even non-empty predictions decode to the wrong pixels and score ~0. I keep your model, weights, sliding-window inference, and RLE logic intact, but add a minimal post-processing step that resizes each predicted slice back to the actual (H,W) of that case/day before encoding. I also ensure the slice count aligns by cropping/padding the Z dimension to the number of slices in `df_` so `Slice` indexing cannot drift. These changes should move the score upward toward your target while preserving your overall approach and producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a *systematic ID↔slice misalignment*: you parse `Slice` from `id` but never use it to select the correct 2D slice from the 3D model output, so most rows get the wrong mask and score collapses. I keep your exact model/inference/RLE approach, but change `segm_rle()` to index the prediction volume by the true numeric slice (using the unique sorted slice list), rather than `slice_id-1`. I also ensure the test dataframe is sorted by `Slice` before encoding and explicitly build the slice-index mapping once per (case,day), which is minimal but directly fixes alignment and should move the score upward toward your target. Submission writing stays identical (merged onto `sample_submission.csv` for correct order/rows).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with a valid-looking submission that is nevertheless *systematically misaligned* with the evaluation IDs: you’re generating NIfTI volumes from PNGs but you never ensure the PNG slice indices match the `Slice` numbers parsed from `test.csv`, so `segm_rle()` often encodes the wrong 2D plane per `id`. I make the smallest score-relevant fix by explicitly building a per-(case,day) mapping from actual PNG slice number → z-index, and then using that mapping when selecting `segm[:, z, ...]` for each row. This preserves your model, weights, sliding-window inference, and RLE encoding, but fixes the core ID↔slice alignment that yields near-random masks. I also add a safety fallback to output empty masks when a slice number is missing from the PNG stack, which avoids invalid RLE while keeping semantics intact.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob, os, gc, re



## === cell 1
sub_df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
if len(sub_df):
    sub = True
else:
    sub = False



## === cell 2
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"

if sub == True:
    path_csv = os.path.join(DATASET_FOLDER, "test.csv")
    df_train = pd.read_csv(path_csv)
    folder = "test"
else:
    path_csv = os.path.join(DATASET_FOLDER, "train.csv")
    df_train = pd.read_csv(path_csv)[:1000]
    df_train = df_train.rename(columns={"segmentation": "predicted"})
    folder = "train"

try:
    display(df_train.head())
except NameError:
    print(df_train.head())



## === cell 3
_SLICE_RE = re.compile(r"(?:^|_)slice_?(\d+)(?:_|$)")


def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")

    m = _SLICE_RE.search(id_)
    if m:
        slice_id = int(m.group(1))
    else:
        nums = re.findall(r"\d+", id_)
        slice_id = int(nums[-1]) if nums else 1

    return {
        "Case": int(case),
        "Day": int(day),
        "Slice": int(slice_id),
    }




## === cell 4
df_train[["Case", "Day", "Slice"]] = df_train["id"].apply(
    lambda x: pd.Series(extract_details(x))
)
try:
    display(df_train.head())
except NameError:
    print(df_train.head())



## === cell 5
train_overview = []
for (case, day), dfg in df_train.groupby(["Case", "Day"]):
    train_overview.append({"Day": int(day), "Case": int(case), "Slices": len(dfg)})
df_train_overview = pd.DataFrame(train_overview)
try:
    display(df_train_overview.head())
except NameError:
    print(df_train_overview.head())



## === cell 6
from PIL import Image


def _slice_num_from_path(p):
    base = os.path.basename(p)
    first = base.split("_")[0]
    try:
        return int(first)
    except Exception:
        nums = re.findall(r"\d+", base)
        return int(nums[0]) if nums else 0


def load_image_volume(img_dir, quant=0.01, return_slice_nums=False):
    imgs = glob.glob(os.path.join(img_dir, "*.png"))
    imgs = sorted(imgs, key=_slice_num_from_path)
    slice_nums = [_slice_num_from_path(p) for p in imgs]

    imgs = [np.array(Image.open(p)).tolist() for p in imgs]
    vol = np.array(imgs)

    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = np.min(vol), np.max(vol)
    if v_max > v_min:
        vol = (vol - v_min) / (v_max - v_min)
    else:
        vol = vol * 0.0
    vol = (vol * 255).astype(np.uint8)
    del imgs
    gc.collect()
    if return_slice_nums:
        return vol, slice_nums
    return vol




## === cell 7
import nibabel as nib



## === cell 8
df_train_overview["vol_path"] = ""
df_train_overview["image"] = ""
df_train_overview["img_dir"] = ""
df_train_overview["H"] = 0
df_train_overview["W"] = 0
df_train_overview["png_slices"] = None
df_train_overview



## === cell 9
for i in range(len(df_train_overview)):
    CASE = df_train_overview["Case"][i]
    DAY = df_train_overview["Day"][i]
    IMAGE_FOLDER = os.path.join(
        "../input/uw-madison-gi-tract-image-segmentation/",
        folder,
        f"case{CASE}",
        f"case{CASE}_day{DAY}",
        "scans",
    )
    vol, slice_nums = load_image_volume(img_dir=IMAGE_FOLDER, return_slice_nums=True)
    print(vol.shape)
    nii1 = nib.Nifti1Image(vol, affine=np.eye(4))
    nii_path = f"./{CASE}_{DAY}_vol.nii.gz"
    df_train_overview.loc[i, "vol_path"] = nii_path
    df_train_overview.loc[i, "image"] = nii_path
    df_train_overview.loc[i, "img_dir"] = IMAGE_FOLDER
    df_train_overview.loc[i, "H"] = int(vol.shape[1])
    df_train_overview.loc[i, "W"] = int(vol.shape[2])
    df_train_overview.at[i, "png_slices"] = slice_nums
    nib.save(nii1, nii_path)
    del vol
    gc.collect()



## === cell 10
df_train_overview



## === cell 11
test_data = []
for i in range(len(df_train_overview)):
    test_data.append({"image": df_train_overview.loc[i, "image"]})



## === cell 12
import json

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

json_string = json.dumps(data1)
print(json_string)



## === cell 13
with open("json_data.json", "w") as outfile:
    json.dump(data1, outfile)



## === cell 14
import sys, subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "monai"])



## === cell 15
import sys, subprocess, os

einops_wheel = "/kaggle/input/uwm-model/einops-0.4.1-py3-none-any.whl"
if os.path.exists(einops_wheel):
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-index",
            f"--find-links=file:///kaggle/input/uwm-model",
            "einops",
        ]
    )
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "einops"])



## === cell 16
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ToTensord,
    EnsureChannelFirstd,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch
import torch.nn.functional as F

print_config()



## === cell 17
sz = (80, 144, 192)



## === cell 18
test_transforms = Compose(
    [
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
)



## === cell 19
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    in_channels=1,
    out_channels=3,  # 3 foreground classes as in original code
    img_size=sz,
    dropout_rate=0.2,
).to(device)




## === cell 20
def find_weight_file():
    candidates = [
        "../input/unetr-raw-size/best_metric_model.pth",
        "/kaggle/input/unetr-raw-size/best_metric_model.pth",
        "../input/uwm-model/best_metric_model.pth",
        "/kaggle/input/uwm-model/best_metric_model.pth",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for p in glob.glob("/kaggle/input/**/best_metric_model.pth", recursive=True):
        return p
    for p in glob.glob("/kaggle/input/**/*.pth", recursive=True):
        return p
    return None


weight_path = find_weight_file()
has_weights = weight_path is not None
print("Weight file:", weight_path)

if has_weights:
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)
else:
    print("WARNING: No weights found. Submission will contain empty masks.")




## === cell 21
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 22
CLASS_TO_CH = {"large_bowel": 0, "small_bowel": 1, "stomach": 2}


def segm_rle(segm, df_vol, out_hw=None, slice_to_z=None):
    df_vol = df_vol.copy()
    df_vol = df_vol.replace(np.nan, "")

    if slice_to_z is None:
        slice_vals = np.array(sorted(df_vol["Slice"].unique().tolist()), dtype=int)
        slice_to_z = {int(s): int(i) for i, s in enumerate(slice_vals.tolist())}

    out_rows = []
    for slice_id, dfg in df_vol.groupby("Slice", sort=True):
        slice_id = int(slice_id)
        z_idx = slice_to_z.get(slice_id, None)
        dfg = dfg.copy()

        for row_i, row in dfg.iterrows():
            ch = CLASS_TO_CH.get(row["class"], None)
            if ch is None or z_idx is None:
                dfg.loc[row_i, "predicted"] = ""
                continue
            if segm.ndim != 4 or z_idx < 0 or z_idx >= segm.shape[1]:
                dfg.loc[row_i, "predicted"] = ""
                continue

            mask = segm[ch, z_idx, :, :]  # [H_pred, W_pred], float in model resolution
            if out_hw is not None:
                out_h, out_w = int(out_hw[0]), int(out_hw[1])
                if (mask.shape[0] != out_h) or (mask.shape[1] != out_w):
                    mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
                    mt = F.interpolate(
                        mt, size=(out_h, out_w), mode="bilinear", align_corners=False
                    )
                    mask = mt.squeeze(0).squeeze(0).numpy()

            dfg.loc[row_i, "predicted"] = rle_encode((mask > 0.5).astype(np.uint8))

        out_rows.append(dfg.loc[:, ["id", "class", "predicted"]])

    del segm
    gc.collect()
    if len(out_rows) == 0:
        return pd.DataFrame(columns=["id", "class", "predicted"])
    return pd.concat(out_rows, axis=0, ignore_index=True)




## === cell 23
datasets = "./json_data.json"
datalist = load_decathlon_datalist(datasets, True, "test")
test_ds = CacheDataset(
    data=datalist,
    transform=test_transforms,
    cache_num=min(16, len(datalist)),
    cache_rate=1.0,
    num_workers=2,
)



## === cell 24
model.eval()
pred_parts = []

for i in range(len(df_train_overview)):
    x_data = test_ds[i]
    path = df_train_overview.loc[i, "vol_path"]

    x_df = df_train_overview[df_train_overview["vol_path"] == path]
    CASE = int(x_df["Case"].iloc[0])
    DAY = int(x_df["Day"].iloc[0])
    out_h = int(x_df["H"].iloc[0])
    out_w = int(x_df["W"].iloc[0])

    df_ = df_train[(df_train["Case"] == CASE) & (df_train["Day"] == DAY)].copy()
    df_ = df_.sort_values(["Slice", "class"]).reset_index(drop=True)
    num_slices_expected = int(df_["Slice"].nunique())

    png_slices = x_df["png_slices"].iloc[0]
    if isinstance(png_slices, (list, tuple)) and len(png_slices) > 0:
        slice_to_z = {int(s): int(z) for z, s in enumerate(png_slices)}
    else:
        slice_to_z = None

    if not has_weights:
        df_empty = df_.loc[:, ["id", "class"]].copy()
        df_empty["predicted"] = ""
        pred_parts.append(df_empty)
        continue

    inputs = torch.unsqueeze(x_data["image"], 0).to(device)

    with torch.no_grad():
        output = sliding_window_inference(inputs, sz, 1, model, overlap=0.8)
    output = output.detach().cpu()
    output = torch.sigmoid(output)

    segm = np.squeeze(output.numpy())  # [C, Z, H, W] in model resolution

    if segm.ndim == 4:
        z = segm.shape[1]
        if z > num_slices_expected:
            segm = segm[:, :num_slices_expected, :, :]
        elif z < num_slices_expected:
            pad = num_slices_expected - z
            segm = np.pad(
                segm,
                ((0, 0), (0, pad), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

    pred_parts.append(segm_rle(segm, df_, out_hw=(out_h, out_w), slice_to_z=slice_to_z))

    del inputs, output, segm
    gc.collect()

pred_df = (
    pd.concat(pred_parts, axis=0, ignore_index=True)
    if len(pred_parts)
    else pd.DataFrame(columns=["id", "class", "predicted"])
)



## === cell 25
df_train["predicted"] = (
    pred_df["predicted"].values
    if len(pred_df) == len(df_train)
    else df_train.get("predicted", "")
)
df_train



## === cell 26
pred_df



## === cell 27
if sub == True:
    sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
else:
    sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))[:1000]
    sub_df = sub_df.rename(columns={"segmentation": "predicted"})



## === cell 28
sub_df = sub_df[["id", "class"]].merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df.to_csv("submission.csv", index=False)
sub_df
