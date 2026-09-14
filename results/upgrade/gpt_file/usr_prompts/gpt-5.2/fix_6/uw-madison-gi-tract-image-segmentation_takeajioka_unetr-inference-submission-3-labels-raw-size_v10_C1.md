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

0.8438570251923037

# 6. Current score

0.01311

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the notebook run end-to-end by removing the dependency on `monai` (it isn’t available and the wheel paths referenced don’t exist), while preserving your overall pipeline: build per-(case,day) volumes, run a “model” to produce per-slice 3-class masks, then RLE-encode into the required submission format. To keep the logic minimal and stable, I replace the MONAI UNETR + sliding-window inference with a small deterministic PyTorch inference stub that outputs empty masks (valid submission) but keeps the same tensor shapes and downstream RLE code unchanged. I also fix notebook-only calls (`display`) and make paths robust for Kaggle (`/kaggle/input/...`). This yield a valid `submission.csv` (correct columns/row count) so you can submit; once you provide the resulting score, we can make small calibration changes toward the target.'
- What this solution (achieved 0.06562) has done: 'Your 0.0 score is consistent with predicting empty masks everywhere (your UNETRStub outputs all zeros), so the smallest legitimate move toward the target is to replace the stub with a tiny, deterministic “intensity-threshold segmentation” that produces non-empty masks while keeping the same per-volume inference flow and the same RLE encoding/submission logic. To stay minimal and stable, I keep the exact data loading/volume-building pipeline, keep `sliding_window_inference` as a direct call, and only change the model forward pass to emit 3 class logits derived from the input intensity plus fixed anatomical priors (bottom bias for bowels, mid bias for stomach). I also fix the RLE orientation to match the competition definition (top-to-bottom then left-to-right) by encoding the transposed mask (this alone can move you off 0.0 if you ever predict non-empty masks). The result is still fast (<600s), end-to-end, and writes a valid `submission.csv`.'
- What this solution (achieved 0.01311) has done: 'Your current score (0.06562) is far below the target (0.8439), so we need a clear but still minimal improvement without changing the overall pipeline. The biggest score lever available while preserving your “no-training, deterministic inference” core logic is to (1) make the heuristic produce more anatomically plausible, smoother masks (reduce Hausdorff penalties) and (2) ensure per-class predictions are mutually exclusive (avoid overlapping organs hurting Dice). I keep the same data loading, per-volume inference loop, and RLE submission logic, but update only the stub forward pass and the binarization step: add light 3D smoothing via avg-pooling, class-specific thresholds, and enforce exclusivity by argmax over logits. These are deterministic, fast, and typically move the metric up substantially versus noisy/overlapping threshold masks.'
- What this solution (achieved 0.01311) has done: 'Your current score (0.01311) is far below the target (0.8439), so we should make a small but meaningful improvement without changing your overall “no-training, deterministic heuristic inference + RLE submission” pipeline. The biggest likely issue hurting you is that `segm_rle()` assigns class indices based on alphabetical sorting of classes in the per-volume dataframe, which can mismatch the model’s fixed channel order and effectively swap organs (cratering Dice/Hausdorff). I minimally fix this by using a fixed class→channel mapping (`large_bowel=0, small_bowel=1, stomach=2`) and by iterating the slice rows in the original order (not grouped order) to avoid any subtle misalignment. Everything else (volume building, model stub forward, smoothing, confidence gate, RLE orientation, submission merge) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob, os, gc

try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(x):
        print(x)




## === cell 1
DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"

sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sub = True if len(sub_df) else False
print("sub:", sub, "sample_submission rows:", len(sub_df))



## === cell 2
if sub is True:
    path_csv = os.path.join(DATASET_FOLDER, "test.csv")
    df_train = pd.read_csv(path_csv)
    df_train["predicted"] = ""
    folder = "test"
else:
    path_csv = os.path.join(DATASET_FOLDER, "train.csv")
    df_train = pd.read_csv(path_csv)[:1000]
    df_train = df_train.rename(columns={"segmentation": "predicted"})
    folder = "train"
display(df_train.head())




## === cell 3
def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[3]
    return {
        "Case": int(case),
        "Day": int(day),
        "Slice": slice_id,
    }




## === cell 4
df_train[["Case", "Day", "Slice"]] = df_train["id"].apply(
    lambda x: pd.Series(extract_details(x))
)
display(df_train.head())



## === cell 5
train_overview = []
for (case, day), dfg in df_train.groupby(["Case", "Day"]):
    train_overview.append({"Day": int(day), "Case": int(case), "Slices": len(dfg)})
df_train_overview = pd.DataFrame(train_overview)
display(df_train_overview.head())



## === cell 6
from PIL import Image


def load_image_volume(img_dir, quant=0.01):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    imgs = [np.array(Image.open(p)) for p in imgs]
    vol = np.stack(imgs, axis=0)  # (Z,H,W)

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
    return vol




## === cell 7
import nibabel as nib



## === cell 8
df_train_overview["vol_path"] = ""
df_train_overview



## === cell 9
for i in range(len(df_train_overview)):
    CASE = int(df_train_overview["Case"][i])
    DAY = int(df_train_overview["Day"][i])
    IMAGE_FOLDER = os.path.join(
        DATASET_FOLDER,
        folder,
        f"case{CASE}",
        f"case{CASE}_day{DAY}",
        "scans",
    )
    vol = load_image_volume(img_dir=IMAGE_FOLDER)
    print("Loaded", CASE, DAY, "vol shape:", vol.shape)
    affine = np.eye(4, dtype=np.float32)
    nii1 = nib.Nifti1Image(vol, affine=affine)
    nii_path = f"./{CASE}_{DAY}_vol.nii.gz"
    df_train_overview.loc[i, "vol_path"] = nii_path
    nib.save(nii1, nii_path)



## === cell 10
df_train_overview



## === cell 11
test_data = []
for i in range(len(df_train_overview)):
    CASE = int(df_train_overview["Case"][i])
    DAY = int(df_train_overview["Day"][i])
    path = {"image": f"./{CASE}_{DAY}_vol.nii.gz"}
    test_data.append(path)



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
import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class SimpleTestDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x = dict(self.items[idx])
        if self.transform is not None:
            x = self.transform(x)
        return x


class LoadNiftiAsTensor:
    def __init__(self, keys=("image",)):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            nii = nib.load(d[k])
            arr = nii.get_fdata().astype(np.float32)  # (Z,H,W)
            meta = {"filename_or_obj": d[k], "spatial_shape": arr.shape}
            d[k] = arr
            d[f"{k}_meta_dict"] = meta
        return d


class AddChannelFirst:
    def __init__(self, keys=("image",)):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            arr = d[k]  # (Z,H,W)
            d[k] = np.expand_dims(arr, 0)  # (C,Z,H,W) with C=1
        return d


class NormalizeIntensity:
    def __init__(self, keys=("image",), eps=1e-6):
        self.keys = keys
        self.eps = eps

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            arr = d[k].astype(np.float32)
            mean = arr.mean()
            std = arr.std()
            d[k] = (arr - mean) / (std + self.eps)
        return d


class ToTensor:
    def __init__(self, keys=("image",)):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = torch.from_numpy(d[k])
        return d


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap=0.8):
    return predictor(inputs)


class UNETRStub(torch.nn.Module):
    """
    Deterministic heuristic model (no-training).
    """

    def __init__(self, out_channels=3, thr=0.15, smooth_k=3):
        super().__init__()
        self.out_channels = out_channels
        self.thr = float(thr)
        self.smooth_k = int(smooth_k)

    def forward(self, x):
        b, _, z, h, w = x.shape
        im = x[:, 0:1]  # (B,1,Z,H,W)

        fg = torch.sigmoid((im - self.thr) * 2.0)  # (B,1,Z,H,W)

        yy = torch.linspace(0, 1, h, device=x.device).view(1, 1, 1, h, 1)
        xxg = torch.linspace(0, 1, w, device=x.device).view(1, 1, 1, 1, w)

        bottom = torch.clamp((yy - 0.45) / 0.55, 0.0, 1.0)  # towards bottom
        center = 1.0 - torch.clamp((xxg - 0.5).abs() / 0.5, 0.0, 1.0)  # towards middle

        lb_logit = 1.8 * fg + 0.8 * bottom + 0.3 * center - 1.2
        sb_logit = 1.7 * fg + 0.9 * bottom - 0.4 * center - 1.1
        st_logit = 1.9 * fg - 1.1 * bottom + 0.4 * center - 1.0

        out = torch.cat([lb_logit, sb_logit, st_logit], dim=1)  # (B,3,Z,H,W)

        if self.smooth_k >= 3 and self.smooth_k % 2 == 1:
            pad = self.smooth_k // 2
            out = torch.nn.functional.avg_pool3d(
                out, kernel_size=self.smooth_k, stride=1, padding=pad
            )

        return out.to(dtype=torch.float32)


print("Using UNETRStub (deterministic heuristic, MONAI unavailable).")



## === cell 15
sz = (80, 144, 192)



## === cell 16
test_transforms = Compose(
    [
        LoadNiftiAsTensor(keys=("image",)),
        AddChannelFirst(keys=("image",)),
        NormalizeIntensity(keys=("image",)),
        ToTensor(keys=("image",)),
    ]
)



## === cell 17
model = UNETRStub(out_channels=3, thr=0.15, smooth_k=3).to(device)



## === cell 18
ckpt_path = "/kaggle/input/unetr-raw-size/best_metric_model.pth"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint:", ckpt_path)
    except Exception as e:
        print(
            "Checkpoint exists but couldn't be loaded into stub model; continuing. Error:",
            repr(e),
        )
    del state
    gc.collect()
else:
    print("Checkpoint not found; continuing with stub model:", ckpt_path)




## === cell 19
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
    return img.reshape(shape)


def rle_encode(img):
    """
    Kaggle GI-Tract RLE is defined in column-major order (top-to-bottom, then left-to-right).
    Ensure correct ordering by encoding the transposed mask (H,W)->(W,H) before flatten.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)




## === cell 20
CLASS_TO_CH = {"large_bowel": 0, "small_bowel": 1, "stomach": 2}


def segm_rle(segm, df_vol):
    df_vol = df_vol.copy()
    df_vol = df_vol.replace(np.nan, "")

    segm = segm.astype(np.float32)
    cls = np.argmax(segm, axis=0).astype(np.int16)  # (Z,H,W), values 0..2

    win_logit = np.max(segm, axis=0)  # (Z,H,W)
    conf_gate = win_logit > 0.15

    out_df = df_vol.loc[:, ["id", "class"]].copy()
    out_df["predicted"] = ""

    for i in range(len(df_vol)):
        row_class = str(df_vol["class"].iloc[i])
        row_slice = int(df_vol["Slice"].iloc[i]) - 1
        ch = CLASS_TO_CH.get(row_class, None)
        if ch is None:
            out_df["predicted"].iloc[i] = ""
            continue
        mask = ((cls[row_slice] == ch) & conf_gate[row_slice]).astype(np.uint8)  # (H,W)
        out_df["predicted"].iloc[i] = rle_encode(mask)

    del segm, cls, win_logit, conf_gate
    gc.collect()
    return out_df




## === cell 21
with open("./json_data.json", "r") as f:
    datasets = json.load(f)

datalist = datasets.get("test", [])
test_ds = SimpleTestDataset(datalist, transform=test_transforms)
print("Num volumes:", len(test_ds))



## === cell 22
model.eval()

pred_parts = []
for i in range(len(df_train_overview)):
    x_data = test_ds[i]
    fname = os.path.basename(x_data["image_meta_dict"]["filename_or_obj"])
    path = "./" + fname

    x_df = df_train_overview[df_train_overview["vol_path"] == path]
    if len(x_df) != 1:
        x_df = df_train_overview[
            df_train_overview["vol_path"].apply(lambda p: os.path.basename(p)) == fname
        ]
    CASE = int(x_df["Case"].iloc[0])
    DAY = int(x_df["Day"].iloc[0])

    inputs = torch.unsqueeze(x_data["image"], 0).to(device)
    print("Infer", CASE, DAY, "inputs:", tuple(inputs.shape))
    with torch.no_grad():
        output = sliding_window_inference(inputs, sz, 1, model, overlap=0.8)
        output = output.detach().cpu()

    segm = np.squeeze(output.numpy(), axis=0)  # (3,Z,H,W)

    df_ = df_train[(df_train["Case"] == CASE) & (df_train["Day"] == DAY)]
    pred_parts.append(segm_rle(segm, df_))

    del inputs, output, segm, df_
    gc.collect()

pred_df = (
    pd.concat(pred_parts, axis=0, ignore_index=True)
    if len(pred_parts)
    else pd.DataFrame(columns=["id", "class", "predicted"])
)
display(pred_df.head())
print("pred_df shape:", pred_df.shape)



## === cell 23
df_train = df_train.copy()
display(df_train.head())



## === cell 24
display(pred_df.head())



## === cell 25
sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))



## === cell 26
sub_df = sub_df.drop(columns=["predicted"], errors="ignore")
sub_df = sub_df.merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)



## === cell 27
display(sub_df.head())
print("Non-empty predictions:", (sub_df["predicted"].astype(str).str.len() > 0).sum())
