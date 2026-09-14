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

0.02384

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I install MONAI (if it isn’t already), import the required classes in the same cell, handle the possible absence of CUDA, guard the model‑weight loading, and adjust the inference loop so that every variable is defined before it’s used. These changes fix the import errors, ensure the test dataset is built correctly, and guarantee that a `submission.csv` file with the required columns is written at the end.'
- What this solution (achieved 0.36305) has done: 'I fixed the import errors by removing the unavailable MONAI dependency, replaced the model‑based inference with a simple intensity‑threshold‑based mask generation, and rewrote the test data loading to use nibabel directly. This keeps the original data‑processing flow while ensuring the script runs end‑to‑end and writes a valid submission.csv file, moving the score away from 0.0 toward the target.'
- What this solution (achieved 0.01781) has done: 'I replace the fixed 0.5 intensity threshold with a data‑driven threshold based on the mean intensity of each volume (a simple adaptive rule that usually yields larger, more accurate masks). After merging predictions I also fill any missing RLE strings with an empty string so the submission format stays valid. These minimal tweaks keep the original pipeline but should raise the Dice/Hausdorff score toward the target.'
- What this solution (achieved 0.00604) has done: 'I replace the simple mean‑based intensity threshold with an Otsu‑derived threshold, which usually yields a better separation between foreground and background while keeping the overall pipeline unchanged. I add a small helper `otsu_threshold` (in the same cell where nibabel is imported) and use it in the inference loop (cell 15) instead of `vol.mean()`. This modest change should raise the Dice/Hausdorff score toward the target without altering the core model logic.'
- What this solution (achieved 0.00621) has done: 'I add a lightweight helper that computes an Otsu threshold for each slice separately and use these per‑slice thresholds instead of a single volume‑wide threshold. This keeps the overall pipeline unchanged while giving a finer‑grained mask that should raise the Dice/Hausdorff score, moving the result toward the target. The rest of the script remains the same, and the final CSV is still written as before.'
- What this solution (achieved 0.00632) has done: 'I keep the overall pipeline unchanged but give each organ its own mask instead of using the identical mask for all three classes. By applying slight offsets to the per‑slice Otsu thresholds we obtain class‑specific binary masks, which should increase overlap with the true organ shapes and move the Dice‑Hausdorff score closer to the target. The rest of the code (loading, RLE encoding, CSV writing) stays the same.'
- What this solution (achieved 0.00562) has done: 'I simplify the thresholding by using the same Otsu‑derived slice threshold for all three organs (removing the class‑specific multipliers) and then apply a lightweight 3‑D binary dilation via a max‑pool operation to slightly enlarge the masks. This keeps the overall pipeline unchanged while providing larger, more complete predictions that should raise the Dice/Hausdorff score toward the target.'
- What this solution (achieved 0.00613) has done: 'I keep the overall pipeline unchanged but give each organ its own mask instead of sharing a common one. By applying small class‑specific offsets to the per‑slice Otsu thresholds (lower for large bowel, higher for stomach) the binary masks become more organ‑specific, which should noticeably raise the Dice/Hausdorff score while still using the same lightweight inference and post‑processing steps.'
- What this solution (achieved 0.03358) has done: 'I correct the volume orientation so that the first dimension truly represents the slice index (by transposing the NIfTI data) and remove the class‑specific threshold offsets, letting each organ use the same Otsu‑derived slice thresholds. This small fix keeps the overall pipeline unchanged while aligning masks with the expected ordering, which should raise the Dice/Hausdorff score toward the target.'
- What this solution (achieved 0.02384) has done: 'The update adds small negative offsets for each organ to produce slightly larger masks and enlarges the binary dilation kernel from 3 to 5 voxels, both of which should increase overlap with the true segmentations and move the Dice / Hausdorff score upward while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob, os, gc




## === cell 1
sub_df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
sub = len(sub_df) > 0




## === cell 2
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"
if sub:
    path_csv = os.path.join(DATASET_FOLDER, "sample_submission.csv")
    df_train = pd.read_csv(path_csv)
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
import nibabel as nib


def otsu_threshold(volume):
    """
    Compute Otsu's threshold for a normalized 3‑D volume (values in [0, 1]).
    """
    hist, bin_edges = np.histogram(volume.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean1 = np.cumsum(hist * bin_mids) / (weight1 + 1e-8)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / (weight2 + 1e-8))[::-1]
    var_between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(var_between)
    return bin_edges[idx]




## === cell 7
def slice_otsu_thresholds(volume):
    """
    Compute an Otsu threshold for each slice (2‑D image) in a 3‑D volume.
    Returns an array of shape (S,) where S is the number of slices.
    Falls back to the slice mean when variance is extremely low.
    """
    S = volume.shape[0]
    thr = np.empty(S, dtype=np.float32)
    for s in range(S):
        slice_img = volume[s]
        if np.std(slice_img) < 1e-3:
            thr[s] = slice_img.mean()
        else:
            thr[s] = otsu_threshold(slice_img)
    return thr




## === cell 8
from PIL import Image


def load_image_volume(img_dir, quant=0.01):
    imgs = sorted(glob.glob(os.path.join(img_dir, f"*.png")))
    imgs = [np.array(Image.open(p)).tolist() for p in imgs]
    vol = np.array(imgs)
    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = np.min(vol), np.max(vol)
    vol = (vol - v_min) / (v_max - v_min + 1e-8)
    vol = (vol * 255).astype(np.uint8)
    del imgs
    gc.collect()
    return vol




## === cell 9
df_train_overview["vol_path"] = ""
for i in range(len(df_train_overview)):
    CASE = df_train_overview.at[i, "Case"]
    DAY = df_train_overview.at[i, "Day"]
    IMAGE_FOLDER = os.path.join(
        DATASET_FOLDER, folder, f"case{CASE}", f"case{CASE}_day{DAY}", "scans"
    )
    vol = load_image_volume(img_dir=IMAGE_FOLDER)
    print(f"Volume shape for case {CASE} day {DAY}: {vol.shape}")
    nii1 = nib.Nifti1Image(vol, affine=None)
    nii_path = f"./{CASE}_{DAY}_vol.nii.gz"
    df_train_overview.at[i, "vol_path"] = nii_path
    nib.save(nii1, nii_path)




## === cell 10
df_train_overview




## === cell 11
test_data = []
for i in range(len(df_train_overview)):
    CASE = df_train_overview.at[i, "Case"]
    DAY = df_train_overview.at[i, "Day"]
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
json_string = json.dumps(data1, indent=2)
with open("json_data.json", "w") as outfile:
    outfile.write(json_string)




## === cell 13
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 14
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




## === cell 15
def segm_rle(segm, df_vol):
    """
    Convert a 4‑D segmentation tensor (C, S, H, W) into a dataframe
    with RLE strings for each slice.
    """
    df = pd.DataFrame(columns=["id", "class", "predicted"])
    df_vol = df_vol.replace(np.nan, "")
    lbs = sorted(df_vol["class"].unique())
    for slice_id, dfg in df_vol.groupby("Slice"):
        slice_idx = int(slice_id) - 1  # slice indices are 0‑based in the tensor
        for _, row in dfg.iterrows():
            label = row["class"]
            lb_index = lbs.index(label)  # map class to channel index
            mask = segm[lb_index, slice_idx, :, :]
            df.loc[len(df)] = {
                "id": row["id"],
                "class": label,
                "predicted": rle_encode(mask > 0.5),
            }
    gc.collect()
    return df




## === cell 16
import torch.nn.functional as F  # for lightweight dilation

class_offsets = {
    "large_bowel": -0.05,
    "small_bowel": -0.03,
    "stomach": -0.02,
}

pred_df = pd.DataFrame(columns=["id", "class", "predicted"])

for idx, row in df_train_overview.iterrows():
    vol_path = row["vol_path"]
    CASE = row["Case"]
    DAY = row["Day"]

    nii = nib.load(vol_path)
    vol_raw = nii.get_fdata()
    vol = np.transpose(vol_raw, (2, 0, 1)).astype(np.float32)

    vol = (vol / (np.max(vol) + 1e-8)).astype(np.float32)

    slice_thr = slice_otsu_thresholds(vol)  # (S,)

    masks = []
    for organ in ["large_bowel", "small_bowel", "stomach"]:
        offset = class_offsets[organ]  # organ‑specific offset
        adj_thr = np.clip(slice_thr + offset, 0.0, 1.0)  # keep thresholds in [0,1]
        mask = (vol > adj_thr[:, None, None]).astype(np.float32)
        masks.append(mask)

    segm = np.stack(masks, axis=0)  # (C=3, S, H, W)

    segm_tensor = torch.from_numpy(segm).unsqueeze(0)  # shape (1, C, S, H, W)

    segm_dilated = F.max_pool3d(segm_tensor, kernel_size=5, stride=1, padding=2)
    segm = segm_dilated.squeeze(0).numpy()

    df_slice = df_train[(df_train["Case"] == CASE) & (df_train["Day"] == DAY)]

    pred_df = pd.concat([pred_df, segm_rle(segm, df_slice)], ignore_index=True)

    gc.collect()
    print(f"Processed case {CASE} day {DAY}")




## === cell 17
if sub:
    sub_template = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
else:
    sub_template = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))[:1000]
    sub_template = sub_template.rename(columns={"segmentation": "predicted"})

if "predicted" in sub_template.columns:
    del sub_template["predicted"]

submission = sub_template.merge(pred_df, on=["id", "class"], how="left")
submission["predicted"] = submission["predicted"].fillna("")
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
