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

No external packages required in the script and installed.

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

0.3428370399198315

# 6. Current score

0.00745

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the failing TensorFlow/Keras imports, replace the model loading with a dummy zero‑mask predictor, adjust the DataGenerator to use Keras’ Sequence class, and keep the rest of the pipeline unchanged so that a valid `submission.csv` is produced. This fixes the import errors, ensures `model` exists, generates predictions (all zeros) that match the expected shapes, and allows the RLE encoding loop to run without crashes, yielding a correctly formatted submission file.'
- What this solution (achieved 0.11059) has done: 'I remove the failing keras imports and replace the dummy all‑zero prediction with a simple intensity‑threshold mask generated directly from the input images. This fixes the import error and provides non‑trivial predictions, moving the score toward the target while preserving the original pipeline structure.'
- What this solution (achieved 0.01682) has done: 'I replace the naïve fixed‑threshold mask in the prediction loop with an Otsu‑based threshold that automatically adapts to each image’s intensity distribution. This small change keeps the overall pipeline unchanged while providing more realistic binary masks, which should raise the Dice score and move the overall metric closer to the target.'
- What this solution (achieved 0.00831) has done: 'I improve the mask generation by applying Otsu threshold individually to each colour channel of the low‑resolution image (after a light Gaussian blur) and then merge the three binary masks. This keeps the overall pipeline unchanged but gives a more detailed segmentation than the previous single‑channel Otsu on the grayscale mean, which should raise the Dice component and move the overall score toward the target.'
- What this solution (achieved 0.00831) has done: 'I adjust the prediction step so that each organ class receives its own mask derived from the Otsu threshold of a distinct colour channel (red → large_bowel, green → small_bowel, blue → stomach) instead of sharing a single combined mask. This small, targeted change keeps the overall pipeline unchanged while providing more class‑specific segmentations, which should raise the Dice component and move the overall score closer to the target.'
- What this solution (achieved 0.00803) has done: 'Implemented a lightweight morphological dilation on each Otsu‑derived channel mask to enlarge predicted regions slightly. This small post‑processing step tends to increase overlap with ground‑truth masks, thereby raising the Dice component and moving the overall score closer to the target without altering the core modeling logic.'
- What this solution (achieved 0.00837) has done: 'I slightly enhance the Otsu‑based mask creation by adding a morphological closing (dilate → erode) and a light opening to reduce noise, which should increase overlap with the ground‑truth masks and improve the Dice component. In addition, I upscale the masks with linear interpolation (instead of nearest‑neighbor) when resizing back to the original image size, giving smoother predictions. These minimal, targeted adjustments keep the overall pipeline unchanged while moving the score closer to the target.'
- What this solution (achieved 0.0078) has done: 'I keep the overall pipeline unchanged but improve the mask generation step: before Otsu thresholding each channel I apply histogram equalization to boost contrast, and I use a stronger morphological closing (2 iterations) followed by opening. This simple preprocessing usually yields larger, more accurate binary masks, which should increase the Dice component and move the overall score closer to the target while preserving the original logic.'
- What this solution (achieved 0.00755) has done: 'I replace the Otsu‑based thresholding in the prediction loop with a simple mean‑intensity threshold per colour channel (and keep a light morphological closing). This small change keeps the overall pipeline intact while producing larger, more realistic binary masks, which should increase the Dice component and move the score upward toward the target.'
- What this solution (achieved 0.00744) has done: 'The update switches the simple mean‑intensity threshold to Otsu’s adaptive threshold (which better separates foreground from background), adds a light dilation step to slightly enlarge the predicted regions, and switches the mask‑up‑sampling interpolation to nearest‑neighbor to keep the binary values clean. These small, targeted changes keep the original pipeline structure while improving mask quality, which should raise the Dice component and move the overall score closer to the target.'
- What this solution (achieved 0.00634) has done: 'I slightly enlarge the predicted masks to increase overlap with the ground‑truth masks, which should raise the Dice component and move the overall score closer to the target. The change is limited to the mask‑generation loop: a larger dilation kernel and extra dilation iterations are applied after the Otsu threshold and closing step for each colour channel. This keeps the core pipeline unchanged while making the predictions less sparse.'
- What this solution (achieved 0.00745) has done: 'I keep the overall pipeline unchanged but improve the mask‑generation step. After Otsu‑thresholding each colour channel I (1) apply a modest closing, (2) keep only the largest connected component (which removes spurious small regions) and (3) use a smaller dilation kernel to avoid over‑expanding the masks. These tweaks give more realistic organ shapes and increase overlap with the ground‑truth, moving the Dice‑based score toward the target while preserving the original logic.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import gc
import cv2
from glob import glob
from tqdm import tqdm




## === cell 1
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5




## === cell 2
df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
DEBUG = False
if df.shape[0] == 0:
    DEBUG = True
if DEBUG:
    df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
    df.pop("segmentation")
    df["predicted"] = ""




## === cell 3
df.rename(columns={"class": "class_name"}, inplace=True)
df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
df["slice"] = df["id"].apply(lambda x: x.split("_")[3])
TRAIN_DIR = (
    "../input/uw-madison-gi-tract-image-segmentation/train"
    if DEBUG
    else "../input/uw-madison-gi-tract-image-segmentation/test"
)

all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
x = all_train_images[0].rsplit("/", 4)[0]  # base path

path_partial_list = []
for i in range(df.shape[0]):
    path_partial_list.append(
        os.path.join(
            x,
            "case" + str(df["case"].values[i]),
            "case" + str(df["case"].values[i]) + "_" + "day" + str(df["day"].values[i]),
            "scans",
            "slice_" + str(df["slice"].values[i]),
        )
    )
df["path_partial"] = path_partial_list

path_partial_list = [p.rsplit("_", 4)[0] for p in all_train_images]
tmp_df = pd.DataFrame({"path_partial": path_partial_list, "path": all_train_images})
df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])

df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
del x, path_partial_list, tmp_df
df.head(5)




## === cell 4
df_train = pd.DataFrame(
    {
        "id": df["id"][::3].reset_index(drop=True),
        "path": df["path"][::3].values,
        "predicted": df["predicted"][::3].values,
        "case": df["case"][::3].values,
        "day": df["day"][::3].values,
        "slice": df["slice"][::3].values,
        "width": df["width"][::3].values,
        "height": df["height"][::3].values,
    }
)
df_train.reset_index(inplace=True, drop=True)
df_train.fillna("", inplace=True)
df_train.head(5)




## === cell 5
print(df_train.shape)
if DEBUG:
    df_train = df_train.sample(frac=0.05).reset_index(drop=True)
print(df_train.shape)




## === cell 6
gc.collect()




## === cell 7
def rle_encode(img):
    """Run‑length encoding for binary mask."""
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    """Decode RLE to mask."""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape(shape[:2])




## === cell 8
class DataGenerator:
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.subset = subset
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, 128, 128, 3), dtype=np.float32)
        y = np.empty((self.batch_size, 128, 128, 3), dtype=np.float32)
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        for i, idx in enumerate(batch_indexes):
            w = self.df.loc[idx, "width"]
            h = self.df.loc[idx, "height"]
            img_path = self.df.loc[idx, "path"]
            img = self._load_image(img_path)
            X[i] = img
            if self.subset == "train":
                for k, cls in enumerate(["large_bowel", "small_bowel", "stomach"]):
                    rle = self.df.loc[idx, cls]
                    mask = rle_decode(rle, shape=(h, w, 1))
                    mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
                    y[i, :, :, k] = mask.squeeze()
        return (X, y) if self.subset == "train" else X

    def _load_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img




## === cell 9
gc.collect()




## === cell 10
def keep_largest_component(mask):
    """Return a binary mask that keeps only the largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask * 255).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask  # no component found
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype(np.float32)
    return largest_mask


pred_generator = DataGenerator(df_train, batch_size=1, subset="test", shuffle=False)
LOGITS = np.zeros((len(df_train), 128, 128, 3), dtype=np.float32)

kernel_close = np.ones((3, 3), np.uint8)
kernel_dilate = np.ones((3, 3), np.uint8)

for i in range(len(pred_generator)):
    X_batch = pred_generator[i]  # (1,128,128,3)
    img = X_batch[0]  # (128,128,3) float32 in [0,1]

    img_uint8 = (img * 255).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img_uint8, (5, 5), 0)

    channel_masks = []
    for c in range(3):
        eq_channel = cv2.equalizeHist(img_blur[:, :, c])

        _, mask_c = cv2.threshold(
            eq_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask_c = (mask_c // 255).astype(np.float32)

        mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        mask_c = cv2.dilate(mask_c, kernel_dilate, iterations=1)

        mask_c = keep_largest_component(mask_c)

        channel_masks.append(mask_c)

    LOGITS[i] = np.stack(channel_masks, axis=-1)

gc.collect()




## === cell 11
print("Logits shape:", LOGITS.shape)




## === cell 12
lbs, sbs, sts = [], [], []
for idx in tqdm(range(len(df_train)), total=len(df_train)):
    h = df_train.loc[idx, "height"]
    w = df_train.loc[idx, "width"]
    for ch, lst in zip(range(3), [lbs, sbs, sts]):
        mask_resized = cv2.resize(
            LOGITS[idx, :, :, ch], (w, h), interpolation=cv2.INTER_NEAREST
        )
        mask_bin = np.rint(mask_resized).astype(np.uint8)
        lst.append(rle_encode(mask_bin))
del LOGITS
gc.collect()




## === cell 13
submission_ids = []
submission_classes = []
submission_rles = []
for idx, row in df_train.iterrows():
    submission_ids.extend([row["id"]] * 3)
    submission_classes.extend(["large_bowel", "small_bowel", "stomach"])
    submission_rles.extend([lbs[idx], sbs[idx], sts[idx]])

sub_df = pd.DataFrame(
    {"id": submission_ids, "class": submission_classes, "predicted": submission_rles}
)
sub_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with", len(sub_df), "rows")
