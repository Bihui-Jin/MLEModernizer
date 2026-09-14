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

0.4484978484787068

# 6. Current score

0.23669

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23202) has done: 'I add a safe import for OpenCV (fallback to Pillow if unavailable) and apply a light Gaussian blur before Otsu thresholding to obtain slightly cleaner masks, which should modestly move the Dice‑Hausdorff score toward the target while keeping the core logic unchanged. The script also ensure the submission CSV is always written.'
- What this solution (achieved 0.23202) has done: 'I keep the overall pipeline but fix the class handling: instead of creating three identical predictions per image, I keep each original row (which already contains the correct organ class) and use its `class_name` when building the submission. This restores the proper id–class correspondence and should raise the Dice‑Hausdorff score toward the target while leaving the core image‑processing unchanged.'
- What this solution (achieved 0.2523) has done: 'I keep the overall pipeline unchanged but add a small post‑processing step that keeps only the largest connected component of each binary mask. This often removes spurious isolated pixels caused by thresholding and can raise the Dice‑Hausdorff score toward the target without altering the core logic. The change is limited to the mask‑creation loop in cell 8.'
- What this solution (achieved 0.21915) has done: 'I increase the morphological kernel size, add a mild dilation step, and remove the aggressive “keep only the largest connected component” filtering (which can discard valid organ fragments). These tweaks keep the overall pipeline unchanged while likely improving recall and thus the Dice‑Hausdorff score, moving it closer to the target. I also safely import Pillow’s ImageFilter for the fallback Gaussian blur.'
- What this solution (achieved 0.23669) has done: 'I shrink the morphological kernel to 3×3 (to avoid eroding thin organ structures) and add a lightweight post‑processing step that keeps only the largest connected component of each mask. This keeps the core thresholding pipeline unchanged while usually removing spurious isolated pixels, which should raise the Dice‑Hausdorff score toward the target.'
- What this solution (achieved 0.23352) has done: 'I slightly reduce the Gaussian blur size to preserve finer edges and make the “keep‑largest‑component” step less aggressive: it only discard smaller fragments when the biggest component covers a reasonable portion of the mask (≥ 30 %). These minimal tweaks keep the overall pipeline unchanged while likely improving Dice and Hausdorff scores, moving the metric closer to the target.'
- What this solution (achieved 0.21942) has done: 'I slightly enlarge the morphological kernel to 5×5 to close gaps more effectively and make the “keep‑largest‑component” filter much less aggressive by requiring the dominant component to cover ≥ 80 % of the mask before replacing it. These minimal tweaks preserve the overall pipeline while likely increasing the Dice‑Hausdorff score, moving it closer to the target.'
- What this solution (achieved 0.22562) has done: 'I reduce the morphological kernel from 5×5 to 3×3 to preserve thin organ structures and relax the “keep‑largest‑component” filter to accept the dominant component when it covers ≥ 50 % of the mask (instead of 80 %). These minimal adjustments keep the core pipeline unchanged while making the masks less eroded and more consistent, which should raise the Dice‑Hausdorff score toward the target.'
- What this solution (achieved 0.23388) has done: 'I lower the dominance threshold for keeping only the largest connected component from 0.50 to 0.20, so the filter is less aggressive and retains more of the predicted mask. This small change is expected to raise the Dice‑Hausdorff score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.00427) has done: 'The changes slightly enlarge the Gaussian blur and morphological kernel, replace the global Otsu threshold with a local adaptive threshold (when OpenCV is available), and make the “keep‑largest‑component” filter less aggressive (dominance ≥ 0.10). These adjustments keep the overall pipeline intact while providing a modest boost in mask quality, aiming to raise the Dice‑Hausdorff score toward the target.'
- What this solution (achieved 0.23669) has done: 'I replace the adaptive thresholding with a standard Otsu binarization (which is more stable for these gray‑scale scans) and make the morphological processing a bit lighter – a 3×3 kernel and a less aggressive “keep‑largest‑component” rule (dominance ≥ 0.05). These minimal tweaks keep the overall pipeline unchanged while giving the masks a chance to better match the organs, moving the Dice‑Hausdorff score toward the target.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
from glob import glob

try:
    import cv2

    _cv2_available = True
except ImportError:
    _cv2_available = False
    from PIL import Image, ImageFilter  # safe fallback for Gaussian blur



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
if DEBUG:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
else:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

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
path_partial_list = []
for i in range(len(all_train_images)):
    path_partial_list.append(str(all_train_images[i].rsplit("_", 4)[0]))

tmp_df = pd.DataFrame()
tmp_df["path_partial"] = path_partial_list
tmp_df["path"] = all_train_images

df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])
df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
del x, path_partial_list, tmp_df
df.head(5)



## === cell 4
df_train = df[
    ["id", "path", "class_name", "case", "day", "slice", "width", "height"]
].copy()
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
    """
    img: 2‑D numpy array (uint8) where 1 = mask, 0 = background
    Returns run‑length encoding as a space‑separated string.
    Empty mask returns an empty string.
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 8
ids = []
classes = []
rles = []

kernel = np.ones((3, 3), np.uint8)

for _, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
    h = row["height"]
    w = row["width"]
    img_path = row["path"]
    if _cv2_available:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        try:
            pil_img = Image.open(img_path).convert("L")
            img = np.array(pil_img)
        except Exception:
            img = None

    if img is None:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        if img.shape != (h, w):
            if _cv2_available:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                img = np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))

        if _cv2_available:
            img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            img_blur = np.array(
                Image.fromarray(img).filter(ImageFilter.GaussianBlur(2))
            )

        if _cv2_available:
            _, mask = cv2.threshold(img_blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            hist, bin_edges = np.histogram(img_blur.ravel(), bins=256, range=(0, 256))
            total = img_blur.size
            sumB, wB, maximum, thresh = 0.0, 0.0, 0.0, 0
            sum1 = np.dot(np.arange(256), hist)
            for i in range(256):
                wB += hist[i]
                if wB == 0:
                    continue
                wF = total - wB
                if wF == 0:
                    break
                sumB += i * hist[i]
                mB = sumB / wB
                mF = (sum1 - sumB) / wF
                between = wB * wF * (mB - mF) ** 2
                if between >= maximum:
                    thresh = i
                    maximum = between
            mask = (img_blur > thresh).astype(np.uint8)

        mask = (
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) if _cv2_available else mask
        )
        mask = (
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) if _cv2_available else mask
        )
        mask = cv2.dilate(mask, kernel, iterations=1) if _cv2_available else mask

        if _cv2_available:
            num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))
            if num_labels > 1:
                component_sizes = np.bincount(labels_im.flatten())[1:]  # skip bg
                largest_label = component_sizes.argmax() + 1
                largest_size = component_sizes.max()
                total_mask = mask.sum()
                if total_mask > 0 and (largest_size / total_mask) >= 0.05:
                    mask = (labels_im == largest_label).astype(np.uint8)

    ids.append(row["id"])
    classes.append(row["class_name"])
    rles.append(rle_encode(mask))



## === cell 9
submission = pd.DataFrame({"id": ids, "class": classes, "predicted": rles})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with shape:", submission.shape)
