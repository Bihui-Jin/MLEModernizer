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

albumentations==2.0.8
cupy-cuda12x==13.6.0
fastai==2.8.5
more-itertools==10.7.0
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scipy==1.15.3
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.841204308399207

# 6. Current score

0.21444

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.25395) has done: 'I fixed the metadata extraction to handle filenames that don’t start with integer dimensions and made the sample‑ID generation consistent with the IDs used in the CSV files. This prevents the `ValueError` during metadata creation and ensures `metadata_dict` is correctly built, allowing the later prediction loop and submission generation to run without errors.'
- What this solution (achieved 0.37276) has done: 'I add a light preprocessing step (CLAHE contrast‑enhancement) and switch to adaptive Gaussian thresholding with a small morphological opening. These inexpensive tweaks keep the original pipeline but usually yield cleaner binary masks, which should raise the Dice/Hausdorff combined score toward the target without altering the overall architecture.'
- What this solution (achieved 0.25651) has done: 'I replace the adaptive‑threshold step with a Gaussian‑blur + Otsu threshold, use a slightly larger elliptical structuring element for opening (to remove noise) and add a simple inversion guard (if the mask covers most of the image we likely got the background). These tweaks keep the same overall pipeline but usually produce cleaner binary masks, which should raise the Dice / Hausdorff combined score toward the target.'
- What this solution (achieved 0.25324) has done: 'I add a modest post‑processing routine that removes tiny isolated regions, applies a closing operation to fill small holes, and relaxes the inversion‑guard threshold. These steps keep the original pipeline but usually produce cleaner binary masks, which should raise the Dice/Hausdorff combined score toward the target. I also ensure the submission column never contains NaN values.'
- What this solution (achieved 0.2417) has done: 'I tighten the post‑processing to keep only the largest connected component (organs are usually the biggest region), raise the minimum area for noise removal, and set the inversion guard back to a 0.5 threshold. These small tweaks stay within the original pipeline while cleaning masks, which should raise the Dice/Hausdorff combined score toward the target.'
- What this solution (achieved 0.18238) has done: 'The changes tighten the preprocessing and post‑processing: larger morphological kernels, a higher minimum component size, hole‑filling, and an adaptive Gaussian threshold (which usually captures organ regions better than plain Otsu). These adjustments stay within the original pipeline while making the binary masks cleaner, which should raise the Dice/Hausdorff combined score toward the target.'
- What this solution (achieved 0.18277) has done: 'I make the post‑processing less aggressive so that true organ pixels are less likely to be removed. By shrinking the morphological kernels, lowering the minimum component size, and disabling the “keep only largest component” step, the binary masks retain more detail which should raise the Dice/Hausdorff combined score toward the target. The core pipeline (CLAHE, blur, adaptive threshold) remains unchanged.'
- What this solution (achieved 0.17009) has done: 'I make modest but targeted tweaks to the existing pipeline: strengthen contrast (higher CLAHE clip limit), use a slightly larger adaptive‑threshold window, increase the minimum component size, and reactivate the “keep largest component” step (which usually isolates the organ and removes noise). These changes stay within the original workflow yet should produce cleaner binary masks, moving the Dice/Hausdorff score upward toward the target.'
- What this solution (achieved 0.21444) has done: 'I modestly improve the classical pipeline by strengthening contrast (higher CLAHE clip limit), switching to Otsu thresholding (which often separates organ tissue better than the current adaptive Gaussian threshold), and loosening the small‑component filter so that genuine organ pixels are kept. These small, targeted tweaks preserve the overall workflow while expected to raise the Dice/Hausdorff combined score toward the target.'

# 9. Code solution

## === cell 0
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
from fastai.vision.all import get_image_files
import scipy.ndimage as ndimage  # added for hole‑filling




## === cell 1
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation")
test_df = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test_df["id"].tolist()
test_classes = test_df["class"].tolist()




## === cell 2
class Metadata:
    """Simple container linking a slice id to its image file and size."""

    def __init__(self, sample_id: str, full_path: str, h: int, w: int):
        self.sample_id = sample_id
        self.full_path = full_path
        self.h = h
        self.w = w

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        """
        Create a Metadata instance from an image path.
        Handles filenames that may not start with numeric height/width.
        """
        parts = path.stem.split("_")
        h = w = None
        if len(parts) >= 2:
            try:
                h, w = int(parts[0]), int(parts[1])
            except ValueError:
                h = w = None

        if h is None or w is None:
            img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
            else:
                h = w = 0  # unknown size – will be handled later

        sample_id = get_sample_id(path)

        return cls(sample_id, str(path), h, w)


def get_case_day(s: Path) -> str:
    import re

    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s: Path, as_number=False):
    import re

    m = re.search(r"slice_\d{4}", str(s))
    if not m:
        return None
    slice_no = m.group().split("_")[-1]
    return int(slice_no) if as_number else m.group()


def get_sample_id(path: Path) -> str:
    """
    Returns the identifier used in the CSV files, e.g.
    'case101_day20_slice_0001'.
    """
    return f"{get_case_day(path)}_{get_slice(path)}"




## === cell 3
test_image_paths = get_image_files(DATA_DIR / "test")
metadata_dict = {get_sample_id(p): Metadata.extract(p) for p in test_image_paths}




## === cell 4
def mask2rle(mask: np.ndarray) -> str:
    """
    Convert a binary mask (numpy 2‑D array) to run‑length encoding string.
    Pixels are read column‑wise (top‑to‑bottom, then left‑to‑right) as required by the competition.
    """
    flat = mask.T.flatten()  # transpose to get column‑wise order
    padded = np.concatenate([[0], flat, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 5
clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))

kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))


def clean_small_components(mask: np.ndarray, min_area: int = 300) -> np.ndarray:
    """
    Remove connected components smaller than *min_area* pixels.
    A modest min_area discards tiny specks while keeping genuine organ tissue.
    """
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        (mask * 255).astype(np.uint8), connectivity=8
    )
    clean_mask = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):  # skip background label 0
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 1
    return clean_mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Retain only the largest connected component.
    Organs typically appear as a single region per slice, so this helps drop residual noise.
    """
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        (mask * 255).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def postprocess_mask(binary: np.ndarray) -> np.ndarray:
    """
    Morphological closing → keep largest component → remove tiny objects → fill holes
    → gentle inversion guard.
    """
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel_close)

    binary = keep_largest_component(binary)

    binary = clean_small_components(binary, min_area=300)

    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)

    if binary.mean() > 0.5:
        binary = 1 - binary
    return binary


preds = []
for idx, row in test_df.iterrows():
    slice_id = row["id"]
    cls_name = row["class"]
    meta = metadata_dict.get(slice_id)
    if meta is None:
        rle = ""
    else:
        img = cv.imread(meta.full_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            rle = ""
        else:
            img_eq = clahe.apply(img)

            img_blur = cv.GaussianBlur(img_eq, (5, 5), 0)

            _, binary = cv.threshold(img_blur, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)

            binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel_open)

            binary = postprocess_mask(binary.astype(np.uint8))

            rle = mask2rle(binary.astype(np.uint8))
    preds.append({"id": slice_id, "class": cls_name, "predicted": rle})

df_preds = pd.DataFrame(preds)




## === cell 6
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
df_submit = df_submit.drop(columns=["predicted"]).merge(
    df_preds, on=["id", "class"], how="left"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")
df_submit.to_csv("submission.csv", index=False)

print(df_submit.head())
