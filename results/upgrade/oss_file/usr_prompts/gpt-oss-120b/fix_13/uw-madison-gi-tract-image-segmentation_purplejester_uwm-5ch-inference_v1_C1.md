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

0.8427207142594262

# 6. Current score

0.00238

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the submission‑file creation bug. The merge currently leaves the original “predicted” column untouched, causing pandas to create “predicted_x”/“predicted_y” and the code later fails when it expects a single “predicted” column. I drop any existing “predicted” column before merging, then merge and safely fill missing values, ensuring a proper CSV is written. This change is minimal, keeps the core model logic untouched, and allow the script to produce a valid `submission.csv` so the score can be evaluated.'
- What this solution (achieved 0.46338) has done: 'Implemented a lightweight fallback mask generator for cases where the pretrained model cannot be loaded.  
- Added a simple central‑square mask (non‑zero) when `learn` is `None`, ensuring predictions contain meaningful RLE strings instead of empty masks.  
- Kept the original inference path unchanged when the model loads successfully.  
- This minimal change preserves core logic while producing a valid submission that yields a non‑zero score, moving the result toward the target.'
- What this solution (achieved 0.37074) has done: 'I keep the overall pipeline unchanged but adjust two areas that can raise the Dice‑Hausdorff score.  
First, the binary threshold applied to the model’s sigmoid output is set to a more conventional 0.5 (the previous 0.4 was overly permissive).  
Second, for the fallback case where the pretrained model cannot be loaded, I replace the tiny central‑square mask with a simple Otsu‑based threshold on each slice (resized to 320×320). This produces a more realistic mask shape and therefore a higher overlap with the ground‑truth masks, moving the score toward the target.'
- What this solution (achieved 0.37242) has done: 'I keep the overall pipeline unchanged but improve the fallback mask generation used when the pretrained model cannot be loaded. Instead of a single Otsu‑threshold mask duplicated for all three classes, I replace it with a lightweight 4‑cluster k‑means segmentation on each slice, assign the three non‑background clusters to the three organ classes, and keep the rest of the code identical. This change preserves the core logic while providing more realistic, class‑specific masks, which should raise the Dice‑Hausdorff score toward the target.'
- What this solution (achieved 0.37074) has done: 'I replace the k‑means fallback segmentation with a simple Otsu threshold on the centre slice (resized to 320×320). The Otsu mask is more realistic than the previous clustering, and the same binary mask is duplicated for the three organ classes. This small change keeps the overall pipeline unchanged while improving the quality of the fallback masks, moving the Dice‑Hausdorff score upward toward the target.'
- What this solution (achieved 0.32449) has done: 'I keep the overall pipeline unchanged but improve the fallback mask generation used when the pretrained model cannot be loaded.  
Instead of copying the same binary Otsu mask to all three organ classes, I create class‑specific masks by extracting the largest connected components from the Otsu mask (largest → large_bowel, second largest → small_bowel, third largest → stomach).  
Each component is refined with a Gaussian blur and a keep‑largest‑component step, then padded and resized as before. This modest change keeps the core logic intact while producing more realistic, class‑specific masks, which should raise the Dice‑Hausdorff score toward the target.'
- What this solution (achieved 0.00462) has done: 'I add a safe search for any available ``.pkl`` model file in the input directory and load it if found (instead of the hard‑coded missing path), and I improve the fallback mask generation by using an adaptive Gaussian threshold (which usually captures organ structures better than a global Otsu threshold). These minimal tweaks keep the overall pipeline unchanged while providing higher‑quality masks when the pretrained model is unavailable, moving the score upward toward the target.'
- What this solution (achieved 0.23413) has done: 'I replace the adaptive‑threshold fallback with an Otsu‑based approach that extracts the three largest connected components and assigns them to the three organ classes. This simple change keeps the overall pipeline untouched but produces more realistic masks when the pretrained model is unavailable, moving the score significantly closer to the target.'
- What this solution (achieved 0.23413) has done: 'I broaden the model‑loading step so that any pretrained ``.pkl`` file anywhere under the Kaggle ``/kaggle/input`` directory is found and loaded. This small change keeps the core inference pipeline unchanged but allows the pretrained model (which gives far better predictions than the simple Otsu fallback) to be used when it exists, moving the score much closer to the target. No other logic is altered.'
- What this solution (achieved 0.00238) has done: 'I replace the simple Otsu‑based fallback mask generator with a lightweight K‑means segmentation that creates three distinct class‑specific masks (large_bowel, small_bowel, stomach). This keeps the overall pipeline unchanged, improves mask realism when the pretrained model cannot be loaded, and is expected to raise the Dice/Hausdorff score toward the target.'

# 9. Code solution

## === cell 0
import logging, sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from fastai.vision.all import *
from more_itertools import windowed, chunked
from skimage.morphology import disk
from scipy.ndimage import binary_opening
from dataclasses import dataclass
import re
from collections import defaultdict
import gc

logging.captureWarnings(True)




## === cell 1
@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int
    w: int

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        case_and_day = path.parents[1].stem
        _, slice_no, h, w, *_ = path.stem.split("_")
        sample_id = f"{case_and_day}_slice_{int(slice_no):04d}"
        return Metadata(sample_id, str(path), int(h), int(w))




## === cell 2
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation")
DEBUG = False

ids_path = DATA_DIR / ("train.csv" if DEBUG else "sample_submission.csv")
TEST_IDS = pd.read_csv(ids_path)["id"].drop_duplicates().tolist()

files_path = DATA_DIR / ("train" if DEBUG else "test")
TEST_FILES = get_image_files(files_path)

METADATA = {m.sample_id: m for m in TEST_FILES.map(Metadata.extract)}




## === cell 3
def get_size(s: Path):
    parts = s.stem.split("_")
    return int(parts[-2]), int(parts[-1])  # h, w


def get_case_day(s: Path):
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s: Path, as_number=False):
    slice_no = re.search(r"slice_(\d{4})", str(s)).group(1)
    return int(slice_no) if as_number else f"slice_{slice_no}"


def get_sample_id(s: Path):
    return f"{get_case_day(s)}_{get_slice(s)}"


def group_case_day_from_files(image_files):
    groups = defaultdict(list)
    for fn in image_files:
        groups[get_case_day(fn)].append(fn)
    groups = {
        k: sorted(v, key=lambda x: int(re.search(r"slice_(\d{4})", str(x)).group(1)))
        for k, v in groups.items()
    }
    return groups


def packed(groups, n_slices_to_merge=3, step_size=2):
    assert n_slices_to_merge % 2 != 0
    mid_idx = n_slices_to_merge // 2
    chunks = []
    for case_day, files in groups.items():
        files = [None] + files + [None]
        for pack in windowed(files, n=n_slices_to_merge, step=step_size):
            pack = list(pack)
            mid = pack[mid_idx]
            last_not_none = max(i for i, x in enumerate(pack) if x is not None)
            for i in range(last_not_none + 1, len(pack)):
                pack[i] = pack[last_not_none]
            first_not_none = min(i for i, x in enumerate(pack) if x is not None)
            for i in range(0, first_not_none):
                pack[i] = pack[first_not_none]
            chunks.append(pack)
    return chunks




## === cell 4
class CreateSample(Transform):
    def encodes(self, pack):
        h, w = get_size(pack[0])
        merged = np.ndarray((w, h, len(pack)), dtype=np.uint8)
        q = 0.01
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = np.min(img), np.max(img)
            img = (img - v_min) / float(v_max - v_min)
            img = (img * 255).astype(np.uint8)
            merged[:, :, i] = img
        return merged


class CreateTarget(Transform):
    def __init__(self, codes=(1, 2, 3)):
        super().__init__()
        self.codes = codes

    def encodes(self, pack):
        fn = get_y(pack)
        mask = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
        channels = [(mask == c).astype(np.uint8) for c in self.codes]
        return np.dstack(channels)

    def decodes(self, mask):
        return mask * 255




## === cell 5
model_path = None
search_root = Path("/kaggle/input")
for p in search_root.rglob("*.pkl"):
    model_path = p
    break

if model_path is not None:
    try:
        learn = load_learner(model_path)
        logging.info(f"Loaded pretrained model from {model_path}")
    except Exception as e:
        logging.warning(f"Could not load pretrained model at {model_path}: {e}")
        learn = None
else:
    logging.info("No .pkl model file found in the input directory.")
    learn = None




## === cell 6
predicted_ids = TEST_IDS  # no special debug filtering needed
len(predicted_ids)




## === cell 7
paths = [Path(METADATA[id_].full_path) for id_ in predicted_ids]




## === cell 8
packs = packed(group_case_day_from_files(paths), n_slices_to_merge=5, step_size=1)




## === cell 9
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")




## === cell 10
def mask2rle(mask):
    """Convert a binary mask to run‑length encoding."""
    pixels = mask.flatten()
    pads = np.array([0])
    pixels = np.concatenate([pads, pixels, pads])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def pad_mask(mask, image_size):
    padded = np.zeros((image_size, image_size), dtype=mask.dtype)
    dh = image_size - mask.shape[0]
    dw = image_size - mask.shape[1]
    top, left = dh // 2, dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return padded




## === cell 11
def keep_largest_component(binary_mask):
    """Return a mask that keeps only the largest connected component."""
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return binary_mask  # no components found
    largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def generate_fallback_masks(img_resized):
    """
    Create three class‑specific masks using K‑means clustering.
    The three clusters with highest mean intensity are treated as
    large_bowel, small_bowel and stomach respectively.
    """
    Z = img_resized.reshape((-1, 1)).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    _, labels, centers = cv.kmeans(Z, K, None, criteria, 3, cv.KMEANS_PP_CENTERS)

    labels = labels.reshape(img_resized.shape)
    sorted_idx = np.argsort(-centers.squeeze())
    organ_cluster_ids = sorted_idx[1:4]  # three brightest clusters

    masks = np.zeros((3, img_resized.shape[0], img_resized.shape[1]), dtype=np.uint8)

    for i, cid in enumerate(organ_cluster_ids):
        comp_mask = (labels == cid).astype(np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        comp_mask = cv.morphologyEx(comp_mask, cv.MORPH_OPEN, kernel, iterations=1)
        masks[i] = comp_mask

    return masks


preds = []
batch_size = 64
chunks = list(chunked(packs, n=batch_size))

sigmoid_thresh = 0.5

for subset in progress_bar(chunks):
    if learn is not None:
        test_dl = learn.dls.test_dl(subset, batch_size=len(subset), device=device)
        logits, *_ = learn.get_preds(dl=test_dl, act=noop)
        labels = (
            (torch.sigmoid(logits) >= sigmoid_thresh).cpu().numpy().astype(np.uint8)
        )
    else:
        num_samples = len(subset)
        labels = np.zeros((num_samples, 3, 320, 320), dtype=np.uint8)
        for idx, pack in enumerate(subset):
            centre = len(pack) // 2
            img_path = pack[centre]
            img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv.resize(img, (320, 320), interpolation=cv.INTER_LINEAR)
            masks = generate_fallback_masks(img_resized)
            labels[idx] = masks

    for pack, mask in zip(subset, labels):
        centre = len(pack) // 2
        test_id = get_sample_id(pack[centre])
        meta = METADATA[test_id]
        h, w = cv.imread(meta.full_path, cv.IMREAD_UNCHANGED).shape[:2]

        for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
            cls_mask = pad_mask(mask[i], 320)
            cls_mask = cv.resize(cls_mask, (w, h), interpolation=cv.INTER_NEAREST)
            cls_mask = keep_largest_component(cls_mask)
            rle = mask2rle(binary_opening(cls_mask, structure=disk(5)))
            preds.append({"id": test_id, "class": name, "predicted": rle})

    gc.collect()




## === cell 12
df_preds = pd.DataFrame(preds)

df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")

if "predicted" in df_submit.columns:
    df_submit = df_submit.drop(columns=["predicted"])

df_submit = df_submit.merge(df_preds, on=["id", "class"], how="left")

df_submit["predicted"] = df_submit["predicted"].fillna("")

df_submit.to_csv("submission.csv", index=False)
print("Submission file written. First rows:")
print(df_submit.head())
