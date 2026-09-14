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

0.8582719138113138

# 6. Current score

0.21815

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fixed the filename parsing so image dimensions are read directly from the file, corrected the metadata extraction, ensured the sample‑submission columns are stripped of whitespace, replaced the dummy predictor with a simple intensity‑based binary mask (so predictions are no longer all zeros), and added the column‑strip step before merging. These changes resolve the crashes, produce a valid submission.csv, and give a non‑trivial score that moves toward the target.'
- What this solution (achieved 0.25464) has done: 'I replace the simple fixed‑threshold in `dummy_predict` with an Otsu‑based threshold that automatically adapts to each slice’s intensity distribution. This generate non‑empty masks for most images, giving a non‑zero Dice score and moving the evaluation metric toward the target without altering any other logic.'
- What this solution (achieved 0.24734) has done: 'I keep the overall pipeline unchanged but improve the binary mask generation and post‑processing.  
- In `dummy_predict` I add a small Gaussian blur before Otsu and a morphological closing step to produce cleaner masks.  
- In the prediction loop I remove the square‑padding step and directly resize the cleaned mask to the original image size, which better preserves the organ shapes.  
These minimal adjustments keep the model logic identical while giving the masks higher quality, which should raise the Dice‑based score toward the target.'
- What this solution (achieved 0.0) has done: 'I keep the overall pipeline unchanged but improve the dummy predictor to output a soft intensity‑based probability map instead of a hard Otsu binary mask, and lower the post‑prediction threshold from 0.4 to 0.3 so that more pixels are kept as positive. This small change provides richer predictions while preserving the existing logic, and is expected to raise the Dice / Hausdorff‑combined score toward the target.'
- What this solution (achieved 0.24734) has done: 'The update replaces the fixed‑threshold predictor with an Otsu‑based threshold (after a light Gaussian blur) so each slice gets an adaptive binary mask. This produces richer, more accurate segmentations while keeping the original pipeline unchanged, helping the Dice/Hausdorff score move toward the target. The rest of the code, including post‑processing and CSV generation, remains the same.'
- What this solution (achieved 0.25583) has done: 'I adjust the dummy predictor to compute the Otsu mask on the original‑resolution image (rather than on the already‑resized version) and then down‑scale that mask to the network size. This preserves more structural detail before the final resize back to the true image dimensions. I also tighten the post‑processing by applying an opening before the closing and use a 0.5 threshold on the logits (the masks are already binary), which should increase Dice while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.25583) has done: 'I lowered the binary‑threshold used when converting the dummy predictor’s probability maps into masks from 0.5 to 0.3, which makes the masks slightly larger and typically improves both Dice and Hausdorff components, moving the score nearer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.20813) has done: 'I keep the overall pipeline unchanged but make two small, score‑positive tweaks:  
1. In `dummy_predict` I add a dilation (kernel 5×5) after the Otsu mask, then a closing operation. This slightly enlarges the organ masks, which usually raises the Dice component without breaking the existing logic.  
2. When converting the logits to binary masks I drop the 0.3 threshold and simply use `(logits > 0)` – the logits are already 0/1 probability maps, so the extra threshold is unnecessary and can discard true positives.  

These minimal changes stay within the original design while nudging the evaluation score toward the target.'
- What this solution (achieved 0.20969) has done: 'I replace the generic Otsu mask with a per‑class mask derived from connected components: after Otsu the binary mask is split into its individual components, the three largest components are assigned to large bowel, small bowel and stomach (largest → large, second → small, third → stomach). This keeps the overall pipeline unchanged while providing more informative class‑specific predictions, which should raise the Dice / Hausdorff score toward the target.'
- What this solution (achieved 0.21815) has done: 'I keep the overall pipeline unchanged but make three small, score‑positive tweaks inside the dummy predictor: use a slightly stronger blur, enlarge the morphological kernels, and turn the binary masks into soft probability maps by weighting them with the normalized blurred intensity. These changes preserve the original logic while giving richer, slightly larger predictions that should raise the Dice‑and‑Hausdorff combined score toward the target.'

# 9. Code solution

## === cell 0
import sys
import gc
import re
from pathlib import Path
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import cv2 as cv
import torch
import torch.nn.functional as F
from fastai.vision.all import get_image_files, ProgressCallback, progress_bar




## === cell 1
class Metadata:
    def __init__(self, sample_id: str, full_path: str, h: int, w: int):
        self.sample_id = sample_id
        self.full_path = full_path
        self.h = h
        self.w = w

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        case_day = re.search(r"case\d+_day\d+", str(path)).group()
        slice_match = re.search(r"slice_(\d{4})", path.stem)
        slice_no = slice_match.group(1) if slice_match else "0000"
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        h, w = img.shape[:2]
        sample_id = f"{case_day}_slice_{int(slice_no):04d}"
        return cls(sample_id, str(path), h, w)




## === cell 2
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation")

sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
sample_sub.columns = sample_sub.columns.str.strip()
TEST_IDS = sample_sub["id"].drop_duplicates().tolist()

TEST_FILES = get_image_files(DATA_DIR / "test")
METADATA = {m.sample_id: m for m in [Metadata.extract(p) for p in TEST_FILES]}




## === cell 3
def get_case_day(s: Path) -> str:
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s: Path, as_number=False):
    slice_no = re.search(r"slice_\d{4}", str(s)).group()
    return int(slice_no.split("_")[-1]) if as_number else slice_no


def get_sample_id(s: Path) -> str:
    return f"{get_case_day(s)}_{get_slice(s)}"


def group_case_day_from_files(image_files):
    groups = defaultdict(list)
    for fn in image_files:
        groups[get_case_day(fn)].append(fn)
    groups = {
        k: sorted(v, key=partial(get_slice, as_number=True)) for k, v in groups.items()
    }
    return groups


def packed(groups, n_slices_to_merge=3, step_size=2):
    assert n_slices_to_merge % 2 != 0, "n_slices_to_merge must be odd"
    mid_idx = n_slices_to_merge // 2
    chunks = []
    for case_day, files in groups.items():
        files = [None] + files + [None]
        for pack in [
            list(p) for p in zip(*[files[i:] for i in range(n_slices_to_merge)])
        ][::step_size]:
            for i, x in enumerate(pack):
                if x is not None:
                    first_real = x
                    break
            for i in range(len(pack)):
                if pack[i] is None:
                    pack[i] = first_real
                else:
                    break
            for i in range(len(pack) - 1, -1, -1):
                if pack[i] is not None:
                    last_real = pack[i]
                    break
            for i in range(len(pack) - 1, -1, -1):
                if pack[i] is None:
                    pack[i] = last_real
                else:
                    break
            chunks.append(pack)
    return chunks


def mask2rle(mask: np.ndarray) -> str:
    """Convert binary mask to RLE (space‑separated)."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def pad_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    padded = np.zeros((image_size, image_size), dtype=mask.dtype)
    dh = image_size - mask.shape[0]
    dw = image_size - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return padded




## === cell 4
paths = [Path(METADATA[id_].full_path) for id_ in TEST_IDS]
groups = group_case_day_from_files(paths)
packs = packed(groups, n_slices_to_merge=5, step_size=1)




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dummy_predict(batch):
    """
    Adaptive intensity‑based predictor with class‑specific soft masks:
    1. Read the central slice, apply a slightly stronger Gaussian blur (5×5) and Otsu.
    2. Dilate (7×7) and close (5×5) to enlarge structures.
    3. Keep up to three largest connected components and assign them to the
       classes (large → large_bowel, second → small_bowel, third → stomach).
    4. For each component create a soft probability map by weighting the binary
       mask with the normalized blurred intensity (preserves intensity information).
    5. Resize each soft map to 320×320 and return as a float tensor.
    """
    batch_size = len(batch)
    preds = torch.zeros((batch_size, 3, 320, 320), device=device, dtype=torch.float32)

    dilate_kernel = np.ones((7, 7), np.uint8)  # larger dilation
    close_kernel = np.ones((5, 5), np.uint8)  # larger closing

    for i, pack in enumerate(batch):
        img_path = pack[len(pack) // 2]  # central slice
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_blur = cv.GaussianBlur(img, (5, 5), 0)

        _, mask_orig = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        mask_dilated = cv.dilate(mask_orig, dilate_kernel, iterations=1)
        mask_closed = cv.morphologyEx(mask_dilated, cv.MORPH_CLOSE, close_kernel)

        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
            mask_closed, connectivity=8
        )
        if num_labels <= 1:
            class_masks = [np.zeros_like(mask_closed, dtype=np.uint8) for _ in range(3)]
        else:
            component_areas = [
                (c, stats[c, cv.CC_STAT_AREA]) for c in range(1, num_labels)
            ]
            component_areas.sort(key=lambda x: x[1], reverse=True)
            class_masks = [np.zeros_like(mask_closed, dtype=np.uint8) for _ in range(3)]
            for cls_idx in range(min(3, len(component_areas))):
                comp_label = component_areas[cls_idx][0]
                class_masks[cls_idx][labels == comp_label] = 255

        intensity_norm = img_blur.astype(np.float32) / 255.0
        intensity_resized = cv.resize(
            intensity_norm, (320, 320), interpolation=cv.INTER_LINEAR
        )

        for c_idx, cls_mask in enumerate(class_masks):
            mask_resized = cv.resize(
                cls_mask, (320, 320), interpolation=cv.INTER_NEAREST
            )
            mask_binary = (mask_resized / 255.0).astype(np.float32)

            prob = intensity_resized * mask_binary

            preds[i, c_idx] = torch.from_numpy(prob).to(device)

    return preds


preds = []

batch_size = 64
chunks = [packs[i : i + batch_size] for i in range(0, len(packs), batch_size)]

for subset in progress_bar(chunks):
    logits = dummy_predict(subset)
    labels = (logits > 0).cpu().numpy().astype(np.uint8)  # binary masks for post‑proc

    for pack, mask in zip(subset, labels):
        test_id = get_sample_id(pack[len(pack) // 2])
        meta = METADATA[test_id]
        h, w = meta.h, meta.w

        for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
            cls_mask = (mask[i] * 255).astype(np.uint8)

            kernel = np.ones((3, 3), np.uint8)
            cls_mask = cv.morphologyEx(cls_mask, cv.MORPH_OPEN, kernel)
            cls_mask = cv.morphologyEx(cls_mask, cv.MORPH_CLOSE, kernel)

            cls_mask = cv.resize(cls_mask, (w, h), interpolation=cv.INTER_NEAREST)
            cls_mask = (cls_mask > 0).astype(np.uint8)

            rle = mask2rle(cls_mask)
            preds.append({"id": test_id, "class": name, "predicted": rle})

    gc.collect()




## === cell 6
df_preds = pd.DataFrame(preds)

df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
df_submit.columns = df_submit.columns.str.strip()
df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left"
)

df_submit["predicted"] = df_submit["predicted"].fillna("")
df_submit.to_csv("submission.csv", index=False)

print(df_submit.head())
