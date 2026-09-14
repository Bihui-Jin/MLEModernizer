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

0.8419090350968389

# 6. Current score

0.00494

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.00494) has done: 'I ensure the pipeline runs end‑to‑end by creating the working directory, dropping rows without a corresponding image path, and writing the submission file to the expected location while safely handling any missing predictions. These small fixes keep the original model and inference logic intact and produce a valid CSV ready for evaluation.'

# 9. Code solution

## === cell 0
from pathlib import Path
import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

try:
    import cupy as cp
except ImportError:
    import numpy as cp  # fallback to numpy with same API used below

KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DATA_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation"

IMG_SIZE = 356
CROP_SIZE = 320
USE_AUGS = True
BATCH_SIZE = 8
NUM_WORKERS = 0  # safe for notebooks / limited resources
ENCODER_NAME = "efficientnet-b3"
GPUS = 1
CHANNELS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THR = 0.45
DEBUG = False



## === cell 1
transforms_val = A.Compose(
    [A.CenterCrop(CROP_SIZE, CROP_SIZE, p=1.0), ToTensorV2(transpose_mask=True)]
)




## === cell 2
class UWDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def resize(self, img, interp):
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=interp)

    def load_slice(self, img_file, diff):
        slice_num = os.path.basename(img_file).split("_")[1]
        filename = img_file.replace(
            "slice_" + slice_num, "slice_" + str(int(slice_num) + diff).zfill(4)
        )
        if os.path.exists(filename):
            return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        imgs = [self.load_slice(row["image_path"], i) for i in range(-2, 3)]
        if imgs[3] is None:
            imgs[3] = imgs[2]
        if imgs[4] is None:
            imgs[4] = imgs[3]
        if imgs[1] is None:
            imgs[1] = imgs[2]
        if imgs[0] is None:
            imgs[0] = imgs[1]

        image = np.stack(imgs, axis=2).astype(np.float32)
        h, w = image.shape[:2]
        max_val = image.max()
        if max_val != 0:
            image /= max_val
        image = self.resize(image, cv2.INTER_AREA)

        if self.transforms:
            data = self.transforms(image=image)
            image = data["image"]
        return {"image": image, "id": row["id"], "h": h, "w": w}




## === cell 3
def extract_metadata_from_id(df):
    df[["case", "day", "slice"]] = df["id"].str.split("_", n=2, expand=True)
    df["case"] = df["case"].str.replace("case", "").astype(int)
    df["day"] = df["day"].str.replace("day", "").astype(int)
    df["slice"] = df["slice"].str.replace("slice_", "").astype(int)
    return df


def extract_metadata_from_path(path_df):
    path_df[["parent", "case_day", "scans", "file_name"]] = path_df[
        "image_path"
    ].str.rsplit("/", n=3, expand=True)
    path_df[["case", "day"]] = path_df["case_day"].str.split("_", expand=True)
    path_df["case"] = path_df["case"].str.replace("case", "")
    path_df["day"] = path_df["day"].str.replace("day", "")
    path_df[["slice", "width", "height", "spacing", "spacing_"]] = (
        path_df["file_name"]
        .str.replace("slice_", "")
        .str.replace(".png", "")
        .str.split("_", expand=True)
    )
    path_df = path_df.drop(
        columns=["parent", "case_day", "scans", "file_name", "spacing_"]
    )
    numeric_cols = ["case", "day", "slice", "width", "height", "spacing"]
    path_df[numeric_cols] = path_df[numeric_cols].apply(pd.to_numeric)
    return path_df




## === cell 4
sub_df = pd.read_csv(INPUT_DATA_DIR / "sample_submission.csv")
test_set_hidden = not bool(len(sub_df))

if test_set_hidden:
    test_df = pd.read_csv(INPUT_DATA_DIR / "train.csv")[:3000]
    test_df = test_df.drop(columns=["class", "segmentation"]).drop_duplicates()
    image_paths = [str(p) for p in (INPUT_DATA_DIR / "train").rglob("*.png")]
else:
    test_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()
    image_paths = [str(p) for p in (INPUT_DATA_DIR / "test").rglob("*.png")]

test_df = extract_metadata_from_id(test_df)

path_df = pd.DataFrame(image_paths, columns=["image_path"])
path_df = extract_metadata_from_path(path_df)

test_df = test_df.merge(path_df, on=["case", "day", "slice"], how="left")
test_df = test_df.dropna(subset=["image_path"])
test_df["image_path"] = test_df["image_path"].astype(str)
print("Prepared test set:", len(test_df))
test_df.head()



## === cell 5
test_df.to_csv(OUTPUT_DIR / "test_preprocessed.csv", index=False)



## === cell 6
test_dataset = UWDataset(test_df, transforms=transforms_val)




## === cell 7
def build_model():
    if smp is not None:
        return smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=CHANNELS,
            classes=3,
            activation=None,
            decoder_use_batchnorm=True,
            decoder_attention_type="scse",
        )
    else:

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(CHANNELS, 3, kernel_size=1)

            def forward(self, x):
                return self.conv(x)

        return SimpleModel()


def load_model(path):
    """
    Load a checkpoint if it exists; otherwise return an
    randomly‑initialized model (so inference can continue).
    """
    model = build_model()
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=DEVICE)
        state = checkpoint.get("state_dict", checkpoint)
        new_state = {}
        for k, v in state.items():
            new_key = k[6:] if k.startswith("model.") else k
            new_state[new_key] = v
        model.load_state_dict(new_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model




## === cell 8
def mask2rle(mask):
    """
    Convert binary mask to run‑length encoding string.
    Works with both cupy and numpy arrays.
    """
    mask = cp.array(mask, dtype=cp.uint8)
    pixels = mask.flatten()
    pad = cp.array([0], dtype=cp.uint8)
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(int(x)) for x in runs)


def pad_mask(mask):
    padded = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=mask.dtype)
    dh = IMG_SIZE - mask.shape[0]
    dw = IMG_SIZE - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1], :] = mask
    return padded


def resize_mask(mask, height, width):
    msk = np.zeros((height, width, 3), dtype=mask.dtype)
    for c in range(3):
        msk[:, :, c] = cv2.resize(
            mask[:, :, c], (width, height), interpolation=cv2.INTER_NEAREST
        )
    return msk


def masks2rles(masks, ids, heights, widths):
    pred_strings, pred_ids, pred_classes = [], [], []
    for idx in range(masks.shape[0]):
        mask = pad_mask(masks[idx])
        mask = resize_mask(mask, int(heights[idx]), int(widths[idx]))
        rle = [mask2rle(mask[..., c]) for c in range(3)]
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * 3)
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 9
@torch.no_grad()
def infer(model_paths, thr):
    test_set = UWDataset(test_df, transforms=transforms_val)
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    all_strings, all_ids, all_classes = [], [], []

    for batch in tqdm(test_loader, desc="Infer"):
        imgs = batch["image"].to(DEVICE, dtype=torch.float)
        ids = batch["id"]
        heights = batch["h"]
        widths = batch["w"]
        if isinstance(heights, torch.Tensor):
            heights = heights.tolist()
        if isinstance(widths, torch.Tensor):
            widths = widths.tolist()

        agg = torch.zeros((imgs.size(0), 3, imgs.size(2), imgs.size(3)), device=DEVICE)
        for p in model_paths:
            model = load_model(p)
            out = torch.nn.Sigmoid()(model(imgs))
            agg += out / len(model_paths)

        masks = (agg.permute(0, 2, 3, 1) > thr).to(torch.uint8).cpu().numpy()
        strings, ids_batch, classes_batch = masks2rles(masks, ids, heights, widths)
        all_strings.extend(strings)
        all_ids.extend(ids_batch)
        all_classes.extend(classes_batch)

    return pd.DataFrame({"id": all_ids, "class": all_classes, "predicted": all_strings})




## === cell 10
model_pths = ["../input/exp01017/expexp010-bestloss-fold0-7.ckpt"]
pred_df = infer(model_pths, THR)



## === cell 11
sub_df = pd.read_csv(INPUT_DATA_DIR / "sample_submission.csv")
sub_df = sub_df.drop(columns=["predicted"])
submission = sub_df.merge(pred_df, on=["id", "class"], how="left")
submission["predicted"] = submission["predicted"].fillna("")
submission_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission written to", submission_path, "with shape:", submission.shape)
