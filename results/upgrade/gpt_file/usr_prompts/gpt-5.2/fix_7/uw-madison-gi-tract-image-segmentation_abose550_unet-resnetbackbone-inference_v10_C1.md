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
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.5540628775178745

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing `segmentation_models_pytorch` import by installing a version compatible with your Torch/Torchvision stack and then importing it reliably. I also fix the checkpoint discovery logic so it finds whatever model files are actually present in `CKPT_DIR` (not only `best_epoch*.bin`), avoiding the current hard failure and enabling end-to-end inference. To ensure a valid submission is always produced even if no checkpoints exist, I add a safe fallback that outputs empty masks (valid RLE strings) rather than crashing; this yields a valid CSV (but lower score) instead of “Not yielded”. These changes are minimal and preserve your model/inference/RLE logic when checkpoints are available.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with producing (mostly) empty masks, which can happen even when checkpoints exist because the current RLE encoding doesn’t binarize the mask and the model is built without the intended encoder/backbone (so weights may not load or outputs are poor). I make two minimal fixes that preserve your inference pipeline: (1) build the Unet with the configured encoder (`resnet34`) so checkpoints trained with that architecture load and behave correctly, and (2) make `mask2rle` explicitly threshold to a binary mask (as required by the competition) before encoding. These changes should move the score upward toward your target by turning valid probabilistic outputs into proper binary segmentations and leveraging the actual trained weights.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that is effectively empty/near-empty after post-processing, even when model checkpoints exist. To move the score upward toward your 0.554 target without changing your model/training logic, I (1) fix the RLE encoding to follow the competition’s required pixel order (column-major / Fortran order) and (2) fix the padding step so predictions are restored to the original image size using the *original crop coordinates* (not recomputed from resized size), which otherwise can misalign masks and destroy Dice/Hausdorff. These are minimal inference/post-processing corrections and keep your architecture, thresholds, and ensembling intact. The code still produces a valid `submission.csv` end-to-end within the same paths.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with generating largely empty/wrongly-ordered RLEs or badly misaligned masks. I keep your model/inference loop intact, but make two minimal, metric-relevant fixes: (1) use the correct RLE pixel order for this competition (row-major / C order: left-to-right then top-to-bottom) and (2) ensure the model input normalization matches what SMP encoders expect (simple ImageNet mean/std) without changing your architecture or thresholding. These changes typically turn “valid but scored as wrong/empty” submissions into meaningful segmentations and should move the score upward toward your target. Everything still runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser



## === cell 1
import os, sys, subprocess, textwrap, glob, shutil, site, pathlib


def _run(cmd):
    return subprocess.run(
        cmd,
        shell=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


print("Python:", sys.version)
print("Site-packages:", site.getsitepackages()[:2])



## === cell 2
modellibs_path = "/kaggle/input/modellibs/libs"
if os.path.isdir(modellibs_path):
    print(_run(f"ls {modellibs_path} | head"))
else:
    print("No /kaggle/input/modellibs/libs found; skipping legacy manual installs.")



## === cell 3
py_site = site.getsitepackages()[0]


def guarded_install_local_pkg(src_dir):
    if os.path.isdir(src_dir):
        print(_run(f"cd {src_dir} && python setup.py -q install"))
    else:
        print(f"Missing {src_dir} (skipping).")


if os.path.isdir(modellibs_path):
    src = "/tmp/efficientnet_pytorch-0.6.3"
    if not os.path.isdir(src):
        _run("rm -rf /tmp/efficientnet_pytorch-0.6.3")
        print(
            _run(
                f"cp -r {modellibs_path}/efficientnet_pytorch-0.6.3/efficientnet_pytorch-0.6.3/ {src}"
            )
        )
    guarded_install_local_pkg(src)

    src = "/tmp/pytorch-image-models-0.4.12"
    if not os.path.isdir(src):
        _run("rm -rf /tmp/pytorch-image-models-0.4.12")
        print(
            _run(
                f"cp -R {modellibs_path}/pytorch-image-models-0.4.12/pytorch-image-models-0.4.12/ {src}"
            )
        )
    guarded_install_local_pkg(src)

    src = "/tmp/pretrained-models.pytorch"
    if not os.path.isdir(src):
        _run("rm -rf /tmp/pretrained-models.pytorch")
        print(_run(f"cp -R {modellibs_path}/pretrained-models.pytorch/ {src}"))
    guarded_install_local_pkg(src)
else:
    print("Skipping manual model-lib installs (not available).")



## === cell 4
wheel_dir = "/kaggle/input/seg-model-pytorch"
if os.path.isdir(wheel_dir):
    whls = glob.glob(os.path.join(wheel_dir, "*.whl"))
    if whls:
        print(_run(f"pip install -q {whls[0]}"))
    else:
        print("No wheel found in /kaggle/input/seg-model-pytorch (skipping).")
else:
    print(
        "No /kaggle/input/seg-model-pytorch found; using (or installing) segmentation_models_pytorch."
    )



## === cell 5
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import torch
from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import gc
import torch.nn as nn
import albumentations as A
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:
    import segmentation_models_pytorch as smp
except ModuleNotFoundError:
    print("segmentation_models_pytorch not found; installing a compatible version...")
    print(
        _run(
            "pip -q install --no-input segmentation-models-pytorch==0.5.0 timm==0.9.12"
        )
    )
    import importlib

    smp = importlib.import_module("segmentation_models_pytorch")

print("Torch:", torch.__version__)
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
print("SMP:", getattr(smp, "__version__", "unknown"))




## === cell 6
class CFG:
    seed = 42
    debug = False
    model_name = "UNET"
    encoder_name = "resnet34"
    encoder_weights = "imagenet"
    train_batch_size = 32
    val_batch_size = 32
    img_size = (224, 224)
    scheduler = "CosineAnnealingLR"
    epochs = 22
    lr = 2e-3
    min_lr = 1e-6
    weight_decay = 1e-6
    T_max = int(30000 / train_batch_size * epochs) + 50
    num_classes = 3
    val_split_percentage = 0.2
    n_accumulate = max(1, 32 // train_batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thr = 0.45




## === cell 7
TRAIN_METADATA_FILE = "../input/uw-madison-gi-tract-image-segmentation/train.csv"
TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train/"
BASE_PATH = "../input/uw-madison-gi-tract-image-segmentation"
CKPT_DIR = "../input/resnettrain2epoch21"
SAMPLE_SUBMISSION_CSV_PATH = (
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE

print("BASE_PATH exists:", os.path.isdir(BASE_PATH))
print("CKPT_DIR exists:", os.path.isdir(CKPT_DIR))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUBMISSION_CSV_PATH))




## === cell 8
def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].split("case")[1]))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].split("day")[1]))
    df["slice"] = df["id"].apply(lambda x: int(x.split("_")[-1]))
    return df


def path2info(row: pd.Series) -> pd.Series:
    path = row["image_path"]
    parts_path = path.replace("\\", "/").split("/")
    fname = parts_path[-1].replace(".png", "")
    parts = fname.split("_")

    if parts[0] == "slice":
        slice_ = int(parts[1])
        width = int(parts[2])
        height = int(parts[3])
    else:
        slice_ = int(parts[0])
        width = int(parts[1])
        height = int(parts[2])

    case_day = parts_path[-3]
    case = int(case_day.split("_")[0].replace("case", ""))
    day = int(case_day.split("_")[1].replace("day", ""))

    row["height"] = height
    row["width"] = width
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    return row




## === cell 9
def load_image(path: str):
    return Image.open(path).convert("RGB")




## === cell 10
sub_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(TRAIN_FILE_CSV_PATH)[: 1000 * 3]
    sub_df = sub_df.drop(columns=["class", "segmentation"]).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()

sub_df = get_metadata(sub_df)
print("debug:", debug, "sub_df rows:", len(sub_df))



## === cell 11
if debug:
    paths = glob(BASE_PATH + "/train/**/*png", recursive=True)
else:
    paths = glob(BASE_PATH + "/test/**/*png", recursive=True)

path_df = pd.DataFrame(paths, columns=["image_path"])
rows = []
for p in tqdm(path_df["image_path"].tolist(), desc="Parse paths"):
    r = pd.Series({"image_path": p})
    r = path2info(r)
    rows.append(r)
path_df = pd.DataFrame(rows)

print("Found images:", len(path_df))
print(path_df.head())



## === cell 12
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")
missing = int(test_df["image_path"].isna().sum())
print("Merged rows:", len(test_df), "Missing image_path:", missing)
if missing:
    print(test_df[test_df["image_path"].isna()].head())
    raise RuntimeError(
        f"Failed to match {missing} ids to image paths. Check path parsing / dataset structure."
    )
test_df.head()




## === cell 13
class TestDataset(Dataset):
    def __init__(self, df, transforms=None, label=None):
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.ids = df["id"].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        id_ = self.ids[index]
        img = load_image(img_path)
        img = np.array(img)
        h, w = img.shape[:2]
        if self.transforms:
            data = self.transforms(image=img)
            img = data["image"]

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img), id_, h, w




## === cell 14
test_transforms = {
    "test": A.Compose(
        [
            A.CenterCrop(*CFG.img_size),
        ],
        p=1.0,
    )
}




## === cell 15
def build_model():
    model = smp.Unet(
        encoder_name=CFG.encoder_name,
        encoder_weights=None,  # keep None at inference to avoid downloading/overriding weights
        in_channels=3,
        classes=CFG.num_classes,
    )
    return model


def load_model(path):
    model = build_model()
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v
        state = new_state
    model.load_state_dict(state, strict=True)
    model.eval()
    return model




## === cell 16
def mask2rle(msk, thr=0.5):
    msk = np.asarray(msk)
    if msk.dtype != np.uint8:
        msk = (msk > thr).astype(np.uint8)
    else:
        msk = (msk > 0).astype(np.uint8)

    pixels = msk.flatten(order="C")
    pad = np.array([0], dtype=np.uint8)
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def masks2rles(msks, ids, heights, widths, resized_hw=CFG.img_size):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    crop_h, crop_w = int(resized_hw[0]), int(resized_hw[1])

    for idx in range(msks.shape[0]):
        height = int(heights[idx])
        width = int(widths[idx])

        top = max(0, (height - crop_h) // 2)
        left = max(0, (width - crop_w) // 2)

        canvas = np.zeros((height, width, 3), dtype=msks.dtype)
        h_end = min(height, top + crop_h)
        w_end = min(width, left + crop_w)

        src_h = h_end - top
        src_w = w_end - left
        if src_h > 0 and src_w > 0:
            canvas[top:h_end, left:w_end, :] = msks[idx][:src_h, :src_w, :]

        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(canvas[..., midx], thr=0.5)

        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 17
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    msks = []
    imgs = []
    pred_strings = []
    pred_ids = []
    pred_classes = []

    have_models = bool(model_paths)

    models = []
    if have_models:
        for p in model_paths:
            m = load_model(p).to(CFG.device)
            models.append(m)
    else:
        print(f"WARNING: No checkpoints found in {CKPT_DIR}. Will submit empty masks.")

    for idx, (img, ids, heights, widths) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer")
    ):
        if have_models:
            img = img.to(CFG.device, dtype=torch.float32)

            size = img.size()
            msk = torch.zeros(
                (size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32
            )

            for model in models:
                out = model(img)
                out = torch.sigmoid(out)
                msk += out / len(models)

            msk = (msk.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().numpy()

            if idx < num_log:
                img_np = img.permute((0, 2, 3, 1)).cpu().numpy()
                imgs.append(img_np[:10])
                msks.append(msk[:10])

            del img
        else:
            bsz = len(ids)
            resized_h, resized_w = int(CFG.img_size[0]), int(CFG.img_size[1])
            msk = np.zeros((bsz, resized_h, resized_w, 3), dtype=np.uint8)

        result = masks2rles(msk, ids, heights, widths, resized_hw=CFG.img_size)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred_strings, pred_ids, pred_classes, imgs, msks




## === cell 18
test_dataset = TestDataset(test_df, transforms=test_transforms["test"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

model_paths = []
if os.path.isdir(CKPT_DIR):
    patterns = ["best_epoch*.bin", "*.bin", "*.pth", "*.pt"]
    for pat in patterns:
        model_paths.extend(glob(os.path.join(CKPT_DIR, pat)))
    model_paths = sorted(set(model_paths))
print("Checkpoints found:", len(model_paths))
if len(model_paths) > 0:
    print("First checkpoints:", model_paths[:3])

pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)



## === cell 19
if len(imgs) and len(msks):
    for img, msk in zip(imgs[0][:2], msks[0][:2]):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow((img - img.min()) / (img.max() - img.min() + 1e-6), cmap="bone")
        plt.axis("off")
        plt.title("image (normalized view)")
        plt.subplot(1, 3, 2)
        plt.imshow(msk * 255)
        plt.axis("off")
        plt.title("mask")
        plt.subplot(1, 3, 3)
        plt.imshow((img - img.min()) / (img.max() - img.min() + 1e-6), cmap="bone")
        plt.imshow(msk * 255, alpha=0.4)
        plt.axis("off")
        plt.title("overlay")
        plt.tight_layout()
        plt.show()



## === cell 20
if "imgs" in globals():
    try:
        del imgs
    except Exception:
        pass
if "msks" in globals():
    try:
        del msks
    except Exception:
        pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 21
pred_df = pd.DataFrame(
    {
        "id": pred_ids,
        "class": pred_classes,
        "predicted": pred_strings,
    }
)

sub_df = pd.read_csv(BASE_PATH + "/sample_submission.csv")
sub_df = sub_df.drop(columns=["predicted"])
sub_df = sub_df.merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(sub_df), "cols:", list(sub_df.columns))
print(sub_df.head())
print(
    "Non-empty predictions:", int((sub_df["predicted"].astype(str).str.len() > 0).sum())
)
