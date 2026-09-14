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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
sklearn-pandas==2.2.0
timm==1.0.19
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7045545232333765

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01328) has done: 'I fix the immediate runtime error by replacing the removed `Image.ANTIALIAS` with the Pillow 11+ compatible resampling enum, and I also make the image preprocessing robust so a single bad crop/failed load won’t crash multiprocessing. Next, I fix the resized-image path mismatch (your code saves into `.../images_resized/` but loads from `.../images_resized/` without the intermediate `test/`), which is why images weren’t found. Finally, since the external ensemble weight file path doesn’t exist in this environment, I keep the same inference semantics but fall back to using a timm pretrained `inception_v4` classifier to ensure a valid end-to-end run and submission CSV is produced.'
- What this solution (achieved 0.11517) has done: 'Your current score is far below the target, so we should improve performance with minimal, low-risk fixes that don’t change the overall approach (single timm model + softmax + argmax). The biggest issue is a strong train/test preprocessing mismatch: you resize/crop test images to 100px but then upscale to 224, which destroys retinal detail and hurts kappa; we resize the preprocessed test images directly to 224 to match the model’s expected scale. Next, we make inference more stable by enabling AMP on CUDA (same semantics, typically tiny numeric differences) and slightly increasing batch size/num_workers for throughput (within the 600s budget), without changing the model or prediction rule. These changes should move the score substantially upward toward your target while keeping the core logic intact.'
- What this solution (achieved 0.17293) has done: 'The current score is far below the target, so we should make small, low-risk fixes that improve correctness and metric-alignment without changing the core approach (single timm model inference with softmax + argmax). The biggest win with minimal change is to use the correct model-specific normalization for Inception-v4 (it is usually trained with Inception-style mean/std, not ImageNet default), which often materially improves performance while keeping the same architecture and prediction rule. I also make the test DataLoader use a slightly larger batch size and more workers for faster, more stable throughput (no change to semantics). Finally, I ensure the preprocessed image path resolution is consistent and deterministic.'
- What this solution (achieved 0.11614) has done: 'Your current score (0.17293) is far below the target (0.70455), so we should improve performance with minimal, low-risk changes that keep your single-model inference core intact. The biggest correctness issue is that you’re using Inception-v4 with `num_classes=5` and timm pretrained weights, which are ImageNet-1000 weights and therefore mismatched to your 5-class head; instead we keep the same model/argmax pipeline but switch to timm’s native ImageNet head and map the 1000-way probabilities to 5 DR classes using a fixed, deterministic binning of the predicted class index. This preserves your inference semantics (softmax + argmax) while producing a much more meaningful prediction signal than random 5-class logits. Additionally, we make the transform use the model’s required input size (299 for inception_v4) to match the pretrained configuration, which is a small but important preprocessing alignment fix. The script still writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.08031) has done: 'Your current score (0.11614) is far below the target (0.70455), so we should improve it with the smallest changes that keep your single timm pretrained inference core intact. The biggest issue is the current “1000-class ImageNet → 5-class DR” mapping: binning argmax indices is essentially arbitrary and throws away almost all signal, so I replace it with a deterministic ordinal mapping based on the expected class index (a smooth 0–4 projection) and round+clip to integer labels. This preserves your evaluation semantics (single model, softmax, deterministic post-process to integer 0–4) while making predictions meaningfully ordinal, which should move kappa upward toward the target. I also ensure the resize/crop stays matched to the model’s pretrained config and keep the submission format/row alignment unchanged.'

# 9. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import timm
from multiprocessing import Pool, cpu_count
import cv2
import shutil

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
for p in ["/kaggle/working/train", "/kaggle/working/test"]:
    try:
        shutil.rmtree(p)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Could not remove {p}: {e}")

model_dir_to_delete = "/kaggle/working/models"
try:
    if os.path.isfile(model_dir_to_delete):
        os.remove(model_dir_to_delete)
except Exception as e:
    print(f"Could not remove {model_dir_to_delete}: {e}")




## === cell 2
def load_data(data_dir):
    test_csv = os.path.join(data_dir, "test.csv")
    test = pd.read_csv(test_csv)

    test_dir = os.path.join(data_dir, "test_images")
    test["file_path"] = test["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )
    test["file_name"] = test["id_code"] + ".png"
    return test




## === cell 3
data_dir = "/kaggle/input/aptos2019-blindness-detection/"
test_df = load_data(data_dir)
test_df.head()




## === cell 4
def crop_img(img, percentage):
    try:
        img_arr = np.array(img.convert("RGB"))
        img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

        nonzero = img_gray[img_gray != 0]
        if nonzero.size == 0:
            return img.convert("RGB")

        threshold = img_gray > 0.1 * np.mean(nonzero)
        row_sums = np.sum(threshold, axis=1)
        col_sums = np.sum(threshold, axis=0)

        rows = np.where(row_sums > img_arr.shape[1] * percentage)[0]
        cols = np.where(col_sums > img_arr.shape[0] * percentage)[0]

        if rows.size == 0 or cols.size == 0:
            return img.convert("RGB")

        min_row, min_col = int(np.min(rows)), int(np.min(cols))
        max_row, max_col = int(np.max(rows)), int(np.max(cols))

        crop_arr = img_arr[min_row : max_row + 1, min_col : max_col + 1]
        if crop_arr.size == 0:
            return img.convert("RGB")

        return Image.fromarray(crop_arr)
    except Exception:
        return img.convert("RGB")




## === cell 5
_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def resize_maintain_aspect(img, desired_size):
    img = img.convert("RGB")
    old_width, old_height = img.size
    if old_height == 0 or old_width == 0:
        return Image.new("RGB", (desired_size, desired_size))

    aspect_ratio = old_width / old_height

    if aspect_ratio > 1:
        new_width = desired_size
        new_height = max(1, int(desired_size / aspect_ratio))
    else:
        new_height = desired_size
        new_width = max(1, int(desired_size * aspect_ratio))

    resized_img = img.resize((new_width, new_height), resample=_RESAMPLE)

    padded_image = Image.new("RGB", (desired_size, desired_size))
    x_offset = (desired_size - new_width) // 2
    y_offset = (desired_size - new_height) // 2
    padded_image.paste(resized_img, (x_offset, y_offset))

    return padded_image




## === cell 6
def save_single(args):
    image_path, output_path_folder, percentage, output_size = args
    try:
        image = Image.open(image_path).convert("RGB")
        cropped_img = crop_img(image, percentage)
        image_resized = resize_maintain_aspect(
            cropped_img, desired_size=int(output_size[0])
        )

        output_image_name = os.path.basename(image_path)
        output_file_path = os.path.join(output_path_folder, output_image_name)
        image_resized.save(output_file_path)
        return True
    except Exception:
        return False




## === cell 7
def fast_image_resize(df, output_path_folder, percentage, output_size=None):
    """Uses multiprocessing to make it fast"""
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=(224,224)")
        return

    os.makedirs(output_path_folder, exist_ok=True)

    jobs = []
    for df_item in range(len(df)):
        image_path = df.file_path.iloc[df_item]
        jobs.append((image_path, output_path_folder, percentage, output_size))

    nproc = min(cpu_count(), 8)
    with Pool(processes=nproc) as p:
        results = list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))
    failed = results.count(False)
    if failed:
        print(
            f"Warning: {failed} images failed to preprocess and were skipped (kept as missing)."
        )




## === cell 8
percentage = 0.01

MODEL_NAME = "tf_efficientnet_b0_ns"

_tmp_model = timm.create_model(MODEL_NAME, pretrained=True)  # get pretrained_cfg safely
cfg = getattr(_tmp_model, "pretrained_cfg", {}) or {}
mean = cfg.get("mean", (0.485, 0.456, 0.406))
std = cfg.get("std", (0.229, 0.224, 0.225))
input_size = cfg.get("input_size", (3, 224, 224))
del _tmp_model

IMG_SIZE = int(input_size[-1])

train_csv = os.path.join(data_dir, "train.csv")
train_images_dir = os.path.join(data_dir, "train_images")
test_csv = os.path.join(data_dir, "test.csv")
test_images_dir = os.path.join(data_dir, "test_images")

resized_train_dir = "/kaggle/working/train/images_resized/"  # kept for compatibility
resized_test_dir = "/kaggle/working/test/images_resized/"

train_df = pd.read_csv(train_csv)
train_df["file_path"] = train_df["id_code"].map(
    lambda x: os.path.join(train_images_dir, f"{x}.png")
)
test_df = pd.read_csv(test_csv)
test_df["file_path"] = test_df["id_code"].map(
    lambda x: os.path.join(test_images_dir, f"{x}.png")
)

os.makedirs(
    resized_train_dir, exist_ok=True
)  # empty dir; dataset will fall back to orig_images_dir
fast_image_resize(
    test_df, resized_test_dir, percentage, output_size=(IMG_SIZE, IMG_SIZE)
)

print(
    "Resized train files:",
    len([f for f in os.listdir(resized_train_dir) if f.endswith(".png")]),
    "Resized test files:",
    len([f for f in os.listdir(resized_test_dir) if f.endswith(".png")]),
    "IMG_SIZE:",
    IMG_SIZE,
)




## === cell 9
class BlindnessDataset(Dataset):
    def __init__(
        self, csv_file, root_dir, transform=None, test=False, orig_images_dir=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.orig_images_dir = orig_images_dir

        if "id_code" not in self.annotations.columns:
            raise ValueError(f"{csv_file} must contain column 'id_code'")
        if (not self.test) and ("diagnosis" not in self.annotations.columns):
            raise ValueError(
                f"{csv_file} must contain column 'diagnosis' for training/val"
            )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = str(self.annotations.loc[idx, "id_code"])
        img_name = os.path.join(self.root_dir, img_id + ".png")
        try:
            image = Image.open(img_name).convert("RGB")
        except Exception:
            if self.orig_images_dir is not None:
                orig_path = os.path.join(self.orig_images_dir, img_id + ".png")
                try:
                    image = Image.open(orig_path).convert("RGB")
                except Exception:
                    image = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            else:
                image = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.loc[idx, "diagnosis"])
            return image, label




## === cell 10
from timm.data import resolve_data_config, create_transform

_tmp_model = timm.create_model(MODEL_NAME, pretrained=True)
data_cfg = resolve_data_config({}, model=_tmp_model)
del _tmp_model

data_cfg["input_size"] = (3, IMG_SIZE, IMG_SIZE)

train_transform = create_transform(
    input_size=data_cfg["input_size"],
    is_training=True,
    mean=data_cfg.get("mean", mean),
    std=data_cfg.get("std", std),
    interpolation=data_cfg.get("interpolation", "bilinear"),
)

eval_transform = create_transform(
    input_size=data_cfg["input_size"],
    is_training=False,
    mean=data_cfg.get("mean", mean),
    std=data_cfg.get("std", std),
    interpolation=data_cfg.get("interpolation", "bilinear"),
)

print(
    "Using timm data_cfg:",
    {
        k: data_cfg[k]
        for k in ["input_size", "mean", "std", "interpolation"]
        if k in data_cfg
    },
)



## === cell 11
full_train_df = pd.read_csv(train_csv)
idx = np.arange(len(full_train_df))
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_frac = 0.15
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_split_csv = "/kaggle/working/train_split.csv"
val_split_csv = "/kaggle/working/val_split.csv"
full_train_df.iloc[trn_idx].to_csv(train_split_csv, index=False)
full_train_df.iloc[val_idx].to_csv(val_split_csv, index=False)

train_dataset = BlindnessDataset(
    train_split_csv,
    resized_train_dir,
    transform=train_transform,
    test=False,
    orig_images_dir=train_images_dir,
)
val_dataset = BlindnessDataset(
    val_split_csv,
    resized_train_dir,
    transform=eval_transform,
    test=False,
    orig_images_dir=train_images_dir,
)

num_workers = min(8, cpu_count())
train_loader = DataLoader(
    train_dataset,
    batch_size=16 if not torch.cuda.is_available() else 32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if num_workers > 0 else False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32 if not torch.cuda.is_available() else 64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if num_workers > 0 else False,
)

test_dataset = BlindnessDataset(
    test_csv,
    resized_test_dir,
    transform=eval_transform,
    test=True,
    orig_images_dir=test_images_dir,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32 if not torch.cuda.is_available() else 64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if num_workers > 0 else False,
)

len(train_dataset), len(val_dataset), len(test_dataset)



## === cell 12
model_paths = {
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/3/inception_v4.pth",
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=5).to(device)
model.train()

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

class_idx = torch.arange(5, device=device, dtype=torch.float32).view(1, -1)

print("Device:", device, "AMP:", use_amp, "MODEL_NAME:", MODEL_NAME)



## === cell 14
validation_scores = {
    "inception_v4": 1.0,
}



## === cell 15
loaded_model_keys = ["inception_v4"]
weights = {"inception_v4": 1.0}
weights




## === cell 16
def quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5
) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    num = (W * O).sum()
    return float(1.0 - num / denom)




## === cell 17
def apply_thresholds(x: np.ndarray, thr: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)
    return np.digitize(x, bins=thr, right=False).astype(np.int64)


def fit_thresholds_by_label_means(
    y_true: np.ndarray, x: np.ndarray, n_classes: int = 5
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    x = np.asarray(x, dtype=np.float32)

    means = np.zeros(n_classes, dtype=np.float32)
    present = np.zeros(n_classes, dtype=bool)
    for c in range(n_classes):
        mask = y_true == c
        if mask.any():
            means[c] = float(x[mask].mean())
            present[c] = True

    if not present.all():
        xs = np.where(present)[0].astype(np.float32)
        ys = means[present].astype(np.float32)
        if len(xs) == 1:
            means[:] = ys[0] + (np.arange(n_classes, dtype=np.float32) - xs[0]) * 0.25
        else:
            means = np.interp(np.arange(n_classes, dtype=np.float32), xs, ys).astype(
                np.float32
            )

    means = np.maximum.accumulate(means)  # enforce monotonicity

    thr = np.array(
        [(means[i] + means[i + 1]) / 2.0 for i in range(n_classes - 1)],
        dtype=np.float32,
    )

    eps = 1e-4
    for i in range(1, len(thr)):
        if thr[i] <= thr[i - 1]:
            thr[i] = thr[i - 1] + eps
    return thr


def get_expected_values(model, loader, device) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                xb = batch[0]
            else:
                xb = batch
            xb = xb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits = model(xb)
                probs = nn.functional.softmax(logits, dim=1)
                expected = (probs * class_idx).sum(dim=1)
            outs.append(expected.float().cpu().numpy())
    model.train()
    return np.concatenate(outs, axis=0)


def evaluate_kappa_with_thresholds(model, loader, device, thr: np.ndarray) -> float:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits = model(xb)
                probs = nn.functional.softmax(logits, dim=1)
                expected = (probs * class_idx).sum(dim=1)
            exp_np = expected.float().cpu().numpy()
            pred = apply_thresholds(exp_np, thr)
            y_pred.append(pred)
            y_true.append(yb.numpy())
    model.train()
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return quadratic_weighted_kappa(y_true, y_pred, n_classes=5)


best_kappa = -1.0
best_path = "/kaggle/working/best_tf_efficientnet_b0_ns_ordinal.pth"

EPOCHS = 2 if torch.cuda.is_available() else 1

thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)  # fallback thresholds
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"train epoch {epoch+1}/{EPOCHS}")
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).to(torch.float32)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            logits = model(xb)
            probs = nn.functional.softmax(logits, dim=1)
            expected = (probs * class_idx).sum(dim=1)
            loss = criterion(expected, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=float(loss.detach().cpu().item()))

    val_expected = get_expected_values(model, val_loader, device)
    val_true = pd.read_csv(val_split_csv)["diagnosis"].astype(np.int64).values
    thr = fit_thresholds_by_label_means(val_true, val_expected, n_classes=5)
    val_kappa = evaluate_kappa_with_thresholds(model, val_loader, device, thr)
    print(f"Epoch {epoch+1}: val_kappa={val_kappa:.5f} thr={thr.tolist()}")

    if (epoch == 0) or (val_kappa > best_kappa):
        best_kappa = val_kappa
        torch.save({"state_dict": model.state_dict(), "thr": thr}, best_path)
        print("Saved checkpoint to:", best_path)

best_thr = thr
try:
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    best_thr = np.asarray(ckpt.get("thr", thr), dtype=np.float32)
except Exception as e:
    print(
        "Warning: failed to load checkpoint, using in-memory model/thr. Error:", repr(e)
    )

model.eval()
models_list = [model]
print("Loaded best thresholds:", np.asarray(best_thr, dtype=np.float32).tolist())



## === cell 18
all_expected = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="infer test"):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            expected_vals = []
            for model_key, m in zip(loaded_model_keys, models_list):
                logits = m(images)
                probs = nn.functional.softmax(logits, dim=1)
                expected = (probs * class_idx).sum(dim=1)  # [B]
                expected_vals.append(weights[model_key] * expected.unsqueeze(0))
            expected_vals = torch.cat(expected_vals, dim=0)  # [M,B]
            weighted_expected = torch.sum(expected_vals, dim=0)  # [B]

        all_expected.extend(weighted_expected.float().cpu().numpy())

all_expected = np.array(all_expected, dtype=np.float32)

final_predictions = apply_thresholds(all_expected, best_thr)
final_predictions = np.clip(final_predictions, 0, 4).astype(int)

print(
    "Pred shape:",
    final_predictions.shape,
    "min/max:",
    int(final_predictions.min()),
    int(final_predictions.max()),
)

sample_path = os.path.join(data_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)
sample_ids = sample_sub["id_code"].astype(str).values

test_ids = pd.read_csv(test_csv)["id_code"].astype(str).values
assert len(test_ids) == len(
    final_predictions
), f"Length mismatch: ids={len(test_ids)} preds={len(final_predictions)}"

pred_map = dict(zip(test_ids.tolist(), final_predictions.tolist()))
ordered_preds = np.array([pred_map[i] for i in sample_ids], dtype=int)

submission_df = pd.DataFrame({"id_code": sample_ids, "diagnosis": ordered_preds})

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
print(
    "Unique ids:",
    submission_df["id_code"].nunique(),
    "Any null ids:",
    submission_df["id_code"].isna().any(),
)
