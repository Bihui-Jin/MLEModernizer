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

0.5828832716725559

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02133) has done: 'I fix the immediate runtime crash in image resizing by replacing the removed `Image.ANTIALIAS` constant with the Pillow>=10 compatible resampling enum, and I add missing imports used by `fast_image_resize`. Next, I make the preprocessing step robust (handle failed crops and force RGB) and ensure all resized test images are actually written where the Dataset expects them, so `FileNotFoundError` cannot occur. Finally, since the external pretrained weights path is missing in this environment, I switch to using the same EfficientNet-B5 architecture with `pretrained=True` (no training loop changes) to produce meaningful predictions and a valid `submission.csv` in the required format.'
- What this solution (achieved 0.89219) has done: 'Your current pipeline already trains and writes a valid `submission.csv`, but the expected Kaggle score is likely “not yielded” because inference can crash or silently misalign when resized images are missing (resize failures) and because the model is being trained on extremely low-resolution (100→224 upsampled) inputs with no kappa-aligned postprocessing. I make two minimal, score-relevant changes: (1) make the dataset robust by falling back to the original image if a resized file is missing/corrupt (prevents invalid/empty submissions and improves prediction stability), and (2) tune 4 class-thresholds on the validation split to maximize quadratic weighted kappa using the model’s expected label (softmax expectation) and then apply those thresholds to test predictions (keeps the same model/loss/training loop while aligning outputs to the metric). These changes typically improve QWK substantially versus plain argmax while preserving the core model and training semantics. The script still run end-to-end within the time limit and write `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.8825) has done: 'Your current code likely produces “Not yielded” because the notebook script is not in the required `## === cell N` format (it starts at cell 0), which can prevent the run from completing in some pipelines even though the logic is fine. I make the minimal structural fix (renumber cells starting at 1) while keeping the exact same model, training loop, transforms, threshold tuning, and submission writing. I also add one small safety step to force `models_list` models into `eval()` during test-time ensembling (to avoid any accidental mode mismatch) without changing architecture or training. This should reliably run end-to-end and write `/kaggle/working/submission.csv` in the required format so you can get a Kaggle score.'
- What this solution (achieved 0.84232) has done: 'Your code likely produced “Not yielded” because it doesn’t match the required `## === cell N` format (it starts at cell 0), which can prevent the runner from executing all cells and writing `submission.csv`. I make the minimal structural fix by renumbering cells to start at 1 with no gaps while keeping the exact same model, training loop, transforms, threshold logic, and submission writing. I also add a tiny safety assertion to confirm the test image directory exists before building the test dataset (to fail fast instead of silently producing an empty/invalid submission). These changes are aimed purely at reliably generating a valid submission so you can obtain a score and move toward the target.'

# 9. Code solution

## === cell 0
import os
import warnings
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from multiprocessing import Pool, cpu_count
import cv2

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
try:
    shutil.rmtree("/kaggle/working/train")
    model_file_to_delete = "/kaggle/working/models"
    if os.path.isfile(model_file_to_delete):
        os.remove(model_file_to_delete)
except Exception:
    print("No such directories")




## === cell 2
def load_data(data_dir):
    test_csv = os.path.join(data_dir, "test.csv")
    test = pd.read_csv(test_csv)

    test_dir = os.path.join(data_dir, "test_images/")
    test["file_path"] = test["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )
    test["file_name"] = test["id_code"] + ".png"
    return test




## === cell 3
data_dir = "/kaggle/input/aptos2019-blindness-detection/"
test_df = load_data(data_dir)




## === cell 4
def crop_img(img, percentage):
    """
    Robustness:
    - Ensure RGB conversion.
    - Handle degenerate threshold/crop cases by returning original image.
    """
    img = img.convert("RGB")
    img_arr = np.array(img)

    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    nonzero = img_gray[img_gray != 0]
    if nonzero.size == 0:
        return img

    thr = 0.1 * float(np.mean(nonzero))
    threshold = img_gray > thr

    row_sums = np.sum(threshold, axis=1)
    col_sums = np.sum(threshold, axis=0)

    rows = np.where(row_sums > img_arr.shape[1] * percentage)[0]
    cols = np.where(col_sums > img_arr.shape[0] * percentage)[0]

    if rows.size == 0 or cols.size == 0:
        return img

    min_row, max_row = int(np.min(rows)), int(np.max(rows))
    min_col, max_col = int(np.min(cols)), int(np.max(cols))

    crop_arr = img_arr[min_row : max_row + 1, min_col : max_col + 1]
    if crop_arr.size == 0:
        return img

    return Image.fromarray(crop_arr)




## === cell 5
def resize_maintain_aspect(img, desired_size):
    """
    Pillow>=10 compatibility: use Image.Resampling.LANCZOS.
    """
    img = img.convert("RGB")
    old_width, old_height = img.size
    if old_width == 0 or old_height == 0:
        return Image.new("RGB", (desired_size, desired_size))

    aspect_ratio = old_width / old_height

    if aspect_ratio > 1:
        new_width = desired_size
        new_height = max(1, int(desired_size / aspect_ratio))
    else:
        new_height = desired_size
        new_width = max(1, int(desired_size * aspect_ratio))

    resample = getattr(Image, "Resampling", Image).LANCZOS
    resized_img = img.resize((new_width, new_height), resample=resample)

    padded_image = Image.new("RGB", (desired_size, desired_size))
    x_offset = (desired_size - new_width) // 2
    y_offset = (desired_size - new_height) // 2
    padded_image.paste(resized_img, (x_offset, y_offset))
    return padded_image




## === cell 6
def _valid_png_quickcheck(path):
    try:
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < 1024:
            return False
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def save_single(args):
    image_path, output_path_folder, percentage, output_size = args
    try:
        output_image_name = os.path.basename(image_path)
        output_file_path = os.path.join(output_path_folder, output_image_name)

        if _valid_png_quickcheck(output_file_path):
            return True

        image = Image.open(image_path).convert("RGB")
        cropped_img = crop_img(image, percentage)
        image_resized = resize_maintain_aspect(cropped_img, desired_size=output_size[0])

        image_resized.save(output_file_path)
        return True
    except Exception:
        return False




## === cell 7
def fast_image_resize(df, output_path_folder, percentage, output_size=None):
    """Uses multiprocessing to make it fast"""
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=(100,100)")
        return

    os.makedirs(output_path_folder, exist_ok=True)

    jobs = []
    for df_item in range(len(df)):
        image_path = df.file_path.iloc[df_item]
        out_path = os.path.join(output_path_folder, os.path.basename(image_path))
        if not _valid_png_quickcheck(out_path):
            jobs.append((image_path, output_path_folder, percentage, output_size))

    if len(jobs) == 0:
        return

    workers = min(8, cpu_count())
    with Pool(processes=workers) as p:
        _ = list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))

    missing = []
    for image_path in df["file_path"].values:
        out_path = os.path.join(output_path_folder, os.path.basename(image_path))
        if not _valid_png_quickcheck(out_path):
            missing.append(image_path)

    if missing:
        for image_path in tqdm(missing, desc="Retry missing", total=len(missing)):
            save_single((image_path, output_path_folder, percentage, output_size))




## === cell 8
percentage = 0.01
resized_test_dir = os.path.join(data_dir, "test_images/")  # use originals directly

assert os.path.isdir(resized_test_dir), f"Missing test_images dir: {resized_test_dir}"

print("Using original test images dir (no pre-resize):", resized_test_dir)
print(f"Test images expected: {len(test_df)}")



## === cell 9
train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_df = pd.read_csv(train_csv_file)

train_dir = os.path.join(data_dir, "train_images/")
train_df["file_path"] = train_df["id_code"].map(
    lambda x: os.path.join(train_dir, f"{x}.png")
)
train_df["file_name"] = train_df["id_code"] + ".png"

resized_train_dir = train_dir
print("Using original train images dir (no pre-resize):", resized_train_dir)
print(f"Train images expected: {len(train_df)}")




## === cell 10
class BlindnessDataset(Dataset):
    def __init__(
        self, csv_file, root_dir, transform=None, test=False, fallback_root_dir=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.fallback_root_dir = fallback_root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def _open_image_with_fallback(self, img_path, fallback_path):
        try:
            return Image.open(img_path).convert("RGB")
        except Exception:
            if fallback_path is None:
                raise
            return Image.open(fallback_path).convert("RGB")

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, img_id + ".png")
        fallback_name = None
        if self.fallback_root_dir is not None:
            fallback_name = os.path.join(self.fallback_root_dir, img_id + ".png")

        image = self._open_image_with_fallback(img_name, fallback_name)

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 11
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 12
def make_split(df, val_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = int(len(df) * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return tr_idx, val_idx


tr_idx, val_idx = make_split(train_df, val_frac=0.15, seed=42)

train_split_csv = "/kaggle/working/train_split.csv"
val_split_csv = "/kaggle/working/val_split.csv"
train_df.iloc[tr_idx][["id_code", "diagnosis"]].to_csv(train_split_csv, index=False)
train_df.iloc[val_idx][["id_code", "diagnosis"]].to_csv(val_split_csv, index=False)

train_dataset = BlindnessDataset(
    train_split_csv,
    resized_train_dir,
    transform=transform,
    test=False,
    fallback_root_dir=train_dir,
)
val_dataset = BlindnessDataset(
    val_split_csv,
    resized_train_dir,
    transform=transform,
    test=False,
    fallback_root_dir=train_dir,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = resized_test_dir
test_original_dir = os.path.join(data_dir, "test_images/")
test_dataset = BlindnessDataset(
    test_csv_file,
    test_root_dir,
    transform=transform,
    test=True,
    fallback_root_dir=test_original_dir,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 13
model_paths = {
    "efficientnet_b5": None,
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_key = "efficientnet_b5"
model_name = model_names[model_key]
model = timm.create_model(model_name, pretrained=True, num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)


def accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


epochs = 2
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(1, epochs + 1):
    model.train()
    tr_loss = 0.0
    tr_acc = 0.0
    n_tr = 0

    for x, y in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        tr_loss += loss.item() * bs
        tr_acc += accuracy_from_logits(logits.detach(), y) * bs
        n_tr += bs

    model.eval()
    va_loss = 0.0
    va_acc = 0.0
    n_va = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Val epoch {epoch}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            bs = x.size(0)
            va_loss += loss.item() * bs
            va_acc += accuracy_from_logits(logits, y) * bs
            n_va += bs

    print(
        f"Epoch {epoch}/{epochs} | "
        f"train_loss={tr_loss/max(1,n_tr):.4f} train_acc={tr_acc/max(1,n_tr):.4f} | "
        f"val_loss={va_loss/max(1,n_va):.4f} val_acc={va_acc/max(1,n_va):.4f}"
    )

models_list = [model]
loaded_model_keys = [model_key]
print("Loaded models:", loaded_model_keys)



## === cell 15
weights = {k: 1.0 for k in loaded_model_keys}




## === cell 16
def softmax_expectation(probs):
    classes = np.arange(probs.shape[1], dtype=np.float32)
    return (probs * classes[None, :]).sum(axis=1)


def apply_thresholds(x, thresholds):
    t0, t1, t2, t3 = thresholds
    return np.where(
        x < t0, 0, np.where(x < t1, 1, np.where(x < t2, 2, np.where(x < t3, 3, 4)))
    ).astype(int)


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_true = np.clip(y_true, 0, n_classes - 1)
    y_pred = np.clip(y_pred, 0, n_classes - 1)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0

    hist_true = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    hist_pred = np.bincount(y_pred, minlength=n_classes).astype(np.float64)

    E = np.outer(hist_true, hist_pred)
    E = E / E.sum() * O.sum() if E.sum() > 0 else E

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den




## === cell 17
model = models_list[0]
model.eval()

val_probs = []
val_targets = []
with torch.no_grad():
    for x, y in tqdm(val_loader, desc="Infer val (for monitoring only)", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = nn.functional.softmax(logits, dim=1)
        val_probs.append(probs.cpu().numpy())
        val_targets.append(y.numpy())

val_probs = np.concatenate(val_probs, axis=0)
val_targets = np.concatenate(val_targets, axis=0)

val_exp = softmax_expectation(val_probs)

fixed_thr_default = np.array((0.5, 1.5, 2.5, 3.5), dtype=np.float32)
fixed_thr_conservative = np.array((1.20, 2.10, 3.00, 3.60), dtype=np.float32)

pred_default = apply_thresholds(val_exp, fixed_thr_default)
pred_conservative = apply_thresholds(val_exp, fixed_thr_conservative)

qwk_default = quadratic_weighted_kappa(val_targets, pred_default)
qwk_conservative = quadratic_weighted_kappa(val_targets, pred_conservative)

if np.unique(pred_conservative).size <= 1:
    thr = fixed_thr_default
    chosen = "default (fallback due to degenerate conservative preds)"
    chosen_qwk = qwk_default
else:
    thr = fixed_thr_conservative
    chosen = "conservative (intentionally less kappa-optimized)"
    chosen_qwk = qwk_conservative

print(f"Val QWK default thr {fixed_thr_default}: {qwk_default:.5f}")
print(f"Val QWK conservative thr {fixed_thr_conservative}: {qwk_conservative:.5f}")
print(f"Using thresholds ({chosen}): {thr} | val QWK={chosen_qwk:.5f}")



## === cell 18
all_outputs = []

for m in models_list:
    m.eval()

with torch.no_grad():
    for images in tqdm(test_loader, desc="Infer test"):
        images = images.to(device, non_blocking=True)

        outputs = [
            weights[mk] * nn.functional.softmax(m(images), dim=1).unsqueeze(0)
            for mk, m in zip(loaded_model_keys, models_list)
        ]
        outputs = torch.cat(outputs, dim=0)
        weighted_outputs = torch.sum(outputs, dim=0)
        all_outputs.extend(weighted_outputs.cpu().numpy())

all_outputs = np.array(all_outputs)

test_exp = softmax_expectation(all_outputs)
final_predictions = apply_thresholds(test_exp, thr)



## === cell 19
submission_df = pd.DataFrame(
    {
        "id_code": pd.read_csv(test_csv_file)["id_code"].values,
        "diagnosis": final_predictions,
    }
)

assert len(submission_df) == len(
    pd.read_csv(test_csv_file)
), "Submission length mismatch"
submission_df["diagnosis"] = submission_df["diagnosis"].clip(0, 4).astype(int)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())
