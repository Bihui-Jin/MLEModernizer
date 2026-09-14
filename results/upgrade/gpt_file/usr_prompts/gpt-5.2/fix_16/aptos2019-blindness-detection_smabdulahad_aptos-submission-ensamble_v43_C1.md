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

0.6738812898573387

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime error caused by the Pillow deprecation of `Image.ANTIALIAS` by switching to the modern `Image.Resampling.LANCZOS`, so multiprocessing resizing works again. Then I make the resizing step robust to per-image failures (so one bad/corrupt image won’t crash the entire pool) and ensure images are saved in the exact folder the Dataset expects. Next, I fix the missing external model weights path by using a timm pretrained backbone as a safe fallback when the `.pth` file is not available, so the notebook can run end-to-end and produce predictions. Finally, I ensure the submission is always written as `submission.csv` with the required `id_code,diagnosis` columns aligned to `test.csv`.'

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

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
for p in [
    "/kaggle/working/train",
    "/kaggle/working/test/images_resized",
    "/kaggle/working/models",
]:
    try:
        shutil.rmtree(p)
    except Exception:
        pass

os.makedirs("/kaggle/working/models", exist_ok=True)
os.makedirs("/kaggle/working/test/images_resized", exist_ok=True)
os.makedirs("/kaggle/working/train/images_resized", exist_ok=True)




## === cell 2
def resolve_data_dir(primary="/kaggle/input/aptos2019-blindness-detection/"):
    candidates = [
        primary,
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "/kaggle/data/",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c
    return primary


data_dir = resolve_data_dir()
print(f"[INFO] Using data_dir={data_dir}")




## === cell 3
def load_data(data_dir_):
    test_csv = os.path.join(data_dir_, "test.csv")
    test = pd.read_csv(test_csv)

    test_dir = os.path.join(data_dir_, "test_images")
    test["file_path"] = test["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )
    test["file_name"] = test["id_code"] + ".png"
    return test


test_df = load_data(data_dir)




## === cell 4
def crop_img(img, percentage):
    img = img.convert("RGB")
    img_arr = np.array(img)

    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    nz = img_gray[img_gray != 0]
    if nz.size == 0:
        return img

    threshold = img_gray > 0.1 * np.mean(nz)
    row_sums = np.sum(threshold, axis=1)
    col_sums = np.sum(threshold, axis=0)

    rows = np.where(row_sums > img_arr.shape[1] * percentage)[0]
    cols = np.where(col_sums > img_arr.shape[0] * percentage)[0]

    if rows.size == 0 or cols.size == 0:
        return img

    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)

    crop_arr = img_arr[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(crop_arr)




## === cell 5
def resize_maintain_aspect(img, desired_size):
    resample = getattr(Image, "Resampling", Image).LANCZOS

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

    resized_img = img.resize((new_width, new_height), resample=resample)

    padded_image = Image.new("RGB", (desired_size, desired_size))
    x_offset = (desired_size - new_width) // 2
    y_offset = (desired_size - new_height) // 2
    padded_image.paste(resized_img, (x_offset, y_offset))
    return padded_image




## === cell 6
def save_single(args):
    image_path, output_path_folder, percentage, output_size = args

    output_image_name = os.path.basename(image_path)
    output_file_path = os.path.join(output_path_folder, output_image_name)
    if os.path.exists(output_file_path):
        return True

    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            cropped_img = crop_img(im, percentage)
            image_resized = resize_maintain_aspect(
                cropped_img, desired_size=output_size[0]
            )
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
        jobs.append((image_path, output_path_folder, percentage, output_size))

    workers = min(4, cpu_count())
    with Pool(processes=workers) as p:
        results = list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))

    failed = len(results) - int(np.sum(results))
    if failed > 0:
        print(
            f"[WARN] Failed to process {failed}/{len(results)} images. They will be handled at inference time."
        )




## === cell 8
percentage = 0.01
print(
    "[INFO] Skipping pre-resize of test images to ensure end-to-end runtime and submission generation."
)



## === cell 9
print(
    "[INFO] Using original test_images/ at inference time (no resized image folder required)."
)




## === cell 10
class BlindnessDataset(Dataset):
    def __init__(
        self, csv_file, root_dir, transform=None, test=False, default_size=224
    ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.default_size = int(default_size)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        try:
            image = Image.open(img_name).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.default_size, self.default_size), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 11
def build_transform_for_model(model_name: str, is_train: bool, img_size: int = 224):
    try:
        from timm.data import resolve_data_config, create_transform

        cfg = resolve_data_config(
            {}, model=timm.create_model(model_name, pretrained=False, num_classes=5)
        )
        cfg["input_size"] = (3, img_size, img_size)
        tfm = create_transform(**cfg, is_training=is_train)
        return tfm
    except Exception as e:
        print(
            f"[WARN] Could not build timm transform for '{model_name}' ({type(e).__name__}: {e}). Using fallback."
        )
        return transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 12
train_csv_file = os.path.join(data_dir, "train.csv")
train_root_dir = os.path.join(data_dir, "train_images")

train_df = pd.read_csv(train_csv_file)

rng = np.random.RandomState(0)
val_frac = 0.15
val_indices = []
for cls in sorted(train_df["diagnosis"].unique()):
    idxs = np.where(train_df["diagnosis"].values == cls)[0]
    rng.shuffle(idxs)
    take = max(1, int(round(len(idxs) * val_frac)))
    val_indices.extend(idxs[:take])
val_indices = np.array(sorted(val_indices))
is_val = np.zeros(len(train_df), dtype=bool)
is_val[val_indices] = True

train_split_path = "/kaggle/working/train_split.csv"
val_split_path = "/kaggle/working/val_split.csv"
train_df.loc[~is_val, ["id_code", "diagnosis"]].to_csv(train_split_path, index=False)
train_df.loc[is_val, ["id_code", "diagnosis"]].to_csv(val_split_path, index=False)


def seed_worker(worker_id):
    worker_seed = 0 + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)




## === cell 13
test_csv_file = os.path.join(data_dir, "test.csv")
test_root_dir = os.path.join(data_dir, "test_images")



## === cell 14
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TIMM_OFFLINE", None)

model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/3/resnet18.pth",
}

model_names = {
    "resnet18": "resnet18.tv_in1k",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}




## === cell 15
def create_timm_model_safe(requested_name: str, pretrained: bool, num_classes: int):
    try:
        return (
            timm.create_model(
                requested_name, pretrained=pretrained, num_classes=num_classes
            ),
            requested_name,
        )
    except Exception as e1:
        fallbacks = [
            "resnet18.tv_in1k",
            "resnet34.tv_in1k",
            "resnet50.tv_in1k",
            "resnet18",
            "resnet50",
        ]
        for fb in fallbacks:
            if fb == requested_name:
                continue
            try:
                m = timm.create_model(
                    fb, pretrained=pretrained, num_classes=num_classes
                )
                print(
                    f"[WARN] Failed to create model '{requested_name}' ({type(e1).__name__}: {e1}). "
                    f"Falling back to timm model '{fb}'."
                )
                return m, fb
            except Exception:
                continue
        raise e1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_model_keys = []
resolved_model_names = {}

for model_key, path in model_paths.items():
    requested_name = model_names[model_key]

    if os.path.exists(path):
        use_pretrained = False
    else:
        use_pretrained = True
        print(
            f"[WARN] Weights not found at {path}. Using timm pretrained weights for {requested_name} (then training)."
        )

    model, used_name = create_timm_model_safe(
        requested_name, pretrained=use_pretrained, num_classes=5
    )
    resolved_model_names[model_key] = used_name

    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

    model.to(device)
    models_list.append(model)
    loaded_model_keys.append(model_key)

print(f"[INFO] Resolved timm model names: {resolved_model_names}")

primary_model_name = resolved_model_names[loaded_model_keys[0]]
train_transform = build_transform_for_model(
    primary_model_name, is_train=True, img_size=224
)
eval_transform = build_transform_for_model(
    primary_model_name, is_train=False, img_size=224
)

train_dataset = BlindnessDataset(
    train_split_path,
    train_root_dir,
    transform=train_transform,
    test=False,
    default_size=224,
)
val_dataset = BlindnessDataset(
    val_split_path,
    train_root_dir,
    transform=eval_transform,
    test=False,
    default_size=224,
)
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=eval_transform, test=True, default_size=224
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=seed_worker,
    generator=g,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=seed_worker,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=seed_worker,
)




## === cell 16
def train_one_model(model, train_loader, val_loader, epochs=2, lr=3e-4):
    model.train()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in tqdm(train_loader, desc=f"train ep{ep+1}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu()) * x.size(0)
            n += x.size(0)
        train_loss = running / max(1, n)

        model.eval()
        v_running = 0.0
        v_n = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"val ep{ep+1}/{epochs}", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = crit(logits, y)
                v_running += float(loss.detach().cpu()) * x.size(0)
                v_n += x.size(0)
        val_loss = v_running / max(1, v_n)
        print(
            f"[INFO] epoch {ep+1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

    model.eval()
    return model


for model_key, model in zip(loaded_model_keys, models_list):
    path = model_paths[model_key]
    if not os.path.exists(path):
        models_list[loaded_model_keys.index(model_key)] = train_one_model(
            model, train_loader, val_loader, epochs=2, lr=3e-4
        )




## === cell 17
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() == 0:
        return 0.0
    E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def predict_proba_single_model(model, loader):
    model.eval()
    probs_all = []
    y_all = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                y_all.append(y.numpy())
            else:
                x = batch
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = nn.functional.softmax(logits, dim=1)
            probs_all.append(probs.detach().cpu().numpy())
    probs_all = np.concatenate(probs_all, axis=0)
    y_all = np.concatenate(y_all, axis=0) if y_all else None
    return probs_all, y_all


def apply_thresholds(exp_values, thr):
    thr = np.asarray(thr, dtype=float)
    exp_values = np.asarray(exp_values, dtype=float)
    return (exp_values[:, None] > thr[None, :]).sum(axis=1).astype(int)


def find_best_thresholds(y_true, exp_values):
    y_true = np.asarray(y_true, dtype=int)
    exp_values = np.asarray(exp_values, dtype=float)

    t = np.array([0.5, 1.5, 2.5, 3.5], dtype=float)

    def score(thr):
        pred = apply_thresholds(exp_values, thr)
        return quadratic_weighted_kappa(y_true, pred, n_classes=5)

    best = score(t)

    for _ in range(2):
        for i in range(4):
            lo = 0.0 if i == 0 else t[i - 1] + 0.05
            hi = 4.0 if i == 3 else t[i + 1] - 0.05
            if hi <= lo:
                continue
            candidates = np.linspace(max(lo, t[i] - 0.5), min(hi, t[i] + 0.5), 21)
            for c in candidates:
                tt = t.copy()
                tt[i] = float(c)
                s = score(tt)
                if s > best:
                    best = s
                    t = tt
    return t, best




## === cell 18
trained_paths = {}
val_kappa_by_model = {}
best_post_by_model = {}  # store either ("argmax", None) or ("threshold", thr)

for model_key, path in model_paths.items():
    model_idx = loaded_model_keys.index(model_key)
    model = models_list[model_idx]

    model.eval()
    val_probs, val_y = predict_proba_single_model(model, val_loader)

    val_pred_argmax = np.argmax(val_probs, axis=1).astype(int)
    k_argmax = quadratic_weighted_kappa(val_y, val_pred_argmax, n_classes=5)

    exp_val = (val_probs * np.arange(5)[None, :]).sum(axis=1)
    thr, k_thr = find_best_thresholds(val_y, exp_val)

    if k_thr > k_argmax:
        best_post_by_model[model_key] = ("threshold", thr)
        val_kappa_by_model[model_key] = float(k_thr)
        print(
            f"[INFO] {model_key} val_kappa(thresholded)={k_thr:.4f} (argmax={k_argmax:.4f})"
        )
    else:
        best_post_by_model[model_key] = ("argmax", None)
        val_kappa_by_model[model_key] = float(k_argmax)
        print(
            f"[INFO] {model_key} val_kappa(argmax)={k_argmax:.4f} (thresholded={k_thr:.4f})"
        )



## === cell 19
for k in loaded_model_keys:
    best_post_by_model.setdefault(k, ("argmax", None))

if not val_kappa_by_model:
    weights = {k: 1.0 / len(loaded_model_keys) for k in loaded_model_keys}
    print("[WARN] No validation kappas were computed; using uniform ensemble weights.")
else:
    eps = 1e-6
    scores = np.array(
        [max(eps, float(val_kappa_by_model.get(k, eps))) for k in loaded_model_keys],
        dtype=float,
    )
    scores = np.clip(scores, eps, None)
    scores = scores / scores.sum()
    weights = {k: float(w) for k, w in zip(loaded_model_keys, scores)}
print(f"[INFO] Ensemble weights: {weights}")
print(
    f"[INFO] Best post-processing per model: { {k: (v[0], None if v[1] is None else np.round(v[1],4).tolist()) for k,v in best_post_by_model.items()} }"
)



## === cell 20
val_probs_ens = None
val_y_ref = None
for model_key, model in zip(loaded_model_keys, models_list):
    probs, val_y = predict_proba_single_model(model, val_loader)
    if val_probs_ens is None:
        val_probs_ens = weights[model_key] * probs
        val_y_ref = val_y
    else:
        val_probs_ens += weights[model_key] * probs

val_pred_ens_argmax = np.argmax(val_probs_ens, axis=1).astype(int)
k_ens_argmax = quadratic_weighted_kappa(val_y_ref, val_pred_ens_argmax, n_classes=5)

exp_val_ens = (val_probs_ens * np.arange(5)[None, :]).sum(axis=1)
ensemble_thresholds, k_ens_thr = find_best_thresholds(val_y_ref, exp_val_ens)

use_ensemble_thresholds = bool(k_ens_thr > k_ens_argmax)
print(
    f"[INFO] ensemble val_kappa(argmax)={k_ens_argmax:.4f} "
    f"val_kappa(thresholded)={k_ens_thr:.4f} use_thresholds={use_ensemble_thresholds}"
)
if use_ensemble_thresholds:
    print(f"[INFO] ensemble learned thresholds={ensemble_thresholds.round(4).tolist()}")



## === cell 21
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="infer", leave=False):
        images = images.to(device, non_blocking=True)

        outputs = []
        for model_key, model in zip(loaded_model_keys, models_list):
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1)
            outputs.append(weights[model_key] * probs.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        weighted_outputs = torch.sum(outputs, dim=0)
        all_outputs.append(weighted_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)

final_predictions = np.argmax(all_outputs, axis=1).astype(int)

if use_ensemble_thresholds:
    exp_test = (all_outputs * np.arange(5)[None, :]).sum(axis=1)
    final_predictions = apply_thresholds(exp_test, ensemble_thresholds)

final_predictions = np.clip(final_predictions, 0, 4).astype(int)



## === cell 22
sample_path = os.path.join(data_dir, "sample_submission.csv")
sample_df = pd.read_csv(sample_path)

test_ids = pd.read_csv(test_csv_file)["id_code"].values
assert len(test_ids) == len(final_predictions), (len(test_ids), len(final_predictions))

pred_map = dict(zip(test_ids.tolist(), final_predictions.tolist()))
submission_df = sample_df.copy()
submission_df["diagnosis"] = submission_df["id_code"].map(pred_map).astype("Int64")

if submission_df["diagnosis"].isna().any():
    n_bad = int(submission_df["diagnosis"].isna().sum())
    print(
        f"[WARN] {n_bad} id_codes in sample_submission not found in predictions; filling with 0."
    )
    submission_df["diagnosis"] = submission_df["diagnosis"].fillna(0).astype(int)
else:
    submission_df["diagnosis"] = submission_df["diagnosis"].astype(int)

assert (
    submission_df.shape[0] == sample_df.shape[0]
), f"Unexpected submission rows: {submission_df.shape[0]}"
assert (
    submission_df["id_code"].tolist() == sample_df["id_code"].tolist()
), "Submission id_code order mismatch vs sample_submission.csv"

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

assert (
    os.path.exists(submission_path) and os.path.getsize(submission_path) > 0
), "submission.csv was not written correctly."

print(
    f"Wrote {submission_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
)
print(submission_df.head())
