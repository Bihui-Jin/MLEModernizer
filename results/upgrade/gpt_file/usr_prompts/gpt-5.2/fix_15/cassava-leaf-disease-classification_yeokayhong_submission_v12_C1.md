# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from torchvision import models, transforms
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
import glob
import random



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/3/efficientnetv2_l_480_8450.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/5/vit_h_14_518_8718_CBP.pth"
)
vit_image_size = 518

model_select = "vit"

if model_select == "vit":
    model_image_size = vit_image_size
if model_select == "en":
    model_image_size = en_image_size




## === cell 2
def invert_square_pad(img: Image.Image) -> Image.Image:
    width, height = img.size
    img = img.transpose(Image.Transpose.ROTATE_180)
    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,  # left
        (max_side - height) // 2,  # top
        (max_side - width) - (max_side - width) // 2,  # right
        (max_side - height) - (max_side - height) // 2,  # bottom
    )
    padded_img = transforms.functional.pad(img, padding, padding_mode="reflect")
    return padded_img




## === cell 3
val_transforms = transforms.Compose(
    [
        v2.Lambda(invert_square_pad),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((model_image_size, model_image_size)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## === cell 4
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def resolve_checkpoint_path(preferred_path: str, search_root: str, patterns):
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    if isinstance(patterns, str):
        patterns = [patterns]

    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(search_root, pat), recursive=False))
        if candidates:
            break

    if not candidates:
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(search_root, pat), recursive=True))
            if candidates:
                break

    candidates = sorted(set(candidates))
    if not candidates:
        return None
    return candidates[-1]


def load_state_dict_safely(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CassavaTrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_dir, transform, max_items=None):
        df = pd.read_csv(csv_path)
        if max_items is not None:
            df = df.iloc[:max_items].reset_index(drop=True)
        self.df = df
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image_id"])
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = int(row["label"])
        return x, y


def finetune_vit_head_from_images(
    vit_model,
    image_size,
    steps=200,
    lr=3e-4,
    batch_size=16,
    max_train_items=4096,
):
    train_csv = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"

    if (not os.path.isfile(train_csv)) or (not os.path.isdir(train_img_dir)):
        print("Train CSV or train_images not found; skipping finetune.")
        return vit_model, False

    weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose(
        [
            v2.Lambda(invert_square_pad),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((image_size, image_size)),
            v2.Normalize(mean, std),
        ]
    )

    ds = CassavaTrainDataset(
        csv_path=train_csv,
        images_dir=train_img_dir,
        transform=train_transform,
        max_items=max_train_items,
    )

    g = torch.Generator()
    g.manual_seed(0)

    cpu_cnt = os.cpu_count() or 2
    num_workers = min(8, cpu_cnt)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=g,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    for p in vit_model.parameters():
        p.requires_grad = False
    for p in vit_model.heads.head.parameters():
        p.requires_grad = True

    vit_model.train()
    vit_model.to(device)

    if device.type == "cuda":
        vit_model = vit_model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(vit_model.heads.head.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    data_iter = iter(dl)
    for step in range(steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        y = torch.as_tensor(y, dtype=torch.long).to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = vit_model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if (step + 1) % 50 == 0 or step == 0:
            with torch.no_grad():
                acc = (logits.argmax(1) == y).float().mean().item()
            print(
                f"Fallback finetune step {step+1}/{steps} - loss {loss.item():.4f} - acc {acc:.4f}"
            )

    vit_model.eval()
    return vit_model, True


vit_model = None
en_model = None

using_imagenet_fallback = False
performed_fallback_finetune = False

if model_select == "vit":
    vit_model_path_resolved = resolve_checkpoint_path(
        vit_model_path,
        "/kaggle/input",
        patterns=[
            "vit_l_cassava/**/vit_h_14_518_8718_CBP.pth",
            "**/vit_h_14_518_8718_CBP.pth",
            "**/*vit*h*14*518*.pth",
            "**/*vit*_h*_14*_518*.pth",
            "**/*vit*.pth",
        ],
    )

    if vit_model_path_resolved is not None:
        vit_model = models.vit_h_14(weights=None, image_size=518)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        state = load_state_dict_safely(vit_model_path_resolved)
        vit_model.load_state_dict(state, strict=True)
        print("Loaded ViT checkpoint:", vit_model_path_resolved)
        using_imagenet_fallback = False
    else:
        weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
        vit_model = models.vit_h_14(weights=weights, image_size=518)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        print(
            "ViT checkpoint not found; using torchvision ImageNet weights and finetuning 5-class head on cassava train_images."
        )
        using_imagenet_fallback = True

        try:
            vit_model, performed_fallback_finetune = finetune_vit_head_from_images(
                vit_model,
                image_size=518,
                steps=200,
                lr=3e-4,
                batch_size=16,
                max_train_items=4096,
            )
        except Exception as e:
            print(
                "Fallback finetune failed; continuing without finetune. Error:", repr(e)
            )
            performed_fallback_finetune = False

    vit_model.to(device)
    vit_model.eval()
    if device.type == "cuda":
        vit_model = vit_model.to(memory_format=torch.channels_last)

if model_select == "en":
    en_model_path_resolved = resolve_checkpoint_path(
        en_model_path,
        "/kaggle/input",
        patterns=[
            "efficientnetv2-large-test/**/*efficientnet*v2*l*480*.pth",
            "**/*efficientnet*v2*l*480*.pth",
            "**/*efficientnet*v2*l*.pth",
            "**/*efficientnet*.pth",
            "**/*efficientnet*.pt",
        ],
    )

    if en_model_path_resolved is not None:
        en_model = models.efficientnet_v2_l(weights=None)
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
        state = load_state_dict_safely(en_model_path_resolved)
        en_model.load_state_dict(state, strict=True)
        print("Loaded EfficientNetV2-L checkpoint:", en_model_path_resolved)
        using_imagenet_fallback = False
    else:
        weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        en_model = models.efficientnet_v2_l(weights=weights)
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
        print(
            "EfficientNet checkpoint not found; using torchvision ImageNet weights with new 5-class head (no finetune path here)."
        )
        using_imagenet_fallback = True

    en_model.to(device)
    en_model.eval()
    if device.type == "cuda":
        en_model = en_model.to(memory_format=torch.channels_last)

if using_imagenet_fallback:
    if model_select == "vit":
        mean = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms().mean
        std = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms().std
    else:
        mean = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms().mean
        std = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms().std

    val_transforms = transforms.Compose(
        [
            v2.Lambda(invert_square_pad),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((model_image_size, model_image_size)),
            v2.Normalize(mean, std),
        ]
    )

print(
    "using_imagenet_fallback:",
    using_imagenet_fallback,
    "performed_fallback_finetune:",
    performed_fallback_finetune,
)



## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_df = pd.read_csv(sample_path)
expected_ids = sample_df["image_id"].tolist()

predict_ids = expected_ids


try:
    from torchvision.io import read_image as tv_read_image

    _HAS_TV_IO = True
except Exception:
    _HAS_TV_IO = False


class CassavaTestDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, image_ids):
        self.images_dir = images_dir
        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, image_name)

        if _HAS_TV_IO:
            t = tv_read_image(image_path)  # C,H,W
            arr = t.permute(1, 2, 0).contiguous().numpy()  # H,W,C uint8
        else:
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.uint8)

        return arr, image_name


def _make_batch_collate(image_size: int, normalize_mean, normalize_std):
    mean = torch.tensor(normalize_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(normalize_std, dtype=torch.float32).view(1, 3, 1, 1)

    pad_cache = {}  # (h,w) -> (pad_l,pad_r,pad_t,pad_b)

    def collate_fn(batch):
        arrs, names = zip(*batch)  # uint8 HWC
        x = torch.from_numpy(np.stack(arrs, axis=0))  # NHWC uint8
        x = x.permute(0, 3, 1, 2).contiguous()  # NCHW
        x = x.to(dtype=torch.float32).div_(255.0)

        x = torch.flip(x, dims=(2, 3))

        n, c, h, w = x.shape
        if h != w:
            key = (h, w)
            pads = pad_cache.get(key)
            if pads is None:
                m = h if h > w else w
                pad_l = (m - w) // 2
                pad_r = (m - w) - pad_l
                pad_t = (m - h) // 2
                pad_b = (m - h) - pad_t
                pads = (pad_l, pad_r, pad_t, pad_b)
                pad_cache[key] = pads
            x = torch.nn.functional.pad(x, pads, mode="reflect")

        x = torch.nn.functional.interpolate(
            x, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        x = (x - mean) / std
        return x, names  # keep tuple of names; order preserved

    return collate_fn


_norm_mean, _norm_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
if using_imagenet_fallback:
    if model_select == "vit":
        _norm_mean = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms().mean
        _norm_std = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms().std
    else:
        _norm_mean = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms().mean
        _norm_std = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms().std

ds_test = CassavaTestDataset(test_data_directory, predict_ids)

cpu_cnt = os.cpu_count() or 2
batch_size = 32 if device.type == "cuda" else 8

if device.type == "cuda":
    num_workers = min(8, cpu_cnt)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
else:
    num_workers = min(4, cpu_cnt)

dl_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    pin_memory_device=("cuda" if torch.cuda.is_available() else ""),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=_make_batch_collate(model_image_size, _norm_mean, _norm_std),
)

with torch.inference_mode():
    if model_select == "vit":
        vit_model.eval()
    else:
        en_model.eval()

predictions = []
image_ids = []

with torch.inference_mode():
    for xb, names in tqdm(dl_test, desc="Test", total=len(dl_test)):
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)

        if model_select == "vit":
            out = vit_model(xb)
        else:
            out = en_model(xb)

        preds = out.argmax(dim=1).cpu().tolist()
        predictions.extend(preds)
        image_ids.extend(list(names))

pred_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

submission_df = sample_df[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    fill_label = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    submission_df["label"] = submission_df["label"].fillna(fill_label).astype(int)
else:
    submission_df["label"] = submission_df["label"].astype(int)

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df), "Expected:", len(sample_df))
