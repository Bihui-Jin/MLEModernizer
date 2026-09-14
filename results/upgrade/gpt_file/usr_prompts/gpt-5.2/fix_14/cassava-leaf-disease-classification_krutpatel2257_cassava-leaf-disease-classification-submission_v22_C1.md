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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
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
import os
import json

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 1
model_path = "../input/en-b4-tta-calr-15-v2/model(13).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

assert os.path.exists(sample_sub_path), f"Missing sample submission: {sample_sub_path}"
assert os.path.isdir(test_images_path), f"Missing test images dir: {test_images_path}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"
assert os.path.isdir(
    train_images_path
), f"Missing train images dir: {train_images_path}"

has_ckpt = os.path.exists(model_path)
print("Checkpoint exists:", has_ckpt, "| path:", model_path)



## === cell 2
if has_ckpt:
    model = models.efficientnet_b4(weights=None)
    eff_weights = None
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)
else:
    eff_weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=eff_weights)

model.to(device)

if has_ckpt:
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    def _remap_keys_for_torchvision_efficientnet(state_dict):
        """
        Map common EfficientNet-PyTorch naming to torchvision EfficientNet naming.
        Minimal remap to make checkpoint loadable for inference.
        """
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            for prefix in ("module.", "model.", "net."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]

            nk = nk.replace("_fc.", "classifier.1.")
            nk = nk.replace("_conv_stem.", "features.0.0.")
            nk = nk.replace("_bn0.", "features.0.1.")
            nk = nk.replace("_blocks.", "features.1.")
            nk = nk.replace("_conv_head.", "features.8.0.")
            nk = nk.replace("_bn1.", "features.8.1.")

            new_sd[nk] = v
        return new_sd

    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        print(
            "Direct load_state_dict failed; attempting key remap. Error:", str(e)[:300]
        )
        state = _remap_keys_for_torchvision_efficientnet(state)
        missing, unexpected = model.load_state_dict(state, strict=False)

    print("Checkpoint loaded.")
    print("Missing keys (count):", len(missing))
    print("Unexpected keys (count):", len(unexpected))

model.eval()



## === cell 3
if eff_weights is not None:
    mean = eff_weights.transforms().mean
    std = eff_weights.transforms().std
    crop_size = 380
    resize_size = 456  # standard for 380 crop
else:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_size = 256
    resize_size = 288

sub_aug = albumentations.Compose(
    [
        albumentations.SmallestMaxSize(max_size=resize_size),
        albumentations.CenterCrop(crop_size, crop_size, p=1.0),
        albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
    ],
    p=1.0,
)

_proto_base_no_norm = [albumentations.SmallestMaxSize(max_size=resize_size)]


def _proto_aug_for_xy(x_min: int, y_min: int):
    return albumentations.Compose(
        _proto_base_no_norm
        + [
            albumentations.Crop(
                x_min=x_min,
                y_min=y_min,
                x_max=x_min + crop_size,
                y_max=y_min + crop_size,
                p=1.0,
            ),
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ],
        p=1.0,
    )


corner = max(resize_size - crop_size, 0)
proto_aug_list = [
    albumentations.Compose(
        _proto_base_no_norm
        + [
            albumentations.CenterCrop(crop_size, crop_size, p=1.0),
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ],
        p=1.0,
    ),
    _proto_aug_for_xy(0, 0),  # top-left
    _proto_aug_for_xy(corner, 0),  # top-right
    _proto_aug_for_xy(0, corner),  # bottom-left
    _proto_aug_for_xy(corner, corner),  # bottom-right
]




## === cell 4
def _load_image_tensor(img_path: str, aug) -> torch.Tensor:
    image = np.array(Image.open(img_path).convert("RGB"))
    image = aug(image=image)["image"]  # float32 HWC normalized
    image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    return image


def _extract_features(x_bchw: torch.Tensor) -> torch.Tensor:
    feats = model.features(x_bchw)  # [B, C, h, w]
    feats = model.avgpool(feats)  # [B, C, 1, 1]
    feats = torch.flatten(feats, 1)  # [B, C]
    return feats


cassava_prototypes = None  # [5, C] in fallback mode

if not has_ckpt:
    train_df = pd.read_csv(train_csv_path).reset_index(drop=True)

    per_class = 4096
    batch_size = (
        16  # keep VRAM bounded; runtime still OK under typical Kaggle GPU limits
    )

    with torch.inference_mode():
        tmp_img = _load_image_tensor(
            os.path.join(train_images_path, train_df.iloc[0]["image_id"]),
            aug=sub_aug,
        )
        tmp_feat = _extract_features(tmp_img.unsqueeze(0).to(device))
        feat_dim = int(tmp_feat.shape[1])

    outlier_drop_frac = 0.12  # drop worst 12% per class; small, deterministic

    per_class_ids = []
    for cls in range(5):
        cls_df = train_df.loc[train_df["label"] == cls, ["image_id"]].copy()
        cls_df = cls_df.sample(frac=1.0, random_state=42 + cls).reset_index(drop=True)
        ids = cls_df["image_id"].head(per_class).tolist()
        per_class_ids.append(ids)

    def _compute_class_prototype(img_ids):
        proto_sum = torch.zeros(feat_dim, device=device)
        proto_cnt = 0.0

        with torch.inference_mode():
            for i in range(0, len(img_ids), batch_size):
                batch_ids = img_ids[i : i + batch_size]

                x_views = []
                view_owner = []
                for owner_idx, img_id in enumerate(batch_ids):
                    p = os.path.join(train_images_path, img_id)
                    for aug in proto_aug_list:
                        x_views.append(_load_image_tensor(p, aug=aug))
                        view_owner.append(owner_idx)

                xs = torch.stack(x_views, dim=0).to(device)
                feats = _extract_features(xs)  # [V, C]
                feats = F.normalize(feats, dim=1)

                view_owner_t = torch.tensor(view_owner, device=device, dtype=torch.long)
                n_imgs = len(batch_ids)

                img_feat_sum = torch.zeros(n_imgs, feat_dim, device=device)
                img_feat_cnt = torch.zeros(n_imgs, device=device)

                img_feat_sum.index_add_(0, view_owner_t, feats)
                img_feat_cnt.index_add_(
                    0,
                    view_owner_t,
                    torch.ones_like(view_owner_t, dtype=img_feat_cnt.dtype),
                )

                img_feats = img_feat_sum / img_feat_cnt.clamp_min(1.0).unsqueeze(1)
                img_feats = F.normalize(img_feats, dim=1)

                proto_sum += img_feats.sum(dim=0)
                proto_cnt += float(n_imgs)

        proto = proto_sum / max(proto_cnt, 1.0)
        proto = F.normalize(proto, dim=0)
        return proto

    def _filter_outliers(img_ids, proto):
        sims = []
        with torch.inference_mode():
            for i in range(0, len(img_ids), batch_size):
                batch_ids = img_ids[i : i + batch_size]
                x = torch.stack(
                    [
                        _load_image_tensor(
                            os.path.join(train_images_path, img_id), aug=sub_aug
                        )
                        for img_id in batch_ids
                    ],
                    dim=0,
                ).to(device)
                feats = _extract_features(x)
                feats = F.normalize(feats, dim=1)
                s = (feats @ proto.unsqueeze(1)).squeeze(1)  # [B]
                sims.append(s.detach().to("cpu"))
        sims = torch.cat(sims, dim=0).numpy()

        n = len(img_ids)
        keep_n = int(round(n * (1.0 - outlier_drop_frac)))
        keep_n = max(1, min(n, keep_n))

        keep_idx = np.argsort(-sims)[:keep_n]  # highest similarity
        keep_ids = [img_ids[j] for j in keep_idx.tolist()]
        return keep_ids, float(np.mean(sims)), float(np.min(sims)), float(np.max(sims))

    class_protos = []
    counts = []
    for cls in range(5):
        ids = per_class_ids[cls]

        proto0 = _compute_class_prototype(ids)
        keep_ids, mean_sim, min_sim, max_sim = _filter_outliers(ids, proto0)
        proto1 = _compute_class_prototype(keep_ids)

        class_protos.append(proto1)
        counts.append(len(keep_ids))

        print(
            f"class {cls}: ids={len(ids)} keep={len(keep_ids)} "
            f"| sim(mean/min/max)={mean_sim:.4f}/{min_sim:.4f}/{max_sim:.4f}"
        )

    cassava_prototypes = torch.stack(class_protos, dim=0)  # [5, C]
    cassava_prototypes = F.normalize(cassava_prototypes, dim=1)
    print("Built cassava prototypes from train set. kept counts:", counts)
    print("Prototype feature dim:", feat_dim)



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)

tta_count = 1  # keep identical semantics to original (no extra TTA on test)

predictions = []
with torch.inference_mode():
    for image_id in sample_sub["image_id"].tolist():
        img_path = os.path.join(test_images_path, image_id)
        image_pred = None

        for _ in range(tta_count):
            x = _load_image_tensor(img_path, aug=sub_aug).to(device)

            if cassava_prototypes is None:
                outputs = model(x.unsqueeze(0))  # [1,5]
            else:
                feats = _extract_features(x.unsqueeze(0))  # [1,C]
                feats = F.normalize(feats, dim=1)  # [1,C]
                outputs = feats @ cassava_prototypes.T  # [1,5]

            image_pred = outputs if image_pred is None else (image_pred + outputs)

        image_pred = image_pred / tta_count
        pred_label = int(torch.argmax(image_pred, dim=1).item())
        predictions.append([image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df = sample_sub[["image_id"]].merge(sub_df, on="image_id", how="left")
assert len(sub_df) == len(sample_sub), "Submission row count mismatch"
assert sub_df["label"].isna().sum() == 0, "Missing predictions for some test images"

sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Wrote submission.csv with rows:", len(sub_df))
print("Submission columns:", list(sub_df.columns))
print("Unique predicted labels:", sorted(sub_df["label"].unique().tolist()))
