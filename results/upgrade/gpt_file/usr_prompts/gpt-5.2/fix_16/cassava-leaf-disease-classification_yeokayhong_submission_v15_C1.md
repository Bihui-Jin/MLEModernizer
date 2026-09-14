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
from torch.utils.data import (
    DataLoader,
)  # unused but kept to preserve original structure
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
import glob

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = (
    False  # benchmark implies potential nondet; keep as original intent
)




## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/4/efficientnet_v2_l_480_8591_ISP_CBP.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/5/vit_h_14_518_8369_base.pth"
)
vit_image_size = 518

model_select = "en"

if model_select == "vit":
    model_image_size = vit_image_size
if model_select == "en":
    model_image_size = en_image_size




## === cell 2
def invert_square_pad(img):
    width, height = img.size

    center_width, center_height = width // 2, height // 2
    top_left = img.crop((0, 0, center_width, center_height))
    top_right = img.crop((center_width, 0, width, center_height))
    bottom_left = img.crop((0, center_height, center_width, height))
    bottom_right = img.crop((center_width, center_height, width, height))

    top_combined = Image.new("RGB", (width, center_height))
    top_combined.paste(bottom_right, (0, 0))
    top_combined.paste(bottom_left, (center_width, 0))

    bottom_combined = Image.new("RGB", (width, center_height))
    bottom_combined.paste(top_right, (0, 0))
    bottom_combined.paste(top_left, (center_width, 0))

    flipped_img = Image.new("RGB", (width, height))
    flipped_img.paste(top_combined, (0, 0))
    flipped_img.paste(bottom_combined, (0, center_height))

    img = flipped_img.copy()
    del top_combined, bottom_combined, flipped_img

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
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

val_transforms = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((model_image_size, model_image_size)),
        v2.CenterCrop((model_image_size, model_image_size)),
        v2.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)

train_transforms_fallback = val_transforms




## === cell 4
def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_prefix(state_dict, prefixes=("module.", "model.", "net.")):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _find_existing_weight_path(weight_path: str):
    if weight_path and os.path.exists(weight_path):
        return weight_path
    if not weight_path:
        return None
    base_dir = os.path.dirname(weight_path)
    if os.path.isdir(base_dir):
        cands = glob.glob(os.path.join(base_dir, "*.pth")) + glob.glob(
            os.path.join(base_dir, "*.pt")
        )
        if len(cands) == 1:
            return cands[0]
        for c in cands:
            if os.path.basename(c) == os.path.basename(weight_path):
                return c
    return None


def _safe_load_state_dict(model, weight_path, device):
    weight_path = _find_existing_weight_path(weight_path)
    if weight_path and os.path.exists(weight_path):
        try:
            state = torch.load(weight_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(weight_path, map_location=device)

        state = _unwrap_state_dict(state)
        state = _strip_prefix(state)

        if isinstance(state, dict) and any(
            k.startswith("model.") for k in state.keys()
        ):
            state = _strip_prefix(state, prefixes=("model.",))

        try:
            model.load_state_dict(state, strict=True)
            return True
        except RuntimeError:
            missing, unexpected = model.load_state_dict(state, strict=False)
            return not (len(missing) > 0 and len(unexpected) > 0)
    return False


@torch.no_grad()
def _init_head_from_imagenet_by_feature_clustering(
    backbone_model, head_linear, train_loader, device, num_classes=5, max_batches=200
):
    backbone_model.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    for bi, (xb, yb) in enumerate(train_loader):
        if bi >= max_batches:
            break
        xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device=device, dtype=torch.long, non_blocking=True)

        feats = backbone_model.features(xb)
        feats = backbone_model.avgpool(feats)
        feats = torch.flatten(feats, 1)  # [B, C]
        if sums is None:
            sums = torch.zeros(
                (num_classes, feats.shape[1]), dtype=feats.dtype, device=device
            )

        for c in range(num_classes):
            m = yb == c
            if m.any():
                sums[c] += feats[m].sum(dim=0)
                counts[c] += int(m.sum().item())

    if sums is None or int(counts.sum().item()) == 0:
        return False

    protos = sums / counts.clamp_min(1).unsqueeze(1)  # [5, C]

    W_im = head_linear.weight.detach().clone()  # [1000, C]
    Wn = torch.nn.functional.normalize(W_im, dim=1)
    Pn = torch.nn.functional.normalize(protos, dim=1)
    sims = Pn @ Wn.T  # [5, 1000]
    nn_idx = torch.argmax(sims, dim=1)  # [5]

    new_W = W_im[nn_idx].contiguous()
    if head_linear.bias is not None:
        b_im = head_linear.bias.detach().clone()
        new_b = b_im[nn_idx].contiguous()
    else:
        new_b = None

    return new_W, new_b


if model_select == "vit":
    vit_model = models.vit_h_14(weights=None, image_size=518)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )
    loaded = _safe_load_state_dict(vit_model, vit_model_path, device)
    if not loaded:
        vit_model = models.vit_h_14(
            weights=models.ViT_H_14_Weights.DEFAULT, image_size=518
        )
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
    vit_model.to(device)
    vit_model.eval()

if model_select == "en":
    en_model = models.efficientnet_v2_l(weights=None)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )
    loaded = _safe_load_state_dict(en_model, en_model_path, device)

    if not loaded:
        en_pre = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.DEFAULT
        )

        pretrained_head = en_pre.classifier[1]  # Linear(?, 1000)
        in_features = pretrained_head.in_features
        en_pre.classifier[1] = torch.nn.Linear(in_features, num_classes)

        for p in en_pre.features.parameters():
            p.requires_grad = False
        for p in en_pre.classifier[0].parameters():
            p.requires_grad = False
        for p in en_pre.classifier[1].parameters():
            p.requires_grad = True

        en_pre.classifier[0] = torch.nn.Identity()

        en_model = en_pre
        en_model.to(device)

        train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
        train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
        train_df = pd.read_csv(train_csv_path)

        class CassavaTrainDS(torch.utils.data.Dataset):
            def __init__(self, df, img_dir, tfm):
                self.df = df.reset_index(drop=True)
                self.img_dir = img_dir
                self.tfm = tfm

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                img_path = os.path.join(self.img_dir, row["image_id"])
                img = Image.open(img_path).convert("RGB")
                x = self.tfm(img)
                y = int(row["label"])
                return x, y

        g = torch.Generator()
        g.manual_seed(42)

        train_ds = CassavaTrainDS(train_df, train_img_dir, train_transforms_fallback)

        _cpu_workers = os.cpu_count() or 2
        _train_workers = min(8, max(2, _cpu_workers // 2))
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=_train_workers,
            pin_memory=torch.cuda.is_available(),
            generator=g,
            drop_last=False,
            persistent_workers=(_train_workers > 0),
            prefetch_factor=4 if _train_workers > 0 else None,
        )

        with torch.no_grad():
            tmp_linear_1000 = pretrained_head.to(device)
            init = _init_head_from_imagenet_by_feature_clustering(
                en_model,
                tmp_linear_1000,
                train_loader,
                device,
                num_classes=num_classes,
                max_batches=200,
            )
            if init is not False:
                new_W, new_b = init
                en_model.classifier[1].weight.data.copy_(new_W)
                if new_b is not None and en_model.classifier[1].bias is not None:
                    en_model.classifier[1].bias.data.copy_(new_b)

        label_counts = (
            train_df["label"].value_counts().reindex(range(num_classes), fill_value=0)
        )
        counts = label_counts.values.astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / counts
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        optimizer = torch.optim.AdamW(
            en_model.classifier[1].parameters(), lr=2e-4, weight_decay=5e-3
        )
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        en_model.features.eval()
        en_model.avgpool.eval()
        en_model.classifier[0].eval()
        en_model.classifier[1].train()

        num_epochs_fallback = 3
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs_fallback
        )

        for epoch in range(num_epochs_fallback):
            pbar = tqdm(
                train_loader,
                desc=f"Fitting fallback classifier ({epoch+1}/{num_epochs_fallback})",
            )
            for xb, yb in pbar:
                xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
                yb = yb.to(device=device, dtype=torch.long, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    feats = en_model.features(xb)
                    feats = en_model.avgpool(feats)
                    feats = torch.flatten(feats, 1)
                    feats = en_model.classifier[0](feats)  # Identity
                feats = feats.detach().clone()

                out = en_model.classifier[1](feats)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

                if (pbar.n + 1) % 100 == 0:
                    pbar.set_postfix(
                        loss=float(loss.item()), lr=optimizer.param_groups[0]["lr"]
                    )

            scheduler.step()

    en_model.to(device)
    en_model.eval()




## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_df = pd.read_csv(sample_path)
test_image_ids = sample_df["image_id"].tolist()


class CassavaTestDS(torch.utils.data.Dataset):
    def __init__(self, image_ids, img_dir, tfm):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        image_path = os.path.join(self.img_dir, image_name)
        img = Image.open(image_path).convert("RGB")
        x = self.tfm(img)
        return x, image_name


def _test_collate(batch):
    xs, ids = zip(*batch)
    return torch.stack(xs, 0), list(ids)


_cpu_workers = os.cpu_count() or 2
_test_workers = min(8, max(2, _cpu_workers // 2))
test_ds = CassavaTestDS(test_image_ids, test_data_directory, val_transforms)
test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=_test_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    persistent_workers=(_test_workers > 0),
    prefetch_factor=4 if _test_workers > 0 else None,
    collate_fn=_test_collate,
)

if model_select == "en":
    en_model = en_model.to(memory_format=torch.channels_last)
elif model_select == "vit":
    vit_model = vit_model  # no-op

n_test = len(test_ds)
pred_arr = np.empty(n_test, dtype=np.int64)
id_arr = [None] * n_test

write_pos = 0
with torch.inference_mode():
    for xb, batch_ids in tqdm(test_loader, desc="Test"):
        if xb.is_cuda:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            xb = xb.to(device=device, dtype=torch.float32)
        if model_select == "en":
            xb = xb.contiguous(memory_format=torch.channels_last)

        if model_select == "vit":
            output = vit_model(xb)
        if model_select == "en":
            output = en_model(xb)

        predicted_class = (
            torch.argmax(output, dim=1).to("cpu", non_blocking=False).numpy()
        )
        bs = predicted_class.shape[0]
        pred_arr[write_pos : write_pos + bs] = predicted_class
        id_arr[write_pos : write_pos + bs] = batch_ids
        write_pos += bs

predictions = pred_arr.tolist()
image_ids = id_arr




## === cell 6
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

submission_df = sample_df[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(0).astype(int)

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
