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

# 5. Target score

0.8644605621033545

# 6. Current score

0.56689

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09268) has done: 'I remove the TensorFlow import that triggers the protobuf `MessageFactory.GetPrototype` error, since it’s unused in this PyTorch-only inference script. I also make the model weight loading robust: if the external `.pth` paths don’t exist, the code fall back to torchvision pretrained weights so it can run end-to-end and produce a valid submission. To fix the CUDA/CPU mismatch error, I ensure the model and input tensors are on the same device and dtype. Finally, I keep the same inference loop and submission schema, only adding a safety check to keep output rows aligned with `sample_submission.csv`.'
- What this solution (achieved 0.11921) has done: 'Your current score is far below the target, so the most likely issue is not the model choice but a preprocessing mismatch: the Cassava EfficientNet/Vit checkpoints typically expect ImageNet normalization, while your pipeline uses `mean=0.5/std=0.5`, which can collapse accuracy. I keep the same model architecture and inference loop, but switch normalization to the correct torchvision pretrained (ImageNet) normalization corresponding to the selected model weights, which is a minimal semantic fix aligned with the original approach. I also ensure the model’s expected resize/crop behavior is consistent by using the selected weights’ recommended transforms when available (still the same single-image loop). These changes should substantially increase accuracy toward your target without changing the core logic.'
- What this solution (achieved 0.75747) has done: 'Your score is far below the target, so the most likely issue is that the custom Cassava checkpoint is not actually being used: when the external `.pth` isn’t found, the code falls back to ImageNet weights but then replaces the classifier head with a fresh random 5-class layer, which collapse accuracy (near-random). I keep the same model (EfficientNet-V2-L) and the same inference loop, but make the “fallback” case produce valid 5-class predictions by using the pretrained backbone as a fixed feature extractor and calibrating a simple linear head on the provided `train.csv` (no change to architecture; just fitting the existing final linear layer). I also make weight loading tolerant to common checkpoint key prefixes (`module.`, `model.`) so the Cassava checkpoint loads if it exists but was previously failing strict key matching. These are minimal, execution-safe changes that should move accuracy substantially toward your target while preserving evaluation semantics and producing the same `submission.csv` format.'
- What this solution (achieved 0.11921) has done: 'Your current score (0.75747) is below the target (0.86446), so we should make a small accuracy-improving fix without changing the overall approach. The biggest low-risk gain here is to make the *fallback head fitting* actually train only the classifier on top of frozen features: right now you freeze parameters but still run the full model in train mode, so BatchNorm/dropout inside the backbone can drift and hurt generalization. I keep the exact same model, transforms, and single-epoch SGD head fitting, but run the feature extractor in `eval()` and compute features under `no_grad()` while training only the final linear layer. This typically improves the fallback quality (and thus Kaggle accuracy) while preserving your core logic and still producing the same `submission.csv`.'
- What this solution (achieved 0.44096) has done: 'I fix the fallback head-fitting path that currently crashes due to a shape mismatch: `efficientnet_v2_l.features(x)` returns a 4D feature map, but your `Linear(1280->5)` expects a pooled 2D tensor. I keep the same model and “train only the last linear layer for 1 epoch” approach, but route features through the model’s built-in `avgpool` + `flatten` (exactly what the original forward does) before the classifier head. I also keep the backbone in eval/no_grad during head fitting (as you intended) and ensure the final inference still writes a valid `submission.csv` aligned to `sample_submission.csv`. These changes are minimal, unblock execution, and should restore the stronger score you were getting before the crash.'
- What this solution (achieved 0.56689) has done: 'Your current score (0.44096) is far below the target (0.86446), so the likely problem is that the fallback “fit a new 5-class head” is too weak: it trains only 1 epoch with plain SGD at a high LR and no class-imbalance handling, which can underfit badly. Keeping the same model, transforms, and the exact same “train only the last linear layer on frozen features” approach, I (1) compute class weights from `train.csv` and use them in `CrossEntropyLoss` to better match the label distribution, and (2) switch the optimizer to AdamW with a conservative LR to stabilize convergence without changing the training loop structure. These are minimal changes confined to the fallback path, and inference/submission formatting stays identical.'

# 9. Code solution

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

torch.manual_seed(42)
np.random.seed(42)



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


def _safe_load_state_dict(model, weight_path, device):
    if weight_path and os.path.exists(weight_path):
        try:
            state = torch.load(weight_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(weight_path, map_location=device)

        state = _unwrap_state_dict(state)
        state = _strip_prefix(state)

        try:
            model.load_state_dict(state, strict=True)
            return True
        except RuntimeError:
            missing, unexpected = model.load_state_dict(state, strict=False)
            return not (len(missing) > 0 and len(unexpected) > 0)
    return False


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
        en_model = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.DEFAULT
        )
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )

        for p in en_model.features.parameters():
            p.requires_grad = False
        for p in en_model.classifier[0].parameters():
            p.requires_grad = False
        for p in en_model.classifier[1].parameters():
            p.requires_grad = True

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

        train_ds = CassavaTrainDS(train_df, train_img_dir, val_transforms)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            generator=g,
            drop_last=False,
        )

        label_counts = (
            train_df["label"].value_counts().reindex(range(num_classes), fill_value=0)
        )
        counts = label_counts.values.astype(np.float32)
        counts = np.maximum(counts, 1.0)  # safety
        weights = counts.sum() / counts  # inverse frequency
        weights = weights / weights.mean()  # keep scale reasonable
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        optimizer = torch.optim.AdamW(
            en_model.classifier[1].parameters(), lr=3e-4, weight_decay=1e-2
        )
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        en_model.features.eval()
        en_model.avgpool.eval()
        en_model.classifier[0].eval()
        en_model.classifier[1].train()

        for xb, yb in tqdm(train_loader, desc="Fitting fallback classifier (1 epoch)"):
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                feats = en_model.features(xb)  # (N, 1280, H, W)
                feats = en_model.avgpool(feats)  # (N, 1280, 1, 1)
                feats = torch.flatten(feats, 1)  # (N, 1280)

            feats = en_model.classifier[0](feats)  # dropout (or identity)
            out = en_model.classifier[1](feats)  # (N, 5)

            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    en_model.to(device)
    en_model.eval()



## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_df = pd.read_csv(sample_path)
test_image_ids = sample_df["image_id"].tolist()

predictions = []
image_ids = []

for image_name in tqdm(test_image_ids, desc="Test"):
    image_path = os.path.join(test_data_directory, image_name)

    image = Image.open(image_path).convert("RGB")
    transformed_image = (
        val_transforms(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    )

    with torch.no_grad():
        if model_select == "vit":
            output = vit_model(transformed_image)
        if model_select == "en":
            output = en_model(transformed_image)
        predicted_class = torch.argmax(output, dim=1)

    predictions.append(int(predicted_class.item()))
    image_ids.append(image_name)



## === cell 6
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

submission_df = sample_df[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(0).astype(int)

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
